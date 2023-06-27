import tensorflow as tf
import json
import gc 
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.mixed_precision as mixed_precision


from .tf_utils.schedules import OneCycleLR, ListedLR
from .tf_utils.callbacks import Snapshot, SWA
from .tf_utils.learners import FGM, AWP

from signet.dataset.utils import get_tfrec_dataset
from signet.models.feature_extractor import Cnn1dMhsaFeatureExtractor
from signet.losses.ctc import CTCLoss
from signet.configs.Conv1D_LSTM_CTC_Loss import Conv1D_LSTM_CTC_Loss
from signet.trainer.utils import ctc_decode
from signet.trainer.callbacks import LevenshteinCallback

from signet.losses.metrics import normalized_levenshtein_distance,word_accuracy
import os

def get_strategy(CFG: Conv1D_LSTM_CTC_Loss):
    if CFG.device == "GPU"  or CFG.device=="CPU":
        ngpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if ngpu>1:
            print("Using multi GPU")
            strategy = tf.distribute.MirroredStrategy()
        elif ngpu==1:
            print("Using single GPU")
            strategy = tf.distribute.get_strategy()
        else:
            print("Using CPU")
            strategy = tf.distribute.get_strategy()
            CFG.device = "CPU"

    AUTO     = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')
    
    return strategy, REPLICAS
    
def train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files, valid_files=None):
    os.makedirs(os.path.join(CFG.output_dir,experiment_name),exist_ok=True)
    strategy, N_REPLICAS = get_strategy(CFG)    
    tf.keras.backend.clear_session()
    gc.collect()
    tf.config.optimizer.set_jit(CFG.is_jit)
        
    if CFG.fp16:
        try:
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(policy)
        except:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
    else:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_global_policy(policy)

    train_ds = get_tfrec_dataset(train_files, CFG,shuffle=32768)
    valid_ds = get_tfrec_dataset(valid_files, CFG, shuffle=False)
    
    num_train = len(train_files)
    num_valid = len(valid_files)
    steps_per_epoch = num_train//CFG.batch_size
    with strategy.scope():
        dropout_step = CFG.dropout_start_epoch * steps_per_epoch
        model = Cnn1dMhsaFeatureExtractor(CFG)

        schedule = OneCycleLR(CFG.lr, CFG.epoch, warmup_epochs=CFG.epoch*CFG.warmup, steps_per_epoch=steps_per_epoch, resume_epoch=CFG.resume, decay_epochs=CFG.epoch, lr_min=CFG.lr_min, decay_type=CFG.decay_type, warmup_type='linear')
        decay_schedule = OneCycleLR(CFG.lr*CFG.weight_decay, CFG.epoch, warmup_epochs=CFG.epoch*CFG.warmup, steps_per_epoch=steps_per_epoch, resume_epoch=CFG.resume, decay_epochs=CFG.epoch, lr_min=CFG.lr_min*CFG.weight_decay, decay_type=CFG.decay_type, warmup_type='linear')
                
        awp_step = CFG.awp_start_epoch * steps_per_epoch
        if CFG.fgm:
            model = FGM(model.input, model.output, delta=CFG.awp_lambda, eps=0., start_step=awp_step)
        elif CFG.awp:
            model = AWP(model.input, model.output, delta=CFG.awp_lambda, eps=0., start_step=awp_step)

        opt = tfa.optimizers.RectifiedAdam(learning_rate=schedule, weight_decay=decay_schedule, sma_threshold=4)#, clipvalue=1.)
        opt = tfa.optimizers.Lookahead(opt,sync_period=5)

        model.compile(
            optimizer=opt,
            loss=[CTCLoss(CFG.blank_index)],
            metrics=[],
            steps_per_execution=steps_per_epoch,
        )
    
        if CFG.summary:
            print()
            model.summary()
            print()
            print(train_ds, valid_ds)
            print()
            schedule.plot()
            print()
        print(f'---------Starting experiment: {experiment_name}---------')
        print(f'train:{num_train} valid:{num_valid}')
        print()
        
        if CFG.resume_path:
            print(f'resume from path {CFG.resume_path}')
            model.load_weights(CFG.resume_path)
            if train_ds is not None:
                model.evaluate(train_ds.take(steps_per_epoch))
            if valid_ds is not None:
                model.evaluate(valid_ds)

        logger = tf.keras.callbacks.CSVLogger(os.path.join(CFG.output_dir,experiment_name,'logs.csv'))
        swa = SWA(os.path.join(CFG.output_dir,experiment_name), CFG.swa_epochs, strategy=strategy, train_ds=train_ds, valid_ds=valid_ds, valid_steps=-(num_valid//-CFG.batch_size))
        levenshtein_cb = LevenshteinCallback(valid_ds,model,CFG,experiment_name)
        callbacks = []
        if CFG.save_output:
            callbacks.append(logger)
            callbacks.append(swa)
            callbacks.append(levenshtein_cb)  

        history = model.fit(
            train_ds,
            epochs=CFG.epoch-CFG.start_epoch,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=valid_ds,
            verbose=CFG.verbose,
            validation_steps=-(num_valid//-CFG.batch_size)
        )

        if CFG.save_output:
            model.load_weights(os.path.join(CFG.output_dir,experiment_name,'best_ckpt.h5'))
            results = evaluate(model,valid_ds,CFG)
            with open(os.path.join(CFG.output_dir,experiment_name,'best_ckpt_results.json'),"w") as file:
                json.dump(results,file)
    print("Compelted training. Model checkpoints and results saved to: ",os.path.join(CFG.output_dir,experiment_name))

def evaluate(model,validation_dataset,CFG:Conv1D_LSTM_CTC_Loss):
    y_trues = [label for _, label in validation_dataset.unbatch().as_numpy_iterator()]
    y_preds = model.predict(validation_dataset)
    predictions = []
    targets = []
    for true_seq,y_pred in zip(y_trues,y_preds):
        pred_seq = ctc_decode(y_pred,CFG.blank_index,CFG.merge_repeated).numpy()
        true_seq = "".join([CFG.idx_to_char(i) for i in true_seq[true_seq!=-1]])
        pred_seq = "".join([CFG.idx_to_char(i) for i in pred_seq[pred_seq!=-1]])
        predictions.append(pred_seq)
        targets.append(true_seq)
    return {
        "normalized_levenshtein_distance": normalized_levenshtein_distance(targets,predictions),
        "word_accuracy": word_accuracy(targets,predictions),
        "targets":targets ,
        "predictions":predictions
    }