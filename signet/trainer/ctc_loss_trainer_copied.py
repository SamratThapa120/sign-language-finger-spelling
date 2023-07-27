import tensorflow as tf
import json
import gc 
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.mixed_precision as mixed_precision


from .tf_utils.schedules import OneCycleLR, ListedLR
from .tf_utils.callbacks import Snapshot, SWA
from .tf_utils.learners import FGM, AWP

from signet.dataset.utils import get_ctc_dataset
from signet.models.feature_extractor import Cnn1dMhsaFeatureExtractor
from signet.losses.ctc import CTCLoss
from signet.trainer.utils import ctc_decode
from signet.trainer.callbacks import LevenshteinCallbackCTCDecoder,WeightDecayCallback
from signet.losses.metrics import normalized_levenshtein_distance,word_accuracy
from signet.dataset.preprocess import Preprocess
import random
import math

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def lrfn(current_step, num_warmup_steps, lr_max, num_cycles, num_training_steps,warmup_method):
    print("current step:",current_step)
    if current_step < num_warmup_steps:
        if warmup_method == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
    
def get_strategy(CFG):
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
    
def train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files, valid_files=None,hp_tuning=False,min_model_size=10000,max_model_size=10000000):
    set_seed(42) 
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

    train_ds = get_ctc_dataset(train_files, CFG,shuffle=32768,repeat=True,augment=True)
    valid_ds = get_ctc_dataset(valid_files, CFG, shuffle=False,repeat=False,augment=False)
    
    num_train = len(train_files)
    num_valid = len(valid_files)
    steps_per_epoch = num_train//CFG.batch_size
    with strategy.scope():
        model = Cnn1dMhsaFeatureExtractor(CFG)

        LR_SCHEDULE = [lrfn(step, num_warmup_steps=CFG.warmup_epochs, lr_max=CFG.lr_max, num_cycles=0.50,num_training_steps=CFG.epoch,warmup_method=CFG.warmup_type) for step in range(CFG.epoch)]
        
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)
        wdcb = WeightDecayCallback(CFG.weight_decay_ratio,model)

        opt = tfa.optimizers.RectifiedAdam(sma_threshold=4)#, clipvalue=1.)
        opt = tfa.optimizers.Lookahead(opt,sync_period=5)

        model.compile(
            optimizer=opt,
            loss=CTCLoss(CFG.blank_index),
            metrics=[],
        )
    # tf.profiler.experimental.start(os.path.join(CFG.output_dir,experiment_name,'log_data'))
    if CFG.summary:
        print()
        model.summary()
        print()
        print(train_ds, valid_ds)
        print()
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
    levenshtein_cb = LevenshteinCallbackCTCDecoder(valid_ds,model,CFG,experiment_name,validation_steps=-(num_valid//-CFG.batch_size))
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(CFG.output_dir,experiment_name,'log_data'),
                                            #  profile_batch=(10,20))
    callbacks = []
    if CFG.save_output:
        callbacks.append(logger)
        callbacks.append(levenshtein_cb)  
        callbacks.append(nan_callback)
        callbacks.append(lr_callback)
        callbacks.append(wdcb)
    if hp_tuning and (model.count_params()<min_model_size or model.count_params()>max_model_size):
        print(f"{ model.count_params()} params. Model too small or big, stopping training")
        return levenshtein_cb.best_metric
    history = model.fit(
        train_ds,
        epochs=CFG.train_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=valid_ds,
        verbose=CFG.verbose,
        validation_steps=-(num_valid//-CFG.batch_size),
        validation_freq=CFG.validation_frequency
    )
    if hp_tuning:
        return levenshtein_cb.best_metric
    # tf.profiler.experimental.stop()
    model.load_weights(os.path.join(CFG.output_dir,experiment_name,'best_ckpt.h5'))
    results = evaluate(model,valid_ds,CFG,validation_steps=-(num_valid//-CFG.batch_size))
    with open(os.path.join(CFG.output_dir,experiment_name,'best_ckpt_results.json'),"w") as file:
        json.dump(results,file)
    print("Completed training. Model checkpoints and results saved to: ",os.path.join(CFG.output_dir,experiment_name))

def evaluate(model,validation_dataset,CFG,validation_steps):
    y_trues = [label for _, label in validation_dataset.unbatch().as_numpy_iterator()]
    y_preds = model.predict(validation_dataset,steps=validation_steps)
    predictions = []
    targets = []
    for true_seq,y_pred in zip(y_trues,y_preds):
        pred_seq = ctc_decode(y_pred,CFG.blank_index,CFG.merge_repeated).numpy()
        true_seq = "".join([CFG.idx_to_char[i] for i in true_seq[true_seq!=-1]])
        pred_seq = "".join([CFG.idx_to_char[i] for i in pred_seq[pred_seq!=-1]])
        predictions.append(pred_seq)
        targets.append(true_seq)
    return {
        "normalized_levenshtein_distance": normalized_levenshtein_distance(targets,predictions),
        "word_accuracy": word_accuracy(targets,predictions),
        "targets":targets ,
        "predictions":predictions
    }
