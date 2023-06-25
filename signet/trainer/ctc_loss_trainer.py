import tensorflow as tf
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

def ctc_decode(y_pred,blank_index=59,merge_repeated=True):
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int32")

    inplen = input_length * tf.ones(shape=(batch_len,), dtype="int32")
    outs = tf.nn.ctc_greedy_decoder(tf.transpose(y_pred,(1,0,2)),inplen,blank_index=blank_index,merge_repeated=merge_repeated)
    decoded = tf.sparse.to_dense(outs[0][0],default_value=-1)
    return decoded
    
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

    train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size, max_len=CFG.max_len, drop_remainder=True, augment=True, repeat=True, shuffle=32768)
    valid_ds = get_tfrec_dataset(valid_files, batch_size=CFG.batch_size, max_len=CFG.max_len, drop_remainder=False, repeat=False, shuffle=False)
    
    num_train = len(train_files)
    num_valid = len(valid_files)
    steps_per_epoch = num_train//CFG.batch_size
    with strategy.scope():
        dropout_step = CFG.dropout_start_epoch * steps_per_epoch
        model = Cnn1dMhsaFeatureExtractor(max_len=CFG.max_len, dropout_step=dropout_step, dim=CFG.dim)

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
    sv_loss = tf.keras.callbacks.ModelCheckpoint(os.path.join(CFG.output_dir,experiment_name,'best_ckpt.h5'), monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=True, mode='min', save_freq='epoch')
    snap = Snapshot(os.path.join(CFG.output_dir,experiment_name), CFG.snapshot_epochs)
    swa = SWA(os.path.join(CFG.output_dir,experiment_name), CFG.swa_epochs, strategy=strategy, train_ds=train_ds, valid_ds=valid_ds, valid_steps=-(num_valid//-CFG.batch_size))
    callbacks = []
    if CFG.save_output:
        callbacks.append(logger)
        callbacks.append(snap)
        callbacks.append(swa)
        callbacks.append(sv_loss)
        
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
        cv = model.evaluate(valid_ds,verbose=CFG.verbose,steps=-(num_valid//-CFG.batch_size))
    valid_ds
    return model, cv, history

def evaluate(model,validation_dataset,CFG:Conv1D_LSTM_CTC_Loss):
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y_true = batch
        y_pred = model.predict(X)
        for true_seq,pred_seq in zip(y_true.numpy(),ctc_decode(y_pred,CFG.blank_index,CFG.merge_repeated).numpy()):
            true_seq = "".join([CFG.idx_to_char(i) for i in true_seq[true_seq!=-1]])
            pred_seq = "".join([CFG.idx_to_char(i) for i in pred_seq[pred_seq!=-1]])
            predictions.append(pred_seq)
            targets.append(true_seq)