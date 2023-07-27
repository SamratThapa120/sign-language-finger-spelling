import sys
import os
import glob
import pandas as pd
import optuna

sys.path.append("./")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.Conv1D_LSTM_CTC_Loss import Conv1D_LSTM_CTC_Loss
from signet.trainer.ctc_loss_trainer import train_conv1d_mhsa_ctc_model

data_root="../dataset/tdf_data"
experiment_name=sys.argv[1]
CFG = Conv1D_LSTM_CTC_Loss() 

train_df = pd.read_csv("../dataset/folds/fold3_train.csv")
valid_df = pd.read_csv("../dataset/folds/fold3_valid.csv")

train_files = train_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))


def objective(trail: optuna.trial.Trial):
    CFG.lr_min = trail.suggest_float("lr_min",1e-7,1e-2,log=True)
    CFG.lr = trail.suggest_float("lr",CFG.lr_min,2e-2,log=True)
    CFG.weight_decay = trail.suggest_float("weight_decay",1e-7,1e-2,log=True)
    CFG.kernel_size = trail.suggest_int("kernel_size",5,20)
    CFG.num_feature_blocks=trail.suggest_int("num_feature_blocks",5,10)
    
    CFG.blocks_dropout = trail.suggest_int("blocks_dropout",1,8)/10

    CFG.flip_lr_probability=trail.suggest_int("flip_lr_probability",0,8)/10
    CFG.random_affine_probability=trail.suggest_int("random_affine_probability",0,8)/10
    CFG.freeze_probability=trail.suggest_int("freeze_probability",0,8)/10

    CFG.dim = 2**trail.suggest_int("dimension",5,8)
    
    results = train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files, valid_files,hp_tuning=True,min_model_size=3000000,max_model_size=10000000)
    if results<0:
        raise optuna.TrialPruned()
    return results

study = optuna.create_study(direction="maximize", study_name='tct_loss_tuning', storage='sqlite:///example.db', load_if_exists=True)
study.optimize(objective, n_trials=200)
