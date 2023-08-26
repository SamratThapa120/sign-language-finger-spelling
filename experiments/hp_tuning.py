import sys
import os
import glob
import pandas as pd
import json
import optuna

sys.path.append("./")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.ctc_loss_with_downsampled_deploy_concataug import ctc_loss_encdec_params
from signet.trainer.ctc_loss_downsampled_posemb_trainer import train_conv1d_mhsa_ctc_model
import yaml

data_root="../dataset/tdf_data"


all_nan = json.load(open("../dataset/folds/allnan.json"))
seq_length = json.load(open("../dataset/folds/seqlen.json"))
handnonulllen = json.load(open("../dataset/folds/handnotnull.json"))

suppdf = pd.read_csv("../dataset/supplemental_metadata.csv")
invseqid = {}
for id in suppdf.sequence_id:
    invseqid[f"{id}.npy"]=0

valid_df = pd.read_csv("../dataset/folds/fold3_valid.csv")
if os.path.exists("../dataset/folds/cache_train.csv"):
    train_df = pd.read_csv("../dataset/folds/cache_train.csv")
else:
    train_df = pd.read_csv("../dataset/folds/fold3_train.csv")
    train_df["seqlen"] = train_df.files.apply(lambda x: seq_length[x.split(".")[0]])
    train_df["labellen"] = train_df.labels.apply(len)
    train_df = train_df[(train_df.seqlen>train_df.labellen*2)& train_df.files.apply(lambda x: x.split(".")[0] not in all_nan)]
    train_df = train_df[train_df.files.apply(lambda x: x not in invseqid)].sample(20000,random_state=1234)
    train_df["nonnullseq"] = train_df.files.apply(lambda x: handnonulllen[x.split(".")[0]])
    train_df.to_csv("../dataset/folds/cache_train.csv",index=False)

train_files = train_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
train_df["tflabel"] = train_files

def objective(trail: optuna.trial.Trial):
    CFG = ctc_loss_encdec_params() 
    CFG.output_dir = '../runs/conv_trans_tuning'
    CFG.train_epochs=60
    CFG.epoch=120
    if trail.number>0:
        CFG.kernel_size = trail.suggest_int("kernel_size",2,20)
        CFG.num_feature_blocks=trail.suggest_int("num_feature_blocks",1,15)
        CFG.kernel_size_downsampling=trail.suggest_int("kernel_size_downsampling",3,16)
        
        CFG.blocks_dropout = trail.suggest_int("blocks_dropout",1,4)/10
        CFG.final_dropout = trail.suggest_int("final_dropout",0,4)/10
        CFG.dim = trail.suggest_categorical("dimension_2",[128,256,384,768])
        CFG.use_conv = trail.suggest_categorical("use_convolution",[True,False])
        CFG.use_transformer = not CFG.use_conv

        if CFG.use_transformer:
            divisors = [i for i in range(2,9) if CFG.dim % i == 0]  # get the divisors of CFG.dim from 2 to 10
            if len(divisors) == 0:  # if no divisors are found, set the divisor to 2 as a default
                divisors = [2]
            CFG.num_heads = trail.suggest_categorical("num_heads", divisors)
    os.makedirs(os.path.join(CFG.output_dir,sys.argv[1]),exist_ok=True)
    with open(os.path.join(CFG.output_dir,sys.argv[1],"current_params.yaml"),"w") as f:
        yaml.dump(CFG.to_dict(), f, default_flow_style=False)
    results = train_conv1d_mhsa_ctc_model(sys.argv[1],CFG,train_files, valid_files,hp_tuning=True,min_model_size=2000000,max_model_size=10000000,train_df=train_df)
    if results==-99:
        raise RuntimeError()
    return results

study = optuna.create_study(direction="maximize", study_name='hptune_convtrans', storage='sqlite:///hptune_convtrans.db', load_if_exists=True)
study.optimize(objective, n_trials=10000,catch=(RuntimeError))