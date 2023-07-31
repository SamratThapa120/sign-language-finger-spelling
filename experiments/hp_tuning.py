import sys
import os
import glob
import pandas as pd
import optuna
import json
import yaml

sys.path.append("./")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.ctc_loss_with_downsampled_tune import ctc_loss_encdec_params
from signet.trainer.ctc_loss_downsampled_trainer import train_conv1d_mhsa_ctc_model

data_root="../dataset/tdf_data"
experiment_name=sys.argv[1]

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

valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
def objective(trail: optuna.trial.Trial):
    CFG = ctc_loss_encdec_params() 
    
    CFG.rem_percent = trail.suggest_float("rem_percent",0.3,2)
    mask = (train_df.nonnullseq*CFG.rem_percent)>train_df.labellen
    train_files = train_df[mask].files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))

    CFG.lr = trail.suggest_float("lr",1e-5,1e-2,log=True)
    CFG.weight_decay = trail.suggest_float("weight_decay",1e-6,1e-2,log=True)
    CFG.kernel_size = trail.suggest_int("kernel_size",5,20)
    CFG.num_feature_blocks=trail.suggest_int("num_feature_blocks",5,7)
    
    CFG.blocks_dropout = trail.suggest_int("blocks_dropout",1,5)/10
    CFG.final_dropout = trail.suggest_int("final_dropout",1,5)/10

    
    CFG.attention_span=trail.suggest_int("attention_span",0,20)
    do_downsample=trail.suggest_categorical("do_downsample", [True,False])
    if do_downsample:
        CFG.kernel_size_downsampling=trail.suggest_int("kernel_size_downsampling",3,17)
        CFG.downsampling_strides=trail.suggest_int("downsampling_strides",2,CFG.kernel_size_downsampling)

    CFG.loss_type=trail.suggest_categorical("loss_type", ["focal","ctc"]) 
    if CFG.loss_type=="focal" :
        CFG.alpha=trail.suggest_float("alpha",0.1,0.999)
        CFG.gamma=trail.suggest_float("gamma",0,5)
    # elif CFG.loss_type=="min_wer" :
    #     CFG.beam_width=trail.suggest_categorical("beam_width",4,16)
    #     print("beam_width:",CFG.beam_width)
    os.makedirs(os.path.join(CFG.output_dir,sys.argv[1]),exist_ok=True)
    with open(os.path.join(CFG.output_dir,sys.argv[1],"current_params.yaml"),"w") as f:
        yaml.dump(CFG.to_dict(), f, default_flow_style=False)

    results = train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files, valid_files,hp_tuning=True,min_model_size=3000000,max_model_size=10000000)
    if results==-99:
        raise optuna.TrialPruned()
    return results

study = optuna.create_study(direction="maximize", study_name='ctc_loss_tuning', storage='sqlite:///ctc_downsampled.db', load_if_exists=True)
study.optimize(objective, n_trials=10000)
