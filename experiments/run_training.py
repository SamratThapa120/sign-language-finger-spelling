import sys
import os
import glob
import pandas as pd
import json

sys.path.append("./")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.ctc_loss_with_downsampled_deploy_concataug import ctc_loss_encdec_params
from signet.trainer.ctc_loss_downsampled_trainer import train_conv1d_mhsa_ctc_model
import yaml

data_root="../dataset/tdf_data"
CFG = ctc_loss_encdec_params() 

CFG.train_epochs=200
CFG.epoch=200
CFG.resume_path="/app/runs/ctc_with_masking/alldata_concat/best_ckpt.h5"
CFG.start_epoch=120

os.makedirs(os.path.join(CFG.output_dir,sys.argv[1]),exist_ok=True)
with open(os.path.join(CFG.output_dir,sys.argv[1],"current_params.yaml"),"w") as f:
    yaml.dump(CFG.to_dict(), f, default_flow_style=False)

all_nan = json.load(open("../dataset/folds/allnan.json"))
seq_length = json.load(open("../dataset/folds/seqlen.json"))
train_df = pd.read_csv("../dataset/folds/foldall_train.csv")

valid_df = pd.read_csv("../dataset/folds/foldall_valid.csv")

train_df["seqlen"] = train_df.files.apply(lambda x: seq_length[x.split(".")[0]])
train_df["labellen"] = train_df.labels.apply(len)
train_df = train_df[(train_df.seqlen>train_df.labellen*2)& train_df.files.apply(lambda x: x.split(".")[0] not in  all_nan)]

train_files = train_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
train_df["tflabel"] = train_files

valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))

train_conv1d_mhsa_ctc_model(sys.argv[1],CFG,train_files, valid_files,train_df=train_df)