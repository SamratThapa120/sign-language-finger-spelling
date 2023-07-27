import sys
import os
import glob
import pandas as pd
import json

sys.path.append("./")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.ctc_loss_with_downsampled_deploy_strongeraugs import ctc_loss_encdec_params
from signet.trainer.ctc_loss_downsampled_trainer import train_conv1d_mhsa_ctc_model

data_root="../dataset/tdf_data"
experiment_name="fold3_poseinfo_moreaugs"
CFG = ctc_loss_encdec_params() 

all_nan = json.load(open("../dataset/folds/allnan.json"))
seq_length = json.load(open("../dataset/folds/seqlen.json"))
train_df = pd.read_csv("../dataset/folds/fold3_train.csv")
valid_df = pd.read_csv("../dataset/folds/fold3_valid.csv")
train_df["seqlen"] = train_df.files.apply(lambda x: seq_length[x.split(".")[0]])
train_df["labellen"] = train_df.labels.apply(len)
train_df = train_df[(train_df.seqlen>train_df.labellen*2)& train_df.files.apply(lambda x: x.split(".")[0] not in  all_nan)]

train_files = train_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))

train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files, valid_files)