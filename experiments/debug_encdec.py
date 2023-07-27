import sys
import os
import glob
import pandas as pd
import json

sys.path.append("./")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.Encoder_Decoder_Base import EncoderDecoderBase
from signet.trainer.centropy_loss_trainer import train_encoder_decoder_centropy_model

data_root="../dataset/tdf_data"
experiment_name="large_fold3_diff"
CFG = EncoderDecoderBase() 

all_nan = json.load(open("../dataset/folds/allnan.json"))
seq_length = json.load(open("../dataset/folds/seqlen.json"))
train_df = pd.read_csv("../dataset/folds/fold3_train.csv")
valid_df = pd.read_csv("../dataset/folds/fold3_valid_difficult10.csv")
train_df["seqlen"] = train_df.files.apply(lambda x: seq_length[x.split(".")[0]])
train_df["labellen"] = train_df.labels.apply(len)
train_df = train_df[(train_df.seqlen>train_df.labellen*2)& train_df.files.apply(lambda x: x.split(".")[0] not in  all_nan)]

train_files = train_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))

train_encoder_decoder_centropy_model(experiment_name,CFG,train_files, valid_files)