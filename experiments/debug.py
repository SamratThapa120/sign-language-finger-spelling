import sys
import os
import glob
import pandas as pd
sys.path.append("./")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.Conv1D_LSTM_CTC_Loss import Conv1D_LSTM_CTC_Loss
from signet.trainer.ctc_loss_trainer import train_conv1d_mhsa_ctc_model
from sklearn.model_selection import GroupKFold

train_df = pd.read_csv("../dataset/train.csv")

ALL_FILENAMES = glob.glob('../dataset/tdf_data/*.tfrecords')
print(len(ALL_FILENAMES))

CFG = Conv1D_LSTM_CTC_Loss()

gkf = GroupKFold(n_splits=5)
for train, test in gkf.split(train_df.sequence_id, groups=train_df.participant_id):
    valid_seqid = set()
    for idx in test:
        valid_seqid.add(train_df.sequence_id[idx])
train_files = []
valid_files = []
for fpth in ALL_FILENAMES:
    if int(os.path.split(fpth)[-1].split(".")[0]) in valid_seqid:
        valid_files.append(fpth)
    else:
        train_files.append(fpth)


experiment_name="simple_exp"

train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files[:5000], valid_files[:1000])