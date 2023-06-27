import sys
import os
import glob
sys.path.append("./")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.Conv1D_LSTM_CTC_Loss import Conv1D_LSTM_CTC_Loss
from signet.trainer.ctc_loss_trainer import train_conv1d_mhsa_ctc_model

ALL_FILENAMES = glob.glob('../dataset/tdf_data/*.tfrecords')
print(len(ALL_FILENAMES))

CFG = Conv1D_LSTM_CTC_Loss()

train_files= ALL_FILENAMES[:100]
valid_files= ALL_FILENAMES[1000:2000]
experiment_name="simple_exp"

train_conv1d_mhsa_ctc_model(experiment_name,CFG,train_files, valid_files)