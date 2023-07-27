import sys
import os
import glob
import pandas as pd
import optuna

sys.path.append("./")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from signet.configs.Encoder_Decoder_tuning import EncoderDecoderBase
from signet.trainer.centropy_loss_trainer import train_encoder_decoder_centropy_model

data_root="../dataset/tdf_data"
experiment_name=sys.argv[1]
CFG = EncoderDecoderBase() 

train_df = pd.read_csv("../dataset/folds/fold3_train.csv")
suppdf = pd.read_csv("../dataset/supplemental_metadata.csv")
invseqid = {}
for id in suppdf.sequence_id:
    invseqid[f"{id}.npy"]=0
train_df = train_df[train_df.files.apply(lambda x: x not in invseqid)].sample(20000,random_state=1234)

valid_df = pd.read_csv("../dataset/folds/fold3_valid_difficult10.csv")
# train_labels = set(train_df['labels'].unique())
# valid_labels = set(valid_df['labels'].unique())

# common_labels = train_labels.intersection(valid_labels)
# common_mask = valid_df['labels'].isin(common_labels)
# common_rows_df = valid_df[common_mask]
# valid_df = valid_df[~common_mask]

train_files = train_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))
valid_files = valid_df.files.apply(lambda x: os.path.join(data_root,x.replace(".npy",".tfrecords")))




def objective(trail: optuna.trial.Trial):
    CFG = EncoderDecoderBase() 
    CFG.lr = trail.suggest_float("lr",1e-5,1e-2,log=True)
    CFG.weight_decay = trail.suggest_float("weight_decay",1e-6,1e-2,log=True)
    CFG.kernel_size = trail.suggest_int("kernel_size",5,20)
    CFG.num_feature_blocks=trail.suggest_int("num_feature_blocks",3,9)
    
    CFG.blocks_dropout = trail.suggest_int("blocks_dropout",1,8)/10
    CFG.final_dropout = trail.suggest_int("final_dropout",1,8)/10

    CFG.flip_lr_probability=trail.suggest_int("flip_lr_probability",0,8)/10
    CFG.random_affine_probability=trail.suggest_int("random_affine_probability",0,8)/10
    CFG.freeze_probability=trail.suggest_int("freeze_probability",0,8)/10
    CFG.tempmask_probability=trail.suggest_int("tempmask_probability",0,8)/10
    CFG.dim = 2**trail.suggest_int("dimension",5,9)

    divisors = [i for i in range(2,9) if CFG.dim % i == 0]  # get the divisors of CFG.dim from 2 to 10
    if len(divisors) == 0:  # if no divisors are found, set the divisor to 2 as a default
        divisors = [2]

    CFG.num_heads = trail.suggest_categorical("num_heads", divisors)
    CFG.decoder_mlp_dropout = trail.suggest_int("decoder_mlp_dropout",1,8)/10
    CFG.decoder_blocks= trail.suggest_int("decoder_blocks",1,5)
    CFG.label_smoothing = trail.suggest_float("label_smoothing",0,0.3)

    predict_pad_token = trail.suggest_categorical("predict_pad_token", [True,False])
    if predict_pad_token:
        print("predict pad token too")
        CFG.NUM_CLASSES +=1
        CFG.loss_pad_index=-1 
    results = train_encoder_decoder_centropy_model(experiment_name,CFG,train_files, valid_files,hp_tuning=True,min_model_size=0,max_model_size=10000000)
    if results==-99:
        raise optuna.TrialPruned()
    return results

study = optuna.create_study(direction="maximize", study_name='encdec_tune', storage='sqlite:///hptune_encdec.db', load_if_exists=True)
study.optimize(objective, n_trials=10000)
