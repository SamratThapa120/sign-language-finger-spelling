import sys
sys.path.append("./")
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from signet.dataset.dataset import SignLanguageDataset
from signet.configs.Conv1D_LSTM_CTC_Loss import Conv1D_LSTM_CTC_Loss
from signet.models.feature_extractor import Cnn1dMhsaFeatureExtractor
from callbacks import LevenshteinCallback
from torchcontrib.optim import SWA

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from signet.trainer.pt_utils.custom_schedulers import OneCycleLR
from signet.trainer.utils import CTCLossBatchFirst,get_logger
from tqdm import tqdm

def train_conv1d_mhsa_ctc_model(rank,world_size, experiment_name, data_root,train_data, valid_data):
    CFG = Conv1D_LSTM_CTC_Loss()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # set an arbitrary free port

    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train_df = pd.read_csv(train_data)
    valid_df = pd.read_csv(valid_data)
    train_dataset = SignLanguageDataset(train_df.files.apply(lambda x: os.path.join(data_root,x)), train_df.labels, CFG)
    valid_dataset = SignLanguageDataset(valid_df.files.apply(lambda x: os.path.join(data_root,x)), valid_df.labels,  CFG)

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=CFG.batch_size, pin_memory=True,num_workers=CFG.num_workers_train)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.val_batch_size, pin_memory=False,num_workers=CFG.num_workers_valid)

    device = torch.device('cuda:' + str(rank))

    dstart = CFG.dropout_start_epoch*len(train_dataset)
    model = Cnn1dMhsaFeatureExtractor(CFG,dropout_start=dstart)
    
    for name, param in model.named_parameters():
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = CTCLossBatchFirst(CFG.blank_index,zero_infinity=CFG.zero_infinity).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=CFG.lr,weight_decay=CFG.weight_decay)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=CFG.lr, rho=CFG.rho, eps=CFG.eps)
    # optimizer = SWA(optimizer)
    # scaler = GradScaler(enabled=CFG.fp16)

    scheduler = OneCycleLR(optimizer,CFG.lr, CFG.epoch, warmup_epochs=CFG.epoch*CFG.warmup,
                           steps_per_epoch=len(train_loader), resume_epoch=CFG.resume, decay_epochs=CFG.epoch,
                             lr_min=CFG.lr_min, decay_type=CFG.decay_type, warmup_type='linear')
    if rank == 0:
        validation_callback = LevenshteinCallback(valid_loader,model,CFG,experiment_name,criterion,device)
        logger = get_logger(os.path.join(CFG.output_dir,experiment_name,'logs.txt'))
        logger.info("Starting training....")
    for epoch in range(CFG.epoch):
        model.train()
        print(f"Epoch [{epoch}/{CFG.epoch}]")
        train_loss = 0
        for idx, (input_features, labels, inp_length, target_length) in tqdm(enumerate(train_loader),total=len(train_loader),disable=rank!=0):
            input_features, labels,inp_length,target_length= input_features.to(device), labels.to(device), inp_length.to(device), target_length.to(device)

            output = model(input_features)
            torch.backends.cudnn.enabled = False
            loss = criterion(output, labels, inp_length, target_length)
            torch.backends.cudnn.enabled = True
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip) 
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        train_loss /= len(train_loader)

        if rank == 0 and (epoch+1)%CFG.validation_freq==0:
            nld,wa,vloss = validation_callback(epoch)
            logger.info(f"Epoch: {epoch}, Norm distance: {nld:.5f}, word accuracy: {wa:.5f}, train loss: {train_loss:.5f}, valid loss: {vloss:.5f} ")
    logger.info("Finished training")

import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import GroupKFold

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument('--validation-data', type=str, required=True, help="Path to the validations data CSV file.")
    parser.add_argument('--data-root', type=str, required=True, help="Path to the numpy data.")
    parser.add_argument('--experiment-name', type=str, default="simple_exp", help="Name of the experiment.")
    return parser.parse_args()

def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_conv1d_mhsa_ctc_model, args=(world_size, args.experiment_name, args.data_root,args.train_data, args.validation_data), nprocs=world_size, join=True)
if __name__ == "__main__":
    main()
