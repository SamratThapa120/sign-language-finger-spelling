import sys
sys.path.append("./")
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from signet.dataset.dataset import SignLanguageDataset
from signet.configs.Conv1D_LSTM_CTC_Loss import Conv1D_LSTM_CTC_Loss
from signet.models.feature_extractor import Cnn1dMhsaFeatureExtractor
from callbacks import LevenshteinCallback
from torchcontrib.optim import SWA

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from signet.trainer.utils import CTCLossBatchFirst
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
    model = Cnn1dMhsaFeatureExtractor(CFG)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = CTCLossBatchFirst(CFG.blank_index,zero_infinity=CFG.zero_infinity)
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    # optimizer = SWA(optimizer)
    scaler = GradScaler(enabled=CFG.fp16)

    scheduler = OneCycleLR(optimizer, max_lr=CFG.lr, epochs=CFG.epoch, steps_per_epoch=len(train_loader))

    if rank == 0:
        validation_callback = LevenshteinCallback(valid_loader,model,CFG,experiment_name,criterion,device)
    
    for epoch in range(CFG.epoch):
        print(f"Epoch [{epoch}/CFG.epoch]")
        train_loss = 0
        for idx, (input_features, labels, inp_length, target_length) in tqdm(enumerate(train_loader),total=len(train_loader),disable=rank!=0):
            input_features, labels = input_features.to(device), labels.to(device)
            
            output = model(input_features)
            loss = criterion(output, labels, inp_length, target_length)
            
            train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        train_loss /= len(train_loader)

        if rank == 0 and (epoch+1)%CFG.validation_freq==0:
            nld,wa,vloss = validation_callback(epoch)
            print(f"Epoch: {epoch}, Norm distance: {nld:.5f}, word accuracy: {wa:.5f}, train loss: {train_loss:.5f}, valid loss: {vloss:.5f} ")
   
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
