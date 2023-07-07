import torch
from signet.trainer.metrics import normalized_levenshtein_distance,word_accuracy
from signet.trainer.utils import GreedyCTCDecoder
import os
import numpy as np
from tqdm import tqdm
class LevenshteinCallback:
    def __init__(self, data_loader,model,CFG,experiment_name,criterion,device):
        assert data_loader.batch_size==1, "Batch size must be one for inference to simulate real-world inference scenario"
        self.data_loader = data_loader
        self.model = model
        self.CFG = CFG
        self.best_metric = -99
        self.experiment_name = experiment_name
        self.decoder = GreedyCTCDecoder(CFG.idx_to_char,CFG.blank_index)
        self.device=device
        self.criterion = criterion
        os.makedirs(os.path.join(CFG.output_dir,experiment_name,"weights"),exist_ok=True)
        # self(-1)
    def __call__(self, epoch):
        self.model.eval()
        predictions = []
        targets = []
        original = []
        valid_loss = []
        with torch.no_grad():
            for idx,(input_features, labels, inp_length, target_length) in tqdm(enumerate(self.data_loader),total=len(self.data_loader)):
                    pred = self.model(input_features[:,:inp_length[0]].to(self.device)).detach().cpu()
                    valid_loss.append(self.criterion(pred,labels,inp_length,target_length).item())
                    gts = "".join([self.CFG.idx_to_char[i] for i in labels[0,:target_length[0]].numpy()])
                    pred,orig = self.decoder(pred[0])
                    original.append(orig)
                    predictions.append(pred)
                    targets.append(gts)
        nld =  normalized_levenshtein_distance(targets,predictions)
        wa = word_accuracy(targets,predictions)
        if nld>= self.best_metric:
            self.best_metric = nld
            fpath = os.path.join(self.CFG.output_dir,self.experiment_name,'best_ckpt.pth')
            torch.save(self.model.module.state_dict(),fpath)
            # Save best model in fpath
            print("Saved best model!!")
            fpath = os.path.join(self.CFG.output_dir,self.experiment_name,"weights",f'epoch-{epoch}-examples.txt')
            with open(fpath,"w") as file:
                for pred,targ,orig in zip(predictions,targets,original):
                    if np.random.rand()<self.CFG.validation_prediction_save_ratio:
                        file.write(f"{targ}\t\t{pred}\t\t{orig}\n")
            
        if epoch%self.CFG.save_frequency==0:
            fpath = os.path.join(self.CFG.output_dir,self.experiment_name,"weights",f'epoch-{epoch}.pth')
        return nld,wa,np.mean(valid_loss)