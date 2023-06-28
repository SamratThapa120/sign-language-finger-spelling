import tensorflow as tf
from signet.losses.metrics import normalized_levenshtein_distance,word_accuracy
from signet.trainer.utils import ctc_decode
import os
import numpy as np
class LevenshteinCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data,model,CFG,experiment_name,validation_steps):
        super(LevenshteinCallback, self).__init__()
        self.validation_data = validation_data
        self.y_trues = [label for _, label in self.validation_data.unbatch().as_numpy_iterator()]

        self.model = model
        self.CFG = CFG
        self.best_metric = -99
        self.experiment_name = experiment_name
        self.spe = validation_steps
        os.makedirs(os.path.join(CFG.output_dir,experiment_name,"weights"),exist_ok=True)
        self.on_epoch_end(-1)
    def on_epoch_end(self, epoch, logs=None):
        y_preds = self.model.predict(self.validation_data,steps = self.spe)
        predictions = []
        targets = []
        for true_seq,y_pred in zip(self.y_trues,y_preds):
            pred_seq = ctc_decode(y_pred,self.CFG.blank_index,self.CFG.merge_repeated).numpy()
            true_seq = "".join([self.CFG.idx_to_char[i] for i in true_seq[true_seq!=-1]])
            pred_seq = "".join([self.CFG.idx_to_char[i] for i in pred_seq[pred_seq!=-1]])
            predictions.append(pred_seq)
            targets.append(true_seq)
        nld =  normalized_levenshtein_distance(targets,predictions)
        wa = word_accuracy(targets,predictions)
        print(f"Epoch: {epoch}, Norm distance: {nld}, word accuracy: {wa} ")
        if nld>= self.best_metric:
            self.best_metric = nld
            fpath = os.path.join(self.CFG.output_dir,self.experiment_name,'best_ckpt.h5')
            self.model.save_weights(fpath)
            print("Saved best model!!")
            fpath = os.path.join(self.CFG.output_dir,self.experiment_name,"weights",f'epoch-{epoch}-examples.txt')
            with open(fpath,"w") as file:
                for pred,targ in zip(predictions,targets):
                    if np.random.rand()<self.CFG.validation_prediction_save_ratio:
                        file.write(f"{targ}\t\t{pred}\n")
        if epoch%self.CFG.save_frequency==0:
            fpath = os.path.join(self.CFG.output_dir,self.experiment_name,"weights",f'epoch-{epoch}.h5')
            self.model.save_weights(fpath)
