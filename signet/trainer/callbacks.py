import tensorflow as tf
from signet.losses.metrics import normalized_levenshtein_distance,word_accuracy
from signet.trainer.utils import ctc_decode
import os
import numpy as np
import logging

def get_logger(path):
    # Create a logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.DEBUG) # Set the logging level

    # Create a file handler
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG) # Set the logging level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) # Set the logging level

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file and console handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class LevenshteinCallbackCTCDecoder(tf.keras.callbacks.Callback):
    def __init__(self, validation_data,model,CFG,experiment_name,validation_steps,trainepochs=100):
        super(LevenshteinCallbackCTCDecoder, self).__init__()
        self.validation_data = validation_data
        self.y_trues = [data[1] for data in self.validation_data.unbatch().as_numpy_iterator()]

        self.model = model
        self.CFG = CFG
        self.best_metric = -99
        self.experiment_name = experiment_name
        self.spe = validation_steps
        os.makedirs(os.path.join(CFG.output_dir,experiment_name,"weights"),exist_ok=True)
        self.logger = get_logger(os.path.join(self.CFG.output_dir,self.experiment_name,'logs.txt'))
        self.trainepochs = trainepochs
        # self.on_epoch_end(-1)
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.trainepochs:  # Epochs are zero-indexed
            self.model.stop_training = True
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
        self.logger.info(f"Epoch: {epoch}, Norm distance: {nld}, word accuracy: {wa} ")
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

class LevenshteinCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data,model,CFG,experiment_name,validation_steps):
        super(LevenshteinCallback, self).__init__()
        self.validation_data = validation_data
        self.y_trues = [label for _, label in self.validation_data.unbatch().as_numpy_iterator()]

        # self.start_token=CFG.start_index
        self.end_token =CFG.end_index
        self.model = model
        self.CFG = CFG
        self.best_metric = -99
        self.experiment_name = experiment_name
        self.spe = validation_steps
        os.makedirs(os.path.join(CFG.output_dir,experiment_name,"weights"),exist_ok=True)
        self.logger = get_logger(os.path.join(self.CFG.output_dir,self.experiment_name,'logs.txt'))

        # self.on_epoch_end(-1)
    def on_epoch_end(self, epoch, logs=None):
        y_preds = self.model.predict(self.validation_data,steps = self.spe)
        predictions = []
        targets = []
        for true_seq,y_pred in zip(self.y_trues,y_preds):
            pred_indices = tf.argmax(y_pred, axis=-1).numpy().tolist()
            
            # Stop decoding once the END_TOKEN has been generated
            if self.end_token in pred_indices:
                end_index = pred_indices.index(self.end_token)
            else:
                end_index=len(pred_indices)

            # if self.start_token in pred_indices:
            #     start_index = pred_indices.index(self.start_token)+1
            # else:
            #     start_index=0         
            pred_seq = pred_indices[:end_index]
            # If true_seq is one-hot encoded, you would need to convert it back to class indices
            # true_seq = tf.argmax(true_seq, axis=-1).numpy().tolist() 
            # If it's already class indices then no need for any conversion, just converting it to list
            true_seq = true_seq.tolist()

            if self.end_token in true_seq:
                end_index = true_seq.index(self.end_token)
            else:
                end_index=len(true_seq)

            # if self.start_token in true_seq:
            #     start_index = true_seq.index(self.start_token)+1
            # else:
            #     start_index=0         
            true_seq = true_seq[:end_index]

            true_seq = "".join([self.CFG.idx_to_char[i] for i in true_seq])
            pred_seq = "".join([self.CFG.idx_to_char[i] for i in pred_seq])
            predictions.append(pred_seq)
            targets.append(true_seq)
        nld =  normalized_levenshtein_distance(targets,predictions)
        wa = word_accuracy(targets,predictions)
        self.logger.info(f"Epoch: {epoch}, Norm distance: {nld}, word accuracy: {wa} ")
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


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio,model):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
        self.model=model
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')