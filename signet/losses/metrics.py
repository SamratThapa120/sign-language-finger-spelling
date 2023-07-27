import Levenshtein as lev
import numpy as np
import tensorflow as tf
def total_levenshtein_distance(y_true, y_pred):
    total_distance = sum(lev.distance(p,t) for t, p in zip(y_true, y_pred))
    return total_distance

def normalized_levenshtein_distance(y_true, y_pred):
    N = sum(len(t) for t in y_true)
    D = total_levenshtein_distance(y_true, y_pred)
    return (N - D) / N

def word_accuracy(y_true, y_pred):
    return (np.array(y_true)==np.array(y_pred)).mean()

# TopK accuracy for multi dimensional output
class TopKAccuracy(tf.keras.metrics.Metric):
    def __init__(self, k, numclasses,numclasses0,**kwargs):
        super(TopKAccuracy, self).__init__(name=f'top{k}acc', **kwargs)
        self.top_k_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)
        self.numclasses =numclasses
        self.numclasses0 = numclasses0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, self.numclasses])
        character_idxs = tf.where(y_true < self.numclasses0)
        y_true = tf.gather(y_true, character_idxs, axis=0)
        y_pred = tf.gather(y_pred, character_idxs, axis=0)
        self.top_k_acc.update_state(y_true, y_pred)

    def result(self):
        return self.top_k_acc.result()
    
    def reset_state(self):
        self.top_k_acc.reset_state()