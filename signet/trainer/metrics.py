import Levenshtein as lev
import numpy as np

def total_levenshtein_distance(y_true, y_pred):
    total_distance = sum(lev.distance(t, p) for t, p in zip(y_true, y_pred))
    return total_distance

def normalized_levenshtein_distance(y_true, y_pred):
    N = sum(len(t) for t in y_true)
    D = total_levenshtein_distance(y_true, y_pred)
    return (N - D) / N

def word_accuracy(y_true, y_pred):
    return (np.array(y_true)==np.array(y_pred)).mean()
