import numpy as np
import math

def np_nan_mean(x, axis=0, keepdims=False):
    return np.nansum(x, axis=axis, keepdims=keepdims) / np.sum(~np.isnan(x), axis=axis, keepdims=keepdims)

def np_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = np_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return np.sqrt(np_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess:
    def __init__(self,CFG):
        self.max_len = CFG.max_len
        self.point_landmarks = CFG.POINT_LANDMARKS

    def __call__(self, inputs):
        if inputs.ndim == 3:
            x = inputs[np.newaxis,...]
        else:
            x = inputs
        
        mean = np_nan_mean(np.take(x, [17], axis=2), axis=(1,2), keepdims=True)
        mean = np.where(np.isnan(mean), 0.5, mean)
        x = np.take(x, self.point_landmarks, axis=2) #N,T,P,C
        std = np_nan_std(x, center=mean, axis=(1,2), keepdims=True)
        
        x = (x - mean)/std

        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = x.shape[1]
        x = x[...,:2]

        dx = np.pad(x[:,1:] - x[:,:-1], [[0,0],[1,0],[0,0],[0,0]]) if x.shape[1] > 1 else np.zeros_like(x)
        dx2 = np.pad(x[:,2:] - x[:,:-2], [[0,0],[2,0],[0,0],[0,0]]) if x.shape[1] > 2 else np.zeros_like(x)

        x = np.concatenate([
            x.reshape((-1,length,2*len(self.point_landmarks))),
            dx.reshape((-1,length,2*len(self.point_landmarks))),
            dx2.reshape((-1,length,2*len(self.point_landmarks))),
        ], axis = -1)
        
        x = np.where(np.isnan(x), 0.0, x)
        
        return x[0]
