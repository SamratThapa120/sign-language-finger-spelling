import tensorflow as tf
import math

@tf.function()
def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

@tf.function()
def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    vector_a = p1 - p2
    vector_b = p3 - p2

    cosine_angle = tf.math.reduce_sum(vector_a*vector_b,axis=3) / (tf.norm(vector_a,axis=3) * tf.norm(vector_b,axis=3))
    angle = tf.acos(tf.clip_by_value(cosine_angle, -1.0, 1.0))
    
    return angle/math.pi

def calculate_length(p1, p2):
    """Calculate the distance between two points."""
    return tf.norm(p1 - p2,axis=3)

def compute_angles(landmarks, connections):
    a,b,c = connections
    angle = calculate_angle(tf.gather(landmarks, a, axis=2)[:,:,:,:2], 
                            tf.gather(landmarks, b, axis=2)[:,:,:,:2], 
                            tf.gather(landmarks, c, axis=2)[:,:,:,:2])
    return angle

def compute_lengths(landmarks, connections):
    a,b = connections
    lengths = calculate_length(tf.gather(landmarks, a, axis=2)[:,:,:,:2], 
                            tf.gather(landmarks, b, axis=2)[:,:,:,:2])
    return lengths

class PreprocessCopied(tf.keras.layers.Layer):
    def __init__(self,CFG,**kwargs):
        super().__init__(**kwargs)
        self.max_len = CFG.max_len
        self.point_landmarks = CFG.POINT_LANDMARKS
    
    @tf.function()
    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
        x = (x - mean)/std
        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)

        length = tf.shape(x)[1]
        # # x = x[...,:2]

        # dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        # # dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        # x = tf.concat([
        #     tf.reshape(x, (-1,length,3*len(self.point_landmarks))),
        #     tf.reshape(dx, (-1,length,3*len(self.point_landmarks))),
        #     # tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
        # ], axis = -1)
        
        return tf.reshape(x, (-1,length,3*len(self.point_landmarks)))
    
class Preprocess(tf.keras.layers.Layer):
    def __init__(self,CFG,**kwargs):
        super().__init__(**kwargs)
        self.max_len = CFG.max_len
        self.point_landmarks = CFG.POINT_LANDMARKS
        self.use_depth = CFG.use_depth
        self.ndims = 3 if self.use_depth else 2
        self.use_angle = CFG.useangle
        self.use_lengths = CFG.uselengths
        self.angleidxs =  (CFG.angle_a,CFG.angle_b,CFG.angle_c)
        self.lengthidxs =  (CFG.length_a,CFG.length_b)

    @tf.function()
    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs
        if self.use_angle:
            angles = compute_angles(x,self.angleidxs)
            
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
        x = (x - mean)/std
        length = tf.shape(x)[1]

        if not self.use_depth:
            x = x[...,:2]

        dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))
        # dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))
        concatterms = [
            tf.reshape(x, (-1,length,self.ndims*len(self.point_landmarks))),
            tf.reshape(dx, (-1,length,self.ndims*len(self.point_landmarks))),
        ]
        if self.use_angle:
            concatterms.append(angles)
        # if self.use_lengths:
        #     concatterms.append(compute_lengths(x,self.lengthidxs))
        x = tf.concat(concatterms, axis = -1)
        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        return x