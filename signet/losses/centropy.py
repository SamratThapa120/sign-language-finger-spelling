import tensorflow as tf

class CategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,padtoken,numclasses,labelsmooting=0.25,name="custom_categorical_loss",**kwargs):
        super().__init__(name=name,**kwargs)
        self.padtoken = padtoken
        self.numclasses = numclasses
        self.labelsmoothing = labelsmooting
        self.loss = tf.losses.CategoricalCrossentropy(label_smoothing=self.labelsmoothing, from_logits=True)
    def call(self,y_true, y_pred):
        # Filter Pad Tokens
        idxs = tf.where(y_true != self.padtoken)
        y_true = tf.gather_nd(y_true, idxs)
        y_pred = tf.gather_nd(y_pred, idxs)
        # One Hot Encode Sparsely Encoded Target Sign
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, self.numclasses, axis=1)
        # Categorical Crossentropy with native label smoothing support
        loss = self.loss(y_true, y_pred)
        loss = tf.math.reduce_mean(loss)
        return loss

class CategoricalCrossEntropyFocalLoss(tf.keras.losses.Loss):
    def __init__(self,padtoken,numclasses,labelsmooting=0.25,alpha=0.25,gamma=2,name="custom_categorical_focal_loss",**kwargs):
        super().__init__(name=name,**kwargs)
        self.padtoken = padtoken
        self.numclasses = numclasses
        self.labelsmoothing = labelsmooting
        self.alpha = alpha
        self.gamma = gamma
        self.loss = tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=self.labelsmoothing, from_logits=True)
    def call(self,y_true, y_pred):
        # Filter Pad Tokens
        idxs = tf.where(y_true != self.padtoken)
        y_true = tf.gather_nd(y_true, idxs)
        y_pred = tf.gather_nd(y_pred, idxs)
        # One Hot Encode Sparsely Encoded Target Sign
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, self.numclasses, axis=1)
        # Categorical Crossentropy with native label smoothing support
        loss = self.loss(y_true, y_pred)
        loss = tf.math.reduce_mean(loss)
        return loss