import tensorflow as tf

class CTCLoss:
    def __init__(self,blank_index=59):
        self.blank_index = blank_index

    def __call__(self,y_true,y_pred):
        labellen = tf.math.count_nonzero(y_true != -1, axis=1)

        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

        inplen = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        return tf.reduce_mean(tf.nn.ctc_loss(y_true,y_pred,labellen,inplen,blank_index=self.blank_index,logits_time_major=False))