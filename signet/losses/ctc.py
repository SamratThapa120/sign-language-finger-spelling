import tensorflow as tf

class CTCLoss:
    def __init__(self,blank_index=59,use_sparse=False):
        self.blank_index = blank_index
        self.use_sparse = use_sparse
    def sparse_calculation(self,y_true,y_pred):
        labellen = tf.math.count_nonzero(y_true != -1, axis=1)

        input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
        batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int32)

        inplen = input_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
        sparse_y_true = tf.keras.backend.ctc_label_dense_to_sparse(tf.cast(y_true, tf.int32), tf.cast(labellen, tf.int32))
        
        return tf.reduce_mean(tf.nn.ctc_loss(sparse_y_true,y_pred,labellen,inplen,blank_index=self.blank_index,logits_time_major=False)) 
      
    def dense_calculation(self,y_true,y_pred):
        labellen = tf.math.count_nonzero(y_true != -1, axis=1)

        input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
        batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int32)

        inplen = input_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
        return tf.reduce_mean(tf.nn.ctc_loss(y_true,y_pred,labellen,inplen,blank_index=self.blank_index,logits_time_major=False))   
    
    def __call__(self,y_true,y_pred):
        if self.use_sparse:
            return self.sparse_calculation(y_true,y_pred)
        else:
            return self.dense_calculation(y_true,y_pred)