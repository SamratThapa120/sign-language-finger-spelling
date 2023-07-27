import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self,blank_index=59,use_sparse=False,name="ctc_loss",**kwargs):
        super().__init__(name=name,**kwargs)
        self.blank_index = blank_index
        self.use_sparse = use_sparse
        
    def call(self,labels,logits):
        label_length = tf.reduce_sum(tf.cast(labels != -1, tf.int32), axis=-1)
        logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
        loss = tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=logit_length,
                blank_index=self.blank_index,
                logits_time_major=False
            )
        loss = tf.reduce_mean(loss)
        return loss
    
class CTCFocalLoss:
    '''
    https://github.com/TeaPoly/CTC-OptimizedLoss/blob/main/ctc_focal_loss.py
    '''

    def __init__(self, alpha=0.5, gamma=0.5, logits_time_major=False, blank_index=-1, lsm_prob=0.0, name="CTCFocalLoss"):
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index
        self.need_logit_length = True
        self.lsm_prob = lsm_prob
        self.gamma = gamma
        self.alpha = alpha
        self.name = name

    def __call__(self,labels,logits):
        label_length = tf.reduce_sum(tf.cast(labels != -1, tf.int32), axis=-1)
        logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
        ctc_loss = tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=logit_length,
                blank_index=self.blank_index,
                logits_time_major=False
            )

        p = tf.math.exp(-ctc_loss)
        focal_ctc_loss = ((self.alpha)*((1-p)**self.gamma)*(ctc_loss))
        loss = tf.reduce_mean(focal_ctc_loss)

        return loss