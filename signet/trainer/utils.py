import tensorflow as tf


def ctc_decode(y_pred,blank_index=59,merge_repeated=True):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.expand_dims(y_pred, 0)
    y_pred = tf.cast(y_pred, tf.float32)
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int32")

    inplen = input_length * tf.ones(shape=(batch_len,), dtype="int32")
    outs = tf.nn.ctc_greedy_decoder(tf.transpose(y_pred,perm=(1,0,2)),inplen,blank_index=blank_index,merge_repeated=merge_repeated)
    decoded = tf.sparse.to_dense(outs[0][0],default_value=-1)
    return decoded[0]