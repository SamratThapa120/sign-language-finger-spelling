
import tensorflow as tf
from .preprocess import Preprocess
from .transforms import augment_fn

def filter_nans_tf(x,landmarks):
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x,landmarks,axis=1)), axis=[-2,-1]))
    print(mask.shape) #prints <unknown>
    mask = tf.reshape(mask, [-1])  
    print(mask.shape) #prints <unknown>
    x = tf.boolean_mask(x, mask, axis=0)
    return x


def preprocess(x,CFG):
    coord = x['coordinates']
    coord = filter_nans_tf(coord,CFG.POINT_LANDMARKS)
    if CFG.augment:
        coord = augment_fn(coord, max_len=CFG.max_len)
    coord = tf.ensure_shape(coord, (None,CFG.ROWS_PER_FRAME,3))
    
    return tf.cast(Preprocess(CFG)(coord)[0],tf.float32), x['sign']

def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        'coordinates': tf.io.FixedLenFeature([], tf.string),
        'sequence_id': tf.io.FixedLenFeature([], tf.int64),
        'sign': tf.io.VarLenFeature(tf.int64),
        'shape': tf.io.VarLenFeature(tf.int64),

    })
    out = {}
    out["shape"] = tf.sparse.to_dense(features['shape'])
    out['coordinates']  = tf.reshape(tf.io.decode_raw(features['coordinates'], tf.float32), out["shape"])
    out['sign'] = tf.sparse.to_dense(features['sign'])
    return out

def get_tfrec_dataset(tfrecords,CFG):
    # Initialize dataset with TFRecords
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.experimental.AUTOTUNE, compression_type='GZIP')
    ds = ds.map(decode_tfrec, tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x: preprocess(x, CFG), tf.data.experimental.AUTOTUNE)

    if CFG.repeat: 
        ds = ds.repeat()
        
    if CFG.shuffle:
        ds = ds.shuffle(CFG.shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = (False)
        ds = ds.with_options(options)

    print(ds.element_spec)
    if CFG.batch_size:
        ds = ds.padded_batch(CFG.batch_size, padding_values=CFG.PAD, padded_shapes=([CFG.max_len,CFG.CHANNELS],[CFG.MAX_WORD_LENGTH]), drop_remainder=CFG.drop_remainder)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        
    return ds