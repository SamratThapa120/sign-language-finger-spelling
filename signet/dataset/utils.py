
import tensorflow as tf
from .preprocess import Preprocess
from .transforms import augment_fn

def filter_nans_tf(x,landmarks):
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x,landmarks,axis=1)), axis=[-2,-1]))
    mask = tf.reshape(mask, [-1])  
    x = tf.boolean_mask(x, mask, axis=0)
    return x

@tf.function()
def preprocess_ctc(x,CFG,augment=True):
    coord = x['coordinates']
    coord = filter_nans_tf(coord, CFG.POINT_LANDMARKS)
    coord = tf.ensure_shape(coord, (None, CFG.ROWS_PER_FRAME, 3))
    if CFG.augment and augment:
        coord = augment_fn(coord,tf.shape(x["sign"])[0],CFG)
    features = tf.cast(Preprocess(CFG)(coord)[0], tf.float32)
    labels = x["sign"]
    
    # pad or truncate features
    seqlen = tf.shape(features)[0]
    if seqlen < CFG.max_len:
        if CFG.use_mask:
            padding = tf.tile(tf.constant([[CFG.PAD[0]]]), [CFG.max_len - seqlen, tf.shape(features)[1]])
        else:
            padding = tf.tile([features[-1, :]], [CFG.max_len - seqlen, 1])
        features = tf.concat([features, padding], axis=0)
    elif seqlen > CFG.max_len:
        features = features[:CFG.max_len, :]

    # pad or truncate labels
    letters = tf.shape(labels)[0]
    if letters < CFG.MAX_WORD_LENGTH:
        padding = tf.fill([CFG.MAX_WORD_LENGTH - letters], -1)
        padding = tf.cast(padding, tf.int64)
        labels = tf.concat([labels, padding], axis=0)
    elif letters > CFG.MAX_WORD_LENGTH:
        labels = labels[:CFG.MAX_WORD_LENGTH]
    return features, labels

@tf.function()
def preprocess_centropy(x,CFG,augment=True):
    coord = x['coordinates']
    coord = filter_nans_tf(coord, CFG.POINT_LANDMARKS)
    coord = tf.ensure_shape(coord, (None, CFG.ROWS_PER_FRAME, 3))
    if CFG.augment and augment:
        coord = augment_fn(coord, CFG)
    features = tf.cast(Preprocess(CFG)(coord)[0], tf.float32)
    labels = x["sign"]
    
    # pad or truncate features
    seqlen = tf.shape(features)[0]
    if seqlen < CFG.max_len:
        padding = tf.tile([features[-1, :]], [CFG.max_len - seqlen, 1])
        features = tf.concat([features, padding], axis=0)
    elif seqlen > CFG.max_len:
        features = features[:CFG.max_len, :]

    # pad or truncate labels
    letters = tf.shape(labels)[0]
    if letters > CFG.ONLY_WORLD_LENGTH:
        labels = labels[:CFG.ONLY_WORLD_LENGTH]
    labels = tf.concat([labels, [CFG.end_index]], axis=0)
    letters = tf.shape(labels)[0]
    if letters < CFG.MAX_WORD_LENGTH:
        padding = tf.fill([CFG.MAX_WORD_LENGTH - letters], CFG.pad_index)
        padding = tf.cast(padding, tf.int64)
        labels = tf.concat([labels, padding], axis=0)
    return features, labels

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
    out['letters_len'] = tf.size(out['sign'])
    out['frame_len'] = tf.gather(out['shape'], 0) 
    return out

def get_ctc_dataset(tfrecords,CFG,shuffle,repeat=False,augment=True):
    # Initialize dataset with TFRecords
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=CFG.num_parallel_reads, compression_type='GZIP')
    ds = ds.map(decode_tfrec, CFG.num_parallel_reads)
    ds = ds.map(lambda x: preprocess_ctc(x, CFG,augment), CFG.num_parallel_reads)

    if repeat: 
        ds = ds.repeat()
        
    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

    if CFG.batch_size:
        ds = ds.batch(CFG.batch_size,num_parallel_calls=CFG.num_parallel_reads)

    ds = ds.prefetch(tf.data.AUTOTUNE)
        
    return ds


def get_centropy_dataset(tfrecords,CFG,shuffle,repeat=False,augment=True):
    # Initialize dataset with TFRecords
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=CFG.num_parallel_reads, compression_type='GZIP')
    ds = ds.map(decode_tfrec, CFG.num_parallel_reads)
    ds = ds.map(lambda x: preprocess_centropy(x, CFG,augment), CFG.num_parallel_reads)

    if repeat: 
        ds = ds.repeat()
        
    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

    if CFG.batch_size:
        ds = ds.batch(CFG.batch_size,num_parallel_calls=CFG.num_parallel_reads)

    ds = ds.prefetch(tf.data.AUTOTUNE)
        
    return ds