import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count
import tensorflow as tf
import numpy as np
import plotly.express as px
cpu_count()
ROOT_DIR = "../dataset"
supp_df = pd.read_csv("../dataset/supplemental_metadata.csv")
train_df = pd.read_csv("../dataset/train.csv")
example =pd.read_parquet('../dataset/train_landmarks/1019715464.parquet')
classes,counts = np.unique([x.split("_")[1] for x in example.columns[1:]],return_counts=True)

def sort_func(x):
    p = x.split("_")[-1]
    return "_".join(x.split("_")[1:-1])+ "{:04d}".format(int(p))

COLUMNS_SEQUENCE=sorted(example.columns[1:],key=sort_func)

ROWS_PER_FRAME = 543

def load_relevant_data_subset(pq_path):
    example = pd.read_parquet(pq_path)
    byseqid = {}
    for seqid,row in example.iterrows():
        if seqid not in byseqid:
            byseqid[seqid] = []
        byseqid[seqid].append(np.array(row[COLUMNS_SEQUENCE]))
    for key in byseqid:
        byseqid[key] = np.stack(byseqid[key]).astype(np.float32).reshape(-1,ROWS_PER_FRAME,3)
    return byseqid
import json
with open('../dataset/character_to_prediction_index.json') as json_file:
    LABEL_DICT = json.load(json_file)
import glob
def process_file(filename,seqid_label,LABEL_DICT):
        print("Working on :",filename)
        print("Completed :",len(glob.glob("../dataset/tdf_data/*")))

        record_bytes = load_relevant_data_subset(filename)
        options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)
        for seqid,coords in record_bytes.items(): 
            tfrecord_name = f"../dataset/tdf_data/{seqid}.tfrecords"
            example = tf.train.Example(features=tf.train.Features(feature={
                'coordinates': tf.train.Feature(bytes_list=tf.train.BytesList(value=[coords.tobytes()])),
                'sequence_id':tf.train.Feature(int64_list=tf.train.Int64List(value=[seqid])),
                'sign':tf.train.Feature(int64_list=tf.train.Int64List(value=[LABEL_DICT[t] for t in seqid_label[seqid]])),
                'shape':tf.train.Feature(int64_list=tf.train.Int64List(value=list(coords.shape))),
                })).SerializeToString()
            with tf.io.TFRecordWriter(tfrecord_name, options=options) as file_writer:
                file_writer.write(example)
                file_writer.close()
            del example
assert pd.concat([train_df,supp_df]).sequence_id.nunique() == len(train_df)+len(supp_df)
os.makedirs("../dataset/tdf_data",exist_ok=True)
full_df = pd.concat([train_df,supp_df])
seqid_to_label = {sid:phr for sid,phr in zip(full_df.sequence_id,full_df.phrase)}
len(seqid_to_label)
def parse_example(example_proto):
    # Define the features within the example
    feature_description = {
        'coordinates': tf.io.FixedLenFeature([], tf.string),
        'sequence_id': tf.io.FixedLenFeature([], tf.int64),
        'sign': tf.io.VarLenFeature(tf.int64),
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the coordinates
    coordinates = tf.io.decode_raw(parsed_example['coordinates'], tf.float32)

    # The 'sign' feature is a variable length feature, we have to convert it from sparse to dense
    sign = tf.sparse.to_dense(parsed_example['sign'])
    
    return coordinates, parsed_example['sequence_id'], sign

def process_tfrecord_file(tfrecord_name):
    # Load the data from the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name, compression_type='GZIP')

    # Parse the data
    parsed_dataset = raw_dataset.map(parse_example)
    
    for coordinates, seqid, sign in parsed_dataset:
        # process data here

        # Note: To convert tensors back to numpy use `.numpy()`
        # e.g., `numpy_seqid = seqid.numpy()`
        print(f"Coordinates: {coordinates.numpy().shape}")
        print(f"Seqid: {seqid}")
        print(f"Sign: {sign}")
    return parsed_dataset

_ = Parallel(n_jobs=18)(
    delayed(process_file)(os.path.join("../dataset",pth),seqid_to_label,LABEL_DICT)
    for pth in full_df.path.unique())

# tfrecord_name = "/app/ThesisProject/dataset/tdf_data/1975433633.tfrecords" # use your tfrecord file path here
# z = process_tfrecord_file(tfrecord_name)