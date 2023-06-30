import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count
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
def process_file(filename, seqid_label, LABEL_DICT):
    print("Working on :", filename)
    print("Completed :", len(glob.glob("../dataset/npy_data/*")))

    record_dict = load_relevant_data_subset(filename)

    for seqid, coords in record_dict.items():
        # Use numpy's save function to save the array to a .npy file
        npy_filename = f"../dataset/npy_data/{seqid}.npy"
        np.save(npy_filename, coords)
        del coords
assert pd.concat([train_df,supp_df]).sequence_id.nunique() == len(train_df)+len(supp_df)
os.makedirs("../dataset/npy_data",exist_ok=True)
full_df = pd.concat([train_df,supp_df])
seqid_to_label = {sid:phr for sid,phr in zip(full_df.sequence_id,full_df.phrase)}


_ = Parallel(n_jobs=18)(
    delayed(process_file)(os.path.join("../dataset",pth),seqid_to_label,LABEL_DICT)
    for pth in full_df.path.unique())