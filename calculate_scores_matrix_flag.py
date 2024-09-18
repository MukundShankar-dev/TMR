import os
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import multiprocessing
from fastdtw import fastdtw
from tqdm import tqdm
from datetime import datetime
from itertools import combinations
import argparse
from itertools import product
import json

def get_df():
    print("reading csv")
    df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR/flag_ref.csv')
    print("finished reading csv")
    return df

def all_pairs(df):      # 6m14 per call
    num_rows = df.shape[0]

    all_indices = np.arange(num_rows, dtype=int)
    all_pairs = combinations(all_indices,2)     # Just converting to list here made this crash

    i = 0
    pairs_lst = np.empty((376710076,2))

    for pair in all_pairs:
        pair_arr = np.array([pair[0],pair[1]])
        pairs_lst[i] = pair_arr
        i += 1
    for j in range(0, 27448):
        pairs_lst[i] = np.array([j,j])
        i += 1

    print("finished generating pair list")
    return pairs_lst

def calculate_pair_distance(pair, df, annotations):      # 26ms per call
    # print('in calculate fn')
    fps = 20.0
    i, j = pair
    base_path = 'datasets/motions/guoh3dfeats/'

    # print('getting keyids')
    motion_1_keyid = df.iloc[pair[0]]['keyids']
    motion_2_keyid = df.iloc[pair[1]]['keyids']

    # print('getting start/end times')
    motion_1_start = int(annotations[motion_1_keyid]["annotations"][0]["start"] * fps)
    motion_1_end = int(annotations[motion_1_keyid]["annotations"][0]["end"] * fps)
    motion_2_start = int(annotations[motion_2_keyid]["annotations"][0]["start"] * fps)
    motion_2_end = int(annotations[motion_2_keyid]["annotations"][0]["end"] * fps)

    # print('loading motions in')
    motion_1 = np.load(f"{base_path}{((df.iloc[pair[0]])['motion path'])}.npy")
    motion_2 = np.load(f"{base_path}{((df.iloc[pair[1]])['motion path'])}.npy")

    # print('slicing motions')
    motion_1 = motion_1[motion_1_start:motion_1_end]
    motion_2 = motion_2[motion_2_start:motion_2_end]

    # print('Using fastdtw on motions afer slicing')
    distance, _ = fastdtw(motion_1, motion_2,dist=euclidean)

    # print('returning')
    return i, j, distance

def worker_function(pairs_chunk, df, annotations, results):
    for pair in pairs_chunk:
        _, _, distance = calculate_pair_distance(pair, df, annotations)
        results.append((pair[0], pair[1], distance))

def calculate_scores(all_pairs, df, num_cores):
    counter = 0
    chunk_size = len(all_pairs) // num_cores
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]

    results_queue = multiprocessing.Queue()

    print('reading json')
    with open('datasets/annotations/humanml3d/annotations.json') as f:
        annotations = json.load(f)
    print('finished reading json')

    with multiprocessing.Manager() as manager:
        results = manager.list()
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.starmap(worker_function, [(chunk, df, annotations, results) for chunk in chunks])

        print('converting results to list')
        results_list = list(results)
        print("Generating DataFrame")
        scores_df = pd.DataFrame(results_list, columns=['i', 'j', 'distance'])

    return scores_df

if __name__ =='__main__':
    print('in main')
    BASE_DUMP_DIR = 'dtw_scores/'
    print('parsing')
    parser = argparse.ArgumentParser(description='Takes in the number of cores to use. The first [i] available cores will be used')
    parser.add_argument('--num_cores', type=int, help='number of cores to use')
    parser.add_argument('--start_idx', type=int, help='number of cores to use')
    parser.add_argument('--per_job_ids', type=int, help='number of cores to use')

    args = parser.parse_args()

    num_cores = args.num_cores

    print('reading df')
    df = get_df() # 147 ms to read (one time operation)
    total_ids = df.shape[0]
    end_idx = min(args.start_idx + args.per_job_ids, total_ids)

    print(f"start_idx: {args.start_idx}, end_idx: {end_idx}")

    ids_to_process = list(range(args.start_idx, end_idx))
    total_ids = np.arange(total_ids, dtype=int)
    pairs = list(product(ids_to_process, total_ids))

    scores_df = calculate_scores(pairs, df, num_cores)

    os.makedirs(BASE_DUMP_DIR, exist_ok=True)
    csv_dump_path = os.path.join(BASE_DUMP_DIR, f'all_dtw_{args.start_idx}_{end_idx}.csv')
    print(f"saving df to {csv_dump_path}")
    scores_df.to_csv(csv_dump_path, index=False)