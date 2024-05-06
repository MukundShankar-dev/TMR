import os
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastdtw import fastdtw
from tqdm import tqdm
from datetime import datetime
from itertools import combinations
import argparse
from itertools import product
# Total pairs: 376710076

def get_df():   # 140ms per call
    print("reading csv")
    df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR/outputs/tmr_humanml3d_guoh3dfeats_old1/new_latents/embeddings.csv')
    print("finished reading csv")
    print(f"head: {df.head()}")
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


def calculate_pair_distance(pair, df):      # 26ms per call
    i, j = pair
    base_path = 'datasets/motions/guoh3dfeats/'
    motion1 = np.load(f"{base_path}{((df.iloc[pair[0]])['motion path'])}.npy")
    motion2 = np.load(f"{base_path}{((df.iloc[pair[1]])['motion path'])}.npy")
    distance, _ = fastdtw(motion1, motion2,dist=euclidean)
    return i, j, distance

def calculate_scores(all_pairs, df, num_cores):
    # motion_scores = np.zeros((len(all_pairs), 27448))
    scores_df = pd.DataFrame(index=np.arange(len(all_pairs)), columns=['i', 'j', 'distance'])
    counter = 0
    # Non-batching approach:
    print("starting to calculate")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(calculate_pair_distance, pair, df) for pair in all_pairs]

        for future in as_completed(futures):
            i, j, distance = future.result()
            scores_df.loc[counter] = [i, j, distance]
            counter += 1
            if counter % 1000 == 0:
                print(f"completed {counter} calculations")
    
    return scores_df

if __name__ =='__main__':
    BASE_DUMO_DIR = 'dtw_scores/'
    parser = argparse.ArgumentParser(description='Takes in the number of cores to use. The first [i] available cores will be used')
    parser.add_argument('--num_cores', type=int, help='number of cores to use')
    parser.add_argument('--start_idx', type=int, help='number of cores to use')
    parser.add_argument('--per_job_ids', type=int, help='number of cores to use')


    args = parser.parse_args()

    num_cores = args.num_cores

    df = get_df() # 147 ms to read (one time operation)
    total_ids = df.shape[0]
    end_idx = min(args.start_idx + args.per_job_ids, total_ids)

    print(f"start_idx: {args.start_idx}, end_idx: {end_idx}")

    ids_to_process = list(range(args.start_idx, end_idx))
    total_ids = np.arange(total_ids, dtype=int)
    pairs = list(product(ids_to_process, total_ids))

    # # # start_time1 = datetime.now()
    # pairs = all_pairs(df)   # Takes 6m14s

    # # start_time2 = datetime.now()
    scores_df = calculate_scores(pairs, df, num_cores)

    # scores_df = matrix_to_df(motion_scores)
    os.makedirs(BASE_DUMO_DIR, exist_ok=True)
    csv_dump_path = os.path.join(BASE_DUMO_DIR, f'all_dtw_{args.start_idx}_{end_idx}.csv')
    print(f"saving df to {csv_dump_path}")
    scores_df.to_csv(csv_dump_path, index=False)

    # end_time = datetime.now()
    # time_diff1 = end_time - start_time1
    # total_seconds1 = time_diff1.total_seconds()
    # hours = int(total_seconds1 // 3600)
    # minutes = int((total_seconds1 % 3600) // 60)
    # seconds = int(total_seconds1 % 60)
    # milliseconds = int((total_seconds1 % 1) * 1000)
    # print(f"total running time: {hours:02d} hours, {minutes:02d} minutes, {seconds:02d} seconds, {milliseconds:02d} milliseconds")
