import pandas as pd
import os
import torch
import json
from tqdm import tqdm
import argparse

def save_dfs_to_json():
    ref_df = pd.read_csv('embeddings.csv')
    files_lst = os.listdir('../TMR_old/dtw_scores')
    dict_to_save = {}

    for file in tqdm(files_lst):
        current_file = pd.read_csv(f'../TMR_old/dtw_scores/{file}')
        for _, row in current_file.iterrows():
            i = int(row['i'])
            j = int(row['j'])

            keyid1 = ref_df.iloc[i]['keyids']
            keyid2 = ref_df.iloc[j]['keyids']

            distance = row['distance']
            
            if keyid1 not in dict_to_save:
                dict_to_save[keyid1] = {}

            dict_to_save[keyid1][keyid2] = distance

    with open('dtw_scores.json', 'w') as f:
        json.dump(dict_to_save, f)

def save_dfs_to_pt():
    files_lst = os.listdir('../TMR_old/dtw_scores')
    torch_matrix = torch.empty(27448, 27448)

    for file in tqdm(files_lst):
        current_file = pd.read_csv(f'../TMR_old/dtw_scores/{file}')
        for _, row in current_file.iterrows():
            i = int(row['i'])
            j = int(row['j'])

            distance = row['distance']
            torch_matrix[i][j] = distance
    torch.save(torch_matrix, 'dtw_scores.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()
    mode = args.mode

    if mode == "json":
        save_dfs_to_json()
    else:
        save_dfs_to_pt()