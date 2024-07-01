import os
import pandas as pd
from tqdm import tqdm
import argparse
import json
import torch
import numpy as np
import random

'''
Generate positive samples and negative samples in a json file. 
Use a threshold provided

{
    "anchor_keyid":
        {
            "anchor_motion_path": anchor_motion_path,

            "positive_sample_keyids": [keyids],
            "positive_sample_distances": [distances],
            "positive_sample_motion_path": positive_sample_path,

            "negative_samples_keyids": [keyids],
            "negative_sample_distances": [distances],
            "negative_sample_motion_path": negative_sample_path

        }
}
'''
def gen_samples_dtw(threshold):
    samples_dict ={}

    main_df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR/embeddings.csv')
    files_dir = '/vulcanscratch/mukunds/downloads/TMR_old/dtw_scores'
    files_list = os.listdir(files_dir)
    num_files = len(files_list)
    
    total_no_samples = 0

    for i in tqdm(range(num_files)):
        counter = 0
        file = files_list[i]
        df = pd.read_csv(f"{files_dir}/{files_list[i]}")
        anchor_idx_list = df['i'].unique()

        for idx in anchor_idx_list:
            anchor_keyid = (main_df.iloc[[idx]])['keyids'].item()
            anchor_motion_path = (main_df.iloc[[idx]])['motion path'].item()

            tmp_df = df[df['i'] == idx]
            positives = tmp_df[tmp_df['distance'].astype('float64') < threshold]
            negatives = tmp_df[tmp_df['distance'].astype('float64') >= threshold]

            # if positives.shape[0] >= 50:
                # positive_samples = positives.sample(50)
            # else:
                # positive_samples = positives.sample(positives.shape[0])
            positive_samples = positives
            
            # breakpoint()
            if positive_samples.shape[0] == 0:
                counter += 1
            
            # if anchor_keyid == main_df.iloc[2]['keyids']:
                # breakpoint()

            if negatives.shape[0] >= 50:
                negative_samples = negatives.sample(50)
            else:
                negative_samples = negatives.sample(negatives.shape[0])
            # negative_samples = negatives

            positive_keyids = positive_samples['j'].map(main_df['keyids'])
            negative_keyids = negative_samples['j'].map(main_df['keyids'])

            positive_motion_paths = positive_samples['j'].map(main_df['motion path'])
            negative_motion_paths = negative_samples['j'].map(main_df['motion path'])

            positive_distances = positive_samples['distance']
            negative_distances = negative_samples['distance']

            samples_dict[anchor_keyid] = {
                "anchor_motion_path": anchor_motion_path,

                "positive_sample_keyids": positive_keyids.to_list(),
                "positive_sample_distances": positive_distances.to_list(),
                "positive_sample_motion_paths": positive_motion_paths.to_list(),

                "negative_sample_keyids": negative_keyids.to_list(),
                "negative_sample_distances": negative_distances.to_list(),
                "negative_sample_motion_paths": negative_motion_paths.to_list()
            }
        
        # print(f"In file {file}, {counter} motions out of 75 have no positive sample...")
        total_no_samples += counter

    with open(f"samples_motion_{threshold}_unsampled.json", "w") as outfile:
        json.dump(samples_dict, outfile)
    print(f"Total number of motions with no positive sample: {total_no_samples}")

    return

'''
Generate positive samples and negative samples in a json file. 
Use a threshold provided

{
    "anchor_keyid":
        {
            "anchor_motion_path": anchor_motion_path,

            "positive_sample_keyids": [keyids],
            "positive_sample_distances": [distances],
            "positive_sample_motion_path": positive_sample_path,

            "negative_samples_keyids": [keyids],
            "negative_sample_distances": [distances],
            "negative_sample_motion_path": negative_sample_path

        }
}
'''
def gen_samples_text(threshold):
    sim_matrix = torch.load('sentence_sim_matrix.pt')
    ref_df = pd.read_csv('../TMR_old/outputs/tmr_humanml3d_guoh3dfeats_old1/new_latents/embeddings.csv')
    np_sim_matrix = sim_matrix.cpu().detach().numpy()
    samples_dict = {}

    # print('initialized values')

    real_threshold = 2 * threshold - 1
    idx = np.argwhere(np_sim_matrix > real_threshold)
    idx = np.argwhere((np_sim_matrix > real_threshold) & (np.arange(np_sim_matrix.shape[0])[:, None] != np.arange(np_sim_matrix.shape[1])))
    all_idx=np.arange(27448)

    # print('going thru ref_df')

    for i, row in tqdm(ref_df.iterrows(), total=ref_df.shape[0]):
        # print(f'i: {i}')
        anchor_keyid = row['keyids']
        anchor_motion_path = row['motion path']
        # pos_sample_idx = np.array([pair[1] for pair in idx if pair[0] == i])
        
        pos_indices = []
        row = np_sim_matrix[i]
        for j, dist in enumerate(row):
            if dist >= 0.95:
                pos_indices.append(j)
        
        pos_sample_idx = np.array(pos_indices)
        
        # if i == 2:
            # breakpoint()

        neg_sample_idx = np.setdiff1d(all_idx, pos_sample_idx)
        neg_sample_idx = np.random.choice(neg_sample_idx, 50, replace=False)
        
        positive_keyids = ref_df['keyids'].iloc[pos_sample_idx].tolist()
        negative_keyids = ref_df['keyids'].iloc[neg_sample_idx].tolist()

        # positive_motion_paths = pos_sample_idx.map(ref_df['motion path'])
        # negative_motion_paths = neg_sample_idx.map(ref_df['motion path'])
        positive_motion_paths = ref_df['motion path'].iloc[pos_sample_idx].tolist()
        negative_motion_paths = ref_df['motion path'].iloc[neg_sample_idx].tolist()


        samples_dict[anchor_keyid] = {
            "anchor_motion_path": anchor_motion_path,

            "positive_sample_keyids": positive_keyids,
            # "positive_sample_distances": positive_distances.to_list(),
            "positive_sample_motion_paths": positive_motion_paths,

            "negative_sample_keyids": negative_keyids,
            # "negative_sample_distances": negative_distances.to_list(),
            "negative_sample_motion_paths": negative_motion_paths
        }

        # print(f'got samples for {anchor_keyid}. pos: {pos_sample_idx}, neg: {neg_sample_idx}')
        # print(f'anchor row: {row["annotations"]}')
        # print(f'pos: {ref_df.iloc[pos_sample_idx[0]]["annotations"]}')
        # print(f'first 5 elements of neg: {ref_df.iloc[neg_sample_idx[0:5]]["annotations"]}')

    with open(f"samples_text_{threshold}_unsampled.json", "w") as outfile:
        json.dump(samples_dict, outfile)

    return

    # for row,col in idx:
        # if row == 1:
            # col_annot = ref_df.iloc[col]['annotations']
            # print(f"column {col} ~~~ {col_annot}")
        # else:
            # continue

def get_most_similar_text(keyid, ref_df, motion_sim_df, text_sim_matrix):
    keyid_idx = ref_df[ref_df['keyids'] == keyid].index[0]

    sim_row = text_sim_matrix[keyid_idx]
    # print(f"keyid_index: {keyid_idx}")
    
    values, indices = torch.sort(sim_row, descending=True)
    max_index = indices[1].item()
    max_val = values[1].item()

    row_annotation = ref_df.iloc[max_index]['annotations']

    dtw_dist = motion_sim_df[(motion_sim_df['i'] == keyid_idx) & (motion_sim_df['j'] == max_index)].iloc[0]['distance']
    
    print(f"Annotation of closest motion using text similarity: {row_annotation}, DTW distance: {dtw_dist}: , text sim: {max_val}")


def get_most_similar_motion(keyid, ref_df, motion_sim_df, text_sim_matrix):
    keyid_idx = ref_df[ref_df['keyids'] == keyid].index[0]
    relevant_rows = motion_sim_df[(motion_sim_df['i'] == keyid_idx) & (motion_sim_df['j'] != keyid_idx)]
    min_index = relevant_rows['distance'].idxmin()
    min_row = relevant_rows.loc[min_index]
    j_idx = min_row['j']
    j_dist = min_row['distance']
    j_idx = int(j_idx)

    # if j_dist == 0.0:
        # breakpoint()

    row_annotation = ref_df.iloc[j_idx]['annotations']

    text_sim = text_sim_matrix[keyid_idx][j_idx]

    print(f"Annotation of closest motion using DTW score: {row_annotation}, DTW distance: {j_dist}, text sim: {text_sim}")

# def gen_samples_both():
#     with open('samples_motion_200_unsampled.json') as jsonfile:
#         motion_samples = json.load(jsonfile)
#     with open('samples_text_0.95_unsampled.json') as jsonfile:
#         text_samples = json.load(jsonfile)

#     ref_df = pd.read_csv('../TMR_old/outputs/tmr_humanml3d_guoh3dfeats_old1/new_latents/embeddings.csv')
#     text_sim_matrix = torch.load('sentence_sim_matrix.pt')
#     motion_sim_df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR_old/dtw_scores/all_dtw_0_75.csv')
    
#     # diffs = []
#     empty_subset = []
#     counter = 0
#     for i, keyid in tqdm(enumerate(motion_samples.keys())):

#         pos_motion_samples = motion_samples[keyid]['positive_sample_keyids']
#         pos_text_samples = text_samples[keyid]['positive_sample_keyids']
        
#         # if len(pos_text_samples) == 1:
#             # print(f"keyid: {keyid}, pos_sample: {pos_text_samples}")

#         common = [common_id for common_id in pos_motion_samples if common_id in pos_text_samples]
#         if len(common) == 0:
#             empty_subset.append(keyid)
#             counter += 1
        
#         if len(empty_subset) == 10:
#             break
#         # diffs.append(max(len(pos_motion_samples) - len(common), len(pos_text_samples) - len(common)))
    
#     for keyid in empty_subset:
#         print(f"Motion annotation: {ref_df[ref_df['keyids'] == keyid].iloc[0]['annotations']}")
#         get_most_similar_text(keyid, ref_df, motion_sim_df, text_sim_matrix)
#         get_most_similar_motion(keyid, ref_df, motion_sim_df, text_sim_matrix)
#         print()

#     print(f"number of samples with no positive pair: {counter} / 27448")
#     print(f"{(counter / 27448) * 100}% of the dataset has no positive M/T sample")
#     # print(f"counter: {counter}")

#     # print(f"mean max diff b/w sample sets: {np.mean(diffs)}")
#     # print(f"there are {len(empty_subset)} keyids with no positive sample")

# def get_tm_samples(idx, distances, similarities):
#     filtered_distances = distances[distances['i'] == idx]
#     motion_ranks = filtered_distances.sort_values(by='distance').index.tolist()
#     motion_ranks = np.array(motion_ranks)
#     motion_ranks = motion_ranks + 1
#     motion_ranks = np.argsort(motion_ranks)
#     motion_ranks = motion_ranks + 1
#     # print(f"length of sorted motion sample indices: {motion_ranks.shape}")

#     sims = -similarities[idx]
#     text_ranks = np.argsort(sims)
#     text_ranks = text_ranks + 1
#     text_ranks = np.argsort(sims)
#     text_ranks = text_ranks + 1
#     # print(f"length of text_ranks: {text_ranks.shape}")

#     combined_ranks = motion_ranks + text_ranks
#     combined_ranks_idx = np.argsort(combined_ranks)
#     pos_samples = combined_ranks_idx[1:21]

#     ### Do 2-5%, 5-8%, 10-13%

#     # neg_sample_slice = combined_ranks[548:1372] # 2-5%
#     index_range = np.arange(548, 1372)
#     neg_sample_indices = np.random.choice(index_range, 20, replace=False)
#     neg_samples = combined_ranks[neg_sample_indices]

#     breakpoint()
    
#     return pos_samples, neg_samples

def get_tm_samples(idx, distances, similarities):
    filtered_distances = distances[distances['i'] == idx]
    motion_ranks = filtered_distances['distance'].argsort().argsort() + 1

    sims = -similarities[idx]
    text_ranks = np.argsort(sims).argsort() + 1

    combined_ranks = motion_ranks + text_ranks
    sorted_indices_by_rank = np.argsort(combined_ranks)

    pos_sample_indices = sorted_indices_by_rank[1:21]
    pos_motion_distances = []
    pos_text_distances = []
    for pos_sample_index in pos_sample_indices:
        motion_distance = filtered_distances[filtered_distances['j'] == pos_sample_index].iloc[0]['distance']
        text_distance = similarities[idx][pos_sample_index]

        pos_motion_distances.append(motion_distance)
        pos_text_distances.append(text_distance)

    num_elements = len(combined_ranks)
    # 2-5, 5-8, 10-13 for negative samples. Positives are always the 20 best
    second_percentile_idx = int(0.1 * num_elements)
    fifth_percentile_idx = int(0.13 * num_elements)

    percentile_index_range = sorted_indices_by_rank[second_percentile_idx:fifth_percentile_idx]
    neg_sample_indices = np.random.choice(percentile_index_range, 20, replace=False)
    
    neg_motion_distances = []
    neg_text_distances = []
    for neg_sample_index in neg_sample_indices:
        motion_distance = filtered_distances[filtered_distances['j'] == neg_sample_index].iloc[0]['distance']
        text_distance = similarities[idx][neg_sample_index]

        neg_motion_distances.append(motion_distance)
        neg_text_distances.append(text_distance)

    return pos_sample_indices, neg_sample_indices, pos_motion_distances, pos_text_distances, neg_motion_distances, neg_text_distances

def gen_samples_both():
    ref_df = pd.read_csv('embeddings.csv')
    sim_matrix = torch.load('sentence_sim_matrix.pt')
    sim_matrix = sim_matrix.detach().cpu().numpy()
    total_motions = 27448
    all_files = []
    
    for i in range(0, total_motions, 75):
        all_files.append(f'all_dtw_{i}_{min(i+75, 27448)}.csv')
    
    all_files_idx = -1
    samples_dict = {}
    for i, row in tqdm(ref_df.iterrows(), total=ref_df.shape[0]):
        anchor_keyid = row['keyids']
        anchor_motion_path = row['motion path']
        if i % 75 == 0:
            all_files_idx += 1
            curr_file = pd.read_csv(f'../TMR_old/dtw_scores/{all_files[all_files_idx]}')
        
        pos_sample_idx, neg_sample_idx, pos_motion_distances, pos_text_distances, neg_motion_distances, neg_text_distances = get_tm_samples(i, curr_file, sim_matrix)
        
        positive_keyids = ref_df['keyids'].iloc[pos_sample_idx].tolist()
        negative_keyids = ref_df['keyids'].iloc[neg_sample_idx].tolist()

        positive_motion_paths = ref_df['motion path'].iloc[pos_sample_idx].tolist()
        negative_motion_paths = ref_df['motion path'].iloc[neg_sample_idx].tolist()

        samples_dict[anchor_keyid] = {
            "anchor_motion_path": anchor_motion_path,

            "positive_sample_keyids": positive_keyids,
            "positive_motion_distance": f"{pos_motion_distances}",
            "positive_text_distance": f"{pos_text_distances}",
            "positive_sample_motion_paths": positive_motion_paths,

            "negative_sample_keyids": negative_keyids,
            "negative_motion_distance": f"{neg_motion_distances}",
            "negative_text_distance": f"{neg_text_distances}",
            "negative_sample_motion_paths": negative_motion_paths
        }
    
    with open(f"samples_both.json", "w") as outfile:
        json.dump(samples_dict, outfile)
    
    return
    

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', help='Similarity score threshold to use when generating samples', type=float, default=0.95)
parser.add_argument('--type', help='text, motion, or both?', type=str)
args = parser.parse_args()

threshold = args.threshold
if threshold > 1:
    threshold = int(threshold)

if args.type == "text":
    gen_samples_text(threshold)
if args.type == "both":
    gen_samples_both()
if args.type == "dtw":
    gen_samples_dtw(threshold)