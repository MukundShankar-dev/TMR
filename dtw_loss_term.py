# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn

# # Look at evaluation metrics from TMR paper
#     # How many testing samples?
#     # What do metrics mean?
#     # Run some iterations, make sure triplet loss is going down. Integrate wandb

# class DTWLoss:
#     def __init__(self, threshold):
#         # df = pd.read_csv('./all_dtw_scores.csv')
#         self.df = pd.read_csv('./final_scores/scores1.csv')
#         # self.df = df[df['keyid'] in batch{"keyids"}]
#         self.threshold = threshold

#     def get_positive_sample(self, keyid):
#         sample_from = self.df[self.df['keyid_1'] == keyid]
#         thresholded = sample_from[sample_from['distance'] <= self.threshold]
#         if thresholded.shape[0] > 0:
#             sample = thresholded.sample()
#             return sample.iloc[0].to_dict()
#         else:
#             sorted_df = sample_from.sort_values(by='distance', ascending=True)
#             return sorted_df.iloc[0].to_dict()

#     def get_negative_sample(self, keyid):
#         sample_from = self.df[self.df['keyid_1'] == keyid]
#         thresholded = sample_from[sample_from['distance'] > self.threshold]
#         if thresholded.shape[0] > 0:
#             sample =  thresholded.sample()
#             return sample.iloc[0].to_dict()
#         else:
#             sorted_df = sample_from.sort_values(by='distance', ascending=True)
#             return sorted_df.iloc[sorted_df.shape[0] - 1].to_dict()
        

#     def pad_array(self, array, max_length):
#         padding_needed = max_length - array.shape[0]
#         if padding_needed > 0:
#             return np.pad(array, pad_width=((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
#         else:
#             return array

#     def __call__(self, batch):
#         # triplet_loss = 0
#         losses = []
        
#         triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

#         for keyid in batch['keyids']:
#             positive_sample = self.get_positive_sample(keyid)
#             negative_sample = self.get_negative_sample(keyid)

#             anchor_motion_path = positive_sample['keyid_1_path']
#             positive_motion_path = positive_sample['keyid_2_path']
#             negative_motion_path = negative_sample['keyid_2_path']

#             base_path = '/vulcanscratch/mukunds/downloads/TMR/datasets/motions/guoh3dfeats/'
            
#             anchor_motion = np.load(f'{base_path}{anchor_motion_path}.npy')
#             positive_motion = np.load(f'{base_path}{positive_motion_path}.npy')
#             negative_motion = np.load(f'{base_path}{negative_motion_path}.npy')

#             max_length = max(anchor_motion.shape[0], positive_motion.shape[0], negative_motion.shape[0])

#             anchor_motion_padded = self.pad_array(anchor_motion, max_length)
#             positive_motion_padded = self.pad_array(positive_motion, max_length)
#             negative_motion_padded = self.pad_array(negative_motion, max_length)

#             # anchor_pos_L2 = np.linalg.norm(anchor_motion_padded - positive_motion_padded)
#             # anchor_neg_L2 = np.linalg.norm(anchor_motion_padded - negative_motion_padded)

#             anchor_motion_padded = torch.from_numpy(anchor_motion_padded)
#             positive_motion_padded = torch.from_numpy(positive_motion_padded)
#             negative_motion_padded = torch.from_numpy(negative_motion_padded)

#             losses.append(triplet_loss(anchor_motion_padded, positive_motion_padded, negative_motion_padded).item())
#             # triplet_loss += anchor_pos_L2 - anchor_neg_L2
#             # losses.append(anchor_pos_L2 - anchor_neg_L2)
        
#         return np.mean(losses).item()

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import random

# Look at evaluation metrics from TMR paper
    # How many testing samples?
    # What do metrics mean?
    # Run some iterations, make sure triplet loss is going down. Integrate wandb

class DTWLoss:
    def __init__(self, threshold):
        f = open("/vulcanscratch/mukunds/downloads/TMR/samples_200.json")
        self.samples = json.load(f)
        self.threshold = threshold


    def get_samples(self, keyid):
        # print("in get_samples function...")
        to_process = self.samples[keyid]
        anchor_motion_path = to_process['anchor_motion_path']
        len_pos_samples = len(to_process['positive_sample_keyids'])
        len_neg_samples = len(to_process['negative_sample_keyids'])

        rand_pos = random.randint(0, len_pos_samples - 1)
        rand_neg = random.randint(0, len_neg_samples - 1)

        positive_sample_keyid = to_process['positive_sample_keyids'][rand_pos]
        positive_sample_distance = to_process['positive_sample_distances'][rand_pos]
        positive_sample_path = to_process['positive_sample_motion_paths'][rand_pos]

        # print(f"positive keyid: {positive_sample_keyid}, dist: {positive_sample_distance}, path: {positive_sample_path}")

        negative_sample_keyid = to_process['negative_sample_keyids'][rand_neg]
        negative_sample_distance = to_process['negative_sample_distances'][rand_neg]
        negative_sample_path = to_process['negative_sample_motion_paths'][rand_neg]
        
        # print(f"negative keyid: {negative_sample_keyid}, dist: {negative_sample_distance}, path: {negative_sample_path}")

        return {
            "anchor_motion_path": anchor_motion_path,

            "positive_sample_keyid": positive_sample_keyid,
            "positive_sample_distance": positive_sample_distance,
            "positive_sample_path": positive_sample_path,
            
            "negative_sample_keyid": negative_sample_keyid,
            "negative_sample_distance": negative_sample_distance,
            "negative_sample_path": negative_sample_path
        }
        

    def pad_array(self, array, max_length):
        padding_needed = max_length - array.shape[0]
        if padding_needed > 0:
            return np.pad(array, pad_width=((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
        else:
            return array

    def __call__(self, batch):
        # triplet_loss = 0
        losses = []
        
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

        for keyid in batch['keyids']:
            try:
                samples = self.get_samples(keyid)
            except:
                continue

            anchor_motion_path = samples['anchor_motion_path']
            # print(f"anchor motion path: {anchor_motion_path}")
            positive_motion_path = samples['positive_sample_path']
            # print(f"positive sample path: {positive_motion_path}")
            negative_motion_path = samples['negative_sample_path']
            # print(f"negative sample path: {negative_motion_path}")

            base_path = '/vulcanscratch/mukunds/downloads/TMR/datasets/motions/guoh3dfeats/'
            
            # try:
            anchor_motion = np.load(f'{base_path}{anchor_motion_path}.npy')
            positive_motion = np.load(f'{base_path}{positive_motion_path}.npy')
            negative_motion = np.load(f'{base_path}{negative_motion_path}.npy')
            # except:
                # continue

            max_length = max(anchor_motion.shape[0], positive_motion.shape[0], negative_motion.shape[0])

            anchor_motion_padded = self.pad_array(anchor_motion, max_length)
            positive_motion_padded = self.pad_array(positive_motion, max_length)
            negative_motion_padded = self.pad_array(negative_motion, max_length)

            # anchor_pos_L2 = np.linalg.norm(anchor_motion_padded - positive_motion_padded)
            # anchor_neg_L2 = np.linalg.norm(anchor_motion_padded - negative_motion_padded)

            anchor_motion_padded = torch.from_numpy(anchor_motion_padded)
            positive_motion_padded = torch.from_numpy(positive_motion_padded)
            negative_motion_padded = torch.from_numpy(negative_motion_padded)

            if torch.isnan(anchor_motion_padded).any() or torch.isnan(positive_motion_padded).any() or torch.isnan(negative_motion_padded).any():
                continue
            losses.append(triplet_loss(anchor_motion_padded, positive_motion_padded, negative_motion_padded).item())
        return np.mean(losses).item()