import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import re
import time

import cProfile
import io
import pstats
import numpy as np

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8, threshold_dtw=200):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.threshold_dtw = threshold_dtw
        self.ref_df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR/embeddings.csv')

        self.all_dfs = [None] * 366
        self.df_indices = {}
        all_files = os.listdir('/vulcanscratch/mukunds/downloads/TMR_old/dtw_scores')
        pattern = re.compile(r"all_dtw_(\d+)_(\d+)\.csv")
        extracted_values = [(int(match.group(1)), int(match.group(2))) for file in all_files if (match := pattern.match(file))]
        
        for i, file in enumerate(all_files):
            self.all_dfs[i] = pd.read_csv(f'/vulcanscratch/mukunds/downloads/TMR_old/dtw_scores/{file}')
            self.df_indices[extracted_values[i][0]] = i

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    # def get_idx(self, keyids):
    #     # start_time = time.time()
    #     keyid_idx = self.ref_df[self.ref_df['keyids'].isin(keyids)].index.to_list()

    #     # time_after_getting_keyids = time.time()

    #     sim_matrix = torch.full((32, 32), torch.inf)  # Initialize with torch.inf for diagonal elements

    #     for i, keyid1 in enumerate(keyid_idx):
    #         start = (keyid1 // 75) * 75
    #         scores_df = self.all_dfs[self.df_indices[start]]
    #         filtered_scores = scores_df[scores_df['i'].isin(keyid_idx) & scores_df['j'].isin(keyid_idx)]
            
    #         # time_after_filtering_scores = time.time()

    #         for _, row in filtered_scores.iterrows():
    #             i_idx = keyid_idx.index(row['i'])
    #             j_idx = keyid_idx.index(row['j'])
    #             distance = row['distance']
    #             if i_idx != j_idx:
    #                 sim_matrix[i_idx][j_idx] = distance
    #                 sim_matrix[j_idx][i_idx] = distance

    #         # breakpoint()

    #     # time_after_making_sim_matrix = time.time()
    #     idx = torch.where(sim_matrix < 200)
    #     # time_after_all_calculations = time.time()
    #     # breakpoint()
    #     return idx

    # def get_idx(self, keyids):
    #     # keyids_set = set(keyids)
    #     keyid_idx = self.ref_df.index[self.ref_df['keyids'].isin(keyids)].tolist()
    #     sim_matrix = torch.full((32, 32), torch.inf)  # Initialize with torch.inf for diagonal elements

    #     keyid_idx_dict = {keyid: idx for idx, keyid in enumerate(keyid_idx)}

    #     for start in set((keyid // 75) * 75 for keyid in keyid_idx):
    #         scores_df = self.all_dfs[self.df_indices[start]]

    #         # Filter using numpy for efficiency
    #         i_values = scores_df['i'].values
    #         j_values = scores_df['j'].values
    #         mask = np.isin(i_values, keyid_idx) & np.isin(j_values, keyid_idx)
    #         filtered_scores = scores_df[mask]

    #         i_array = filtered_scores['i'].values
    #         j_array = filtered_scores['j'].values
    #         distance_array = filtered_scores['distance'].values

    #         for i, j, distance in zip(i_array, j_array, distance_array):
    #             i_idx = keyid_idx_dict.get(i)
    #             j_idx = keyid_idx_dict.get(j)
    #             if i_idx is not None and j_idx is not None and i_idx != j_idx:
    #                 sim_matrix[i_idx][j_idx] = distance
    #                 sim_matrix[j_idx][i_idx] = distance

    #     idx = torch.where(sim_matrix < 200)
    #     return idx

    def get_idx(self, keyids):
        keyids_set = set(keyids)
        keyid_idx = self.ref_df.index[self.ref_df['keyids'].isin(keyids)].tolist()
        size = len(keyid_idx)
        sim_matrix = torch.full((size, size), torch.inf)  # Initialize with torch.inf for diagonal elements

        keyid_idx_dict = {keyid: idx for idx, keyid in enumerate(keyid_idx)}

        for start in set((keyid // 75) * 75 for keyid in keyid_idx):
            scores_df = self.all_dfs[self.df_indices[start]]

            # Filter using numpy for efficiency
            i_values = scores_df['i'].values            # [i, j, distance]
            j_values = scores_df['j'].values
            # NOTE: 
            mask = np.isin(i_values, keyid_idx) & np.isin(j_values, keyid_idx)
            filtered_scores = scores_df[mask]

            i_array = filtered_scores['i'].values
            j_array = filtered_scores['j'].values
            distance_array = filtered_scores['distance'].values

            # Cache dictionary lookups
            i_indices = np.array([keyid_idx_dict.get(i) for i in i_array])
            j_indices = np.array([keyid_idx_dict.get(j) for j in j_array])

            valid_mask = (i_indices != j_indices)
            i_indices = i_indices[valid_mask]
            j_indices = j_indices[valid_mask]
            distance_array = distance_array[valid_mask]

            sim_matrix[i_indices, j_indices] = torch.tensor(distance_array, dtype=sim_matrix.dtype)
            sim_matrix[j_indices, i_indices] = torch.tensor(distance_array, dtype=sim_matrix.dtype)

        idx = torch.where(sim_matrix < self.threshold_dtw)
        return idx
    
    # x = text latents, y = motion latents.
    def __call__(self, x, y, keyids, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature
        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            # real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values, mask them by putting -inf in the sim_matrix
            # selfsim = sent_emb @ sent_emb.T
            # selfsim_nodiag = selfsim - selfsim.diag().diag()

            # Try implementing both with DTW scores AND with combined ranks
            # idx = torch.where(selfsim_nodiag > real_threshold_selfsim)

            # pr = cProfile.Profile()
            # pr.enable()
            start_time=time.time()
            idx = self.get_idx(keyids)
            end_time=time.time()
            # pr.disable()
            # s = io.StringIO()
            # sortby = 'cumulative'
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        breakpoint()
        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"

class DTWLoss:
    def __init__(self, margin=0.1):        
        self.margin = margin

    def __call__(self, anchor_latent, positive_latent, negative_latent):
        triplet_loss = 0
        triplet_loss = nn.TripletMarginLoss(margin = self.margin, p=2, eps=1e-7)
        return triplet_loss(anchor_latent, positive_latent, negative_latent)
    
# Use cosine scores instead to bound losses?
class TripletLossCosine:
    def __init__(self, margin=0.1):
        self.margin = margin

    def __call__(self, anchor, positive, negative):

        pos_similarity = F.cosine_similarity(anchor, positive)
        neg_similarity = F.cosine_similarity(anchor, negative)

        losses = torch.relu(-pos_similarity + neg_similarity + self.margin)

        return losses.mean()

# class DifferentiableDTWLoss(nn.Module):
#     def __init__(self):
#         super(DifferentiableDTWLoss, self).__init__()

#     def __call__(self, x, y):
#         x_logits = torch.nn.functional.normalize(x, dim=-1)
#         y_logits = torch.nn.functional.normalize(y, dim=-1)
        
#         n, m = x.size(0), y.size(0)
#         cost = torch.zeros((n+1, m+1), device=x.device)
#         cost[1:, 0] = float('inf')
#         cost[0, 1:] = float('inf')

#         for i in range(1, n+1):
#             for j in range(1, m+1):
#                 dist = torch.norm(x[i-1] - y[j-1])
#                 cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
#         return cost[-1, -1]
    
# class FastDifferentiableDTWLoss(nn.Module):
#     def __init__(self):
#         super(FastDifferentiableDTWLoss, self).__init__()

#     def forward(self, x, y):
#         n, m = x.size(0), y.size(0)
#         cost = torch.zeros((n+1, m+1), device=x.device)
#         cost[1:, 0] = float('inf')
#         cost[0, 1:] = float('inf')

#         # Compute the cost matrix
#         for i in range(1, n+1):
#             cost_i_j = torch.cdist(x[i-1:i], y, p=2) + torch.min(
#                 torch.stack([cost[i-1, 1:], cost[i-1, :-1], cost[i, :-1]]), dim=0).values
#             cost[i, 1:] = cost_i_j

#         return cost[-1, -1]