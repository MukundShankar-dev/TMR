import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import re

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
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
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

    def get_idx(self, keyids):
        keyid_idx = self.ref_df[self.ref_df['keyids'].isin(keyids)].index.to_list()
        sim_matrix = torch.empty((32, 32))
        
        for i, keyid1 in enumerate(keyid_idx):
            start = (keyid1 // 75) * 75
            scores_df = self.all_dfs[self.df_indices[start]]
            for j, keyid2 in enumerate(keyid_idx):
                if sim_matrix[i][j] != None:
                    distance = scores_df[(scores_df['i'] == keyid1) & (scores_df['j'] == keyid2)].iloc[0]['distance']
                    sim_matrix[i][j] = distance
                    sim_matrix[j][i] = distance

        idx = torch.where(sim_matrix < 200)
        return idx

    # x = text latents, y = motion latents.
    def __call__(self, x, y, keyid, sent_emb=None):
        bs, device = len(x), x.device
        
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature
        # TODO add a flag here
        # TODO instead of checking text sim in text, check it in the DTW space
        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            # NOTE Here, instead of thresholding this way, take thresholds with DTW
                # where DTW similarity is below a certain threshold
            # Try implementing both with DTW scores AND with combined ranks
            # idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            idx = self.get_idx(keyid)
            # breakpoint()
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

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