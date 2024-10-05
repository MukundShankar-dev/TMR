import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

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


# class InfoNCE_with_filtering:
#     def __init__(self, temperature=0.7, threshold_selfsim=0.8, threshold_dtw=200, mode="text"):
#         self.temperature = temperature
#         self.threshold_selfsim = threshold_selfsim
#         self.threshold_dtw = threshold_dtw
#         self.mode = mode

#         if mode == "motion":
#             self.ref_df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR/embeddings.csv')
#             # self.all_scores = torch.load('/vulcanscratch/mukunds/downloads/TMR/dtw_scores.pt')

#     def get_sim_matrix(self, x, y):
#         x_logits = torch.nn.functional.normalize(x, dim=-1)
#         y_logits = torch.nn.functional.normalize(y, dim=-1)
#         sim_matrix = x_logits @ y_logits.T
#         return sim_matrix

#     def get_idx(self, keyids):
#         # keyid_idx = self.ref_df.index[self.ref_df['keyids'].isin(keyids)].tolist()
#         # size = len(keyid_idx)
#         sim_matrix = torch.full((size, size), torch.inf)

#         # sim_matrix = self.all_scores[torch.tensor(keyid_idx)][:, torch.tensor(keyid_idx)]
#         sim_matrix.fill_diagonal_(torch.inf)
#         # idx = torch.where(sim_matrix < self.threshold_dtw)
#         return idx
    
#     # x = text latents, y = motion latents.
#     def __call__(self, x, y, keyids, sent_emb=None):
#         bs, device = len(x), x.device
#         sim_matrix = self.get_sim_matrix(x, y) / self.temperature
#         if sent_emb is not None and self.threshold_selfsim:

#             if self.mode == "text":
#                 # put the threshold value between -1 and 1
#                 real_threshold_selfsim = 2 * self.threshold_selfsim - 1
#                 # Filtering too close values, mask them by putting -inf in the sim_matrix
#                 selfsim = sent_emb @ sent_emb.T
#                 selfsim_nodiag = selfsim - selfsim.diag().diag()

#                 # Try implementing both with DTW scores AND with combined ranks
#                 idx = torch.where(selfsim_nodiag > real_threshold_selfsim)

#             if self.mode == "motion":
#                 idx = self.get_idx(keyids)
            
#             sim_matrix[idx] = -torch.inf

#         labels = torch.arange(bs, device=device)

#         total_loss = (
#             F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
#         ) / 2

#         return total_loss

#     def __repr__(self):
#         return f"Constrastive(temp={self.temp})"

class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"

class DTWLoss:
    def __init__(self, margin=0.15):        
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