import os
import json
import torch
import numpy as np
import pandas as pd
from src.model.tmr import get_sim_matrix
from src.config import read_config
from src.data.collate import collate_text_motion
from src.load import load_model_from_cfg
from tqdm import tqdm
import yaml
from src.model.metrics import all_contrastive_metrics
from torch.utils.data import Dataset

class FlagDataSet(Dataset):
    def __init__(self, split="train"):
        self.split = split
        with open('flag_action_annotations.json', 'r') as file:
            self.annotations_ref = json.load(file)
        with open('datasets/annotations/flag3d/sent_embeddings/sentence-transformers/all-mpnet-base-v2_index.json', 'r') as file:
            self.sent_annot_idx = json.load(file)
        self.all_sent_annotations = np.load('datasets/annotations/flag3d/sent_embeddings/sentence-transformers/all-mpnet-base-v2.npy')
        with open('datasets/annotations/flag3d/token_embeddings/distilbert-base-uncased_index.json', 'r') as file:
            self.token_annot_idx = json.load(file)
        self.token_annot_slice = np.load('datasets/annotations/flag3d/token_embeddings/distilbert-base-uncased_slice.npy')
        self.all_token_annotations = np.load('datasets/annotations/flag3d/token_embeddings/distilbert-base-uncased.npy')

        with open(f"datasets/annotations/flag3d/splits/{split}.txt", "r") as file:
            self.keyids_list = [int(line.strip()) for line in file.readlines()]
        ref_df = pd.read_csv("flag_ref.csv")
        self.ref_df = ref_df[ref_df['keyids'].isin(self.keyids_list)]

    def __len__(self):
        return len(self.ref_df)

    def __getitem__(self, idx):
        row = self.ref_df.iloc[idx]
        device = "cuda:0"
        curr_motion = np.load(row['motion path'])
        curr_motion = torch.from_numpy(curr_motion).to(device)
        motion_x_dict = {"x": curr_motion, "length": len(curr_motion)}
        action_id = row['motion path'][49:53]
        action_annotation = self.annotations_ref[action_id]
        sent_emb = torch.from_numpy(self.all_sent_annotations[self.sent_annot_idx[action_annotation]]).to(device)
        token_begin, token_end = self.token_annot_slice[self.token_annot_idx[action_annotation]]
        text_x_dict = {
            "x": torch.from_numpy(self.all_token_annotations[token_begin:token_end]).to(device),
            "length": token_end - token_begin
        }

        return {
            "motion_x_dict": motion_x_dict,
            "sent_emb": sent_emb,
            "text_x_dict": text_x_dict,
            "keyid": row['keyids'],
            "text": row['annotations']
        }



    # def __getitem__(self, idx):
    #     row = self.ref_df.iloc[idx]
    #     device = "cuda:0"
    #     curr_motion = np.load(row['motion path'])
    #     curr_motion = torch.from_numpy(curr_motion).to(torch.float).to(device)
    #     motion_x_dict = {"x": curr_motion, "length": len(curr_motion)}
    #     action_id = row['motion path'][49:53]
    #     action_annotation = self.annotations_ref[action_id]
    #     sent_emb = (self.all_sent_annotations[self.sent_annot_idx[action_annotation]])
    #     token_begin, token_end = self.token_annot_slice[self.token_annot_idx[action_annotation]]
    #     text_x_dict = {"x": torch.from_numpy(self.all_token_annotations[token_begin:token_end]).to(torch.float).to(device), "length": token_end - token_begin}
    #     return {
    #         "motion_x_dict": motion_x_dict,
    #         "sent_emb": sent_emb,
    #         "text_x_dict": text_x_dict,
    #         "keyid": row['keyids'],
    #         "text": row['annotations']
    #     }