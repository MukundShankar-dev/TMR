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

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def load_flag_motions():
    all_data = []
    device = "cuda:0"

    ref_df = pd.read_csv("flag_ref.csv")
    with open('flag_action_annotations.json', 'r') as file:
        annotations_ref = json.load(file)
    with open('datasets/annotations/flag3d/sent_embeddings/sentence-transformers/all-mpnet-base-v2_index.json', 'r') as file:
        sent_annot_idx = json.load(file)
    all_sent_annotations = np.load('datasets/annotations/flag3d/sent_embeddings/sentence-transformers/all-mpnet-base-v2.npy')
    with open('datasets/annotations/flag3d/token_embeddings/distilbert-base-uncased_index.json', 'r') as file:
        token_annot_idx = json.load(file)
    token_annot_slice = np.load('datasets/annotations/flag3d/token_embeddings/distilbert-base-uncased_slice.npy')
    all_token_annotations = np.load('datasets/annotations/flag3d/token_embeddings/distilbert-base-uncased.npy')

    for i, row in tqdm(ref_df.iterrows(), desc="Loading flag motions"):
        # if i >= 0:
            # if i < 3600:
            curr_motion = np.load(row['motion path'])
            curr_motion = torch.from_numpy(curr_motion).to(torch.float).to(device)
            motion_x_dict = {"x": curr_motion, "length": len(curr_motion)}
            action_id = row['motion path'][49:53]
            action_annotation = annotations_ref[action_id]

            sent_emb = (all_sent_annotations[sent_annot_idx[action_annotation]])

            token_begin, token_end = token_annot_slice[token_annot_idx[action_annotation]]
            text_x_dict = {"x": torch.from_numpy(all_token_annotations[token_begin:token_end]).to(torch.float).to(device), "length": token_end - token_begin}
            all_data.append({"motion_x_dict": motion_x_dict, "sent_emb": sent_emb, "text_x_dict": text_x_dict})
            # else:
                # break
    return all_data

if __name__ == '__main__':
    all_data = load_flag_motions()
    all_metrics = []
    batch_size = 256
    nsplit = len(all_data) // batch_size
    all_data_splitted = np.array_split(all_data, nsplit)

    cfg = read_config("outputs/flag_few_shot_test")
    cfg["use_wandb"] = False
    ckpt_name = "last-v1"
    device = "cuda"
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    latent_texts = []
    latent_motions = []
    sent_embs = []

    with torch.inference_mode():
        for data in tqdm(all_data_splitted, desc="Calculating metrics"):
            # print("batch.")
            # print("getting batch.")
            batch = collate_text_motion(data, device=model.device)

            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            sent_emb = batch["sent_emb"]

            # Encode both motion and text
            # print("encoding.")
            latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)

            # print("latent_texts/motions appending")
            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)

    # for i in range(6):
        # sim_matrix, sent_embs = get_latent_text_motion(motion_batches[i], token_embd_batches[i], sent_embds_batches[i])
        normal_metrics = all_contrastive_metrics(sim_matrix, None, threshold=None)
        all_metrics.append(normal_metrics)

        threshold_metrics = all_contrastive_metrics(sim_matrix, sent_embs, threshold=0.95)
        all_metrics.append(threshold_metrics)

    save_metric("flag_normal.yaml", normal_metrics)
    save_metric("flag_threshold.yaml", threshold_metrics)