# import os
# from src.load import load_model_from_cfg
# from src.config import read_config
# from torch.utils.data import DataLoader
# from train_flag_dataloader import FlagDataSet
# from src.data.collate import collate_text_motion
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# test_dataset = FlagDataSet(split="test")
# test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=1, collate_fn=collate_text_motion, shuffle=False)
# data_iter = iter(test_dataloader)
# print(f"length: {len(data_iter)}")
# file_1 = np.load('datasets/motions/flag_subset_pose_data/M001P001A001R001.npy')
# file_2 = np.load('datasets/motions/flag_subset_pose_data/M001P001A001R002.npy')
# dtw_score_original = fastdtw(file_1, file_2, dist=euclidean)

file_1 = np.load('datasets/motions/flag_subset_pose_data/M001P001A001R001.npy')
print(f"file_1 shape: {file_1.shape}")
file_2 = np.load('datasets/motions/flag_subset_pose_data/M001P001A002R001.npy')
print(f"file_2 shape: {file_2.shape}")

file_1_flattened = file_1.reshape(file_1.shape[0], -1)
file_2_flattened = file_2.reshape(file_2.shape[0], -1) 

dtw_score_feats = fastdtw(file_1_flattened, file_2_flattened, dist=euclidean)

# print(f"original files score: {dtw_score_original}; feature files score: {dtw_score_feats}")
print(f"score: {dtw_score_feats[0]}")