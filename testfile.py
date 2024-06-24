import pandas as pd
import numpy as np
import torch

def get_tm_samples(idx, distances, similarities):
    filtered_distances = distances[distances['i'] == idx]
    motion_ranks = filtered_distances.sort_values(by='distance').index.tolist()
    motion_ranks = np.array(motion_ranks)
    print(f"length of sorted motion sample indices: {motion_ranks.shape}")

    sims = similarities[idx]
    text_ranks = np.argsort(sims)
    print(f"length of text_ranks: {text_ranks.shape}")

    combined_ranks = motion_ranks + text_ranks
    combined_ranks_idx = np.argsort(combined_ranks)
    pos_samples = combined_ranks_idx[:21]
    neg = combined_ranks_idx[-20:]
    
    return

arr1 = np.array([0, 2, 4, 3, 1])
arr1 = arr1 + 1
print(f"pre-sort1: {arr1}")
arr2 = np.array([2, 4, 7, 5, 3])
print(f"pre-sort2: {arr2}")

sorted1 = np.argsort(arr1)
sorted1 = np.argsort(sorted1)
print(f"post-sort1: {sorted1}")
sorted2 = np.argsort(arr2)
sorted2 = np.argsort(sorted2)
print(f"post-sort2: {sorted2}")

combined_sort = sorted1 + sorted2
print(f"combined: {combined_sort}")

print(combined_sort[:2])