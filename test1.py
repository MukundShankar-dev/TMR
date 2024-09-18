# import torch
import numpy as np
import json

"""
Test out reading the sentence embeddings
"""



if __name__ == '__main__':
    # Load the sentence embeddings
    emb = np.load('datasets/annotations/humanml3d/sent_embeddings/sentence-transformers/all-mpnet-base-v2.npy')
    # emb = np.load('datasets/annotations/humanml3d/token_embeddings/distilbert-base-uncased.npy')
    with open('datasets/annotations/humanml3d/sent_embeddings/sentence-transformers/all-mpnet-base-v2_index.json', 'r') as f:
        index = json.load(f)
    breakpoint()
