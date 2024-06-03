import torch
import numpy as np
import pandas as pd

"""

"""

def test_text_similarities():
    sim_matrix = torch.load('sentence_sim_matrix.pt')
    ref_df = pd.read_csv('../TMR_old/outputs/tmr_humanml3d_guoh3dfeats_old1/new_latents/embeddings.csv')

    row_annot = ref_df.iloc[1]['annotations']
    np_sim_matrix = sim_matrix.cpu().detach().numpy()

    real_threshold = 2 * 0.95 - 1
    idx = np.argwhere(np_sim_matrix > real_threshold)
    idx = np.argwhere((np_sim_matrix > real_threshold) & (np.arange(np_sim_matrix.shape[0])[:, None] != np.arange(np_sim_matrix.shape[1])))

    print(f"{row_annot} is similar to:")
    for row,col in idx:
        if row == 1:
            col_annot = ref_df.iloc[col]['annotations']
            print(f"column {col} ~~~ {col_annot}")
        else:
            continue

if __name__ == '__main__':
    test_text_similarities()