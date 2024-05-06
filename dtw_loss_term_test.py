import pandas as pd
import numpy as np
from tqdm import tqdm 
import time
from dtw_loss_term import DTWLoss

# df = pd.read_csv('final_scores/scores1.csv')
df = pd.read_csv('/vulcanscratch/mukunds/downloads/TMR/outputs/tmr_humanml3d_guoh3dfeats_old1/new_latents/embeddings.csv')

dtw_loss = DTWLoss(100)
batch_mean_losses = []

# df["keyid_1"] = df["keyid_1"].astype(str)
# print(f"head: {df.head()}")

for j in tqdm(range(10)):
    losses = []
    sample = df['keyids'].sample(100)
    batch = {"keyids": sample.to_list()}
    # batch = {"keyids": [row['keyids'] for idx,row in sample.iterrows()]}
    # print(f"batch: {batch}")
    times = []
    
    for i in range(10):
        start_time = time.time()
        losses.append(dtw_loss(batch)) 
        end_time = time.time()

        times.append(end_time - start_time)

    batch_mean_loss = np.mean(losses)
    batch_mean_losses.append(batch_mean_loss)
    print(f"mean loss for batch {j+1}: {batch_mean_loss}")
    print(f"max loss for batch {j+1}: {max(losses)}")
    print(f"min loss for batch {j+1}: {min(losses)}")
    print('~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~')

    print(f"mean time taken for one batch: {np.mean(times)}")

print(f"mean loss per call: {np.mean(batch_mean_losses)}")