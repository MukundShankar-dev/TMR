import os
import pandas as pd
from tqdm import tqdm

import json

# count = 0
# files = os.listdir('dtw_scores')
# num_files = len(files)
# print(f"number of files: {num_files}")
# missing_files = []
# for i in tqdm(range(num_files)):
#     file = files[i]
#     df = pd.read_csv(f'dtw_scores/{file}')
#     # print(df.shape[0])
#     if df.shape[0] != 2744800:
#         print(file)
#         count += 1

# print(f'total misshapen files: {count}')

# for i in range(0, 27448, 100):
#     if f"all_dtw_{i}_{i+100}.csv" not in files:
#         if f"all_dtw_{i}_{i+50}" in files and f"all_dtw_{i+50}_{i+100}" in files:
#             continue
#         else:
#             if f"all_dtw_{i}_{i+50}" not in files:
#                 print(f"i: {i}, i+50: {i+50}, i+100: {i+100}")
#                 missing_files.append(f"all_dtw_{i}_{i+50}.csv")
#                 count += 1
#             if f"all_dtw_{i+50}_{i+100}" not in files:
#                 print(f"i: {i}, i+50: {i+50}, i+100: {i+100}")
#                 missing_files.append(f"all_dtw_{i+50}_{i+100}.csv")
#                 count += 1

# print(f"missing file count: {count}")
# print(f"missing files list:\n{missing_files}")

# df = pd.read_csv('dtw_scores/all_dtw_0_100.csv')
# print(f"head: {df.head()}")

# print("~~~~~~~")

# df = df[df['i'] != df['j']]
# new_df = df[df['distance'] == 0]
# # print('shape: ' + str(new_df.shape[0]))
# print("head of where distance == 0:")
# print(new_df.head())

f = open('samples_200.json')
file = json.load(f)
f.close()
print(f"number of keys: {len(file.keys())}")