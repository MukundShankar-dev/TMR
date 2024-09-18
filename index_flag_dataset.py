import numpy
import os
import pandas as pd
import joblib
import json

def generate_ids_and_splits():
    print('Retrieving motion data')
    base_dir='datasets/motions/flag_subset_guoh3dfeats/'
    file_list = [f for f in os.listdir(base_dir) if f != 'M' and os.path.isfile(os.path.join(base_dir, f))]
    num_digits = len(str(len(file_list)))

    print('Retrieving annotations')
    annotation_dict = json.load(open('flag_action_annotations.json', 'r'))
    print('Generating key ids')
    key_ids = [str(i).zfill(num_digits) for i in range(1, len(file_list) + 1)]
    motion_paths = [os.path.join(base_dir, file_name) for file_name in file_list]
    annotations = [annotation_dict[file_name[8:12]] for file_name in file_list]

    print('Generating DataFrame')
    df = pd.DataFrame({
        'keyids': key_ids,
        'motion path': motion_paths,
        'annotations': annotations
    })

    print('Saving IDs, paths, and annotations to ./flag_ref.csv')
    df.to_csv('flag_ref.csv', index=False)

    print('Generating train and test splits')
    train_split = key_ids[:int(len(key_ids) * 0.7)] # 70% of the data
    test_split = key_ids[int(len(key_ids) * 0.7):]  # 30% of the data

    print('Saving train and test splits to ./flag_train_split.txt and ./flag_test_split.txt')
    with open('flag_train_split.txt', 'w') as f:
        for item in train_split:
            f.write("%s\n" % item)
    
    with open('flag_test_split.txt', 'w') as f:
        for item in test_split:
            f.write("%s\n" % item)

if __name__ == '__main__':
    generate_ids_and_splits()

### Order of steps FOR INFERENCE (NO TRAINING) ###
# Get motion data ready (all the way through to guoh3dfeats)    √
# Get annotation data ready (just need the Action annotations)
# Then, we can make annotations.json
# Define splits
# Change dataloader
# Inference + Metrics

### Order of steps FOR TRAINING ###
# Get motion data ready (all the way through to guoh3dfeats)
# Get annotation data ready (need mpnet embeddings, distilbert embeddings, and the annotations themselves)
# Then, we can make annotations.json
# Define splits
# Change dataloader
# Train