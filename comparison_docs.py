import pandas as pd
import argparse
import os
import ast
import json

def get_metrics(direction, protocol):
    vanilla_file_path = f'old_outputs/tmr_humanml3d_guoh3dfeats_vanilla_model/contrastive_metrics_2/{direction}_{protocol}_keyid_metrics.csv'
    ours_file_path = f'outputs/new_infonce/contrastive_metrics_2/{direction}_{protocol}_keyid_metrics.csv'

    vanilla_df = pd.read_csv(vanilla_file_path)
    ours_df = pd.read_csv(ours_file_path)
    df_cols = ['keyid', 'annotations', 'vanilla_R01', 'ours_R01', 'vanilla_R02', 'ours_R02', 'vanilla_R03', 'ours_R03', 'vanilla_R05', 'ours_R05', 'vanilla_R10', 'ours_R10']
    new_df = df = pd.DataFrame(columns=df_cols)

    num_motions = len(vanilla_df)

    for i in range(num_motions):
        vanilla_row = vanilla_df.iloc[i]
        ours_row = ours_df.iloc[i]
        keyid = vanilla_row['keyid']
        keyid_annotations = vanilla_row['annotations']
        new_df.loc[len(df.index)] = [keyid, keyid_annotations, vanilla_row['R01'],ours_row['R01'], vanilla_row['R02'],ours_row['R02'], vanilla_row['R03'],ours_row['R03'], vanilla_row['R05'],ours_row['R05'], vanilla_row['R10'],ours_row['R10']]

    return new_df

def save_annotations(df, filename, retrieval_file):
    ref_df = pd.read_csv('embeddings.csv')
    
    with open(f'comparitive_metrics/annotations/{filename}.txt', 'w') as file:
        for idx, row in df.iterrows():
            annotations_lst = ast.literal_eval(row['annotations'])
            file.write('anchor:\n')
            file.write(annotations_lst[0])
            file.write('\n')
            file.write('retrieved:\n')
            try:
                file.write(ref_df[ref_df['keyids'] == retrieval_file[row['keyid']]].iloc[0]['annotations'])
            except:
                pass
            file.write('\n\n')
    return

def get_hit_df(df):
    for idx, row in df.iterrows():
        ours_hit = row['vanilla_R01']

def process_metrics(direction, protocol):
    file_path = f"combined_metrics/{direction}_{protocol}.csv"
    df = pd.read_csv(file_path)
    
    both = df[(df['vanilla_R10'] == True) & (df['ours_R10'] == True)]
    only_vanilla = df[(df['vanilla_R10'] == True) & (df['ours_R10'] == False)]
    only_ours = df[(df['vanilla_R10'] == False) & (df['ours_R10'] == True)]
    neither = df[(df['vanilla_R10'] == False) & (df['ours_R10'] == False)]

    retrieved_items_dict = f"{protocol}_{direction}_retrievals"
    with open(f"outputs/new_infonce/contrastive_metrics_2/{retrieved_items_dict}") as file:
        retrieval_file = json.load(file)

    save_annotations(both, f"{direction}_{protocol}_both", retrieval_file)
    save_annotations(only_vanilla, f"{direction}_{protocol}_vanilla", retrieval_file)
    save_annotations(only_ours, f"{direction}_{protocol}_ours", retrieval_file)
    save_annotations(neither, f"{direction}_{protocol}_neither", retrieval_file)

    print(f'saved files for {direction}, {protocol}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--direction", type=str, default="all")
    parser.add_argument("--protocols", type=str, default="all")
    parser.add_argument("--mode", type=str, default="save")

    args = parser.parse_args()
    mode = args.mode
    
    if args.direction == "all":
        directions = ["t2m", "m2t"]
    if args.protocols == "all":
        protocols = ["normal", "threshold_0.95"]

    if mode == "save":
        save_dir = 'combined_metrics'
        os.makedirs(save_dir, exist_ok=True)
    
        for direction in directions:
            for protocol in protocols:
                combined_metrics = get_metrics(direction, protocol)
                save_path = f'{save_dir}/{direction}_{protocol}.csv'
                combined_metrics.to_csv(save_path)
                print(f"doc done. saved in {save_path}")

    elif mode == "read":
        for direction in directions:
            for protocol in protocols:
                process_metrics(direction, protocol)