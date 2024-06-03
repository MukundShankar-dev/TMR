import pandas as pd
import argparse
import os
import ast

def get_metrics(direction, protocol):
    vanilla_file_path = f'old_outputs/tmr_humanml3d_guoh3dfeats_vanilla_model/contrastive_metrics_2/{direction}_{protocol}_keyid_metrics.csv'
    ours_file_path = f'old_outputs/tmr_cos_loss_0.15/contrastive_metrics_2/{direction}_{protocol}_keyid_metrics.csv'

    vanilla_df = pd.read_csv(vanilla_file_path)
    ours_df = pd.read_csv(ours_file_path)
    df_cols = ['keyid', 'annotations', 'vanilla_R01', 'ours_R01', 'vanilla_R02', 'ours_R02', 'vanilla_R03', 'ours_R03', 'vanilla_R05', 'ours_R05', 'vanilla_R10', 'ours_R10']
    new_df = df = pd.DataFrame(columns=df_cols)

    num_motions = len(vanilla_df)

    for i in range(num_motions):
        vanilla_row = vanilla_df.iloc[i]
        ours_row = ours_df.iloc[i]
        # rows are formatted as: keyid,R01,R02,R03,R05,R10,annotations

        keyid = vanilla_row['keyid']
        keyid_annotations = vanilla_row['annotations']

        new_df.loc[len(df.index)] = [keyid, keyid_annotations, vanilla_row['R01'],ours_row['R01'], vanilla_row['R02'],ours_row['R02'], vanilla_row['R03'],ours_row['R03'], vanilla_row['R05'],ours_row['R05'], vanilla_row['R10'],ours_row['R10']]

    return new_df

def process_metrics(direction, protocol):
    file_path = f"combined_metrics/{direction}_{protocol}.csv"
    df = pd.read_csv(file_path)
    
    both = df[(df['vanilla_R01'] == True) & (df['ours_R01'] == True)]
    only_vanilla = df[(df['vanilla_R01'] == True) & (df['ours_R01'] == False)]
    only_ours = df[(df['vanilla_R01'] == False) & (df['ours_R01'] == True)]
    neither = df[(df['vanilla_R01'] == False) & (df['ours_R01'] == False)]

    # print(f"both: {both.shape}")
    # print(f"vanilla: {only_vanilla.shape}")
    # print(f"ours: {only_ours.shape}")
    # print(f"neither: {neither.shape}")

    print('where both work:')
    both_sample = both.sample(5)
    for idx, row in both_sample.iterrows():
        annotations = ast.literal_eval(row['annotations'])
        print(annotations[0])
        
    print("\n\n where only the vanilla model works:")
    vanilla_sample = only_vanilla.sample(5)
    for idx, row in vanilla_sample.iterrows():
        annotations = ast.literal_eval(row['annotations'])
        print(annotations[0])

    print("\n\n where only our model works:")
    ours_sample = only_ours.sample(5)
    for idx, row in ours_sample.iterrows():
        annotations = ast.literal_eval(row['annotations'])
        print(annotations[0])

    print("\n\n where neither model works:")
    neither_sample = neither.sample(5)
    for idx, row in neither_sample.iterrows():
        annotations = ast.literal_eval(row['annotations'])
        print(annotations[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--direction", type=str, default="all")
    parser.add_argument("--protocols", type=str, default="all")
    parser.add_argument("--mode", type=str, default="save")

    args = parser.parse_args()
    mode = args.mode

    if mode == "save":
        save_dir = 'combined_metrics'
        os.makedirs(save_dir, exist_ok=True)

        if args.direction == "all":
            directions = ["t2m", "m2t"]
        if args.protocols == "all":
            protocols = ["normal", "threshold_0.95"]
        
        for direction in directions:
            for protocol in protocols:
                combined_metrics = get_metrics(direction, protocol)
                save_path = f'{save_dir}/{direction}_{protocol}.csv'
                combined_metrics.to_csv(save_path)
                print(f"doc done. saved in {save_path}")

    elif mode == "read":
        process_metrics(args.direction, args.protocols)