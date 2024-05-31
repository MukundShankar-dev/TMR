import pandas as pd
import argparse
import pprint

def read_csv(run_dir, protocol):
    t2m_df = pd.read_csv(f"{run_dir}/contrastive_metrics_2/t2m_{protocol}_keyid_metrics.csv")
    m2t_df = pd.read_csv(f"{run_dir}/contrastive_metrics_2/m2t_{protocol}_keyid_metrics.csv")
    return t2m_df, m2t_df

def print_annotations(df):
    best_at_1 = df[df['R01'] == True]
    # print(f"number of things hit at R01: {best_at_1.shape}")
    print("r01 stuff:")
    pprint.pprint(list(best_at_1['annotations'])[0:5])

    # best_at_2 = df[df['R02'] == True]
    # print(f"number of things hit at R02: {best_at_2.shape}")

    # best_at_3 = df[df['R03'] == True]
    # print(f"number of things hit at R03: {best_at_3.shape}")
    # best_at_5 = df[df['R05'] == True]
    # print(f"number of things hit at R05: {best_at_5.shape}")

    # best_at_10 = df[df['R10'] == True]
    # print(f"number of things hit at R10: {best_at_10.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--protocol', type=str)

    args = parser.parse_args()
    run_dir = args.run_dir
    protocol = args.protocol

    t2m_df, m2t_df = read_csv(run_dir, protocol)

    print_annotations(t2m_df)
    # print_annotations(m2t_df)