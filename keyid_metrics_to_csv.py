import pandas as pd
import json
import argparse

def read_data(file_path, protocol):
    with open(f"{file_path}/contrastive_metrics_2/{protocol}_keyid_metrics.json") as json_file:
        to_return = json.load(json_file)
    return to_return

def get_annotations():
    with open("datasets/annotations/humanml3d/annotations.json") as json_file:
        to_return = json.load(json_file)
    return to_return



if __name__  == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--protocol', type=str)

    args = parser.parse_args()

    data_dict = read_data(args.run_dir, args.protocol)
    t2m_data = data_dict['t2m']
    m2t_data = data_dict['m2t']

    all_annotations = get_annotations()

    keyids = t2m_data.keys()
    for keyid in keyids:
        keyid_annotations = all_annotations[keyid]
        # print(f"keyid_annotations: {keyid_annotations}")
        annotations = []
        for annotation in keyid_annotations['annotations']:
            # print(f'annotation: {annotation}')
            annotations.append(annotation['text'])
        # annotations = [annotation['text'] for annotation in keyid_annotations]
        t2m_data[keyid]['annotations'] = annotations
        m2t_data[keyid]['annotations'] = annotations
    
    df1 = pd.DataFrame.from_dict(t2m_data, orient='index')
    df2 = pd.DataFrame.from_dict(m2t_data, orient='index')

    df1.to_csv(f"{args.run_dir}/contrastive_metrics_2/t2m_{args.protocol}_keyid_metrics.csv")
    df2.to_csv(f"{args.run_dir}/contrastive_metrics_2/m2t_{args.protocol}_keyid_metrics.csv")