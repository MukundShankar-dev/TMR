import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

import numpy as np
import pandas as pd

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def test(cfg: DictConfig):
    print("instantiating train dataset")
    train_dataset = instantiate(cfg.data, split="train")
    print("instantiating train dataloader")
    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    for idx, batch in enumerate(train_dataloader):
        continue
    
    print(f"number of batches in val {idx + 1}")

    print("instantiating train dataset")
    val_dataset = instantiate(cfg.data, split="val")
    print("instantiating train dataloader")
    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    for idx, batch in enumerate(val_dataloader):
        continue
    
    print(f"number of batches in val {idx + 1}")


def get_tm_samples(idx, distances):
    filtered_distances = distances[distances['i'] == idx]
    motion_ranks = filtered_distances.sort_values(by='distance').index.tolist()
    motion_ranks = np.array(motion_ranks)
    motion_ranks = motion_ranks + 1
    motion_ranks = np.argsort(motion_ranks)
    motion_ranks = motion_ranks + 1
    
    print(motion_ranks)

if __name__ == "__main__":
    # get_tm_samples()

    data = {"i": [0, 0, 0, 0, 0], "j": [0, 1, 2, 3, 4], "distance": [0.0, 123.0, 234.0, 345.0, 456.0]}
    df = pd.DataFrame.from_dict(data)
    
    get_tm_samples(0, df)