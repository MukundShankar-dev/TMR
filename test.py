import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

import numpy as np

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def test(cfg: DictConfig):

    train_dataset = instantiate(cfg.data, split="train")
    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    for idx, batch in enumerate(train_dataloader):
        continue
    
    print(f"number of batches {idx + 1}")

if __name__ == "__main__":
    test()