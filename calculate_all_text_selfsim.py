import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config
import torch
from tqdm import tqdm


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):

    dataset = instantiate(cfg.data, split="all")

    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    embs = []
    for data in enumerate(dataloader):
        embs.append(data['sent_emb'])
    embs = torch.cat(embs, axis=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embs = embs.to(device)
    text_selfsim_matrix = embs @ embs.T
    torch.save(text_selfsim_matrix, "/vulcanscratch/mukunds/downloads/TMR/sentence_sim_matrix.pt")

if __name__ == "__main__":
    train()