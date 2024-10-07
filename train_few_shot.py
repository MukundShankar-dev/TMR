# import os
# from src.load import load_model_from_cfg
# from src.config import read_config
# from torch.utils.data import DataLoader
# from train_flag_dataloader import FlagDataSet
# from src.data.collate import collate_text_motion
# import logging
# import hydra
# from omegaconf import DictConfig
# from hydra.utils import instantiate
# from src.config import read_config, save_config
# logger = logging.getLogger(__name__)

# cfg = read_config("outputs/flag_few_shot_test")
# ckpt = cfg.ckpt

# import src.prepare  # noqa
# import pytorch_lightning as pl

# pl.seed_everything(cfg.seed)

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)


# print("Loading the datasets")
# train_dataset = FlagDataSet("train")
# val_dataset = FlagDataSet("val")

# print("Loading the dataloaders")
# train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True, collate_fn=collate_text_motion, persistent_workers=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=1, shuffle=False, collate_fn=collate_text_motion, persistent_workers=True)

# print("Resuming training")
# logger.info("Resuming training")
# logger.info(f"The config is loaded from: \n{cfg.resume_dir}")

# logger.info("Loading the model")
# model = instantiate(cfg.model)

# logger.info("Training")
# trainer = instantiate(cfg.trainer)
# trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)

import os
from src.load import load_model_from_cfg
from src.config import read_config
from torch.utils.data import DataLoader
from train_flag_dataloader import FlagDataSet
from src.data.collate import collate_text_motion
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config
import pytorch_lightning as pl
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    mp.set_start_method('spawn', force=True)

    # cfg = read_config("outputs/")
    # ckpt = "last"
    ckpt = None

    import src.prepare  # noqa

    pl.seed_everything(cfg.seed)

    print("Loading the datasets")
    train_dataset = FlagDataSet("test")
    val_dataset = FlagDataSet("val")

    print("Loading the dataloaders")
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=7, shuffle=True, collate_fn=collate_text_motion, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=7, shuffle=False, collate_fn=collate_text_motion, persistent_workers=True)

    print("Resuming training")
    logger.info("Resuming training")
    logger.info(f"The config is loaded from: \n{cfg.resume_dir}")

    logger.info("Loading the model")
    model = instantiate(cfg.model)

    # cfg.trainer.accumulate_grad_batches = 2
    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)



if __name__ == "__main__":
   train()