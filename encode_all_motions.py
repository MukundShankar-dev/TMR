import os
from omegaconf import DictConfig
import logging
import hydra
import json
import pandas as pd
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)

def x_dict_to_device(x_dict, device):
    import torch

    for key, val in x_dict.items():
        if isinstance(val, torch.Tensor):
            x_dict[key] = val.to(device)
    return x_dict

def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))

@hydra.main(version_base=None, config_path="configs", config_name="encode_dataset")
def encode_all(cfg: DictConfig) -> None:
    print("in encode_all function")
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    cfg_data = cfg.data

    choices = HydraConfig.get().runtime.choices
    data_name = choices.data

    print("importing all libraries")
    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything

    print("constructing dataframe:")
    df = pd.DataFrame(columns=['keyids', 'annotations', 'motion path', 'length', 'mask_path'])

    print(f"imported all. now reading config from run_dir. run_dir is: {run_dir}")
    cfg = read_config(run_dir)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    # save_dir = os.path.join(run_dir, "latents")

    print("assigning save_dir")
    # save_dir = os.path.join(run_dir, "new_latents/embeddings")
    save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    mask_dir = os.path.join(run_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    print("Instantiating dataloader")
    dataset = instantiate(cfg_data, split="all")
    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )
    seed_everything(cfg.seed)

    count = 0
    all_keyids = {}

    print("running inference")
    with torch.inference_mode():
        for batch in dataloader:
            count += 1

            keyids = batch["keyid"]
            annotations = batch['text']

            motion_x_dict = batch["motion_x_dict"]
            mask = motion_x_dict["mask"]

            for i in range(len(keyids)):
                keyid = keyids[i]
                # length = (motion_x_dict["length"])[i]
                length = (motion_x_dict["length"])
                path = os.path.join(save_dir, f"{keyid}")

                mask_path = mask_dir + '/' + keyid + '.pt'
                torch.save(mask, mask_path)

                if keyid in all_keyids:
                    all_keyids[keyid] += 1
                else:
                    all_keyids[keyid] = 0
                # ['keyids', 'annotations', 'motion path']
                df.loc[len(df.index)] = [keyid, annotations[i], batch['motion_path'][i]]

    csv_path=os.path.join(run_dir, 'new_latents','embeddings.csv')
    df.to_csv(csv_path, index=False)

    print(f"There were {count} batches. csv saved at {csv_path}, embeddings are in {save_dir}")

if __name__ == "__main__":
    print("running main now")
    encode_all()
