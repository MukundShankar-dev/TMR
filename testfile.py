import os
from src.load import load_model_from_cfg
from src.config import read_config
from torch.utils.data import DataLoader
from train_flag_dataloader import FlagDataSet
from src.data.collate import collate_text_motion

test_dataset = FlagDataSet(split="test")
test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=1, collate_fn=collate_text_motion, shuffle=False)
data_iter = iter(test_dataloader)
print(f"length: {len(data_iter)}")