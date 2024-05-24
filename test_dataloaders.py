from hydra.utils import instantiate
from src.config import read_config

run_dir = "/vulcanscratch/mukunds/downloads/TMR/outputs/tmr_humanml3d_guoh3dfeats"
cfg = read_config(run_dir)

# print(f"cfg: {cfg.data}")

dataset = instantiate(cfg.data, split="test")
print(f"length of dataset: {len(dataset)}")
list1 = []
list2 = []

for i in dataset:
    list1.append(i['keyid'])
    # break

for i in dataset:
    list2.append(i['keyid'])
    # break

print(len(list1))
print(len(list2))
print(list1 == list2)