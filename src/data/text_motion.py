import os
import codecs as cs
import orjson  # loading faster than json
import json
import random

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())

def load_samples(lower, upper):
    json_path = f"/vulcanscratch/mukunds/downloads/TMR/samples_both_{lower}_{upper}.json"
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        use_dtw: bool = True,
        lower_neg_sample: int = 5,
        upper_neg_sample: int = 8
    ):
        if tiny:
            split = split + "_tiny"

        self.lower = lower_neg_sample
        self.upper = upper_neg_sample

        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        self.use_dtw = use_dtw
        if self.use_dtw:
            self.samples = load_samples(self.lower, self.upper)

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = split == "train"
        self.is_val = split == "val"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def get_positive_sample(self, keyid):
        pos_samples = self.samples[keyid]["positive_sample_keyids"]
        num_samples = len(pos_samples)
        idx = random.randint(0, num_samples-1)
        motion_keyid = pos_samples[idx]
        pos_motion = self.motion_loader(
            path=self.annotations[motion_keyid]["path"],
            start=self.annotations[motion_keyid]["annotations"][0]["start"],
            end=self.annotations[motion_keyid]["annotations"][0]["end"],
        )
        return pos_motion

    def get_negative_sample(self, keyid):
        neg_samples = self.samples[keyid]["negative_sample_keyids"]
        num_samples = len(neg_samples)
        idx = random.randint(0, num_samples-1)
        motion_keyid = neg_samples[idx]
        neg_motion = self.motion_loader(
            path=self.annotations[motion_keyid]["path"],
            start=self.annotations[motion_keyid]["annotations"][0]["start"],
            end=self.annotations[motion_keyid]["annotations"][0]["end"],
        )
        return neg_motion

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        text = annotation["text"]
        text_x_dict = self.text_to_token_emb(text)
        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )
        sent_emb = self.text_to_sent_emb(text)
        
        if self.use_dtw:
            if self.is_training or self.is_val:
                positive_sample = self.get_positive_sample(keyid)
                negative_sample = self.get_negative_sample(keyid)

                output = {
                    "motion_x_dict": motion_x_dict,
                    "text_x_dict": text_x_dict,
                    "text": text,
                    "keyid": keyid,
                    "sent_emb": sent_emb,
                    "positive_sample_x_dict": positive_sample,
                    "negative_sample_x_dict": negative_sample
                }
            else:
                output = {
                "motion_x_dict": motion_x_dict,
                "text_x_dict": text_x_dict,
                "text": text,
                "keyid": keyid,
                "sent_emb": sent_emb,
            }
        else:
            output = {
                "motion_x_dict": motion_x_dict,
                "text_x_dict": text_x_dict,
                "text": text,
                "keyid": keyid,
                "sent_emb": sent_emb,
            }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
