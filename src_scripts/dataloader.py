import pandas as pd
import numpy as np
import random
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch
from dataset import BengaliDataset

seed = 42
random.seed(seed)
np.random.seed(seed)

df = pd.read_csv("input/train.csv")


def Bengali_dataloader(phase, img_height, img_width, mean, std, batch):
    split_ratio = 0.2
    ful_size = len(df)
    index = list(range(ful_size))
    split = int(np.floor(split_ratio * ful_size))
    np.random.shuffle(index)
    val_idx = index[:split]
    tr_idx = index[split:]

    tr_sampler = SubsetRandomSampler(tr_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    if phase == "train":
        sampler = tr_sampler
        idx = tr_idx
    else:
        sampler = val_sampler
        idx = val_idx

    init_dataset = BengaliDataset(df.iloc[idx], phase, img_height, img_width, mean, std)
    init_dataloader = DataLoader(
        init_dataset, batch_size=batch, sampler=sampler, drop_last=True
    )
    return init_dataloader
