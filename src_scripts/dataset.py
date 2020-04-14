from albumentations import Resize, Normalize, ShiftScaleRotate, Compose
from albumentations.pytorch import ToTensor
from PIL import Image
import joblib
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch


class BengaliDataset:
    def __init__(self, phase, img_height, img_width, mean, std):
        df = pd.read_csv("input/train.csv")
        df = df[
            ["image_id", "grapheme_root", "vowel_diacritic", "consonant_diacritic",]
        ]

        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        self.aug = self.augmentation(mean, std, phase)

    @staticmethod
    def augmentation(mean, std, phase):
        empty = []
        if phase == "train":
            empty.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.9
                    )
                ]
            )
        empty.extend([Normalize(mean=mean, std=std), ToTensor()])
        empty = Compose(empty)
        return empty

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image = joblib.load(f"input/image_pickles/{self.image_ids[item]}.pkl")
        image = image.reshape(137, 236)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.aug(image=image)["image"].type(torch.float)
        return {
            "image": image,
            "grapheme_root": torch.tensor(self.grapheme_root[item], dtype=torch.long),
            "vowel_diacritic": torch.tensor(
                self.vowel_diacritic[item], dtype=torch.long
            ),
            "consonant_diacritic": torch.tensor(
                self.consonant_diacritic[item], dtype=torch.long
            ),
        }
