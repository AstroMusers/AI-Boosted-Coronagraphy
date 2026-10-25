import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tdpy.verbosity import print

class Augmentation():
    def __init__(self):
        print("Augmentation class init")
    
    def normalize(self, img, sanity_checks=False):
        # print("Normalization")
        if sanity_checks:
            if not np.isfinite(img).all():
                raise ValueError("Image must contain only finite values")
            if img.max() <= img.min():
                raise ValueError("Image must have a nonzero range")
        return (img - img.min()) / (img.max() - img.min())

    def rotate90(self, img, times=1):
        # print("Counterclockwise rotation")
        return np.rot90(img, k=times)
    
    def flip(self, img, horizontal=True, vertical=True):
        # print("Flip")
        return np.flipud(np.fliplr(img)) if horizontal and vertical else (np.flipud(img) if vertical else (np.fliplr(img) if horizontal else img))

    def shift(self, img, right_shift, down_shift):
        # print("Shift")
        return np.roll(np.roll(img, right_shift, axis=1), down_shift, axis=0)


class AugmentationGPU(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.transforms import v2

        self.transforms = v2.Compose(
            [
                v2.RandomRotation(degrees=[0,90,180,270]),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5)
            ]
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.transforms(x)
        return x


def load_csv(filepath):
    print(f"Reading from {filepath}...")
    return pd.read_csv(filepath)

def save_csv(dataframe, filepath):
    print(f"Writing to {filepath}...")
    dataframe.to_csv(filepath, index=False)

def train_test_split(data, test_size=0.2):
    pass
