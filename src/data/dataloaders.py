import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import os
import glob
from src.utils import get_dataset_dir
from tdpy.verbosity import print

class SynDatasetLabel(Dataset):

    def __init__(self, image_paths, args):

        self.image_paths = image_paths
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        ])
        self.args = args

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        
        if 'fc' in image_path.split('/')[-1]:
            label = torch.squeeze(torch.Tensor([1]))
        else:
            label = torch.squeeze(torch.Tensor([0]))

        print(f"Reading from {image_path}...")
        image = np.load(image_path).astype(np.float32)
        sanity_checks = getattr(self.args, "sanity_checks", False)
        if sanity_checks:
            if not np.isfinite(image).all():
                raise ValueError("Classifier image must contain only finite values")
            if image.size == 0 or np.ptp(image) == 0:
                raise ValueError("Classifier image must have a nonzero range")

        if self.args.apply_lowpass:
            filtered_img = self.apply_low_pass(image)

        image = self.transform(image)
        if sanity_checks and not bool(torch.isfinite(image).all()):
            raise ValueError("Transformed classifier image must contain only finite values")

        filtered_image = 0
        if self.args.apply_lowpass:
            filtered_image = self.transform(filtered_img)
            if sanity_checks and not bool(torch.isfinite(filtered_image).all()):
                raise ValueError("Low-pass classifier image must contain only finite values")

        batch = (image, label, filtered_image, image_path)

        return batch

class SynthDatasetv2(Dataset):
    def __init__(self, dataset_path=None, data_type='train', num_injection=10):
        if dataset_path is None:
            dataset_path = os.path.join(get_dataset_dir(), "torch_dataset_injection")
        psf_dicts = []
        labels = []

        pattern = os.path.join(dataset_path, data_type, '*.pth')
        print(f"Reading from {pattern}...")
        psf_paths = glob.glob(pattern)

        for psf_path in psf_paths:
            print(f"Reading from {psf_path}...")
            torch_psf = torch.load(psf_path, weights_only=False)

            if type(torch_psf) == tuple:
                psf_dicts.append(torch_psf[0])
                labels.append(torch.ones(torch_psf[0].shape[0]))
            else:
                psf_dicts.append(torch_psf.repeat(num_injection, 1, 1))
                labels.append(torch.zeros(torch_psf.shape[0]*num_injection))

        self.psf_dicts = torch.cat(psf_dicts)
        self.labels = torch.cat(labels)

    def __len__(self):
        return self.labels.shape[0]
        
    def __getitem__(self, idx):
        return self.psf_dicts[idx], self.labels[idx].type(torch.LongTensor)