import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import os
import glob

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

        image = np.load(image_path).astype(np.float32)

        if self.args.apply_lowpass:
            filtered_img = self.apply_low_pass(image)

        image = self.transform(image)

        batch = (image, label, self.transform(filtered_img) if self.args.apply_lowpass else 0, image_path)

        return batch

class SynthDatasetv2(Dataset):
    def __init__(self, dataset_path='/data/scratch/bariskurtkaya/dataset/torch_dataset_injection/', data_type='train', num_injection=10):
        psf_dicts = []
        labels = []

        psf_paths = glob.glob(os.path.join(dataset_path, data_type, '*.pth'))

        for psf_path in psf_paths:
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