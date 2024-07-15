import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


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
