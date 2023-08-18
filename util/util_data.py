import torch
from torch.utils.data import Dataset
from torchvision import transforms
from itertools import product
from PIL import Image
import numpy as np



class SynDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths
        self.transform  = transforms.Compose([
        transforms.ToTensor(),
        ])

    def __len__(self,):

        return len(self.image_paths)
    
    def __getitem__(self, index):
        
        image = self.image_paths[index]
        image = np.load(image)
        image = self.transform(image)

        return image