import torch
from torch.utils.data import Dataset
from torchvision import transforms
from itertools import product
from PIL import Image



class SynDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths
        self.transform  = transforms.Compose([
        transforms.ToTensor(),
        ])


    def __len__(self,):

        return len(self.image_paths)

    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        image      = Image.open(image_path).convert('L')
        image = self.transform(image)

        if torch.isnan(image).any().item():
            torch.nan_to_num(image)
            
        return image