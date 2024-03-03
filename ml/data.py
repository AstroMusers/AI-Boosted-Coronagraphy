import numpy as np
import torch
from torch.utils.data import Dataset



class NIRCamDataset(Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = np.load(self.dataset[idx])
        data = data.astype(np.float32)
        data = (data - np.min(data))/(np.max(data) - np.min(data))

        scaled_data = np.arcsinh(data * 2**16).astype(np.float32)
        scaled_data = (scaled_data - np.min(scaled_data))/(np.max(scaled_data) - np.min(scaled_data))

        scaled_data = 2*scaled_data - 1
        data= 2*data - 1
        stacked_data = np.stack((data, scaled_data), axis=0)
        data_class = 1 if 'fc' in self.dataset[idx].split('-')[-1] else 0 # 1 for exoplanet and 0 for non-exoplanet

        return torch.from_numpy(stacked_data).to(self.device), torch.tensor(data_class).to(self.device)



class NIRCamDataset_ViT(Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, data):
        data -= np.mean(data, axis=0, keepdims=True)
        data /= np.std(data, axis=0, keepdims=True)

        return data
    
    def __getitem__(self, index):
        data = np.load(self.dataset[index])
        data = data.astype(np.float32)

        data = (data - np.min(data))/(np.max(data) - np.min(data))

        data = np.expand_dims(data, 0)
        data = np.repeat(data, 3, axis=0)
 
        data_class = 1 if 'fc' in self.dataset[index].split('-')[-1] else 0

        return torch.from_numpy(data).to(self.device), torch.tensor(data_class).to(self.device)
        