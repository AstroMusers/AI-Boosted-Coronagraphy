import numpy as np

from glob import glob

import pickle5 as pickle

from convAE_train import InjectionDataset, set_device, DataLoader, Encoder

import torch

class LatentSpace():
    def __init__(self, encoder_dir, data_dir, batch_size, device) -> None:
        self.device = device

        self.encoder = self.__get_encoder(encoder_dir=encoder_dir)

        self.dataset = InjectionDataset(data_dir=data_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        latent_space = self.__create_latent()

        save_dir = f'{"/".join(encoder_dir.split("/")[:-1])}/latent_v1/{encoder_dir.split("/")[-1].split("_")[0]}_latent_space.npy'

        self.__save_latent(latent_space, save_dir)


    def __get_encoder(self, encoder_dir):
        with open(f'{encoder_dir}', 'rb') as fin:
            encoder = pickle.load(fin).to(self.device)
        
        return encoder

    def __create_latent(self):
        latent_space = []

        for image_batch, label in self.dataloader:
            self.encoder.eval()
            with torch.no_grad():
                latent = self.encoder(image_batch.to(self.device))
                latent = latent.cpu().numpy()
                # add labels into latent one by one
                for idx in range(latent.shape[0]):
                    latent_space.append((latent[idx], label[idx]))
        
        return np.array(latent_space)
    
    def __save_latent(self, latent_space, save_dir):
        np.save(save_dir, latent_space)


if __name__ == '__main__':
    
    PROPOSAL_ID = '1386'
    INSTRUMENT = 'NIRCAM'
    data_dir = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/injections/test/*.npy'

    enc_epoch = 149
    encoder_dir = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/models/modelv1/{enc_epoch}_enc.pickle'

    batch_size = 1024
    device = set_device(3)

    with open(f'{encoder_dir}', 'rb') as fin:
        encoder = pickle.load(fin).to(device)

    latent_space = LatentSpace(encoder_dir, data_dir, batch_size, device)
    