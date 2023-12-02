import numpy as np

from glob import glob

import pickle5 as pickle

from convAE_train import InjectionDataset, set_device, DataLoader, Encoder

import torch

from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.getcwd()))


from util.util_data import *
from util.util_dirs import *
from util.util_train import *
from ml.models import Exonet, VAExonet


class LatentSpace():
    def __init__(self, encoder_dir, data_dir, batch_size, device) -> None:
        print('Latent Space Class Created.')
        self.device = device

        self.encoder = self.__get_encoder(encoder_dir=encoder_dir)
        print('Encoder Loaded.')

        print('Dataset Loading...')
        dataset = InjectionDataset(data_dir=data_dir)
        print('Dataset Loaded.')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print('Creating Latent Space...')
        latent_space = self.__create_latent(dataloader)

        save_dir = f'{"/".join(encoder_dir.split("/")[:-1])}/latent_v1/train_{encoder_dir.split("/")[-1].split("_")[0]}_latent_space.npy'

        self.__save_latent(latent_space, save_dir)


    def __get_encoder(self, encoder_dir):

        # with open(f'{encoder_dir}', 'rb') as fin:
        #     encoder = pickle.load(fin).to(self.device)
        encoder = torch.load(encoder_dir)
        return encoder

    def __create_latent(self, dataloader):
        latent_space = []
        print('Creating Latent Space')
        for _ ,data in enumerate(tqdm(dataloader)):
            image_batch, label = data
            self.encoder.eval()
            with torch.no_grad():
                x, latent = self.encoder(image_batch.to(self.device))
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
    #data_dir = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/injections/train/*.npy'
    
    inj = glob.glob('/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/injections2/fc1_coron_injections_train/*fc1.npy')[:128]
    no_inj = glob.glob('/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/injections2/fc1_coron_injections_train/*[!fc1].npy')[:128]

    train_paths = inj + no_inj

    #enc_epoch = 374
    #encoder_dir = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/models/modelv1/{enc_epoch}_enc.pickle'

    encoder_dir  = "/data/scratch/sarperyurtseven/results/plots/training_results/coron_injections_ae/0/model_epoch-29.pt"
    batch_size = 2048
    device = set_device(3)

    with open(f'{encoder_dir}', 'rb') as fin:
        encoder = pickle.load(fin).to(device)

    latent_space = LatentSpace(encoder_dir, data_dir, batch_size, device)
    