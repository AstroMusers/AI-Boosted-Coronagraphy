import matplotlib.pyplot as plt

from glob import glob

import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from tqdm import tqdm

# import optuna

import wandb

import pickle5 as pickle

import sys
sys.path.append("..")


class InjectionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.class_map = {'star_only': 0, 'star_exo': 1}
        self.img_dim = (320, 320)

        self.data = []

        self.__prepare_dataset()

        self.data = np.array(self.data)


    def __len__(self):
        print(self.data.shape[0])
        return self.data.shape[0]

    def __prepare_dataset(self) -> None:
        npy_files_dirs = glob(f'{self.data_dir}')

        for _, npy_dir in enumerate(npy_files_dirs):

            if 'fc' in npy_dir.split('-')[-1] and 'y' in npy_dir.split('-')[-2] and 'x' in npy_dir.split('-')[-3]:
                self.data.append([np.load(npy_dir), 'star_exo'])
            else:
                self.data.append([np.load(npy_dir), 'star_only'])

    def __getitem__(self, idx):
        psf_img, class_name = self.data[idx]

        img_tensor = torch.from_numpy(psf_img)
        img_tensor = img_tensor.float()

        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


def set_device(device_numb:int=3):
    device = torch.device(f'cuda:{device_numb}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        current_device = torch.cuda.current_device()
        print('Selected GPU Name:', torch.cuda.get_device_name(current_device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB')
        print('Max Memmory Cached:', round(torch.cuda.max_memory_reserved(current_device)/1024**3,1), 'GB')

    return device


class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        # cnn1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(0.1)) # 320, 1 -> 320, 32 -> 160, 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1)) # 160, 32 -> 80, 64
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1)) # 80,64 -> 40, 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1)) # 40, 128 -> 20, 128
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1)) # 20, 128 -> 10, 64
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        #FC layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(10*10*64, 1000),
            nn.LeakyReLU(0.1))
        
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #FC 
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 10*10*32),
            nn.LeakyReLU(0.1))
        
        #Unflatten
        self.unflatten = nn.Unflatten(dim=1, 
            unflattened_size=(32, 10, 10))
        
        #unconv
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(0.1)) # 10,32 -> 10,16 -> 34, 512
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16, affine=False),
            nn.LeakyReLU(0.1)) #34,512 -> 36,256
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16, affine=False),
            nn.LeakyReLU(0.1))        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8, affine=False),
            nn.LeakyReLU(0.1)) # 76,128 -> 78,128

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)) # 320,64 -> 320,1
        
        
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = x.view(-1, 320, 320)
        return x


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        #print(encoded_data.shape)
        # Decode data
        decoded_data = decoder(encoded_data)
        #print(decoded_data.shape)
        
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print(f'\t {str(256*len(train_loss))} partial train loss (single batch): {loss.data}')
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for image_batch, labels in train_loader:
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            rec_img = decoder(encoded_data)
        
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            plt.imshow(image_batch[i].cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img[i].cpu().squeeze().numpy(), cmap='gist_gray')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Reconstructed images')
    plt.savefig('enc_dec_outputs.png')


if __name__ == '__main__':

    print('PyTorch version:', torch.__version__)
    print('CUDA version', torch.version.cuda)
    print('cuDNN version', torch.backends.cudnn.version())

    print('Dataset loading...')
    PROPOSAL_ID = '1386'
    INSTRUMENT = 'NIRCAM'
    data_dir = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/injections/train/*.npy'

    dataset = InjectionDataset(data_dir=data_dir)
    print('Dataset loaded!')

    batch_size = 1024
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = set_device()

    torch.manual_seed(42)

    print('Model loading...')
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    print('Model loaded!')

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    loss_fn = torch.nn.MSELoss()

    lr= 0.001

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    num_epochs = 2500
    diz_loss = {'train_loss':[]}

    print('wandb login...')
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        'optim': 'Adam',
        'loss': 'MSE',
        'batch_size': batch_size,
        "architecture": "VGG-19_custom",
        "dataset": "Injection_v1.0",
        "epochs": num_epochs,
        }
    )


    print('Training...')
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
        
        print('\n EPOCH {}/{} \t train loss {} '.format(epoch + 1, num_epochs,train_loss))
        
        if (epoch+1) % 25 == 0:
            with open(f"/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/models/{epoch}_enc.pickle", "wb") as fout:
                pickle.dump(encoder, fout)
            with open(f"/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/models/{epoch}_dec.pickle", "wb") as fout:
                pickle.dump(decoder, fout)
        
        wandb.log({"loss": train_loss})

        diz_loss['train_loss'].append(train_loss)

    print(diz_loss)
    print('Training completed!')
    
    plot_ae_outputs(encoder, decoder, 10)
    wandb.finish()
