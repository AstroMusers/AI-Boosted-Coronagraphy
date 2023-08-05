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

from util.util_main import get_filename_from_dir, get_dataset_dir
from notebooks.visualization_helpers import get_stage3_products

from astropy.io import fits



class InjectionDataset(Dataset):
    def __init__(self):
        PROPOSAL_ID = '1386'
        INSTRUMENT = 'NIRCAM'
        self.real_psfs_dir = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/mastDownload/JWST/'

        self.class_map = {'star_only' : 0, 'star_exo': 1}
        self.img_dim = (320, 320)


        #self.real_psfstacks = {}
        self.real_psfstacks = []
        self.injected_psfs = []

        self.__prepare_real_psf()
        self.__prepare_injected_psf()

        self.real_psfstacks = np.array(self.real_psfstacks) 
        self.injected_psfs = np.array(self.injected_psfs)

        self.data = np.concatenate((self.real_psfstacks, self.injected_psfs), axis=0)

        for _, psf in enumerate(self.data):
            psf[0] = (psf[0] - np.min(psf[0])) / (np.max(psf[0])- np.min(psf[0]))


    def __len__(self):
        return self.data.shape[0]

    def __create_psfstacks_dict(self) -> None:
        for _, dir in enumerate(self.psfstacks_nircam_dirs):
            fits_name = get_filename_from_dir(dir)
            #self.real_psfstacks[fits_name] = fits.open(dir)[1].data
            psfstack = fits.open(dir)[1].data

            for _, psf in enumerate(psfstack):
                self.real_psfstacks.append([psf, 'star_only'])

            del fits_name

    def __prepare_real_psf(self) -> None:
        self.psfstacks_nircam_dirs = get_stage3_products(
            suffix='psfstack', directory=self.real_psfs_dir)

        self.__create_psfstacks_dict()

    def __prepare_injected_psf(self) -> None:
        injected_psfs_dir = glob(f'{get_dataset_dir()}/PSF_INJECTION/*.png')

        for _, dir in enumerate(injected_psfs_dir):
            #self.injected_psfs.append([ dir.split('/')[-1], cv.imread(dir, 0)])
            self.injected_psfs.append([cv.imread(dir, 0), 'star_exo'])

    def __getitem__(self, idx):
        psf_img, class_name = self.data[idx]

        psf_img = np.nan_to_num(psf_img)

        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(psf_img)

        img_tensor = img_tensor.float()

        #img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


def set_device():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 320, 1 -> 320, 64 -> 160, 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1), 
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 160, 64 -> 80, 64
        
        #cnn2
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1)) # 80,64 -> 78, 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1)) # 78, 128 -> 76, 128
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #76,128 -> 76,128 -> 38,128
        
        #cnn3
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256, affine=False),
            nn.LeakyReLU(0.1)) # 38,128 -> 36,256
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512, affine=False),
            nn.LeakyReLU(0.1)) # 36,256 -> 34,512
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 34, 512 -> 32,64 -> 16,64
        
        # #cnn4
        # self.layer9 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1)) # 36,256 -> 34,512
        # self.layer10 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1)) # 34,256 -> 32,512
        # self.layer11 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1)) # 32,256 -> 30,512
        # self.layer12 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1)) # 30,256 -> 28,512
        
        # self.layer13 = nn.Sequential(
        #     nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1)) # 28,512 -> 28,512
        
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        #FC layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16*16*64, 1000),
            nn.LeakyReLU(0.1))
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.LeakyReLU(0.1))
        # self.fc2= nn.Sequential(
        #     nn.Linear(4096, 1000))
        
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # x = self.layer9(x)
        # x = self.layer10(x)
        # x = self.layer11(x)
        # x = self.layer12(x)
        # x = self.layer13(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #FC 
        # self.fc2= nn.Sequential(
        #     nn.Linear(1000, 4096))
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.LeakyReLU(0.1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 16*16*64),
            nn.LeakyReLU(0.1))
        
        #Unflatten
        self.unflatten = nn.Unflatten(dim=1, 
            unflattened_size=(64, 16, 16))
        
        #unconv4
        # self.layer13 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512, affine=False),
        #     nn.LeakyReLU(0.1))
        # self.layer12 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1))
        # self.layer11 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1))
        # self.layer10 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1))
        # self.layer9 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0),
        #     nn.LeakyReLU(0.1),
        #     nn.Upsample(scale_factor=2, mode='nearest'))
        
        #unconv3
        self.layer8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512, affine=False),
            nn.LeakyReLU(0.1)) # 16,64 -> 32,64 -> 34, 512
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256, affine=False),
            nn.LeakyReLU(0.1)) #34,512 -> 36,256
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1),
             nn.Upsample(scale_factor=2, mode='nearest')) # 36,256 -> 38,128 -> 76,128
        
        #unconv2
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1)) # 76,128 -> 78,128
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1)) # 78,128 -> 80,128
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1),) # 80,128 -> 160,128 -> 160,64
        
        #unconv1
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1)) # 160,64 -> 320,64 -> 320,64
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)) # 320,64 -> 320,1
        
        
        
        
    def forward(self, x):
        # x = self.fc2(x)
        # x = self.fc1(x)
        x = self.fc(x)
        x = self.unflatten(x)
        # x = self.layer13(x)
        # x = self.layer12(x)
        # x = self.layer11(x)
        # x = self.layer10(x)
        # x = self.layer9(x)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        #x = torch.sigmoid(x) # for synth2
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
    dataset = InjectionDataset()
    print('Dataset loaded!')

    batch_size = 64
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

    num_epochs = 2000
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
            with open(f"/data/scratch/bariskurtkaya/dataset/models_injection/{epoch}_enc.pickle", "wb") as fout:
                pickle.dump(encoder, fout)
            with open(f"/data/scratch/bariskurtkaya/dataset/models_injection/{epoch}_dec.pickle", "wb") as fout:
                pickle.dump(decoder, fout)
        
        wandb.log({"loss": train_loss})

        diz_loss['train_loss'].append(train_loss)

    print(diz_loss)
    print('Training completed!')
    plot_ae_outputs(encoder, decoder, 10)
    wandb.finish()
