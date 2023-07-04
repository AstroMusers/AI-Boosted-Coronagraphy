import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import torchvision
from torchvision import transforms


import cv2 as cv

import numpy as np

from glob import glob

from tqdm import tqdm
import pickle5 as pickle

from warnings import filterwarnings
filterwarnings('ignore')


class SynthDataset(Dataset):
    def __init__(self):
        self.imgs_path = '/data/scratch/bariskurtkaya/synthetic_dataset/synth2_dataset/'
        self.data = []
        
        for img_path in glob(self.imgs_path + '/*.png'):
            if 'exo' in img_path:
                self.data.append([img_path, 'star_exo'])
            else:
                self.data.append([img_path, 'star_only'])
        
        #print(self.data[0])
        self.class_map = {'star_only' : 0, 'star_exo': 1}
        self.img_dim = (28, 28)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img_path, class_name = self.data[idx]
        img = cv.imread(img_path, 0)
        
        #noise = np.random.normal(0,10,img.shape)
        #noisy_img = img + noise
                
        #img = cv.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        
        img_tensor = img_tensor.float()
        
        #img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

def set_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('Total GPU count:', torch.cuda.device_count())
        print('Selected GPU index:', torch.cuda.current_device())
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
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()) # 160, 1 -> 158, 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 158, 64 -> 156, 64 -> 78,64
        
        #cnn2
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()) # 78,64 -> 76, 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()) # 76, 128 -> 74, 128
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 74,128 -> 72,128 -> 36,128
        
        #cnn3
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()) # 36,128 -> 34,256
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()) # 34,128 -> 32,256
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 32, 256 -> 32,256 -> 16,256
        
        #cnn4
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()) # 16,256 -> 14,512
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()) # 14,256 -> 12,512
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()) # 12,256 -> 10,512
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()) # 10,256 -> 8,512
        
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()) # 8,512 -> 8,512
        
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        #FC layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8*8*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, 1000))
        
        
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
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #FC 
        self.fc2= nn.Sequential(
            nn.Linear(1000, 4096))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 8*8*512),
            nn.ReLU())
        
        #Unflatten
        self.unflatten = nn.Unflatten(dim=1, 
            unflattened_size=(512, 8, 8))
        
        #unconv4
        self.layer13 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer11 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        #unconv3
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
             nn.Upsample(scale_factor=2, mode='nearest'))
        
        #unconv2
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        #unconv1
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=0))
        
        
        
        
    def forward(self, x):
        x = self.fc2(x)
        x = self.fc1(x)
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.layer13(x)
        x = self.layer12(x)
        x = self.layer11(x)
        x = self.layer10(x)
        x = self.layer9(x)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        #x = torch.sigmoid(x) # for synth2
        x = x.view(-1, 160, 160)
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
    #plt.show() 


if __name__ == '__main__':
    dataset = SynthDataset()
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = set_device()
    loss_fn = torch.nn.MSELoss()

    lr= 0.001

    torch.manual_seed(42)

    encoder = Encoder()
    decoder = Decoder()

    encoder.to(device)
    decoder.to(device)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)


    num_epochs = 200
    diz_loss = {'train_loss':[]}
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
        
        print('\n EPOCH {}/{} \t train loss {} '.format(epoch + 1, num_epochs,train_loss))
        
        if epoch % 25 == 0:
            with open(f"/data/scratch/bariskurtkaya/synthetic_dataset/synth2_model/{epoch}_enc.pickle", "wb") as fout:
                pickle.dump(encoder, fout)
            with open(f"/data/scratch/bariskurtkaya/synthetic_dataset/synth2_model/{epoch}_dec.pickle", "wb") as fout:
                pickle.dump(decoder, fout)
        
        diz_loss['train_loss'].append(train_loss)
    
    plot_ae_outputs(encoder, decoder, 10)






