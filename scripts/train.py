
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from itertools import product
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
from PIL import Image
import os


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
    
def calculate_conv_dims(input_size,paddings:list,kernels:list,strides:list,maxpool:list):
    
    outputs = []
    outputs.append(input_size)
    for i in range(len(paddings)):
        
        output_size = (input_size + (2*paddings[i]) - (kernels[i] - 1) - 1)/strides[i] + 1
        if maxpool[i] != 0:
            output_size = (output_size  + (2*paddings[i]) - (maxpool[i]-1)-1)/2 +1
        
        outputs.append(int(output_size))
        input_size = output_size
        
    print(outputs)
    return outputs


def calculate_convtrans_dim(input_size,paddings:list,kernels:list,strides:list):
    outputs = []
    outputs.append(input_size)
    for i in range(len(paddings)):
        
        output_size = (input_size - 1) * strides[i]  -  2 * paddings[i] + kernels[i] - 1 + 1
        outputs.append(int(output_size))
        input_size = output_size
        
    print(outputs)
    return outputs


class Exonet(nn.Module):
    
    def __init__(self, convdim_enc_outputs:list, convdim_dec_outputs:list, kernels_enc:list, strides_enc:list, kernels_dec:list, strides_dec:list):
        
        super(Exonet,self).__init__()
        
        self.convdim_enc = convdim_enc_outputs
        self.convdim_dec = convdim_dec_outputs
        self.kernels_enc = kernels_enc
        self.strides_enc = strides_enc
        self.kernels_dec = kernels_dec
        self.strides_dec = strides_dec
        self.C       = 8 
        
        self.encoder  = nn.Sequential(
                        
            nn.Conv2d(in_channels=1, out_channels=self.C, stride=self.strides_enc[0], kernel_size=self.kernels_enc[0]), #1
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=self.C, out_channels=self.C*2, stride=self.strides_enc[1], kernel_size=self.kernels_enc[1]), #2
            nn.BatchNorm2d(self.C*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=self.C*2, out_channels=self.C*2, stride=self.strides_enc[2], kernel_size=self.kernels_enc[2]), #3
            nn.BatchNorm2d(self.C*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=self.C*2, out_channels=self.C*2, stride=self.strides_enc[3], kernel_size=self.kernels_enc[3]), #4 
            nn.BatchNorm2d(self.C*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=self.C*2, out_channels=self.C*4, stride=self.strides_enc[4], kernel_size=self.kernels_enc[4]), #5
            nn.BatchNorm2d(self.C*4),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=self.C*4, out_channels=self.C*8, stride=self.strides_enc[5], kernel_size=self.kernels_enc[5]), #6
            nn.BatchNorm2d(self.C*8),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=self.C*8, out_channels=self.C*16, stride=self.strides_enc[6], kernel_size=self.kernels_enc[6]), #7
            nn.BatchNorm2d(self.C*16),
            nn.LeakyReLU(),
            
        
        ) 
        
        self.fc1 = nn.Sequential(
        
                nn.Linear((self.C*16)*convdim_outputs[-1]**2,4096),
                nn.SiLU(),
                nn.Linear(4096,2048),
                nn.SiLU(),
                nn.Linear(2048,1024),
                nn.SiLU(),
        )

        self.latent = nn.Linear(1024,1024)

        self.fc2   = nn.Sequential(

                nn.Linear(1024,2048),
                nn.SiLU(),
                nn.Linear(2048,4096),
                nn.SiLU(),
                nn.Linear(4096,(self.C*16)*convdim_outputs[-1]**2),
                nn.SiLU(),

        )

        self.decoder = nn.Sequential(

                        
            nn.ConvTranspose2d(in_channels=self.C*16, out_channels=self.C*8, stride=self.strides_dec[0], kernel_size=self.kernels_dec[0]), #1
            nn.BatchNorm2d(self.C*8),
            nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=self.C*8, out_channels=self.C*4, stride=self.strides_dec[1], kernel_size=self.kernels_dec[1]), #2
            nn.BatchNorm2d(self.C*4),
            nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=self.C*4, out_channels=self.C*2, stride=self.strides_dec[2], kernel_size=self.kernels_dec[2]), #3
            nn.BatchNorm2d(self.C*2),
            nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, stride=self.strides_dec[3], kernel_size=self.kernels_dec[3]), #4 
            nn.BatchNorm2d(self.C*2),
            nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C, stride=self.strides_dec[4], kernel_size=self.kernels_dec[4]), #5
            nn.BatchNorm2d(self.C),
            nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=self.C, out_channels=self.C, stride=self.strides_dec[5], kernel_size=self.kernels_dec[5]), #6
            nn.BatchNorm2d(self.C),
            nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=self.C, out_channels=1, stride=self.strides_dec[6], kernel_size=self.kernels_dec[6]), #7
            nn.BatchNorm2d(1),
            nn.SiLU(),
            
        ) 
        
    def forward(self,x):
        
        bs       = x.size(0)

        x       = self.encoder(x)
        x       = x.view(x.size(0),-1)

        x       = self.fc1(x)
        latents = self.latent(x)
        x       = self.fc2(latents)

        x       = x.view(bs,self.C*16,convdim_outputs[-1],convdim_outputs[-1])
        x       = self.decoder(x)
        
        return x
        


def train(model,train_dataloader,optimizer,device,loss_fn,EPOCH=800):
    

    save_path = os.path.join('/home/sarperyn/sarperyurtseven/ProjectFiles', 'training_results3')
    model_save_path = os.path.join('/home/sarperyn/sarperyurtseven/ProjectFiles', 'models')

        
    model.train()
    model = model.to(device)
    
    for epoch in tqdm(range(EPOCH)):
        
        batch_loss = 0
        
        for idx, batch in enumerate(train_dataloader):
            
            batch = batch.to(device)
            
            recons = model(batch)

            
            loss = loss_fn(recons,batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()

                
                
            batch_loss = batch_loss / (idx + 1)
            
        
            if idx % 150 == 0:
                plot_results(batch, recons, save_path, epoch, idx)

            print(f'{batch_loss}')

        if (epoch+1) % 20 == 0:
                torch.save(model, os.path.join(model_save_path,f'model_exp-aug2_epoch-{epoch}.pt'))

        print('Epoch:',epoch)

            


def plot_results(imgs, recons, save_path, epoch, idx):

    bs = 8 #imgs.size(0)
    fig, axes = plt.subplots(nrows=2,ncols=bs,figsize=(bs*4,20))

    for i, (row,col) in enumerate(product(range(2),range(bs))):

        if row == 0:
            axes[row][col].imshow(np.transpose(imgs[col].detach().cpu().numpy(),(1,2,0)))
            if col == 0:
                axes[row][col].set_ylabel('Original Image',fontsize=15,fontweight='bold')
        
        elif row == 1:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(),(1,2,0)))

            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image',fontsize=15,fontweight='bold')

            
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{idx}.jpg'),format='jpg',bbox_inches='tight',pad_inches=0,dpi=100)
    plt.show() 



injections = glob.glob('/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/injections/*.png')[:35000]
augmentations = glob.glob('/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/sci_imgs/*')[:35000]

test = glob.glob('/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/sci_imgs/*')[35000:]

with open(r'text_dirs.txt', 'w') as fp:
    for item in test:
        fp.write("%s\n" % item)
    print('Done')

image_paths = injections + augmentations


syndata        = SynDataset(image_paths=image_paths)
syndata_loader = DataLoader(dataset=syndata, batch_size=256, shuffle=True)

kernels_enc = [7,7,7,7,7,7,7]
paddings_enc= [0,0,0,0,0,0,0]
strides_enc = [1,2,1,2,1,2,1]
maxpool = [0,0,0,0,0,0,0]

kernels_dec  = [7,7,7,7,7,7,10]
paddings_dec = [0,0,0,0,0,0,0]
strides_dec  = [1,2,1,2,1,2,1]


convdim_outputs = calculate_conv_dims(320,paddings_enc,kernels_enc,strides_enc,maxpool)

convtrans_outputs = calculate_convtrans_dim(24,paddings_dec,kernels_dec,strides_dec)


model = Exonet(convdim_enc_outputs=convdim_outputs, 
               convdim_dec_outputs=convtrans_outputs,
               kernels_enc=kernels_enc, 
               strides_enc=strides_enc, 
               kernels_dec=kernels_dec, 
               strides_dec=strides_dec)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  
loss_fn   = torch.nn.CrossEntropyLoss(reduction='sum')
device    = 'cuda:2'

train(model=model,train_dataloader=syndata_loader,optimizer=optimizer,loss_fn=loss_fn,device=device)



