import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms


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
        


