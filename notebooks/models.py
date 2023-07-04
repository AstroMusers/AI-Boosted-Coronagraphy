import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import numpy as np




class Exonet(nn.Module):
    
    def __init__(self,convdim_outputs:list,kernels:list,strides:list):
        
        super(Exonet,self).__init__()
        
        self.convdim = convdim_outputs
        self.kernels = kernels
        self.strides = strides
        self.C       = 8 

        
        self.exonet  = nn.Sequential(
                        
            nn.Conv2d(in_channels=1,out_channels=self.C,stride=strides[0],kernel_size=kernels[0]), #1
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C,stride=strides[1],kernel_size=kernels[1]), #2
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C,stride=strides[2],kernel_size=kernels[2]), #3
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C*2,stride=strides[3],kernel_size=kernels[3]), #4 
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=strides[4],kernel_size=kernels[4]), #5
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=strides[5],kernel_size=kernels[5]), #6
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=strides[6],kernel_size=kernels[6]), #7
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=strides[7],kernel_size=kernels[7]), #8
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=strides[8],kernel_size=kernels[8]), #9
            nn.ReLU(),
        
        ) 
        
        self.linear = nn.Sequential(
        
                nn.Linear((self.C*2)*convdim_outputs[-1]**2,4096),
                nn.ReLU(),
                nn.Linear(4096,1024),
                nn.ReLU(),
                nn.Linear(1024,2),
                nn.Softmax()
        )
        
        
    def forward(self,x):
        
        x = self.exonet(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        
        x = self.linear(x)
        
        return x



class VExocoder(nn.Module):

    def __init__(self,convdim_outputs_e:list,kernels_e:list,strides_e:list,convdim_outputs_d:list,kernels_d:list,strides_d:list,latent_dim:int,device):

        super(Exocoder,self).__init__()

        self.convdim           = convdim_outputs_e
        self.kernels           = kernels_e
        self.strides           = strides_e
        self.convtranspose     = convdim_outputs_d
        self.kernelsd          = kernels_d
        self.stridesd          = strides_d
        self.latent_dim        = latent_dim
        self.device            = device 

        self.C                 = 8


        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=1,out_channels=self.C,stride=self.strides[0],kernel_size=self.kernels[0]), #1
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C,stride=self.strides[1],kernel_size=self.kernels[1]), #2
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C,stride=self.strides[2],kernel_size=self.kernels[2]), #3
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C*2,stride=self.strides[3],kernel_size=self.kernels[3]), #4 
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[4],kernel_size=self.kernels[4]), #5
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[5],kernel_size=self.kernels[5]), #6
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[6],kernel_size=self.kernels[6]), #7
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[7],kernel_size=self.kernels[7]), #8
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[8],kernel_size=self.kernels[8]), #9
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear((self.C*2)*self.convdim[-1]**2,self.latent_dim),
            nn.ReLU(),

        )

        self.mean   = nn.Linear(self.latent_dim,(self.C*2)*self.convdim[-1]**2)
        self.logvar = nn.Linear(self.latent_dim,(self.C*2)*self.convdim[-1]**2)

        self.decoder = nn.Sequential(

            nn.Unflatten(1,(self.C*2,self.convdim[-1],self.convdim[-1])),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[0], stride=self.stridesd[0]), #1
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[1], stride=self.stridesd[1]), #2
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[2], stride=self.stridesd[2]), #3
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[3], stride=self.stridesd[3]), #4
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[4], stride=self.stridesd[4]), #5
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C, kernel_size=self.kernelsd[5], stride=self.stridesd[5]), #6
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C, out_channels=1, kernel_size=self.kernelsd[6], stride=self.stridesd[6]), #7
        )

    def reparametrize(self,mean,logvar):

        eps = torch.randn(mean.size(0),mean.size(1)).to(self.device)
        z = mean + torch.exp(logvar/2) * eps
        
        return z

    def forward(self,x):

        x = self.encoder(x)

        mean = self.mean(x)
        logv = self.logvar(x)

        z = self.reparametrize(mean,logv)

        x_recon = self.decoder(z)

        return x_recon, z, mean, logv
        

def vae_loss(x_recon,x,mean,logv):

    recons = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    kl /= 512 * 160 * 160
    
    return recons + kl 


class Exocoder(nn.Module):

    def __init__(self,convdim_outputs_e:list,kernels_e:list,strides_e:list,convdim_outputs_d:list,kernels_d:list,strides_d:list,latent_dim:int,device):

        super(Exocoder,self).__init__()

        self.convdim           = convdim_outputs_e
        self.kernels           = kernels_e
        self.strides           = strides_e
        self.convtranspose     = convdim_outputs_d
        self.kernelsd          = kernels_d
        self.stridesd          = strides_d
        self.latent_dim        = latent_dim
        self.device            = device 

        self.C                 = 8


        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=1,out_channels=self.C,stride=self.strides[0],kernel_size=self.kernels[0]), #1
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C,stride=self.strides[1],kernel_size=self.kernels[1]), #2
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C,stride=self.strides[2],kernel_size=self.kernels[2]), #3
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C,out_channels=self.C*2,stride=self.strides[3],kernel_size=self.kernels[3]), #4 
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[4],kernel_size=self.kernels[4]), #5
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[5],kernel_size=self.kernels[5]), #6
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[6],kernel_size=self.kernels[6]), #7
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[7],kernel_size=self.kernels[7]), #8
            nn.ReLU(),
            
            nn.Conv2d(in_channels=self.C*2,out_channels=self.C*2,stride=self.strides[8],kernel_size=self.kernels[8]), #9
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear((self.C*2)*self.convdim[-1]**2,self.latent_dim),
            nn.ReLU(),

        )
        
        self.linear  = nn.Linear(self.latent_dim,(self.C*2)*self.convdim[-1]**2)

        self.decoder = nn.Sequential(

            nn.Unflatten(1,(self.C*2,self.convdim[-1],self.convdim[-1])),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[0], stride=self.stridesd[0]), #1
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[1], stride=self.stridesd[1]), #2
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[2], stride=self.stridesd[2]), #3
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[3], stride=self.stridesd[3]), #4
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C*2, kernel_size=self.kernelsd[4], stride=self.stridesd[4]), #5
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C*2, out_channels=self.C, kernel_size=self.kernelsd[5], stride=self.stridesd[5]), #6
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.C, out_channels=1, kernel_size=self.kernelsd[6], stride=self.stridesd[6]), #7
        )

    def forward(self,x):

        z = self.encoder(x)
        
        x_recon = self.decoder(z)

        return x_recon, z 
