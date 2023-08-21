import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm
import os

from util.util_data import *
from util.util_dirs import *
from util.util_train import *


class AExonet(nn.Module):
    
    def __init__(self, args, convdim_enc_outputs:list, convdim_dec_outputs:list, kernels_enc:list, strides_enc:list, kernels_dec:list, strides_dec:list):
        
        super(AExonet,self).__init__()
        
        self.convdim_enc = convdim_enc_outputs
        self.convdim_dec = convdim_dec_outputs
        self.kernels_enc = kernels_enc
        self.strides_enc = strides_enc
        self.kernels_dec = kernels_dec
        self.strides_dec = strides_dec
        self.C           = 8 


        self.l1_loss     = nn.L1Loss()
        self.l2_loss     = nn.MSELoss()


        self.recons_loss = self.l1_loss if args.loss_type == 'l1' else self.l2_loss
        self.args        = args
        
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
        
                nn.Linear((self.C*16)*self.convdim_enc[-1]**2,4096),
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
                nn.Linear(4096,(self.C*16)*self.convdim_enc[-1]**2),
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

        x       = x.view(bs,self.C*16,self.convdim_enc[-1],self.convdim_enc[-1])
        x       = self.decoder(x)
        
        return x
    
    def train_model(self, model, train_dataloader, EPOCH=800):

        save_path = os.path.join(PLOTS_SAVE_PATH, f'training_results', str(self.args.idx))
        save_dict_as_yaml((self.args), os.path.join(PLOTS_SAVE_PATH, 'args.yml'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        model.train()
        model = model.to(self.args.device)

        adam_optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)  
        sgd_optim  = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        optimizer  = sgd_optim if self.args.optim == 'sgd' else adam_optim 

        total_loss = 0

        with tqdm(total=len(train_dataloader)*EPOCH) as tt:

            for epoch in tqdm(range(EPOCH)):
                
                batch_loss = 0
            
                for idx, batch in enumerate(train_dataloader):
                    
                    batch = batch.float().to(self.args.device)
                    
                    recons = model(batch)

                    loss = self.recons_loss(recons,batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item()
                    

                    wandb.log({
                        "batch_loss":batch_loss
                    })

                    print(f'{batch_loss}')
                    if idx % 150 == 0:
                        plot_results(batch, recons, save_path, epoch, idx)

                total_loss += batch_loss

                if (epoch+1) % 30 == 0:
                        torch.save(model, os.path.join(save_path,f'model_epoch-{epoch}.pt'))

                total_loss = batch_loss / idx
                print('Epoch:',epoch)
                wandb.log({
                        "total_loss": total_loss
                })


class VAExonet(nn.Module):
        
    def __init__(self, args, convdim_enc_outputs:list, convdim_dec_outputs:list, kernels_enc:list, strides_enc:list, kernels_dec:list, strides_dec:list):
        
        super(VAExonet,self).__init__()
        
        self.convdim_enc = convdim_enc_outputs
        self.convdim_dec = convdim_dec_outputs
        self.kernels_enc = kernels_enc
        self.strides_enc = strides_enc
        self.kernels_dec = kernels_dec
        self.strides_dec = strides_dec
        self.C           = 8 

        self.l1_loss     = nn.L1Loss()
        self.l2_loss     = nn.MSELoss()
        self.recons_loss = self.l1_loss if args.loss_type == 'l1' else self.l2_loss
        

        self.args = args

        
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
        
                nn.Linear((self.C*16)*self.convdim_enc[-1]**2,4096),
                nn.SiLU(),
                nn.Linear(4096,2048),
                nn.SiLU(),
                nn.Linear(2048,1024),
                nn.SiLU(),
        )


        self.sigma  = nn.Linear(1024,1024)
        self.mu     = nn.Linear(1024,1024)
        self.latent = nn.Linear(1024,1024)

        self.fc2   = nn.Sequential(

                nn.Linear(1024,2048),
                nn.SiLU(),
                nn.Linear(2048,4096),
                nn.SiLU(),
                nn.Linear(4096,(self.C*16)*self.convdim_enc[-1]**2),
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

        #ENCODER
        x       = self.encoder(x.float())
        x       = x.view(x.size(0),-1)
        x       = self.fc1(x)

        #LATENTS
        z_mean, z_logvar = self.mu(x), self.sigma(x)
        encoded = self.reparametrization(z_mean, z_logvar)
        latents = self.latent(encoded)

        #DECODER
        x       = self.fc2(latents)
        x       = x.view(bs,self.C*16,self.convdim_enc[-1],self.convdim_enc[-1])
        decoded       = self.decoder(x)

        return decoded, encoded, z_mean, z_logvar
    

    def loss_fn(self, decoded, batch, z_mean, z_logvar):
        

        recons = self.recons_loss(decoded,  batch)
        kl     = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        total_loss = recons + kl

        return total_loss
    
    def reparametrization(self, z_mean, z_logvar):

        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(self.args.device)

        z   = z_mean + torch.exp(z_logvar/2) * eps

        return z

    def train_model(self, model, train_dataloader, EPOCH=800):

        save_path = os.path.join(PLOTS_SAVE_PATH, f'training_results', str(self.args.idx))
        save_dict_as_yaml((self.args), os.path.join(PLOTS_SAVE_PATH, 'args.yml'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.train()
        model = model.to(self.args.device)

        adam_optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)  
        sgd_optim  = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        optimizer  = sgd_optim if self.args.optim == 'sgd' else adam_optim 

        with tqdm(total=len(train_dataloader)*EPOCH) as tt:

            total_loss = 0
            for epoch in range(EPOCH):

                batch_loss = 0

                for idx, batch in enumerate(train_dataloader):
                    
                    batch = batch.to(self.args.device)

                    decoded, encoded, z_mean, z_logvar = model(batch)

                    loss = self.loss_fn(decoded=decoded,
                                        batch=batch,
                                        z_mean=z_mean,
                                        z_logvar=z_logvar)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item()
                    tt.update()


                batch_loss = batch_loss / (idx+1)
                if idx % 150 == 0:
                    plot_results(batch, decoded, save_path, epoch, idx)

                total_loss += batch_loss
                print('Batch_loss:', batch_loss) 
            
            total_loss = total_loss / epoch 
            print('Total_loss:', total_loss) 


