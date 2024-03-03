import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm
import os
from collections import defaultdict
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from abc import abstractmethod
import cv2

from util.util_data import *
from util.util_dirs import *
from util.util_train import *


########################################################################
########################################################################
########################################################################
######################################################################## 
############################ TRAIN CLASS ###############################
########################################################################
########################################################################
########################################################################
########################################################################

class Train():

    def __init__(self, model, train_dataloader, args, EPOCH=2000):

        self.model            = model
        self.train_dataloader = train_dataloader
        self.EPOCHS           = EPOCH
        self.args             = args

        self.l1_loss     = nn.L1Loss()
        self.l2_loss     = nn.MSELoss()

        self.recons_loss = self.l1_loss if self.args.loss_type == 'l1' else self.l2_loss

    def set_optimizer(self, model):
                
        adam_optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)  
        sgd_optim  = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        optimizer  = sgd_optim if self.args.optim == 'sgd' else adam_optim 

        return optimizer

    def set_save_dir(self):

        save_path = os.path.join(PLOTS_SAVE_PATH, f'training_results', str(self.args.idx))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        save_dict_as_yaml((self.args), os.path.join(save_path, 'args.yml'))
    
        return save_path

    def set_initials(self):
                
        save_path = self.set_save_dir()

        # Set self.model    
        model = self.model
        model = model.train()
        model = model.to(self.args.device)

        optimizer = self.set_optimizer(model)

        return model, optimizer, save_path


    def train_VAE(self):

        model, optimizer, save_path = self.set_initials()

        if self.args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.98)
                
        with tqdm(total=len(self.train_dataloader)*self.EPOCHS) as tt:
            
            for epoch in tqdm(range(self.EPOCHS)):

                epoch_dict = defaultdict(lambda:0.0)
                print(f'Epoch {epoch} starts.')
                mu_log = 0
                log_var_log = 0
                for idx, (batch, _, filtered_batch, img_paths) in enumerate(self.train_dataloader):
                    
                    batch_dict = defaultdict(lambda:0.0)
                    batch_dict['epoch'] = epoch
                    batcht = batch
                    
                    # Batches to device
                    if self.args.apply_lowpass:
                        batch = filtered_batch.float().to(self.args.device)

                    else:
                        batch = batch.float().to(self.args.device)
                   

                    recons, z, mu, log_var = model(batch)
            
                    # Compute Loss
                    loss_dict = model.loss_function(recons, filtered_batch, mu, log_var)
                    loss = loss_dict['loss']
                    recons_loss = loss_dict['recons_loss']
                    kld_loss    = loss_dict['kld']
                    
                    # Back Prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if self.args.model != 'ae':
                        mu_log  +=  torch.norm(mu)
                        log_var_log  +=  torch.norm(log_var)
                        batch_dict['loss/kld_loss'] += kld_loss.item() 
                    
                    batch_dict['loss/batch_loss'] += loss.item() 
                    batch_dict['loss/recons_loss'] += recons_loss.item() 
                        

                    print(f"Epoch {epoch}/{self.EPOCHS}  Batch {idx}/{len(self.train_dataloader)}")

                    if self.args.wandb:
                        wandb.log(batch_dict)

                    if idx % 150 == 0:
                        if self.args.apply_lowpass:
                            plot_results(batcht, recons, filtered_batch, save_path, epoch, idx, img_paths)
                        else:
                            plot_results(batcht, recons, save_path, epoch, idx, img_paths)
                
                if self.args.model != 'ae':
                    epoch_dict['latent/mu'] = mu_log/idx
                    epoch_dict['latent/log_var'] = log_var_log/idx

                epoch_dict['loss/epoch_loss'] = batch_dict['loss/batch_loss'] / (idx)
                print(f"Epoch {epoch}/{self.EPOCHS}  Loss:{epoch_dict['loss/epoch_loss']}")


                if (epoch+1) % 20 == 0:
                        torch.save(model, os.path.join(save_path,f'model_epoch-{epoch}.pt'))

                if self.args.wandb:
                    wandb.log(epoch_dict)
    
                if self.args.scheduler:
                    scheduler.step()

########################################################################
########################################################################
########################################################################
######################################################################## 
############################### VAE ####################################
########################################################################
########################################################################
########################################################################
########################################################################
    

class VAE(nn.Module):
    
    def __init__(self, 
                 args, 
                 in_channels:int,
                 latent_dim:int,
                 convdim_enc_outputs:list, 
                 kernels:list, 
                 strides:list, 
                 hidden_dims:list = None):
        
        super(VAE,self).__init__()
        
        self.convdim_enc = convdim_enc_outputs
        self.kernels     = kernels
        self.strides     = strides
        self.hidden_dims = hidden_dims
        self.latent_dim  = latent_dim
        self.in_channels = in_channels 
        self.args        = args

        if self.hidden_dims is None:
            self.hidden_dims = [2**(i+4) for i in range(len(kernels))]

        
        #Encoder
        modules = []
        for idx in range(len(self.hidden_dims)):

            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.hidden_dims[idx],
                              kernel_size=self.kernels[idx],
                              stride=self.strides[idx],
                              padding = 1),
                    nn.BatchNorm2d(self.hidden_dims[idx]),
                    nn.LeakyReLU()))
            
            self.in_channels = self.hidden_dims[idx]

        self.encoder = nn.Sequential(*modules)

        #Latent
        if self.args.model == 'ae':

            self.fc_latent = nn.Linear((self.hidden_dims[-1])*self.convdim_enc[-1]**2, self.latent_dim)
            self.decoder_input = nn.Linear(self.latent_dim, (self.hidden_dims[-1])*self.convdim_enc[-1]**2)


        else:
            self.fc_mu = nn.Linear((self.hidden_dims[-1])*self.convdim_enc[-1]**2, self.latent_dim)
            self.fc_var = nn.Linear((self.hidden_dims[-1])*self.convdim_enc[-1]**2, self.latent_dim)
            self.decoder_input = nn.Linear(self.latent_dim, (self.hidden_dims[-1])*self.convdim_enc[-1]**2)


        #Decoder
        modules = []
        self.hidden_dims.reverse()
        self.kernels.reverse()
        self.strides.reverse()
        

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=self.kernels[i],
                                       stride=self.strides[i],
                                       padding=1,
                                       output_padding= 0 if strides[i]== 1 else 1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               1,#self.hidden_dims[-1],
                                               kernel_size=kernels[-1],
                                               stride=strides[-1],
                                               padding=1,
                                               output_padding=1),
                            #nn.BatchNorm2d(self.hidden_dims[-1]),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(in_channels = self.hidden_dims[-1], 
                            #           out_channels= self.hidden_dims[-1],
                            #           kernel_size= 3, 
                            #           padding= 0),
                            # nn.BatchNorm2d(self.hidden_dims[-1]),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(in_channels = self.hidden_dims[-1], 
                            #           out_channels= in_channels,
                            #           kernel_size= 3, 
                            #           padding= 0),
                            nn.Tanh())
    

    def encode(self, input):

        result = self.encoder(input)
        result = result.view(result.size(0),-1)

        if self.args.model =='ae':

            z = self.fc_latent(result)

            return z
        
        else:

            mu = self.fc_mu(result)
            log_var = self.fc_var(result)

            return [mu, log_var]
    
    def decode(self, z, *args):
        
        bs     = args[0]
        result = self.decoder_input(z)
        result = result.view(bs, self.hidden_dims[0], self.convdim_enc[-1], self.convdim_enc[-1])
        result = self.decoder(result)
        result = self.final_layer(result)

        return result
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
        
    def forward(self, input, mu=None, log_var=None):

        bs          = input.size(0)

        if self.args.model == 'ae':

            z = self.encode(input)

        else:

            mu, log_var = self.encode(input)
            z           = self.reparameterize(mu, log_var)

        recons      = self.decode(z, bs)

        return [recons, z, mu, log_var]

    def loss_function(self,
                    *args):

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0 if self.args.model == 'ae' else self.args.kld_weight

        recons_loss = F.mse_loss(recons, input)

        kld_loss = 0 if self.args.model == 'ae' else torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'recons_loss':recons_loss.detach(), 'kld': 0 if self.args.model == 'ae' else -kld_loss.detach()}
    

########################################################################
########################################################################
########################################################################
######################################################################## 
###########################  CLASSIFIER  ###############################
########################################################################
########################################################################
########################################################################
########################################################################

class ExoClassifier(nn.Module):
    
    def __init__(self,
                 args, 
                 in_channels:int,
                 latent_dim:int,
                 convdim_enc_outputs:list, 
                 kernels:list, 
                 strides:list,
                 hidden_dims:list = None):
        
        super(ExoClassifier,self).__init__()
        
        self.convdim_enc = convdim_enc_outputs
        self.kernels     = kernels
        self.strides     = strides
        self.latent_dim  = latent_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels 
        self.args        = args

        if self.hidden_dims is None:
            self.hidden_dims = [2**(i+4) for i in range(len(kernels))]

        
        #Encoder
        modules = []
        for idx in range(len(self.hidden_dims)):

            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.hidden_dims[idx],
                              kernel_size=self.kernels[idx],
                              stride=self.strides[idx],
                              padding = 1),
                    nn.BatchNorm2d(self.hidden_dims[idx]),
                    nn.LeakyReLU()))
            
            self.in_channels = self.hidden_dims[idx]

        self.encoder = nn.Sequential(*modules)


        fcs = []
        fc_nodes = [2**(i*2) for i in reversed(range(1,6,2))]
        nodes = (self.hidden_dims[-1])*self.convdim_enc[-1]**2

        for idx in range(len(fc_nodes)):
            fcs.append(nn.Sequential(
                nn.Linear(nodes, fc_nodes[idx]),
                nn.LeakyReLU()
            ))
            
            nodes = fc_nodes[idx]

        self.fc_layers = nn.Sequential(*fcs)

        self.final_layer = nn.Linear(fc_nodes[-1],2)

    def encode(self, input):

        result = self.encoder(input)
        result = result.view(result.size(0),-1)

        return result

    def forward(self, x):
        
        z = self.encode(x)
        z = self.fc_layers(z)
        z = self.final_layer(z)
        output = F.log_softmax(z)

        return output


    def train_model(self, model, train_dataloader):


        save_path = os.path.join(PLOTS_SAVE_PATH, f'training_results', str(self.args.idx), 'models')
        save_dict_as_yaml((self.args), os.path.join(PLOTS_SAVE_PATH, 'args.yml'))

        if not os.path.exists(save_path):
           os.makedirs(save_path)

        model.train()
        model = model.to(self.args.device)

        adam_optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)  
        sgd_optim  = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        optimizer  = sgd_optim if self.args.optim == 'sgd' else adam_optim 
        EPOCH = self.args.epoch

        with tqdm(total=len(train_dataloader)*EPOCH) as tt:

            total_loss = 0
            for epoch in range(EPOCH):

                batch_loss = 0

                for idx, (image, label, filtered_batch, _) in enumerate(train_dataloader):
                    

                    if self.args.apply_lowpass:
                        batch = filtered_batch.to(self.args.device).float()

                    else:
                        batch = image.to(self.args.device).float()

                    label = label.type(torch.LongTensor).to(self.args.device)

                    output = model(batch)
                    
                    loss = F.nll_loss(output, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item()
                    tt.update()


                batch_loss = batch_loss / (idx+1)
                
                torch.save(model, os.path.join(save_path, f'model.pt'))

                if idx % 300 == 0:
                    plot_inputs_classifier(batch, save_path, epoch, idx)

                total_loss += batch_loss
                print('Batch_loss:', batch_loss) 
            
            total_loss = total_loss / epoch 
            print('Total_loss:', total_loss) 