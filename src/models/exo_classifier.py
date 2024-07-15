import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm
import os
from torch import nn

from ..utils.data_utils import *
from ..utils.variable_utils import *
from ..utils.viz_utils import *


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


        save_path = os.path.join(PLOT_SAVE_PATH, f'training_results', str(self.args.idx), 'models')
        save_dict_as_yaml((self.args), os.path.join(PLOT_SAVE_PATH, 'args.yml'))

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