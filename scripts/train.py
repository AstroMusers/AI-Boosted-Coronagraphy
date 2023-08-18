
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import glob
import os
import sys
import wandb
sys.path.append(os.path.dirname(os.getcwd()))


from util.util_data import *
from util.util_dirs import *
from util.util_train import *
from ml.ae_models import Exonet


args = arg_parser()
init_wandb(args)


train_paths = glob.glob(os.path.join(INJECTIONS,'*.npy'))[:95000]
print(len(train_paths))
test_paths = glob.glob(os.path.join(INJECTIONS,'*.npy'))[95000:]
get_test_dirs(test_paths)


syndata        = SynDataset(image_paths=train_paths)
syndata_loader = DataLoader(dataset=syndata, batch_size=args.batch_size, shuffle=True)

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

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  
loss_fn   = torch.nn.MSELoss(reduction='sum')
device    = args.device
    

def train(model,train_dataloader,optimizer,device,loss_fn,EPOCH=800):

    save_path = os.path.join(PLOTS_SAVE_PATH, f'training_results', str(args.idx))
    save_dict_as_yaml((args), os.path.join(PLOTS_SAVE_PATH, 'args.yml'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model.train()
    model = model.to(device)
    
    total_loss = 0

    for epoch in tqdm(range(EPOCH)):
        
        batch_loss = 0
        
        for idx, batch in enumerate(train_dataloader):
            
            batch = batch.float().to(device)
            
            recons = model(batch)

            loss = loss_fn(recons,batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            total_loss += batch_loss

            wandb.log({
                "batch_loss":batch_loss
            })

            print(f'{batch_loss}')
            if idx % 150 == 0:
                plot_results(batch, recons, save_path, epoch, idx)

        if (epoch+1) % 30 == 0:
                torch.save(model, os.path.join(save_path,f'model_epoch-{epoch}.pt'))

        total_loss = batch_loss / idx
        print('Epoch:',epoch)
        wandb.log({
                "total_loss": total_loss
        })

            

train(model=model,train_dataloader=syndata_loader,optimizer=optimizer,loss_fn=loss_fn,device=device)



