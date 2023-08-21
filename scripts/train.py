
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
from ml.models import AExonet, VAExonet


args = arg_parser()
init_wandb(args)


train_paths = glob.glob(os.path.join(INJECTIONS,'train/*.npy'))
print(len(train_paths))
test_paths = glob.glob(os.path.join(INJECTIONS,'test/*.npy'))
print(len(test_paths))

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


model = VAExonet(args=args,
               convdim_enc_outputs=convdim_outputs, 
               convdim_dec_outputs=convtrans_outputs,
               kernels_enc=kernels_enc, 
               strides_enc=strides_enc, 
               kernels_dec=kernels_dec, 
               strides_dec=strides_dec)
    

model.train_model(model=model,train_dataloader=syndata_loader)



