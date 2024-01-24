
from torch.utils.data import DataLoader
import glob
import os
import sys
import random
sys.path.append(os.path.dirname(os.getcwd()))

from util.util_data import *
from util.util_dirs import *
from util.util_train import *
from models import ExoClassifier


### TRAIN classifier script

def train_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--idx',type=str, default='classifier_lp/0')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--epoch',type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', action='store_true', help='If true set scheduler')
    parser.add_argument('--train_folder', type=str, default='fc5_train', help='Train folder name')
    parser.add_argument('--apply_lowpass', action='store_true', help='If true apply low pass filter to the input images')
    args = parser.parse_args()
    return args

def get_arc_params():

    kernels_enc = [7,5,5,3]
    paddings_enc= [1,1,1,1]
    strides_enc = [2,1,1,2]

    maxpool = [0,0,0,0,0,0,0]

    kernels_dec = list(reversed(kernels_enc))
    paddings_dec= [1,1,1,1]
    strides_dec = list(reversed(strides_enc))

    convdim_outputs = calculate_conv_dims(80,paddings_enc,kernels_enc,strides_enc,maxpool)

    convtrans_outputs = calculate_convtrans_dim(convdim_outputs[-1],paddings_dec,kernels_dec,strides_dec)

    return convdim_outputs, kernels_enc, strides_enc


args = train_arg_parser()
init_wandb(args)

injected = glob.glob(os.path.join(INJECTIONS, args.train_folder, '*fc5.npy'))[:20000]
not_injected = glob.glob(os.path.join(INJECTIONS, args.train_folder, '*[!fc5].npy'))[:20000]

train_paths = injected + not_injected
random.shuffle(train_paths)
print(len(train_paths))

syndata        = SynDatasetLabel(image_paths=train_paths, args=args)
syndata_loader = DataLoader(dataset=syndata, batch_size=args.batch_size, shuffle=True)

convdim_outputs, kernels_enc, strides_enc = get_arc_params()


model = ExoClassifier(args=args,
               in_channels=1,
               latent_dim=8,
               convdim_enc_outputs=convdim_outputs, 
               kernels=kernels_enc, 
               strides=strides_enc)

model.train_model(model=model,train_dataloader=syndata_loader)