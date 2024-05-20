
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


def train_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--idx',type=str, default='classifier/b')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--epoch',type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', action='store_true', help='If true set scheduler')
    parser.add_argument('--apply_lowpass', action='store_true', help='If true apply low pass filter to the input images')
    parser.add_argument('--train_pids', type=str, default='1386', help='Choose the pids to train the model on (use * for all programs)')
    parser.add_argument('--train_filters', type=str, default='*', help='Choose the filters to train the model on (f300m, f277w, f356w, f444w) (use * for all filters)')
    parser.add_argument('--mode', type=str, default='train', help='Choose the mode (train, test)')
    args = parser.parse_args()
    return args

def get_arc_params():

    kernels_enc = [7,5,5,3]
    paddings_enc= [1,1,1,1]
    strides_enc = [2,1,1,2]

    maxpool = [0 for i in range(len(kernels_enc))]

    convdim_outputs = calculate_conv_dims(80,paddings_enc,kernels_enc,strides_enc,maxpool)

    return convdim_outputs, kernels_enc, strides_enc

def get_paths(args):

    train_pids = args.train_pids.split(' ')
    train_filters = args.train_filters.split(' ')
    mode = args.mode

    if len(train_pids) == 1:
        train_pids = train_pids[0]

        if len(train_filters) == 1:
            train_filters = train_filters[0]
            injected     = glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{train_filters}*fc*.npy'))
            not_injected = glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{train_filters}*[!fc].npy')) 
            not_injected = list(set(not_injected) - set(injected))

        else:
            injected = []
            not_injected = []

            for f in train_filters:
                injected     += glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{f}*fc*.npy'))
                not_injected += glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{f}*[!fc].npy'))
                not_injected = list(set(not_injected) - set(injected))

    else:
        injected = []
        not_injected = []

        if len(train_filters) == 1:
            for pid in train_pids:
                injected     += glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{train_filters}*fc*.npy'))
                not_injected += glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{train_filters}*[!fc].npy'))

        else:
            for pid in train_pids:
                for f in train_filters:
                    injected     += glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{f}*fc*.npy'))
                    not_injected += glob.glob(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{f}*[!fc].npy'))

    print("INJECTED:",len(injected))
    print("NOT INJECTED:",len(not_injected))

    paths = injected + not_injected
    random.shuffle(paths)
    print("#Samples:",len(paths))

    return paths


args = train_arg_parser()
init_wandb(args)
train_paths = get_paths(args=args)

syndata        = SynDatasetLabel(image_paths=train_paths, args=args)
syndata_loader = DataLoader(dataset=syndata, batch_size=args.batch_size, shuffle=True)

convdim_outputs, kernels_enc, strides_enc = get_arc_params()


model = ExoClassifier(args=args,
               in_channels=1,
               latent_dim=16,
               convdim_enc_outputs=convdim_outputs, 
               kernels=kernels_enc, 
               strides=strides_enc)

model.train_model(model=model,train_dataloader=syndata_loader)