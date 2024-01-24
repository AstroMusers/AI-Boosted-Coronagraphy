
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
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import umap
import einops
from sklearn.decomposition import PCA
sys.path.append(os.path.dirname(os.getcwd()))


from util.util_data import *
from util.util_dirs import *
from util.util_train import *
from ml.models import VAE


### Visualize Latent Space script

def plot_sample_comps(comps, index, num_points, num_chairs):


    plt.figure(figsize=(15,15))

    increment = num_points//num_chairs
    # neigh = index.split('_')[-3]
    # dist  = index.split('_')[-2]

    colors = []
    for i in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:num_chairs]:
        colors.extend([i]*increment)

    plt.scatter(comps[:,0],comps[:,1],color=colors)
    plt.colorbar()
    plt.savefig(f'latent{index}.png',format='png',dpi=100)
    plt.show()
    plt.close()
    print(f"saved_latent{index}")


def inference_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--results_dir',type=str, default='/data/scratch/sarperyurtseven/results/training_results')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model', type=str, default='ae')
    parser.add_argument('--inference_folder', type=str, default='fc5_injections_test', help='Train folder name')
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

    convdim_outputs = calculate_conv_dims(120,paddings_enc,kernels_enc,strides_enc,maxpool)

    convtrans_outputs = calculate_convtrans_dim(convdim_outputs[-1],paddings_dec,kernels_dec,strides_dec)


    return convdim_outputs, kernels_enc, strides_enc

def prepare_latents(latent_vecs):

    latent_vecs = torch.stack(latent_vecs)
    print('LATENT_SHAPE',latent_vecs.shape)
    latent_vecs = einops.rearrange(latent_vecs, ' a b c -> (a b) c')
    latent_vecs = latent_vecs.detach().cpu().numpy()

    return latent_vecs

def get_testset(args):

    inj = glob.glob(os.path.join(INJECTIONS, args.inference_folder, '*fc5.npy'))[:args.batch_size]
    no_inj = glob.glob(os.path.join(INJECTIONS, args.inference_folder, '*[!fc5].npy'))[:args.batch_size]

    print("INJ:",len(inj))
    print("No INJ:",len(no_inj))
    test_paths = inj + no_inj

    return test_paths

def get_latest_models(exp_files):

    def sort_key(dir):
        
        return int(dir.split('/')[-1].split('.')[0].split('-')[-1])

    latest_model_paths = []
    for dir in exp_files:

        model = sorted(glob.glob(dir+'/*.pt'), key=sort_key)[-1]
        latest_model_paths.append(model)

    return latest_model_paths


def get_model_paths(args):

    main_dir   = args.results_dir
    exp_files  = sorted(glob.glob(main_dir+'/*ae*'+'/*'))
    model_paths = get_latest_models(exp_files)
    
    return model_paths 



def get_results():

    args = inference_arg_parser()

    model_paths = get_model_paths(args)

    print(args.inference_folder)

    test_paths = get_testset(args)
    syndata        = SynDatasetLabel(image_paths=test_paths, args=args)
    syndata_loader = DataLoader(dataset=syndata, batch_size=args.batch_size, shuffle=True)
    print(args.batch_size)

    convdim_outputs, kernels_enc, strides_enc = get_arc_params()

    model = VAE(args=args,
                in_channels=1,
                latent_dim=8,
                convdim_enc_outputs=convdim_outputs, 
                kernels=kernels_enc, 
                strides=strides_enc)

    model_dir = '/data/scratch/sarperyurtseven/results/training_results/ae_filter/1/model_epoch-1499.pt'
    model = torch.load(model_dir, map_location=args.device)

    latent_vecs = []

    with torch.no_grad():

        for idx, (image, _, filtered_images, image_paths) in enumerate(syndata_loader):
            
            if args.apply_lowpass:
                batch = filtered_images.float().to(args.device)

            else:
                batch = image.float().to(args.device)

            bs    = batch.size(0)
            
            if args.model == 'ae':
                z = model.encode(batch)
            else:
                mu, log_var = model.encode(input)
                z           = model.reparametrize(mu, log_var)

            latent_vecs.append(z)

    
    latent_vecs = prepare_latents(latent_vecs)

    print('Latent:',latent_vecs.shape)
    print("UMAP STARTS")

    n_neighbors = 256
    min_dist = 0.01
    fit   = umap.UMAP(n_neighbors=n_neighbors,
            min_dist=min_dist)   
    umap_comps = fit.fit_transform(latent_vecs)

    print("UMAP DONE")

    pca = PCA(n_components=2)
    pc_comps = pca.fit_transform(latent_vecs)

    index = '_'.join(model_dir.split('/')[-3:])

    plot_sample_comps(umap_comps, f'_{index}_neigh-{n_neighbors}_min_dist-{min_dist}_umap', args.batch_size*2, 2)
    plot_sample_comps(pc_comps, f'_{index}_pca', args.batch_size*2, 2)

get_results()