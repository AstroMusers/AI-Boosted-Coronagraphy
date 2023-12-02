import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import os
import wandb
import argparse
import yaml
import astropy.io.fits as fits
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u


def init_wandb(args):
    group_name, exp_name = args.idx.split('/')
    wandb.init(
    mode =   'online'if args.wandb else 'disabled',
    project = 'jwst-dev',
    entity  = 'jwst-princeton',
    group   = f'{group_name}',
    name    = f'{exp_name}',
    config = vars(args)
)

def save_dict_as_yaml(dic, path):
    with open(path, 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)

def calculate_conv_dims(input_size,paddings:list,kernels:list,strides:list,maxpool:list):
    
    outputs = []
    outputs.append(input_size)
    for i in range(len(paddings)):
        
        output_size = (input_size + (2*paddings[i]) - (kernels[i] - 1) - 1)/strides[i] + 1
        if maxpool[i] != 0:
            output_size = (output_size  + (2*paddings[i]) - (maxpool[i]-1)-1)/2 +1
        
        outputs.append(int(output_size))
        input_size = output_size
        
    print(outputs)
    return outputs


def calculate_convtrans_dim(input_size,paddings:list,kernels:list,strides:list):
    outputs = []
    outputs.append(input_size)
    for i in range(len(paddings)):
        
        output_size = (input_size - 1) * strides[i]  -  2 * paddings[i] + kernels[i] - 1 + 1
        outputs.append(int(output_size))
        input_size = output_size
        
    print(outputs)
    return outputs


def get_test_dirs(test):
    
    with open(r'text_dirs.txt', 'w') as fp:
        for item in test:
            fp.write("%s\n" % item)
        print('Done')


def plot_results_filtered(imgs, recons, filtered, save_path, epoch, idx, image_paths):

    bs = 8 #imgs.size(0)
    step = 20*0.06259530358142339
    step = round(step,2)
    labels = step*np.array([-2., -1., 0., 1., 2.])
    axis_points = np.linspace(0,80,5)


    fig, axes = plt.subplots(nrows=4,ncols=bs,figsize=(bs*1.5,7.5))

    for i, (row,col) in enumerate(product(range(4),range(bs))):
        
        idx_x  = image_paths[i].rfind('x')

        if idx_x == -1:
            x, y = 0, 0
        else:
            x = int(image_paths[i][idx_x+1:idx_x+3])
            y = int(image_paths[i][idx_x+5:idx_x+7])

        if row == 0:
            
            axes[row][col].imshow(np.transpose(imgs[col],(1,2,0)), interpolation='nearest', cmap='inferno')


            if idx_x != -1:
                axes[row][col].text(x, y, s="\u25CF", fontsize=12, color='green', alpha=.2, ha='center', va='center')

            if col == 0:
                axes[row][col].set_ylabel('Original Image',fontsize=8,fontweight='bold')
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)

            else:
                axes[row][col].set_yticks([])
                axes[row][col].set_xticks([])

        elif row == 1:
            axes[row][col].imshow(np.transpose(filtered[col].detach().cpu().numpy(),(1,2,0)), interpolation='nearest', cmap='inferno')

            if idx_x != -1:
                axes[row][col].text(x, y, s="\u25CF", fontsize=12, color='green', alpha=.2, ha='center', va='center')

            if col == 0:
                axes[row][col].set_ylabel('Filtered Image',fontsize=8,fontweight='bold')
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)
        
            else:
                axes[row][col].set_yticks([])
                axes[row][col].set_xticks([])

        elif row == 2:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(),(1,2,0)), interpolation='nearest', cmap='inferno')

            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image',fontsize=8,fontweight='bold', )
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)
            
            else:
                axes[row][col].set_yticks([])
                axes[row][col].set_xticks([])
        
        elif row == 3:
            residue = filtered[col].detach().cpu().numpy() - recons[col].detach().cpu().numpy()
            axes[row][col].imshow(np.transpose(residue,(1,2,0)), interpolation='nearest', cmap='inferno')

            if idx_x != -1:
                axes[row][col].text(x, y, s="\u25CF", fontsize=12, color='green', alpha=.2, ha='center', va='center')# ha='center', va='center'

            if col == 0:
                axes[row][col].set_ylabel('Residual Image',fontsize=8,fontweight='bold')
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)

            else:
                axes[row][col].set_yticks([])
                #axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{idx}.jpg'),format='jpg',bbox_inches='tight',pad_inches=0,dpi=200)
    plt.close()

def plot_results(imgs, recons, save_path, epoch, idx, image_paths):

    bs = 8 #imgs.size(0)
    step = 20*0.06259530358142339
    step = round(step,2)
    labels = step*np.array([-2., -1., 0., 1., 2.])
    axis_points = np.linspace(0,80,5)


    fig, axes = plt.subplots(nrows=3,ncols=bs,figsize=(bs*1.5,7.5))

    for i, (row,col) in enumerate(product(range(3),range(bs))):
        
        idx_x  = image_paths[i].rfind('x')

        if idx_x == -1:
            x, y = 0, 0
        else:

            x = int(image_paths[i][idx_x+1:idx_x+3])
            y = int(image_paths[i][idx_x+5:idx_x+7])

        if row == 0:

            axes[row][col].imshow(np.transpose(imgs[col],(1,2,0)), interpolation='nearest', cmap='inferno')


            if idx_x != -1:
                axes[row][col].text(x, y, s="\u25CF", fontsize=12, color='green', alpha=.2, ha='center', va='center')

            if col == 0:
                axes[row][col].set_ylabel('Original Image',fontsize=8,fontweight='bold')
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)

            else:
                axes[row][col].set_yticks([])
                axes[row][col].set_xticks([])


        elif row == 1:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(),(1,2,0)), interpolation='nearest', cmap='inferno')

            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image',fontsize=8,fontweight='bold', )
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)
            
            else:
                axes[row][col].set_yticks([])
                axes[row][col].set_xticks([])
        
        elif row == 2:
            residue = imgs[col].detach().cpu().numpy() - recons[col].detach().cpu().numpy()
            axes[row][col].imshow(np.transpose(residue,(1,2,0)), interpolation='nearest', cmap='inferno')

            if idx_x != -1:
                axes[row][col].text(x, y, s="\u25CF", fontsize=12, color='green', alpha=.2, ha='center', va='center')# ha='center', va='center'

            if col == 0:
                axes[row][col].set_ylabel('Residual Image',fontsize=8,fontweight='bold')
                axes[row][col].set_yticks(axis_points,labels, fontsize=4, rotation=0)
                axes[row][col].set_xticks(axis_points,labels, fontsize=4, rotation=0)

            else:
                axes[row][col].set_yticks([])
                #axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{idx}.jpg'),format='jpg',bbox_inches='tight',pad_inches=0,dpi=200)
    plt.close()



def plot_inputs_classifier(imgs, save_path, epoch, idx):

    bs = 8 #imgs.size(0)

    fig, axes = plt.subplots(nrows=1,ncols=bs,figsize=(bs*1.5,3.5))

    for i, (row,col) in enumerate(product(range(1),range(bs))):
        
        if row == 0:

            axes[row][col].imshow(np.transpose(imgs[col],(1,2,0)), interpolation='nearest', cmap='inferno')
            axes[row][col].set_yticks([])
            axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{idx}.jpg'),format='jpg',bbox_inches='tight',pad_inches=0,dpi=200)
    plt.close()