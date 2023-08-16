import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import os
import wandb
import argparse
import yaml


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



def plot_results(imgs, recons, save_path, epoch, idx):

    bs = 8 #imgs.size(0)
    fig, axes = plt.subplots(nrows=2,ncols=bs,figsize=(bs*4,20))

    for i, (row,col) in enumerate(product(range(2),range(bs))):

        if row == 0:
            axes[row][col].imshow(np.transpose(imgs[col].detach().cpu().numpy(),(1,2,0)))
            if col == 0:
                axes[row][col].set_ylabel('Original Image',fontsize=15,fontweight='bold')
        
        elif row == 1:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(),(1,2,0)))

            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image',fontsize=15,fontweight='bold')

            
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{idx}.jpg'),format='jpg',bbox_inches='tight',pad_inches=0,dpi=100)
    plt.show() 


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--idx',type=str, default='1')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_arguemnt('--lr', type=int, default='0.001')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    return args


def init_wandb(args):
    group_name, exp_name = args.idx.split('/')
    wandb.init(
    mode =   'online'if args.wandb else 'disabled',
    project = 'jwst-dev',
    entity  = 'jwst-princeton',
    group   = f'syn',
    name    = f'syn-runs',
    config = vars(args)
)


def save_dict_as_yaml(dic, path):
    with open(path, 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)

# python train.py --device cuda:0 --idx 1 --seed1