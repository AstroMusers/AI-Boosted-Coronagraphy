import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import cv2 as cv
import h5py
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from generate_samples import visualize_syn_data
from warnings import filterwarnings
from syndata import get_exo_locations
from itertools import product
import seaborn as sns
import pandas as pd
filterwarnings('ignore')


def visualize_data_locations(data,exo_locations):
    _, axes = plt.subplots(nrows=5,ncols=5,figsize=(20,20))

    for i, (row,col) in enumerate(product(range(5),range(5))):
        
        star_circle = plt.Circle( (70, 90 ),
                                      20 ,
                                      fill =False ,color='red')
        
        exo_circle = plt.Circle( (exo_locations[i][0]+3, exo_locations[i][1]+3),
                                     3 ,
                                     fill = False ,color='blue')
        
        axes[row][col].imshow(data[i],cmap='gray',clim=(0,124))
        axes[row][col].add_artist(star_circle)
        axes[row][col].add_artist(exo_circle)
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])
        
    _.patch.set_facecolor('#423f3b')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()  
    
    
    
def visualize_data(data):
    _, axes = plt.subplots(nrows=5,ncols=5,figsize=(20,20))

    for i, (row,col) in enumerate(product(range(5),range(5))):
                
        axes[row][col].imshow(data[i],cmap='gray',clim=(0,124))
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])
        
    _.patch.set_facecolor('#423f3b')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()  
    
        
    
def label_data(arr:np.ndarray,label:str):
    star_label    = torch.Tensor([0,1])
    exo_label     = torch.Tensor([1,0])
    #nothing_label = torch.Tensor([0,0,1])
    
    labelled = []
    
    if label == 'star':
        label_stack = torch.stack((star_label,star_label))
        labelled.append(arr)
            
        for i in range(2,len(arr)):
            label_stack = torch.vstack((label_stack,star_label))
        labelled.append(label_stack)
    
    elif label == 'exo':
        
        label_stack = torch.stack((exo_label,exo_label))
        labelled.append(arr)
        for i in range(2,len(arr)):
            label_stack = torch.vstack((label_stack,exo_label))
        labelled.append(label_stack)
        
    if label == 'nothing':
        
        label_stack = torch.stack((nothing_label,nothing_label))
        labelled.append(arr)
            
        for i in range(2,len(arr)):
            label_stack = torch.vstack((label_stack,nothing_label))
        labelled.append(label_stack)
            
    return labelled


def get_train_test():
    

    DIR = '/home/sarperyn/sarperyurtseven/ProjectFiles/dataset/'
    h5_files = glob(os.path.join(DIR,'NIRCAM/**/*.h5'))
    data_1441 = h5py.File(h5_files[0],'r')
    data_1386 = h5py.File(h5_files[1],'r')
    keys_1386 = [x for x in data_1386.keys()]
    final_1386 = np.concatenate((np.array(data_1386[keys_1386[0]]),np.array(data_1386[keys_1386[1]])))

    for i in range(len(keys_1386)-2):

        final_1386 = np.concatenate((final_1386,np.array(data_1386[keys_1386[i+2]])))


    augmented_1386   = np.load(f'{DIR}/augmented/augmented.npy')
    #augmented_erased = np.load(f'{DIR}/augmented/augmented_erased_data.npy')
    blurred_exo      = np.load(f'{DIR}/augmented/blurred_exo.npy')
    blurred_star     = np.load(f'{DIR}/augmented/blurred_star.npy')
    #blurred_nothing  = np.load(f'{DIR}/augmented/blurred_nothing.npy')
    noised_exo       = np.load(f'{DIR}/augmented/noised_exo.npy')
    noised_star      = np.load(f'{DIR}/augmented/noised_star.npy')


    exo_f250m = np.concatenate((np.load(f'{DIR}/synthetic/exo1_f250m.npy'),np.load(f'{DIR}/synthetic/exo2_f250m.npy')),axis=0)
    exo_f300m = np.concatenate((np.load(f'{DIR}/synthetic/exo1_f300m.npy'),np.load(f'{DIR}/synthetic/exo2_f300m.npy')),axis=0)
    exo_f356w = np.concatenate((np.load(f'{DIR}/synthetic/exo1_f356w.npy'),np.load(f'{DIR}/synthetic/exo2_f356w.npy')),axis=0)
    exo_f410m = np.concatenate((np.load(f'{DIR}/synthetic/exo1_f410m.npy'),np.load(f'{DIR}/synthetic/exo2_f410m.npy')),axis=0)
    exo_f444w = np.concatenate((np.load(f'{DIR}/synthetic/exo1_f444w.npy'),np.load(f'{DIR}/synthetic/exo2_f444w.npy')),axis=0)

    exo     = np.concatenate((exo_f250m,exo_f300m,exo_f356w,exo_f410m,noised_exo,blurred_exo,exo_f444w),axis=0)
    star    = np.concatenate((final_1386,augmented_1386,noised_star),axis=0)
    
    print('Exo data:',exo.shape)
    print('Star data:',star.shape)


    np.random.shuffle(exo)
    np.random.shuffle(star)
    
    train_star    = star[:int(len(star)*0.9)]
    train_exo     = exo[:int(len(exo)*0.9)]

    test_star     = star[int(len(star)*0.9):]
    test_exo      = exo[int(len(exo)*0.9):]
    
    return train_star, train_exo, test_star, test_exo




def get_labelled_data(train_star,train_exo,test_star,test_exo):
    
    train_star_labelled    = label_data(train_star,label='star')
    train_exo_labelled     = label_data(train_exo,label='exo')


    test_star_labelled     = label_data(test_star,label='star')
    test_exo_labelled      = label_data(test_exo,label='exo')
    
    
    return train_star_labelled, train_exo_labelled, test_star_labelled, test_exo_labelled

