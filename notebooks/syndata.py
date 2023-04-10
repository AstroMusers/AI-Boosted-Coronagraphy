import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from visualization_helpers import *
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from copy import deepcopy
from tqdm import tqdm
import h5py
import copy



def gaus(x, a, m, s):
    return np.sqrt(a)*np.exp(-(x-m)**2/(2*s**2))



#########################################
def generate_star_exoplanet(nsample):
    
    xx_s, yy_s = np.meshgrid(np.arange(200), np.arange(200))
    xx_p, yy_p = np.meshgrid(np.arange(10), np.arange(10))

    a_star = np.random.randint(35,40)
    #a_exo = np.random.randint(5,10)

    star_stack = 1.6*(gaus(xx_s, 200, 100, a_star)*gaus(yy_s, 200, 100, a_star))**2
    exo_stack  = 0.78*(gaus(xx_p, 10, 5, 5)*gaus(yy_p, 10, 5, 5))**4
    
    star_stack = np.expand_dims(star_stack,axis=2)
    exo_stack = np.expand_dims(exo_stack,axis=2)
    
    
    for i in tqdm(range(nsample-1)):
        
        xx_s, yy_s = np.meshgrid(np.arange(200), np.arange(200))
        xx_p, yy_p = np.meshgrid(np.arange(10), np.arange(10))

        a_star = np.random.randint(35,40)
        #a_exo = np.random.randint(5,10)

        star_n = 1.6*(gaus(xx_s, 200, 100, 40)*gaus(yy_s, 200, 100, 40))**2
        exo_n = 0.2*(gaus(xx_p, 10, 5, 3)*gaus(yy_p, 10, 5, 3))**5
        
        star_n = np.expand_dims(star_n,axis=2)
        exo_n = np.expand_dims(exo_n,axis=2)
        
        #print(star_stack.shape)
        #print(exo_stack.shape)
        star_stack = np.concatenate((star_stack,star_n),axis=2)
        exo_stack  = np.concatenate((exo_stack,exo_n),axis=2)
        
    return star_stack,exo_stack


#########################################
def generate_datasetSE(nsample):
    
    s,e = generate_star_exoplanet(nsample=nsample)
    
    star_loc = np.random.randint(40,85,nsample)

    x_1 = np.random.randint(125,145,nsample) 
    y_1 = np.random.randint(125,145,nsample) 
    x_2 = np.random.randint(165,170,nsample) 
    y_2 = np.random.randint(165,170,nsample) 
    
    x,y = sample([x_1,y_1,x_2,y_2],2)
    
    raw_skewed = np.maximum(0.0, np.expm1(np.random.normal(4, 1.75, (320,320,nsample)))).astype('uint16')
    
    #raw_skewed[60:260,60:260]  = raw_skewed[60:260,60:260:] + s
    
    for i in tqdm(range(nsample)):
        
        raw_skewed[star_loc[i]:star_loc[i]+200,star_loc[i]:star_loc[i]+200,i:i+1]  = raw_skewed[star_loc[i]:star_loc[i]+200,star_loc[i]:star_loc[i]+200,i:i+1] + s[:,:,i:i+1]
        raw_skewed[y[i]:y[i]+10,x[i]:x[i]+10,i:i+1]  = raw_skewed[y[i]:y[i]+10,x[i]:x[i]+10,i:i+1] + e[:,:,i:i+1]
        
    noise = np.random.normal(20,3,(320,320,nsample)) + np.random.exponential(scale=6,size=(320,320,nsample)) + np.squeeze(np.random.dirichlet(alpha=(10,),size=(320,320,nsample)))
    
    data = raw_skewed + 3*noise
    
    return data

#########################################
def generate_star(nsample):
    
    xx_s, yy_s = np.meshgrid(np.arange(200), np.arange(200))
    xx_p, yy_p = np.meshgrid(np.arange(10), np.arange(10))
    a_star = np.random.randint(35,40)
    star_stack = 1.6*(gaus(xx_s, 200, 100, a_star)*gaus(yy_s, 200, 100, a_star))**2
    star_stack = np.expand_dims(star_stack,axis=2)
    
    for i in tqdm(range(nsample-1)):
        
        xx_s, yy_s = np.meshgrid(np.arange(200), np.arange(200))
        xx_p, yy_p = np.meshgrid(np.arange(10), np.arange(10))

        a_star = np.random.randint(35,40)

        star_n = 1.6*(gaus(xx_s, 200, 100, 40)*gaus(yy_s, 200, 100, 40))**2
        
        star_n = np.expand_dims(star_n,axis=2)
        
        star_stack = np.concatenate((star_stack,star_n),axis=2)
        
    return star_stack


#########################################
def generate_datasetS(nsample):
    
    s = generate_star(nsample=nsample)
    star_loc = np.random.randint(40,95,nsample)
    raw_skewed = np.maximum(0.0, np.expm1(np.random.normal(4, 1.75, (320,320,nsample)))).astype('uint16')
    
    for i in tqdm(range(nsample)):
        raw_skewed[star_loc[i]:star_loc[i]+200,star_loc[i]:star_loc[i]+200,i:i+1]  = raw_skewed[star_loc[i]:star_loc[i]+200,star_loc[i]:star_loc[i]+200,i:i+1] + s[:,:,i:i+1]
        
    noise = np.random.normal(20,3,(320,320,nsample)) + np.random.exponential(scale=6,size=(320,320,nsample)) + np.squeeze(np.random.dirichlet(alpha=(10,),size=(320,320,nsample)))
    data = raw_skewed + 3*noise
    
    return data
    
    
    

    
def visualize_syn_data(data,exo_locations):
    _, axes = plt.subplots(nrows=5,ncols=5,figsize=(20,20))

    for i, (row,col) in enumerate(product(range(5),range(5))):
        
        star_circle = plt.Circle( (150, 170 ),
                                      20 ,
                                      fill =False ,color='red')
        
        exo_circle = plt.Circle( (exo_locations[i][0]+3, exo_locations[i][1]+3),
                                     3 ,
                                     fill = False ,color='blue')
        
        axes[row][col].imshow(data[i],cmap='gray',clim=(0,255))
        axes[row][col].add_artist(star_circle)
        axes[row][col].add_artist(exo_circle)
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])
        
    _.patch.set_facecolor('#423f3b')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()  

    
    
def get_exo_locations():

    exo_locs = []
    count = 0
    star_count = 0
    for x in range(100,200,4):

        for y in range(130,210,4):

            if (150 < y < 190) and (125 < x < 165):
                star_count +=1
                continue
                #print('we are in star')
            count +=1
            exo_locs.append((x,y))
            
    return exo_locs    


def create_exoplanet():
    
    xx_p, yy_p = np.meshgrid(np.arange(6), np.arange(6))
    exo_planet  = 1.85*(gaus(xx_p, 6, 3, 3)*gaus(yy_p, 6, 3, 3))**3
    
    return exo_planet



def put_exo2image(data):
    
    exo_planet = create_exoplanet()
    exo_locs = get_exo_locations() # get exoplanet location x-y
    
    samples = copy.deepcopy(data)
    sample1 = samples[0]
    sample1 = np.expand_dims(sample1,axis=0)
    
    
    sample1[0][exo_locs[0][1]:exo_locs[0][1]+6,exo_locs[0][0]:exo_locs[0][0]+6]  = sample1[0][exo_locs[0][1]:exo_locs[0][1]+6,exo_locs[0][0]:exo_locs[0][0]+6] + exo_planet
    sample2 = np.expand_dims(copy.deepcopy(data[15]),axis=0)
    sample2[0][exo_locs[1][1]:exo_locs[1][1]+6,exo_locs[1][0]:exo_locs[1][0]+6]  = sample1[0][exo_locs[1][1]:exo_locs[1][1]+6,exo_locs[1][0]:exo_locs[1][0]+6] + exo_planet
    exo_stack = np.concatenate((sample1,sample2),axis=0)  

    for i in range(2,len(exo_locs)):

        sample = np.expand_dims(copy.deepcopy(data[15]),axis=0)
        sample[0][exo_locs[i][1]:exo_locs[i][1]+6,exo_locs[i][0]:exo_locs[i][0]+6]  = sample[0][exo_locs[i][1]:exo_locs[i][1]+6,exo_locs[i][0]:exo_locs[i][0]+6] + exo_planet
        #print(sample.shape)
        #print(exo_stack.shape)
        exo_stack = np.concatenate((exo_stack,sample),axis=0)
        
    return exo_stack





#########################################
def augment(tensor):
        
    #### ROTATION
    #####################################################################
    angles = [30,45,60,75,90,105,120,135,180,200,210,245,275,310,340]
    rotated_imgs = [T.RandomRotation(degrees=d)(tensor) for d in angles]
    
    rot1 = np.array(rotated_imgs[0])
    rot2 = np.array(rotated_imgs[1])
    rotated_stack = np.concatenate((rot1,rot2),axis=0)   
    
    for i in range(2,len(rotated_imgs)):
    
        rotated_stack = np.concatenate((rotated_stack,np.array(rotated_imgs[i])),axis=0) 
    #####################################################################
    
    
    
    
    #### FLIPPING    
    #####################################################################
    flipped_imgs = [T.RandomHorizontalFlip(p=1)(tensor)]
    #####################################################################
    
    
    
    #### FLIP + ROTATE
    #####################################################################
    flipped_rotated_imgs = [T.RandomRotation(degrees=d)(flipped_imgs[0]) for d in angles]
    
    flip_rot1  = np.array(flipped_rotated_imgs[0])
    flip_rot2  = np.array(flipped_rotated_imgs[1])
    flipped_rotated_stack = np.concatenate((flip_rot1,flip_rot2),axis=0) 
    
    for i in range(2,len(flipped_rotated_imgs)):
    
        flipped_rotated_stack = np.concatenate((flipped_rotated_stack,np.array(flipped_rotated_imgs[i])),axis=0) 
    
    
    augmented_data = np.concatenate((rotated_stack,flipped_rotated_stack,flipped_imgs[0]),axis=0)
    
    return augmented_data




def add_gaussian_blur(exo,star):
    

    copy_exo = deepcopy(exo)
    copy_star = deepcopy(star)
    
    

    blurred_exo = gaussian_filter(copy_exo, sigma=0.8)
    blurred_star = gaussian_filter(copy_star, sigma=1.2)
    
    return blurred_exo, blurred_star





def add_noise(exo,star):
    
    exo_nsample = exo.shape[0]
    star_nsample = star.shape[0]
    
    noise1 = np.random.normal(14,3,(exo_nsample,320,320)) + np.random.exponential(scale=6,size=(exo_nsample,320,320)) + np.squeeze(np.random.dirichlet(alpha=(10,),size=(exo_nsample,320,320)))
    noise2 = np.random.normal(15,3,(star_nsample,320,320)) + np.random.exponential(scale=6,size=(star_nsample,320,320)) + np.squeeze(np.random.dirichlet(alpha=(10,),size=(star_nsample,320,320)))
    
    noised_exo  = exo + noise1
    noised_star = star + noise2
    
    
    return noised_exo, noised_star
    