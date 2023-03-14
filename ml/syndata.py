import matplotlib.pyplot as plt

from PIL import Image

from tqdm.auto import tqdm

import numpy as np
from numpy.random import Generator, MT19937, SeedSequence



mean = [0,0]
cov = [[3,0], [0,3]]

mean_exo = [0,0]
cov_exo = [[0.25,0], [0, 0.25]]


w,h = 28,28
my_dpi = 10


synthetic_dataser_dir_train = '/home/sarperyn/sarperyurtseven/ProjectFiles/dataset/synthetic/Train/train/'
synthetic_dataser_dir_test  = '/home/sarperyn/sarperyurtseven/ProjectFiles/dataset/synthetic/Test/test/'



def create_synthetic_data(img_name:str, width:int = 28, height:int = 28, dpi:float = 10, mean: [] = [0,0], cov: [] = [[3,0], [0,3]], xlimit: int = 30, ylimit: int = 30):
    
    rand_pts  = np.random.randint(500,900)
    
    pts = np.random.multivariate_normal(mean, cov, rand_pts)
    
    plt.rcParams["figure.facecolor"] = 'black'
    plt.rcParams['figure.figsize'] = [width/dpi, height/dpi]
    plt.axis('off')
    
    plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.1, color='white')

    plt.xlim(-xlimit, xlimit)
    plt.ylim(-ylimit, ylimit)
    
    plt.savefig(img_name,format='jpeg', dpi=dpi, facecolor='black')
    #plt.show()
    
def save_star_figs(nsample:int):
    
    for idx in range(1,nsample+1):
        img_name = synthetic_dataser_dir_train + 'trial' + str(idx) + '.jpeg'
        create_synthetic_data(img_name = img_name, width=w, height=h, dpi=my_dpi, mean=mean, cov=cov)
    
        if idx % 500 == 0:
            print(f'{idx}. is generated')
    
    print('Stars are generated...')
        
        
def create_exop_synth_data(img_name:str, width:int = 28, height:int = 28, dpi:float = 10, mean: [] = [0,0], cov: [] = [[3,0], [0,3]], mean_exo: [] = [0,0], cov_exo: [] = [[0.25,0], [0,0.25]], xlimit: int = 30, ylimit: int = 30):
    
    rand_pts_star  = np.random.randint(500,800)
    rand_pts_exo   = np.random.randint(20,80)
    pts = np.random.multivariate_normal(mean, cov, 500)
    pts_exo = np.random.multivariate_normal(mean_exo, cov_exo, rand_pts_exo) 
    
    dist_x = np.random.randint(5,12)
    dist_y = np.random.randint(5,12)
    
    x_add = np.random.randint(1,dist_x) * np.random.choice((-1, 1))
    y_add = np.random.randint(1,dist_y) * np.random.choice((-1, 1))
    
    pts_exo.T[0] = pts_exo.T[0] + x_add
    pts_exo.T[1] = pts_exo.T[1] + y_add
    
    #print(x_add, y_add)
    
    all_pts = np.append(pts, pts_exo, axis = 0)
    
    plt.rcParams["figure.facecolor"] = 'black'
    plt.rcParams['figure.figsize'] = [width/dpi,
                                      height/dpi]
    plt.axis('off')
    
    plt.plot(all_pts[:, 0], all_pts[:, 1], '.', alpha=0.4, color='white')
    
    plt.xlim(-xlimit, xlimit)
    plt.ylim(-ylimit, ylimit)
    

    plt.savefig(img_name, dpi=dpi, facecolor='black')
    #plt.show()

def save_starexo_figs(nsample:int):
    
    for idx in range(1,nsample+1):
        
        img_name = synthetic_dataser_dir_train + 'trial_exo' + str(idx) + '.jpeg'
        create_exop_synth_data(img_name = img_name, width=w, height=h, dpi=my_dpi, mean=mean, cov=cov)
        
        if idx % 500 == 0:
            print(f'{idx}. is generated')
    
    print('Stars-exos are generated...')
        

#save_star_figs(nsample=2500)

save_starexo_figs(nsample=2500)