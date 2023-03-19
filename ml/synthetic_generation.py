import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm

import numpy as np


# Just star
def create_synthetic_data(img_name:str, width:int = 160, height:int = 160, dpi:float = 10, mean: [] = [0,0], cov: [] = [[3,0], [0,3]], xlimit: int = 14, ylimit: int = 14):
    plt.cla()
    pts = np.random.multivariate_normal(mean, cov, 1750)
    
    plt.rcParams['figure.figsize'] = [width/dpi, height/dpi]
    plt.axis('off')
    
    plt.plot(pts[:, 0], pts[:, 1], 'o', alpha=0.5, color='white')

    plt.xlim(-xlimit, xlimit)
    plt.ylim(-ylimit, ylimit)
    
    plt.savefig(img_name, dpi=dpi, facecolor='black')
    

# Star + Exoplanet
def create_exop_synth_data(img_name:str, width:int = 160, height:int = 160, dpi:float = 10, mean: [] = [0,0], cov: [] = [[3,0], [0,3]], mean_exo: [] = [0,0], cov_exo: [] = [[0.25,0], [0,0.25]], xlimit: int = 14, ylimit: int = 14):
    plt.cla()
    all_pts = []
    
    pts = np.random.multivariate_normal(mean, cov, 1750)
    pts_exo = np.random.multivariate_normal(mean_exo, cov_exo, 250) 
    
    x_add = np.random.randint(4,10) * np.random.choice((-1, 1))
    y_add = np.random.randint(4,10) * np.random.choice((-1, 1))
    
    pts_exo.T[0] = pts_exo.T[0] + x_add
    pts_exo.T[1] = pts_exo.T[1] + y_add
        
    all_pts = np.append(pts, pts_exo, axis = 0)
        
    plt.rcParams['figure.figsize'] = [width/dpi, height/dpi]
    plt.axis('off')
    
    plt.plot(all_pts[:, 0], all_pts[:, 1], 'o', alpha=0.5, color='white')

    plt.xlim(-xlimit, xlimit)
    plt.ylim(-ylimit, ylimit)
    
    plt.savefig(img_name, dpi=dpi, facecolor='black')


    
if __name__ == "__main__":
    
    mean = [0,0]
    cov = [[2.5,0], [0,2.5]]

    w,h = 160,160
    my_dpi = 33.5

    matplotlib.use('Agg')

    synthetic_dataser_dir = '/data/scratch/bariskurtkaya/synthetic_dataset/'
    
    #for star
    for idx in tqdm(range(15000)):
        img_name = synthetic_dataser_dir + 'trial' + str(idx)
        create_synthetic_data(img_name = img_name, width=w, height=h, dpi=my_dpi, mean=mean, cov=cov)
    
    #for exop + star
    for idx in tqdm(range(10000)):
        img_name = synthetic_dataser_dir + 'trial_exo' + str(idx)
        create_exop_synth_data(img_name = img_name, width=w, height=h, dpi=my_dpi, mean=mean, cov=cov)