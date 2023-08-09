from astropy.io import fits
from glob import glob
import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from jwst import datamodels
import math
import PIL
from IPython.display import display
from visualization_helpers import get_headers

def get_hdu(fits_,data):
    
    pri_data = []
    sci_data = []
    err_data = []
    dq_data  = [] 
    con_data = []
    wht_data = []

    for i,path in enumerate(fits_):
        fits_file = fits.open(path)
        
        if data == 'psf':
            pri = fits_file[0].header
            sci = fits_file[1].data
            dq  = fits_file[2].data 
            err = fits_file[3].data
            
            dq_data.append(dq)
            
        elif data == 'i2d':
            
            pri = fits_file[0].header
            sci = fits_file[1].data
            err = fits_file[2].data 
            con = fits_file[3].data
            wht = fits_file[4].data
                
            con_data.append(con)
            wht_data.append(wht)
            
        elif data == 'psfsub':
            
            pri = fits_file[0].header
            sci = fits_file[1].data
            err = fits_file[2].data 
            dq  = fits_file[3].data
            
            dq_data.append(dq)
            
        pri_data.append(pri)
        sci_data.append(sci)
        err_data.append(err)

        
    return pri_data,sci_data,err_data,dq_data,con_data,wht_data


def get_stage3_products(suffix,directory):
    return glob(os.path.join(directory, f'*{suffix}.fits'))


def get_hdu_data(fits_dirs,suffix):
    
    if suffix == 'psfstack':
        header,sci,err,dq,con,wht = get_hdu(fits_dirs,data='psf')
    
    elif suffix == 'i2d':
        header,sci,err,dq,con,wht = get_hdu(fits_dirs,data='i2d')
        
    elif suffix == 'psfsub':
        header,sci,err,dq,con,wht = get_hdu(fits_dirs,data='psfsub')


    hdu_dict = {
        'header':header,
        'sci':sci,
        'err':err,
        'dq':dq,
        'con':con,
        'wht':wht
    }
    
    return hdu_dict


def save_hdu_data(hdu_dict,instrume,proposal_id,file_name,hdu,header):
    
    filters = get_headers(header,'FILTER')
    targets = get_headers(header,'TARGPROP')

    data = hdu_dict[hdu]
    DIR = f'/data/scratch/bariskurtkaya/dataset/{instrume}/{proposal_id}/sci_imgs'
    if not os.path.exists(DIR):
        print('sss')
        os.makedirs(DIR)

    for i,d in enumerate(data):
        
        save_name = file_name[i].split('/')[-1][:-5]
        #np.save(os.path.join(DIR,f'{proposal_id}_{filters[i]}_{targets[i]}_{suffix}_{hdu}_{i}.npy'),d)
        npsf = d.shape[0]
        for j in range(npsf):
            im = PIL.Image.fromarray(d[j])
            im = im.convert("L")
            im.save(os.path.join(DIR, save_name + f'_{hdu}_{j}.jpg'))



INSTRUME = 'NIRCAM'
PROPOSAL_ID = '1386'

directory_1386_nircam = f'/data/scratch/bariskurtkaya/dataset/NIRCAM/{PROPOSAL_ID}/mastDownload/JWST'

psfstacks_nircam_1386 = get_stage3_products(suffix='psfstack',directory=directory_1386_nircam)

#print(psfstacks_nircam_1386[0].split('/')[-1][:-5])
header,sci,err,dq,con,wht = get_hdu(psfstacks_nircam_1386,data='psf')

hdu_dict = get_hdu_data(psfstacks_nircam_1386,suffix='psfstack')
save_hdu_data(hdu_dict,INSTRUME,PROPOSAL_ID,psfstacks_nircam_1386,'sci',header)