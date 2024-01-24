from astropy.io import fits
from glob import glob
import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import itertools
from astropy.wcs import WCS
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import random
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from itertools import product
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
import math
import re
import cv2

### HElPER functions to visualize fits files


def get_lower_products(suffix, directory):
    return glob.glob(os.path.join(directory, f'**/*{suffix}.fits'))


def get_stage3_products(suffix, directory):
    return glob.glob(os.path.join(directory, f'*{suffix}.fits'))


def get_axis_labels(w):

    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)

    return y_labels,x_labels

def pixel_to_arcsec_nircam(axis_length,wavelength):
    
    short_wavelengths = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N', 'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']

    long_wavelengths = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M']
    
    if wavelength in short_wavelengths:
        x = 1 / 0.031 
    
    elif wavelength in long_wavelengths:
        x = 1 / 0.063

    else:
        raise ValueError('Wavelength not found!')
    
    zero_point = axis_length / 2
    pos_current_point = zero_point
    neg_current_point = zero_point
    
    arcsec_axis_points = [zero_point]

    while (pos_current_point + x) < axis_length:
        pos_current_point += x
        neg_current_point -= x
        #print(pos_current_point)
        arcsec_axis_points.append(pos_current_point)
        arcsec_axis_points.append(neg_current_point)
        
def get_hdu(fits_, product):
    
    pri_data = []
    sci_data = []
    err_data = []
    dq_data  = [] 
    con_data = []
    wht_data = []

    for i, path in enumerate(fits_):
        fits_file = fits.open(path)
        
        if product == 'psf':
            pri = fits_file[0].header
            sci = fits_file[1].data
            dq  = fits_file[2].data 
            err = fits_file[3].data
            
            dq_data.append(dq)
            
        elif product == 'i2d':
            
            pri = fits_file[0].header
            sci = fits_file[1].data
            err = fits_file[2].data 
            con = fits_file[3].data
            wht = fits_file[4].data
                
            con_data.append(con)
            wht_data.append(wht)
            
        elif product == 'psfsub':
            
            pri = fits_file[0].header
            sci = fits_file[1].data
            err = fits_file[2].data 
            dq  = fits_file[3].data
            
            dq_data.append(dq)
            
        pri_data.append(pri)
        sci_data.append(sci)
        err_data.append(err)

        
    return pri_data, sci_data, err_data, dq_data, con_data, wht_data


def get_sci(fits_):
    
    header = []
    sci_data = []
    
    for i,path in enumerate(fits_):
        fits_file = fits.open(path)
        hd = fits_file[0].header
        sci = fits_file[1].data
        
        header.append(hd)
        sci_data.append(sci)
        
    return header,sci_data
    
    
def pixel2wcs(fits_,ispsf=False): 
    file = fits.open(fits_)
    sci = file[1].data
    axs_length = np.max(file[1].data.shape)
    axis_point = np.arange(3)
    axis_points = np.round(axis_point * axs_length/2)
    
    if ispsf:
        w = WCS(file[1].header,naxis=2)
    else:
        w = WCS(file[1].header)
        
    sky = w.pixel_to_world((axis_points), (axis_points))
    world_coords = w.pixel_to_world_values((axis_points), (axis_points))
    
    return world_coords, axis_points


def RA2time(degree):
    
    if isinstance(degree, np.ndarray):
        time_list = []
        
        for i in degree:
            
            hour_frac, hour = math.modf(i/15)
            min_frac, minute = math.modf(hour_frac*60)
            sec_frac, seconds = math.modf(min_frac*60)
            seconds = round(seconds+sec_frac,1)
            
            d2t = (int(hour),int(minute),seconds)
            time_list.append(d2t)
            
        return time_list
            
        
    elif isinstance(degree, np.float64):
        
        hour_frac, hour = math.modf(i/15)
        min_frac, minute = math.modf(hour_frac*60)
        sec_frac, seconds = math.modf(min_frac*60)
        
        
        return (int(hour),int(minute),seconds)

    
    
def create_axis_label(times,fixed):
    
    fixed_time = times[fixed]
    labels = []
    
    lbl = ''
    for i in range(len(times)):
        
        if i == fixed:
            for j in times[i]:
                lbl += str(j)+':'
            
            lbl = lbl[:-1]
            labels.append(lbl)
            
        else:
            labels.append(times[i][2])
            
    
    return labels
    
def create_declination_labels(labelish,fixed=1):
    
    labels = []
    
    for i in range(len(labelish)):
        
        if i == fixed:
            
            labels.append(labelish[i])
        else:
            labels.append(labelish[i][4:])
            
            
    return labels


def get_headers(header,header_name):
    
    headers = []
    for i in range(len(header)):
        
        headers.append(header[i][f'{header_name.upper()}'])
    
    return headers


def check_hdu_dims(file,hdu:int):
    
    dims = []
    
    for i in range(len(file)):
    
        data = fits.open(file[i])
        #print(data[hdu].data.shape)
        dims.append(data[hdu].data.shape)
        
        
    return dims

def plot_psfaligns(psfaligns,title,w,filtrs,detectors,axis_points):
    
    # times = RA2time(w[0])
    # y_labels = create_axis_label(times,fixed=2)
    # x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    # x_labels = create_declination_labels(x_labelish)

    y_labels, x_labels = get_axis_labels(w)
    
    for data in range(len(psfaligns)):
        
        nints = psfaligns[data].shape[0]
        npsfs = psfaligns[data].shape[1]
        nrow  = math.gcd(nints,npsfs)
        ncol  = npsfs//nints
                      
        for ints in range(nints):
                      
            if nints == 2:
                _, axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(36,8))
        
            else:
                _, axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(36,16))
                      
            for psfs, (row,col) in enumerate(itertools.product(range(nrow),range(ncol))):
                
                axes[row][col].imshow(psfaligns[data][ints][psfs],clim=(0,50),cmap='gray')
                if (row == 1) & (col == 0) & (nrow == 2):
                    axes[row][col].set_yticks(axis_points,y_labels,rotation=45)
                    axes[row][col].set_xticks(axis_points,x_labels,rotation=70)
                    axes[row][col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                    axes[row][col].set_ylabel('RA',fontsize=15,fontweight='bold')
                    
                elif (row == 3) & (col == 0) & (nrow == 4):
                    
                    axes[row][col].set_yticks(axis_points,y_labels,rotation=45)
                    axes[row][col].set_xticks(axis_points,x_labels,rotation=70)
                    axes[row][col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                    axes[row][col].set_ylabel('RA',fontsize=15,fontweight='bold')
                    
                    
                else:
                    axes[row][col].set_yticks([])
                    axes[row][col].set_xticks([])
                      
            if ints == 0:
                plt.text(0.85, 1, filtrs[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
                plt.text(0.85, 0.95, detectors[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
                
            plt.text(0.10, 1, ints+1, fontsize=15,fontweight='bold',transform=plt.gcf().transFigure)
            _.patch.set_facecolor('#423f3b')
            plt.subplots_adjust(wspace=0,hspace=0)
            plt.suptitle(title,y=1,x=0.5,fontsize=25,fontweight='bold')
            plt.show()   
    
    
def plot_psfstack(psfstack,ncol,nrow,title,w,axis_points,filtrs,instrume,program,targprop):
    
    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)
    
    for data in range(len(psfstack)):
        
        if instrume[data] == 'NIRCAM':
            _, axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(36,8))
        
        elif instrume[data] == 'MIRI':
            _, axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(28,8))

        for psf,(row,col) in enumerate(itertools.product(range(nrow),range(ncol))):
            
            axes[row][col].imshow(psfstack[data][psf],cmap='gray')
 
            
            
            if (row == 1) & (col == 0):

                axes[row][col].set_yticks(axis_points,y_labels)
                axes[row][col].set_xticks(axis_points,x_labels)
                axes[row][col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                axes[row][col].set_ylabel('RA',fontsize=15,fontweight='bold')
            
            else:
                axes[row][col].set_yticks([])
                axes[row][col].set_xticks([])
                
        plt.text(0.9, 1, instrume[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        plt.text(0.9, 0.95, filtrs[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        plt.text(0.9, 0.05, targprop[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        plt.text(0.9, 0, program[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        _.patch.set_facecolor('#423f3b')
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.suptitle(title,y=1,x=0.5,fontsize=25,fontweight='bold')
        plt.show()

        
        
def plot_i2d(data,ncols,title,w,axis_points,filtrs,instrume,program,targprop,nrows=1,save=False):

    # times = RA2time(w[0])
    # y_labels = create_axis_label(times,fixed=2)
    # x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    # x_labels = create_declination_labels(x_labelish)
    
    y_labels, x_labels = get_axis_labels(w)

    _,axes = plt.subplots(nrows=1, ncols=ncols, figsize=(40,10))
    
    for index,(row,col) in enumerate(itertools.product(range(nrows),range(ncols))):
            
            axes[col].imshow(np.arcsinh(data[index]),cmap='gray',origin='lower')
            axes[col].text(310, 7, filtrs[index],fontweight='bold')    
            if index != 0:
                axes[col].set_yticks([])
                axes[col].set_xticks([])
            
            else:
                axes[col].set_yticks(axis_points,y_labels,rotation=45)
                axes[col].set_xticks(axis_points,x_labels,rotation=70)
                axes[col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                axes[col].set_ylabel('RA',fontsize=15,fontweight='bold')
        
        
        
    plt.text(0.9, 1, instrume[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0.95, filtrs[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0.05, targprop[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0, program[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    _.patch.set_facecolor('#423f3b')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.suptitle(title,y=1,x=0.5,fontsize=20,fontweight='bold')
    plt.show()

    if save:
        plt.savefig(f'{title}.png')
    
    
    
def plot_psfaligns(psfaligns,title,w,filtrs,instrume,program,targprop,axis_points):
    
    # times = RA2time(w[0])
    # y_labels = create_axis_label(times,fixed=2)
    # x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    # x_labels = create_declination_labels(x_labelish)
    
    y_labels, x_labels = get_axis_labels(w)
    
    for data in range(len(psfaligns)):
        
        nints = psfaligns[data].shape[0]
        npsfs = psfaligns[data].shape[1]
        nrow  = math.gcd(nints,npsfs)
        ncol  = npsfs//nints
                      
        for ints in range(nints):
                      
            if nints == 2:
                _, axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(36,8))
        
            else:
                _, axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(36,16))
                      
            for psfs, (row,col) in enumerate(itertools.product(range(nrow),range(ncol))):
                
                axes[row][col].imshow(psfaligns[data][ints][psfs],clim=(0,50),cmap='gray')
                if (row == 1) & (col == 0) & (nrow == 2):
                    axes[row][col].set_yticks(axis_points,y_labels,rotation=45)
                    axes[row][col].set_xticks(axis_points,x_labels,rotation=70)
                    axes[row][col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                    axes[row][col].set_ylabel('RA',fontsize=15,fontweight='bold')
                    
                elif (row == 3) & (col == 0) & (nrow == 4):
                    
                    axes[row][col].set_yticks(axis_points,y_labels,rotation=45)
                    axes[row][col].set_xticks(axis_points,x_labels,rotation=70)
                    axes[row][col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                    axes[row][col].set_ylabel('RA',fontsize=15,fontweight='bold')
                    
                else:
                    axes[row][col].set_yticks([])
                    axes[row][col].set_xticks([])
                      
            if ints == 0:
                plt.text(0.9, 0.95, filtrs[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
                plt.text(0.9, 1, instrume[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
                plt.text(0.9, 0.05, targprop[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
                plt.text(0.9, 0, program[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
                
                
            
            plt.text(0.10, 1, ints+1, fontsize=15,fontweight='bold',transform=plt.gcf().transFigure)
            _.patch.set_facecolor('#423f3b')
            plt.subplots_adjust(wspace=0,hspace=0)
            plt.suptitle(title,y=1,x=0.5,fontsize=25,fontweight='bold')
            plt.show()    
        
        
def plot_psfsubs(psfsubs,title,w,filtrs,instrume,program,targprop,axis_points):
    
    # times = RA2time(w[0])
    # y_labels = create_axis_label(times,fixed=2)
    # x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    # x_labels = create_declination_labels(x_labelish)

    y_labels, x_labels = get_axis_labels(w)
    
    for data in range(len(psfsubs)):
        
        ncol = psfsubs[data].shape[0]
        
        if ncol == 2:
            _, axes = plt.subplots(nrows=1,ncols=ncol,figsize=(16,8))
        
        else:
            _, axes = plt.subplots(nrows=1,ncols=ncol,figsize=(24,8))
        
        for psub, (row,col) in enumerate(itertools.product(range(1),range(ncol))):
            
            axes[col].imshow(np.arcsinh(psfsubs[data][psub]),clim=(0,1),cmap='gray',origin='lower')
            
            if psub != 0:
                axes[col].set_yticks([])
                axes[col].set_xticks([])
            
            else:
                axes[col].set_yticks(axis_points,y_labels,rotation=45)
                axes[col].set_xticks(axis_points,x_labels,rotation=70)
                axes[col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                axes[col].set_ylabel('RA',fontsize=15,fontweight='bold')
        
        plt.text(0.9, 1, instrume[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        plt.text(0.9, 0.95, filtrs[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        plt.text(0.9, 0.05, targprop[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        plt.text(0.9, 0, program[data], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
        _.patch.set_facecolor('#423f3b')
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.suptitle(title,y=1,x=0.5,fontsize=25,fontweight='bold')
        plt.show()
        

        
def plot_i2d_mir(data,ncols,title,w,axis_points,filtrs,instrume,program,targprop,nrows=1,save=False):

    # times = RA2time(w[0])
    # y_labels = create_axis_label(times,fixed=2)
    # x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    # x_labels = create_declination_labels(x_labelish)

    y_labels, x_labels = get_axis_labels(w)
    
    _,axes = plt.subplots(nrows=1, ncols=ncols, figsize=(35,7))
    
    for index,(row,col) in enumerate(itertools.product(range(nrows),range(ncols))):
            
            axes[col].imshow(np.arcsinh(data[index]),cmap='gray',origin='lower')
            axes[col].text(280, 15, filtrs[index],fontweight='bold')  
            
            if index != 0:
                axes[col].set_yticks([])
                axes[col].set_xticks([])
            
            else:
                axes[col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                axes[col].set_ylabel('RA',fontsize=15,fontweight='bold')
                

    plt.text(0.9, 1, instrume[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0.95, filtrs[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0.05, targprop[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0, program[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    _.patch.set_facecolor('#423f3b')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()
    
    if save:
        plt.savefig(f'{title}_mir.png')
    
def get_wcs(fits):
    return WCS(fits[1].header, naxis=2)

def get_ra_dec(fits):
    ra  = fits[0].header['TARG_RA']
    dec = fits[0].header['TARG_DEC'] 
    return ra, dec

def get_skycoord(ra, dec):
    ra = Longitude(ra, unit=u.deg)
    dec = dec * u.deg
    sky_coord = SkyCoord(ra, dec, frame='icrs')
    sky_coord = SkyCoord(frame=ICRS, ra=ra, dec=dec)
    return sky_coord

def skycoord_to_pixel(wcs,skycoord):
    x, y = wcs.world_to_pixel(skycoord)
    return  y, x 


def rotate_point(x, y, angle_degrees, center=(40, 40)):
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)

    # Calculate the distance from the center to the point
    dx = x - center[0]
    dy = y - center[1]

    # Rotate the point
    new_x = center[0] + dx * math.cos(angle_radians) + dy * math.sin(angle_radians)
    new_y = center[1] - dx * math.sin(angle_radians) + dy * math.cos(angle_radians)

    return new_x, new_y


def flip_point(x, y, flipud:bool, fliplr:bool):

    h, w = 80, 80


    if flipud and fliplr:
        new_x = h - 1 - x 
        new_y = w - 1 - y  

    elif (flipud == True) and (fliplr == False):
        new_x = w - 1 - x  
        new_y = y #w - 1 - y  

    elif (flipud == False) and (fliplr == True):
        new_x = x  
        new_y = h - 1 - y 

    else:
        new_x, new_y = x, y


    return new_x, new_y


def find_new_coordinates_after_shift(original_x, original_y, right_shift, down_shift):
    h, w = 80, 80
    new_x = (original_x + right_shift) % h
    new_y = (original_y + down_shift) % w
    return new_x, new_y

def calculate_distance(x1, y1, x2, y2):
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_augmentation_info(info):

    infos = []
    for inf in info:  

        numeric_info = []
        for lst in inf.split('/')[-1].split('-')[6:-3]:

            numeric_info.append(re.findall(r'\d+', lst)[0])

        infos.append(numeric_info)

    return infos


def get_psf_info(injection_dirs):

    #root_dir = "/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/mastDownload/JWST/"
    root_dir = "/data/scratch/sarperyurtseven/dataset/NIRCAM/1386/mastDownload/JWST/"
    sew = set()
    star_location_info = []
    for i in range(len(injection_dirs)):
        
        psf_name = '-'.join(injection_dirs[i].split('/')[-1].split('-')[:4]) + '_psfstack.fits'

        complete_dirs = os.path.join(root_dir, psf_name)
        psf_ = fits.open(complete_dirs)
        wcs = get_wcs(psf_)
        ra, dec = get_ra_dec(psf_)
        sew.add((ra,dec))
        sky_coord = get_skycoord(ra, dec)
        x, y = skycoord_to_pixel(wcs, sky_coord)
        star_location_info.append((x,y))
        
    return star_location_info

def do_transformations(infos, locations):

    transformed_list = []
    for idx, info in enumerate(infos):

        if len(info) == 6:

            y = 54-4#int(locations[idx][0]) 
            x = 36-5#int(locations[idx][1])

            rotate     = int(info[0])
            flip       = int(info[1])
            vertical   = int(info[2])
            horizontal = int(info[3])
            vshift     = int(info[4])
            hshift     = int(info[5])

            x, y = rotate_point(x, y, rotate*90)

            x, y = flip_point(x, y, flipud=True if flip == 1 or flip == 3 else False, fliplr=True if flip == 2 or flip == 3 else False)
            x, y = find_new_coordinates_after_shift(x, y, right_shift=hshift if horizontal == 2 else -hshift, down_shift=vshift if vertical == 2 else -vshift)

            transformed_list.append((int(x), int(y)))

        else:
            y = 54#int(locations[idx][0]) 
            x = 36#int(locations[idx][1])
            transformed_list.append((y, x))

    return transformed_list

def get_array(nps):

    arrays = []

    for arr in nps[:25]:

        img = np.load(arr)
        arrays.append(img)

    arrays = np.concatenate(np.expand_dims(arrays, axis=0))

    return arrays        


def apply_low_pass(array):

    if len(array.shape) == 2:
    
        r = 50
        ham = np.hamming(80)[:,None] 
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r 
        f = cv2.dft(array.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
        f_filtered = ham2d * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) 
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img*255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        output = filtered_img

    elif len(array.shape) == 3:

        output = []
        r = 50 
        ham = np.hamming(80)[:,None] 
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r 

        for i in range(array.shape[0]):

            f = cv2.dft(array[i].astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
            f_filtered = ham2d * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) 
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img*255 / filtered_img.max()
            filtered_img = filtered_img.astype(np.uint8)
            output.append(filtered_img)
        
        output = np.concatenate(np.expand_dims(output, axis=0))


    return output


def get_wcs(fits):
    return WCS(fits[1].header, naxis=2)

def get_ra_dec(fits):
    ra  = fits[0].header['TARG_RA']
    dec = fits[0].header['TARG_DEC'] 
    return ra, dec

def get_skycoord(ra, dec):
    ra = Longitude(ra, unit=u.deg)
    dec = dec * u.deg
    sky_coord = SkyCoord(ra, dec, frame='icrs')
    sky_coord = SkyCoord(frame=ICRS, ra=ra, dec=dec)
    return sky_coord

def skycoord_to_pixel(wcs,skycoord):
    x, y = wcs.world_to_pixel(skycoord)
    return  y, x 


def rotate_point(x, y, angle_degrees, center=(40, 40)):
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)

    # Calculate the distance from the center to the point
    dx = x - center[0]
    dy = y - center[1]

    # Rotate the point
    new_x = center[0] + dx * math.cos(angle_radians) + dy * math.sin(angle_radians)
    new_y = center[1] - dx * math.sin(angle_radians) + dy * math.cos(angle_radians)

    return new_x, new_y


def flip_point(x, y, flipud:bool, fliplr:bool):

    h, w = 80, 80


    if flipud and fliplr:
        new_x = h - 1 - x 
        new_y = w - 1 - y  

    elif (flipud == True) and (fliplr == False):
        new_x = w - 1 - x  
        new_y = y #w - 1 - y  

    elif (flipud == False) and (fliplr == True):
        new_x = x  
        new_y = h - 1 - y 

    else:
        new_x, new_y = x, y


    return new_x, new_y


def find_new_coordinates_after_shift(original_x, original_y, right_shift, down_shift):
    h, w = 80, 80
    new_x = (original_x + right_shift) % h
    new_y = (original_y + down_shift) % w
    return new_x, new_y

def calculate_distance(x1, y1, x2, y2):
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_augmentation_info(info):

    infos = []
    for inf in info:  

        numeric_info = []
        for lst in inf.split('/')[-1].split('-')[6:-3]:

            numeric_info.append(re.findall(r'\d+', lst)[0])

        infos.append(numeric_info)

    return infos


def get_psf_info(injection_dirs):

    #root_dir = "/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/mastDownload/JWST/"
    root_dir = "/data/scratch/sarperyurtseven/dataset/NIRCAM/1386/mastDownload/JWST/"
    sew = set()
    star_location_info = []
    for i in range(len(injection_dirs)):
        
        psf_name = '-'.join(injection_dirs[i].split('/')[-1].split('-')[:4]) + '_psfstack.fits'

        complete_dirs = os.path.join(root_dir, psf_name)
        psf_ = fits.open(complete_dirs)
        wcs = get_wcs(psf_)
        ra, dec = get_ra_dec(psf_)
        sew.add((ra,dec))
        sky_coord = get_skycoord(ra, dec)
        x, y = skycoord_to_pixel(wcs, sky_coord)
        star_location_info.append((x,y))
        
    return star_location_info


def do_transformations(infos, locations):

    transformed_list = []
    for idx, info in enumerate(infos):

        if len(info) == 6:

            y = 54-4#int(locations[idx][0]) 
            x = 36-5#int(locations[idx][1])

            rotate     = int(info[0])
            flip       = int(info[1])
            vertical   = int(info[2])
            horizontal = int(info[3])
            vshift     = int(info[4])
            hshift     = int(info[5])

            x, y = rotate_point(x, y, rotate*90)

            x, y = flip_point(x, y, flipud=True if flip == 1 or flip == 3 else False, fliplr=True if flip == 2 or flip == 3 else False)
            x, y = find_new_coordinates_after_shift(x, y, right_shift=hshift if horizontal == 2 else -hshift, down_shift=vshift if vertical == 2 else -vshift)

            transformed_list.append((int(x), int(y)))

        else:
            y = 54#int(locations[idx][0]) 
            x = 36#int(locations[idx][1])
            transformed_list.append((y, x))

    return transformed_list

def get_array(nps):

    arrays = []

    for arr in nps[:25]:

        img = np.load(arr)
        arrays.append(img)

    arrays = np.concatenate(np.expand_dims(arrays, axis=0))

    return arrays        


def apply_low_pass(array):

    if len(array.shape) == 2:
    
        r = 50
        ham = np.hamming(80)[:,None] 
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r 
        f = cv2.dft(array.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
        f_filtered = ham2d * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) 
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img*255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        output = filtered_img

    elif len(array.shape) == 3:

        output = []
        r = 50 
        ham = np.hamming(80)[:,None] 
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r 

        for i in range(array.shape[0]):

            f = cv2.dft(array[i].astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
            f_filtered = ham2d * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) 
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img*255 / filtered_img.max()
            filtered_img = filtered_img.astype(np.uint8)
            output.append(filtered_img)
        
        output = np.concatenate(np.expand_dims(output, axis=0))


    return output