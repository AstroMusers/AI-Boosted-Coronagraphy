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




def get_lower_products(suffix,directory):
    return glob(os.path.join(directory, f'**/*{suffix}.fits'))


def get_stage3_products(suffix,directory):
    return glob(os.path.join(directory, f'*{suffix}.fits'))


def plot_psfaligns(psfaligns,title,w,filtrs,detectors,axis_points):
    
    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)
    
    
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
    
    
def plot_rateints_calints(data,ncols,nrows):
    
    nints = data[0].shape[0]
    
    _,axes = plt.subplots(nrows=nrows*nints, ncols=ncols*nints, figsize=(25,10))
    for nint in range(nints):
        for index,(row,col) in enumerate(itertools.product(range(nrows),range(ncols))):
            axes[row][col].imshow(np.sin(data[nint][index]),clim=(-1,1),cmap='gray') 
            
            if index != 0:
                axes[row][col].set_yticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.suptitle(title,y=0.75,x=0.5,fontsize=20)
    plt.show()

    
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
    
    if isinstance(degree,np.ndarray):
        time_list = []
        
        for i in degree:
            
            hour_frac, hour = math.modf(i/15)
            min_frac, minute = math.modf(hour_frac*60)
            sec_frac, seconds = math.modf(min_frac*60)
            seconds = round(seconds+sec_frac,1)
            
            d2t = (int(hour),int(minute),seconds)
            time_list.append(d2t)
            
        return time_list
            
        
    elif isinstance(degree,np.float64):
        
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

def pixel_to_arcsec_nircam(axis_length,wavelength):
    
    short_wavelengths = ['F187N','F212N','F182M','F210M','F200W']
    
    if wavelength in short_wavelengths:
        x = 1 / 0.031 
    
    else:
        x = 1 / 0.063
    
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
    
    
    
    return arcsec_axis_points



def plot_calints(calints,sci_calints,header_calints,title,instrume):
    
        
    for data in range(len(calints)):
        
        world_coords, axis_points = pixel2wcs(calints[data],ispsf=True)
        filtrs = get_filters(header_calints)
        
        times = RA2time(world_coords[0])
        y_labels = create_axis_label(times,fixed=2)
        x_labelish = [str(round(x_label,3)) for x_label in world_coords[1]]
        x_labels = create_declination_labels(x_labelish)
        
        ncol = sci_calints[data].shape[0]
        
        if instrume == 'NIRCAM':
            _, axes = plt.subplots(nrows=1,ncols=ncol,figsize=(36,16))
        
        elif instrume == 'MIRI':
            _, axes = plt.subplots(nrows=1,ncols=ncol,figsize=(20,15))
            
        
        
        for ints, (row,col) in enumerate(itertools.product(range(1),range(ncol))):
            
            if ncol == 1:
                axes.imshow(sci_calints[data][ints],clim=(0,50),cmap='gray')
                
                if (row == 0) & (col == 0):
                    axes.set_yticks(axis_points,y_labels,rotation=45)
                    axes.set_xticks(axis_points,x_labels,rotation=70)
                    axes.set_xlabel('DEC',fontsize=15,fontweight='bold')
                    axes.set_ylabel('RA',fontsize=15,fontweight='bold')
                else:
                    axes.set_yticks([])
                    axes.set_xticks([])
                
                
            if ncol > 1:
                axes[col].imshow(sci_calints[data][ints],clim=(0,75),cmap='gray')
        
                if (row == 0) & (col == 0):
                    axes[col].set_yticks(axis_points,y_labels,rotation=45)
                    axes[col].set_xticks(axis_points,x_labels,rotation=70)
                    axes[col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                    axes[col].set_ylabel('RA',fontsize=15,fontweight='bold')
                else:
                    axes[col].set_yticks([])
                    axes[col].set_xticks([])

        #plt.text(0.9, 1, filtrs[data], fontsize=25,fontweight='bold',transform=plt.gcf().transFigure)
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
        
        y_axis_arcsec = [ y for y in np.sort(pixel_to_arcsec_nircam(320,wavelength=filtrs[data]))]
        
        negative = [-x for x in range((len(y_axis_arcsec)//2)+1)]
        positive = [ x for x in range((len(y_axis_arcsec)//2)+1)]
        arcsec_labels = negative + positive
        arcsec_labels = arcsec_labels[1:]
        arcsec_labels.sort()
        
        #print(y_axis_arcsec)
        #print(type(y_axis_arcsec))

        for psf,(row,col) in enumerate(itertools.product(range(nrow),range(ncol))):
            
            axes[row][col].imshow(psfstack[data][psf],clim=(0,50),cmap='gray')
 
            
            
            if (row == 1) & (col == 0):

                axes[row][col].set_yticks(axis_points,y_labels)
                axes[row][col].set_xticks(axis_points,x_labels)
                axes[row][col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                axes[row][col].set_ylabel('RA',fontsize=15,fontweight='bold')
            
                #axes[row][col].yaxis.tick_right()
                #axes[row][col].set_yticks(y_axis_arcsec,arcsec_labels,rotation=15)
                #axes[row][col].yaxis.set_label_position("right")
                #axes[row][col].set_ylabel('PIX2ARCSEC',fontsize=15,fontweight='bold')
                #axes[row][col].set_xticks([])
            
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

    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)
    
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
    
    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)
    
    
    for data in range(len(psfaligns)//2):
        
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
    
    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)
    
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

    times = RA2time(w[0])
    y_labels = create_axis_label(times,fixed=2)
    x_labelish = [str(round(x_label,3)) for x_label in w[1]]
    x_labels = create_declination_labels(x_labelish)
    
    _,axes = plt.subplots(nrows=1, ncols=ncols, figsize=(35,7))
    
    for index,(row,col) in enumerate(itertools.product(range(nrows),range(ncols))):
            
            axes[col].imshow(np.arcsinh(data[index]),cmap='gray',origin='lower')
            axes[col].text(280, 15, filtrs[index],fontweight='bold')  
            
            if index != 0:
                axes[col].set_yticks([])
                axes[col].set_xticks([])
            
            else:
                #axes[col].set_yticks(axis_points,y_labels,rotation=45)
                #axes[col].set_xticks(axis_points,x_labels,rotation=70)
                axes[col].set_xlabel('DEC',fontsize=15,fontweight='bold')
                axes[col].set_ylabel('RA',fontsize=15,fontweight='bold')
                

    plt.text(0.9, 1, instrume[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0.95, filtrs[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0.05, targprop[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    plt.text(0.9, 0, program[index], fontsize=20,fontweight='bold',transform=plt.gcf().transFigure)
    _.patch.set_facecolor('#423f3b')
    plt.subplots_adjust(wspace=0,hspace=0)
    #plt.suptitle(title,y=0.88,x=0.5,fontsize=20,fontweight='bold')
    plt.show()
    
    if save:
        plt.savefig(f'{title}_mir.png')
    
