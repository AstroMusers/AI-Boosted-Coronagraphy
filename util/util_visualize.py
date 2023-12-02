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


class VisualizeFITS():

    def __init__(self,):


        pass

    def get_lower_products(self, suffix, directory):
        return glob(os.path.join(directory, f'*/*{suffix}.fits'))
    
    def get_stage3_products(self, suffix, directory):
        return glob(os.path.join(directory, f'*{suffix}.fits'))

    
    def get_axis_labels(self, w):

        times = self.RA2time(w[0])
        y_labels = self.create_axis_label(times,fixed=2)
        x_labelish = [str(round(x_label,3)) for x_label in w[1]]
        x_labels = self.create_declination_labels(x_labelish)

        return y_labels,x_labels

    def get_hdu(self, fits_, product):
    
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

    def get_sci(self, fits_):
    
        header = []
        sci_data = []
        
        for i,path in enumerate(fits_):
            fits_file = fits.open(path)
            hd = fits_file[0].header
            sci = fits_file[1].data
            
            header.append(hd)
            sci_data.append(sci)
            
        return header, sci_data

    def pixel2wcs(self, fits_,ispsf=True): 

        file = fits.open(fits_)
        axs_length = np.max(file[1].data.shape)
        axis_point = np.arange(3)
        axis_points = np.round(axis_point * axs_length/2)
        
        if ispsf:
            w = WCS(file[1].header,naxis=2)
        else:
            w = WCS(file[1].header)
            
        world_coords = w.pixel_to_world_values((axis_points), (axis_points))
        
        return world_coords, axis_points

    def RA2time(self, degree):
    
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

    def create_axis_label(self, times,fixed):
    
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

    def create_declination_labels(self, labelish, fixed=1):
    
        labels = []
        
        for i in range(len(labelish)):
            
            if i == fixed:
                
                labels.append(labelish[i])
            else:
                labels.append(labelish[i][4:])
                
                
        return labels

    def get_headers(self, header,header_name):
        
        headers = []
        for i in range(len(header)):
            
            headers.append(header[i][f'{header_name.upper()}'])
        
        return headers


    def check_hdu_dims(self, file:list, hdu:int):
        
        dims = []
        
        for i in range(len(file)):
        
            data = fits.open(file[i])
            dims.append(data[hdu].data.shape)

        return dims
        
    def plot_psfstack(self, psfstack, ncol, nrow, title, w, axis_points, filtrs, instrume, program, targprop):
        
        times = self.A2time(w[0])
        y_labels = self.create_axis_label(times,fixed=2)
        x_labelish = [str(round(x_label,3)) for x_label in w[1]]
        x_labels = self.create_declination_labels(x_labelish)
        

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
