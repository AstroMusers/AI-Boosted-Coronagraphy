import os

if not os.environ.get('WEBBPSF_PATH'):
    os.environ['WEBBPSF_PATH'] = '/data/webbpsf-data'

import sys
sys.path.append("..")

from glob import glob

import warnings
import math

import webbpsf

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.utils.exceptions import AstropyWarning

from util.util_main import get_filename_from_dir, get_dataset_dir
from scripts.visualization_helpers import get_stage3_products

import numpy as np

from time import time

from augmentation import Augmentation

from tqdm import tqdm

warnings.simplefilter('ignore', category=AstropyWarning)

class Injection():
    def __init__(self, psf_directory: str, is_save_original:bool, is_save_augmented:bool, is_save_injected:bool) -> None:
        print('Injection process has been started...')

        self.psf_directory = psf_directory

        self.is_save_original = is_save_original
        self.is_save_augmented = is_save_augmented
        self.is_save_injected = is_save_injected

        self.psfstacks_nircam_dirs = get_stage3_products(
            suffix='psfstack', directory=psf_directory)

        self.psfstacks = {}
        self.__create_psfstacks_dict()

        self.augmentation = Augmentation()

    def apply_injection(self, injection_count:int=2, aug_count:int=20, normalize_psf:bool=False, inject_filename:str='injections_train', flux_coefficients = [100, 1000, 10000]):

        for filter_key in self.psfstacks.keys():
            max_pixel_distance, min_pixel_distance = self.__get_max_min_pixel_distance(
                self.psfstacks[filter_key])
            
            print(f'{filter_key} max pixel distance: {max_pixel_distance}, min pixel distance: {min_pixel_distance}')
            fov = np.round(np.sqrt(self.psfstacks[filter_key][1].header['PIXAR_A2']), 3) * self.psfstacks[filter_key][1].data.shape[1]
            print(fov)

            pix_per_arcsec = np.sqrt(self.psfstacks[filter_key][1].header['PIXAR_A2'])
            max_pix = int(1.5 / pix_per_arcsec)
            size    = 80
            psf_res = (320 - size) // 2

            detector = self.psfstacks[filter_key][0].header['DETECTOR']
            filter = self.psfstacks[filter_key][0].header['FILTER']

            generated_psf = self.__generate_psf_model(detector=detector, filter=filter,  fov=fov*2, save=True)

            if self.psfstacks[filter_key][0].header['CHANNEL'] == 'LONG':
                generated_psf_selection = 1
            else:
                generated_psf_selection = 0
            
            for psf_idx, psf in enumerate(tqdm(self.psfstacks[filter_key][1].data)):
                psf = self.__nan_elimination(psf)

                if normalize_psf:
                    psf = self.augmentation.normalize(psf)
                
                psf = psf[psf_res:psf_res + size, psf_res:psf_res + size]
                norm_psf  = psf

                filename = f'{"/".join(self.psf_directory.split("/")[:-3])}/injections/{inject_filename}/{filter_key}-psf{psf_idx}'
                os.makedirs(os.path.join("/".join(self.psf_directory.split("/")[:-3]), f'injections/{inject_filename}'), exist_ok=True)

                if self.is_save_original:
                    self.__save_psf_to_npy(
                        filename=f'{filename}.npy',
                        psf=norm_psf)

                if not self.__is_psf_empty(psf):
                    self.__injection(
                        psf=norm_psf, 
                        generated_psf=generated_psf[generated_psf_selection].data, 
                        max_pixel_distance=max_pixel_distance, 
                        min_pixel_distance=min_pixel_distance, 
                        max_pix = max_pix,
                        injection_count=injection_count, 
                        flux_coefficients=flux_coefficients,
                        filename=filename
                        )

                    for _ in range(aug_count):
                        # We should consider the shift effect on injection.
                        aug_psf, aug_filename, aug_comp_psf = self.__augmentation(norm_psf, generated_psf[generated_psf_selection].data, filename) 
                        self.__injection(
                            psf=aug_psf,
                            generated_psf=aug_comp_psf, 
                            max_pixel_distance=max_pixel_distance, 
                            min_pixel_distance=min_pixel_distance, 
                            max_pix = max_pix,
                            injection_count=injection_count, 
                            flux_coefficients=flux_coefficients,
                            filename=aug_filename
                        )

                    for _ in range(aug_count):
                        aug_psf, aug_filename, aug_comp_psf = self.__augmentation(norm_psf, generated_psf[generated_psf_selection].data, filename) 

                else:
                    pass
            
    def __injection(self, 
                    psf, 
                    generated_psf, 
                    max_pixel_distance:int, 
                    min_pixel_distance:int, 
                    max_pix:int,
                    injection_count:int=10, 
                    flux_coefficients:list=[100, 1000, 10000], 
                    filename:str=''
        ):
        for _ in range(injection_count):
            if max_pixel_distance > psf.shape[0]:
                max_x = psf.shape[0]
            else:
                max_x = max_pixel_distance
                
            if max_pixel_distance > psf.shape[1]:
                max_y = psf.shape[1]
            else:
                max_y = max_pixel_distance

            max_pix = int(max_pix)
            # print(max_pix)
            # print(min_pixel_distance)
            # print(psf.shape[0]/2)
            # print(min_pixel_distance)

            # random_x = np.random.randint(min_pixel_distance, np.floor(max_x - (psf.shape[0]/2)))
            # random_y = np.random.randint(min_pixel_distance, np.floor(max_y - (psf.shape[1]/2)))

            # print(random_x)
            # print(random_y)

            random_x = np.random.randint(min_pixel_distance, max_pix//2)
            random_y = np.random.randint(min_pixel_distance, max_pix//2)

            # print(random_x)
            # print(random_y)
            # input()

            rand_sign_x = 1 if np.random.rand() < 0.5 else -1
            rand_sign_y = 1 if np.random.rand() < 0.5 else -1

            x = int((psf.shape[0]/2) + rand_sign_x*random_x)
            y = int((psf.shape[1]/2) + rand_sign_y*random_y)

            
            for flux_coefficient in flux_coefficients:
                temp_psf = np.copy(generated_psf * (np.max(psf) / ( flux_coefficient * np.max(generated_psf))))

                injected = np.copy(temp_psf[
                    int(temp_psf.shape[0]//2 - x): int(temp_psf.shape[0]//2 - x + psf.shape[0]),
                    int(temp_psf.shape[1]//2 - y): int(temp_psf.shape[1]//2 - y + psf.shape[1])
                ] + psf)

                if self.is_save_injected:
                    self.__save_psf_to_npy(
                        filename=f'{filename}-x{y}-y{x}-fc{flux_coefficient}.npy', 
                        psf=injected
                    )
        
        del temp_psf, injected, max_x, max_y, random_x, random_y, x, y, filename
    
    def __save_psf_to_npy(self, filename, psf):
        return np.save(filename, psf)

    def __augmentation(self, psf, computed_psf, filename:str=''):
        rotate_rand = np.random.randint(0, 4)
        # If it equals to 0 then it will not rotate the image.
        # If rotate rand equals to 1 then it will just rotate the image one time with 90 degrees.
        flip_rand = np.random.randint(0, 3)
        # If flip rand equals to 1 then it will just flip the image horizontally. 
        # If flip rand equals to 2 then it will just flip the image vertically. 
        # If flip rand equals to 3 then it will flip the image both horizontally and vertically.
        vertical_shift_rand = np.random.randint(0, 3)
        # 1 -> up
        # 2 -> down
        horizontal_shift_rand = np.random.randint(0, 3)
        # 1 -> left
        # 2 -> right
        vertical_shift_pixel_rand = np.random.randint(1, 10)
        horizontal_shift_pixel_rand = np.random.randint(1, 10)

        augmented    = self.augmentation.rotate90(psf, times=rotate_rand)
        aug_comp_psf = self.augmentation.rotate90(computed_psf, times=rotate_rand)

        augmented    = self.augmentation.flip(augmented, horizontal=True if flip_rand == 1 or flip_rand == 3 else False, vertical=True if flip_rand == 2 or flip_rand == 3 else False)
        aug_comp_psf = self.augmentation.flip(aug_comp_psf, horizontal=True if flip_rand == 1 or flip_rand == 3 else False, vertical=True if flip_rand == 2 or flip_rand == 3 else False)

        augmented = self.augmentation.shift(augmented, right_shift=horizontal_shift_pixel_rand if horizontal_shift_rand == 2 else -horizontal_shift_pixel_rand, down_shift=vertical_shift_pixel_rand if vertical_shift_rand == 2 else -vertical_shift_pixel_rand)

        filename=f'{filename}-aug-rot{rotate_rand}-flip{flip_rand}-vshift{vertical_shift_rand}-hshift{horizontal_shift_rand}-vshiftp{vertical_shift_pixel_rand}-hshiftp{horizontal_shift_pixel_rand}'
        
        if self.is_save_augmented:
            self.__save_psf_to_npy(filename=f'{filename}.npy', psf=augmented)
    
        del rotate_rand, flip_rand, vertical_shift_rand, horizontal_shift_rand, vertical_shift_pixel_rand, horizontal_shift_pixel_rand
        return augmented, filename, aug_comp_psf

    def __nan_elimination(self, psf):
        return np.nan_to_num(psf)

    def __generate_psf_model(self, detector:str, filter:str, fov:float, coron_mask:str='', pupil_mask:str='', save:bool=True):
        
        if detector == 'NRCALONG':
            detector = 'NRCA5'
        elif detector == 'NRCBLONG':
            detector = 'NRCB5'

        psf_dir = get_dataset_dir() + f'/PSF_SAMPLES/{detector}-{filter}_{coron_mask}_{pupil_mask}-{fov}.fits'
        psf_dir_glob = glob(psf_dir)

        if psf_dir_glob != []:
            generated_psf = fits.open(psf_dir)
            print(f'{detector}-{filter} PSF with {fov} arcsec fov collected from cache.')
            del psf_dir, psf_dir_glob
        else:
            if 'NRC' in detector:
                wpsf = webbpsf.NIRCam()
            elif 'MIR' in detector:
                wpsf = webbpsf.MIRI()
            else:
                raise ValueError('Detector is not valid!')
            

            time_start = time()
            wpsf.detector   = detector
            wpsf.filter     = filter
            if coron_mask != '':
                wpsf.image_mask = coron_mask.replace('MASKA','MASK')
            if pupil_mask != '':
                wpsf.pupil_mask = pupil_mask

            if save:
                generated_psf = wpsf.calc_psf(fov_arcsec=fov, oversample=2, outfile=f'{psf_dir}')
            else:
                generated_psf = wpsf.calc_psf(fov_arcsec=fov, oversample=2)
            time_end = time()

            print(f'{detector}-{filter} PSF generated respect to {fov} arcsec fov in {np.round(time_end-time_start, 2)} seconds.')
            del psf_dir, psf_dir_glob, wpsf, time_start, time_end

        return generated_psf
        
    def __create_psfstacks_dict(self) -> None:
        for _, dir in enumerate(self.psfstacks_nircam_dirs):
            fits_name = get_filename_from_dir(dir)
            self.psfstacks[fits_name] = fits.open(dir)

            del fits_name

    def __get_ra_dec_with_uncertainty_from_header(self, psfstack):
        return psfstack[0].header['TARG_RA'], psfstack[0].header['TARG_DEC'], psfstack[0].header['TARGURA'], psfstack[0].header['TARGUDEC']

    def __get_max_min_pixel_distance(self, psfstack):
        ra, dec, u_ra, u_dec = self.__get_ra_dec_with_uncertainty_from_header(
            psfstack)
        coord, width, height = SkyCoord(ra=ra, dec=dec, unit=(
            u.deg, u.deg), frame='icrs'), u.Quantity(u_ra, u.deg), u.Quantity(u_dec, u.deg)

        query_results = Gaia.query_object_async(
            coordinate=coord, width=width, height=height)

        min_idx = np.argmin(query_results['dist'])
        target_star = query_results[min_idx]

        distance = (target_star['parallax'] *
                    u.marcsec).to(u.parsec, equivalencies=u.parallax())

        query_exoplanets = NasaExoplanetArchive.query_criteria(
            table="pscomppars", where="discoverymethod like 'Imaging'")

        exoplanet_extremes = {
            'max': np.nanmax(query_exoplanets['pl_orbsmax']),
            'min': np.nanmin(query_exoplanets['pl_orbsmax'])
        }

        one_pix_side_length_arcsec = np.sqrt(
            psfstack[1].header['PIXAR_A2']) * u.arcsec
        max_pixel_distance = exoplanet_extremes['max'] / (
            distance.to(u.au) * np.tan(one_pix_side_length_arcsec))
        min_pixel_distance = exoplanet_extremes['min'] / (
            distance.to(u.au) * np.tan(one_pix_side_length_arcsec))

        del ra, dec, u_ra, u_dec, coord, width, height, query_results, min_idx, target_star, distance, query_exoplanets, exoplanet_extremes, one_pix_side_length_arcsec
        return np.floor(max_pixel_distance.value), np.floor(min_pixel_distance.value) + 1
    
    def __is_psf_empty(self, psf):
        width, height = psf.shape[0], psf.shape[1]
        psd = self.__create_psd(psf)
        if psd[int(width/2) -1][int(height/2) -1] < 0.55:
            return True
        else:
            return False

    def __create_psd(self, psf):
        psd = np.abs(np.fft.fftshift(np.fft.fft2(psf)))**2
        psd = np.log10(psd)
        psd = psd/psd.max()
        return psd

    def __rotate_point(x, y, angle_degrees, center=(160, 160)):
        # Convert angle to radians
        angle_radians = math.radians(angle_degrees)

        # Calculate the distance from the center to the point
        dx = x - center[0]
        dy = y - center[1]

        # Rotate the point
        new_x = center[0] + dx * math.cos(angle_radians) - dy * math.sin(angle_radians)
        new_y = center[1] + dx * math.sin(angle_radians) + dy * math.cos(angle_radians)

        return new_x, new_y


    def __flip_point(img, x, y, flipud:bool, fliplr:bool):

        if flipud and fliplr:
            new_x = img.shape[1] - 1 - x 
            new_y = img.shape[0] - 1 - y  

        elif (flipud == True) and (fliplr == False):
            new_x = x  
            new_y = img.shape[0] - 1 - y  

        elif (flipud == False) and (fliplr == True):
            new_x = img.shape[1] - 1 - x  
            new_y = y 


        return new_x, new_y

    
    def __find_new_coordinates_after_shift(img, original_x, original_y, right_shift, down_shift):
        new_x = (original_x + right_shift) % img.shape[1]
        new_y = (original_y + down_shift) % img.shape[0]
        return new_x, new_y

    def __calculate_distance(x1, y1, x2, y2):
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


if __name__ == '__main__':

    np.random.seed(42)

    PROPOSAL_ID = '1386'
    INSTRUMENT = 'NIRCAM'
    psf_directory = f'/data/scratch/sarperyurtseven/dataset/{INSTRUMENT}/{PROPOSAL_ID}/mastDownload/JWST/'
    
    is_save_original = True
    is_save_augmented = True
    is_save_injected = True

    injection = Injection(psf_directory=psf_directory, is_save_original=is_save_original, is_save_augmented=is_save_augmented, is_save_injected=is_save_injected)

    injection_count = 2
    aug_count = 20
    flux_coefficients = [5]
    normalize_psf = True
    filename = f'fc{flux_coefficients[0]}_test'

    injection.apply_injection(injection_count=injection_count, aug_count=aug_count, inject_filename=filename, normalize_psf=normalize_psf, flux_coefficients=flux_coefficients)

