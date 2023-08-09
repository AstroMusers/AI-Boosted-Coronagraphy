import os

if not os.environ.get('WEBBPSF_PATH'):
    os.environ['WEBBPSF_PATH'] = '/data/webbpsf-data'

import sys
sys.path.append("..")

from glob import glob

import warnings

import webbpsf
import PIL

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.utils.exceptions import AstropyWarning

from util.util_main import get_filename_from_dir, get_dataset_dir
from notebooks.visualization_helpers import get_stage3_products

import numpy as np

import matplotlib.pyplot as plt

from time import time

warnings.simplefilter('ignore', category=AstropyWarning)

class Injection():
    def __init__(self, psf_directory: str) -> None:
        print('Injection process has been started...')

        self.psfstacks_nircam_dirs = get_stage3_products(
            suffix='psfstack', directory=psf_directory)

        self.psfstacks = {}
        self.__create_psfstacks_dict()

    def apply_injection(self, injection_count:int=10, flux_coefficients = [1, 2, 5, 10, 100, 1000, 10000]):

        for filter_key in self.psfstacks.keys():
            max_pixel_distance, min_pixel_distance = self.__get_max_min_pixel_distance(
                self.psfstacks[filter_key])
            
            print(f'{filter_key} max pixel distance: {max_pixel_distance}, min pixel distance: {min_pixel_distance}')

            fov = np.round(np.sqrt(self.psfstacks[filter_key][1].header['PIXAR_A2']), 3) * self.psfstacks[filter_key][1].data.shape[1]
            detector = self.psfstacks[filter_key][0].header['DETECTOR']
            filter = self.psfstacks[filter_key][0].header['FILTER']

            generated_psf = self.__generate_psf_model(detector=detector, filter=filter, fov=fov*2, save=True)

            if self.psfstacks[filter_key][0].header['CHANNEL'] == 'LONG':
                generated_psf_selection = 1
            else:
                generated_psf_selection = 0
            
            for idx, psf in enumerate(self.psfstacks[filter_key][1].data):
                psf = self.__nan_elimination(psf)
                if not self.__is_psf_empty(psf):
                    self.__injection(idx, psf, generated_psf[generated_psf_selection].data, max_pixel_distance, min_pixel_distance, filter_key=filter_key, injection_count=injection_count, flux_coefficients=flux_coefficients)
                else:
                    pass
            
    def __injection(self, idx, psf, generated_psf, max_pixel_distance:int, min_pixel_distance:int, filter_key:str, injection_count:int=10, flux_coefficients:list=[1, 2, 5, 10, 100, 1000, 10000]):
        dataset_dir = get_dataset_dir()

        for _ in range(injection_count):
            if max_pixel_distance > psf.shape[0]:
                max_x = psf.shape[0]
            else:
                max_x = max_pixel_distance
                
            if max_pixel_distance > psf.shape[1]:
                max_y = psf.shape[1]
            else:
                max_y = max_pixel_distance

            random_x = np.random.randint(min_pixel_distance, np.floor(max_x - (psf.shape[0]/2)))
            random_y = np.random.randint(min_pixel_distance, np.floor(max_y - (psf.shape[1]/2)))

            rand_sign_x = 1 if np.random.rand() < 0.5 else -1
            rand_sign_y = 1 if np.random.rand() < 0.5 else -1

            x = int((psf.shape[0]/2) + rand_sign_x*random_x)
            y = int((psf.shape[1]/2) + rand_sign_y*random_y)
            
            for flux_coefficient in flux_coefficients:
                temp_psf = np.copy(generated_psf * (np.max(psf) / ( flux_coefficient * np.max(generated_psf))))

                injected = np.copy(temp_psf[
                    int(temp_psf.shape[0]/2 - x): int(temp_psf.shape[0]/2 - x + psf.shape[0]),
                    int(temp_psf.shape[1]/2 - y): int(temp_psf.shape[1]/2 - y + psf.shape[1])
                ] + psf)
                                    
                #filename = f'{dataset_dir}/PSF_INJECTION/{filter_key}-psf{idx}-x{x}-y{y}-fc{flux_coefficient}.png'
                filename = f'/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/injections/{filter_key}-psf{idx}-x{x}-y{y}-fc{flux_coefficient}.npy'
                #plt.imsave(fname=filename, arr=injected, cmap='gray')
                # im = PIL.Image.fromarray(injected)
                # im = im.convert("L")
                # im.save(filename)
                np.save(filename,injected)

        
        del temp_psf, injected, max_x, max_y, random_x, random_y, x, y, filename
        
    def __nan_elimination(self, psf):
        return np.nan_to_num(psf)

    def __generate_psf_model(self, detector:str, filter:str, fov:float, save:bool=True):
        
        if detector == 'NRCALONG':
            detector = 'NRCA5'
        elif detector == 'NRCBLONG':
            detector = 'NRCB5'

        psf_dir = get_dataset_dir() + f'/PSF_SAMPLES/{detector}-{filter}-{fov}.fits'
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
            wpsf.detector = detector
            wpsf.filter = filter

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
        if psd[int(width/2) -1][int(height/2) -1] < 0.85:
            return True
        else:
            return False

    def __create_psd(self, psf):
        psd = np.abs(np.fft.fftshift(np.fft.fft2(psf)))**2
        psd = np.log10(psd)
        psd = psd/psd.max()
        return psd
    


if __name__ == '__main__':

    PROPOSAL_ID = '1386'
    INSTRUMENT = 'NIRCAM'
    psf_directory = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/mastDownload/JWST/'

    injection = Injection(psf_directory=psf_directory)

    injection_count = 4
    flux_coefficients = [100, 1000, 10000]

    injection.apply_injection(injection_count=injection_count, flux_coefficients=flux_coefficients)
    print('ss')