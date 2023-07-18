import os

if not os.environ.get('WEBBPSF_PATH'):
    os.environ['WEBBPSF_PATH'] = '/data/webbpsf-data'

import sys
sys.path.append("..")

from glob import glob

import warnings

import webbpsf

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

    def apply_injection(self, injection_count:int=10):

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
            
            for _, psf in enumerate(self.psfstacks[filter_key][1].data):
                psf = self.__nan_elimination(psf)
                self.__injection(psf, generated_psf[generated_psf_selection].data, max_pixel_distance, min_pixel_distance, filename=filter_key, injection_count=injection_count)
            

    def __injection(self, psf, generated_psf, max_pixel_distance:int, min_pixel_distance:int, filename:str, injection_count:int=10):
        # Function is broken. It needs to be fixed. At value of 160 the exoplanet and the star are not seperated.
        # Max should be max - shape/2 and there should be a sign generator to determine the left or right side of the image.
        # Also, there should be a flux controller that controls the flux of the exoplanet according to the star.
        for idx in range(injection_count):
            if max_pixel_distance > psf.shape[0]:
                max_x = psf.shape[0]
            else:
                max_x = max_pixel_distance
            
            if max_pixel_distance > psf.shape[1]:
                max_y = psf.shape[1]
            else:
                max_y = max_pixel_distance

            random_x = np.random.randint(min_pixel_distance, max_x)
            random_y = np.random.randint(min_pixel_distance, max_y)

            injected = generated_psf[
                psf.shape[0]-random_x : (psf.shape[0] * 2) - random_x,
                psf.shape[1]-random_y : (psf.shape[1] * 2) - random_y
            ] + psf

            filename = f'{filename}-{random_x}-{random_y}.png'

            plt.imsave(filename=filename, arr=injected, cmap='gray')

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


if __name__ == '__main__':

    PROPOSAL_ID = '1386'
    INSTRUMENT = 'NIRCAM'
    psf_directory = f'/data/scratch/bariskurtkaya/dataset/{INSTRUMENT}/{PROPOSAL_ID}/mastDownload/JWST/'

    injection = Injection(psf_directory=psf_directory)

    injection.apply_injection(injection_count=10)
