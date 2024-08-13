import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import warnings
import webbpsf
import math

import torch
import torch.nn as nn

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.utils.exceptions import AstropyWarning

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

if not os.environ.get('WEBBPSF_PATH'):
    os.environ['WEBBPSF_PATH'] = '/data/webbpsf-data'

from src.utils import get_filename_from_dir, get_dataset_dir, get_stage3_products

from src.utils.seed import seed_everything
from src.utils.time import timing

warnings.simplefilter('ignore', category=AstropyWarning)

class PSFDatasetGPU_Base(nn.Module):
    @timing
    def __init__(self, 
                psf_directory: str,
                device='cuda:0') -> None:
        super().__init__()

        self.psf_directory = psf_directory
        self.DEVICE = device

        self.get_exoplanet_extremes()

    def get_exoplanet_extremes(self):
        query_exoplanets = NasaExoplanetArchive.query_criteria(
            table="pscomppars", where="discoverymethod like 'Imaging'")

        self.exoplanet_extremes = {
            'max': np.nanmax(query_exoplanets['pl_orbsmax']),
            'min': np.nanmin(query_exoplanets['pl_orbsmax'])
        }

    @timing
    def prepare_dataset(self, 
                        save_folder:str='/data/scratch/bariskurtkaya/dataset/torch_dataset', 
                        pid:str='test',
                        instrume:str='NIRCAM') -> None:
        dataset_dict_arr = []

        self.psfstacks_nircam_dirs = get_stage3_products(suffix='psfstack', directory=self.psf_directory)
        self.psfstacks = {}
        self.__create_psfstacks_dict()

        if len(self.psfstacks.keys()) == 0:
            print('No PSF Stack has been found!')
        
        else:
            for filter_key in self.psfstacks.keys():
                max_pixel_distance, min_pixel_distance = self.__get_max_min_pixel_distance(self.psfstacks[filter_key])

                print(f'{filter_key} max pixel distance: {max_pixel_distance}, min pixel distance: {min_pixel_distance}')
                fov = np.round(np.sqrt(self.psfstacks[filter_key][1].header['PIXAR_A2']), 3) * self.psfstacks[filter_key][1].data.shape[1]
                detector = self.psfstacks[filter_key][0].header['DETECTOR']
                filter = self.psfstacks[filter_key][0].header['FILTER']

                # We aren't using for injection
                # pupil_mask = self.psfstacks[filter_key][0].header['PUPIL']
                # image_mask = self.psfstacks[filter_key][0].header['CORONMSK']
                pupil_mask = ''
                image_mask = ''

                generated_psf = self.__generate_psf_model(detector=detector, filter=filter, pupil_mask=pupil_mask, image_mask=image_mask, fov=fov*2, save=True)

                if self.psfstacks[filter_key][0].header['CHANNEL'] == 'LONG':
                    generated_psf_selection = 1
                else:
                    generated_psf_selection = 0

                psfs = self.__nan_elimination(torch.from_numpy(self.psfstacks[filter_key][1].data.astype(np.float32)).to(self.DEVICE))

                dataset_dict_arr.append(
                    {
                        "filter_key": str(filter_key),
                        "psfs": psfs,
                        "generated_psf": torch.from_numpy(generated_psf[generated_psf_selection].data.astype(np.float32)).to(self.DEVICE),
                        "max": max_pixel_distance,
                        "min": min_pixel_distance,
                        'pid': pid,
                        'instrume': instrume
                    }
                )                
            
            print(f'{save_folder}/{pid}.pth') 
            os.makedirs(save_folder, exist_ok=True)
            torch.save(dataset_dict_arr, f'{save_folder}/{pid}.pth')

    def __nan_elimination(self, psf):
        return torch.nan_to_num(psf)

    def __generate_psf_model(self, detector:str, filter:str, pupil_mask:str, image_mask:str, fov:float, save:bool=True):
        
        if detector == 'NRCALONG':
            detector = 'NRCA5'
        elif detector == 'NRCBLONG':
            detector = 'NRCB5'

        psf_dir = get_dataset_dir() + f'/PSF_SAMPLES/{detector}-{filter}-{fov}-{pupil_mask}-{image_mask}.fits'
        psf_dir_glob = glob(psf_dir)

        if psf_dir_glob != []:
            generated_psf = fits.open(psf_dir)
            print(f'{detector}-{filter} PSF with {fov} arcsec fov collected from cache.')
            del psf_dir, psf_dir_glob
        else:
            os.makedirs(get_dataset_dir() + f'/PSF_SAMPLES/', exist_ok=True)
            if 'NRC' in detector:
                wpsf = webbpsf.NIRCam()
            elif 'MIR' in detector:
                wpsf = webbpsf.MIRI()
            else:
                raise ValueError('Detector is not valid!')
            
            wpsf.detector = detector
            wpsf.filter = filter
            if image_mask != '':
                wpsf.image_mask = image_mask.replace('MASKA','MASK') # MASKA335R -> MASK335R
            if pupil_mask != '':
                wpsf.pupil_mask = pupil_mask

            if save:
                generated_psf = wpsf.calc_psf(fov_arcsec=fov, oversample=2, outfile=f'{psf_dir}')
            else:
                generated_psf = wpsf.calc_psf(fov_arcsec=fov, oversample=2)

            print(f'{detector}-{filter} PSF generated respect to {fov} arcsec fov')
            del psf_dir, psf_dir_glob, wpsf

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

        one_pix_side_length_arcsec = np.sqrt(
            psfstack[1].header['PIXAR_A2']) * u.arcsec
        max_pixel_distance = self.exoplanet_extremes['max'] / (
            distance.to(u.au) * np.tan(one_pix_side_length_arcsec))
        min_pixel_distance = self.exoplanet_extremes['min'] / (
            distance.to(u.au) * np.tan(one_pix_side_length_arcsec))

        del ra, dec, u_ra, u_dec, coord, width, height, query_results, min_idx, target_star, distance, one_pix_side_length_arcsec
        return torch.from_numpy(np.floor(max_pixel_distance.value.astype(np.float32))), torch.from_numpy(np.ceil(min_pixel_distance.value.astype(np.float32)))


@timing
def main():
    seed_everything(42)
    
    data_path = '/data/scratch/bariskurtkaya/dataset/NIRCAM'
    save_folder = '/data/scratch/bariskurtkaya/dataset/torch_dataset'
    from glob import glob
    pids = [path.split('/')[-1] for path in glob(f'{data_path}/*')]
    
    instrume: str = 'NIRCAM'

    injection_GPU = PSFDatasetGPU_Base(psf_directory='')

    for pid in pids:
        psf_directory = f'/data/scratch/bariskurtkaya/dataset/{instrume}/{pid}/mastDownload/JWST/'

        try:
            injection_GPU.psf_directory = psf_directory
            injection_GPU.prepare_dataset(
                save_folder=save_folder,
                pid=pid,
                instrume=instrume
            )

        except Exception as err:
            print(f'Error: {err}')
            raise Exception


if __name__ == '__main__':
    main()