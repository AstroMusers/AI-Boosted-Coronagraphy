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

from torchvision.transforms import v2
from torchvision.transforms.v2._utils import query_size

warnings.simplefilter('ignore', category=AstropyWarning)
torch.set_warn_always(True)

FIRST_STEP = False

class ModifiedRandomCrop(v2.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super().__init__(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)
        
    def _transform(self, inpt, params):
        if params["needs_pad"]:
            fill = self._get_fill(self._fill, type(inpt))
            inpt = self._call_kernel(v2.functional.pad, inpt, padding=params["padding"], fill=fill, padding_mode=self.padding_mode)

        if params["needs_crop"]:
            inpt = self._call_kernel(
                v2.functional.crop, inpt, top=params["top"], left=params["left"], height=params["height"], width=params["width"]
            )

        return inpt.unsqueeze(0), torch.Tensor([np.ceil(params["width"] - params["left"]), np.ceil(params["height"] - params["top"])])

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
                size = self.psfstacks[filter_key][1].data.shape[1] if self.psfstacks[filter_key][1].data.shape[1] > self.psfstacks[filter_key][1].data.shape[2] else self.psfstacks[filter_key][1].data.shape[2]
                fov = np.round(np.sqrt(self.psfstacks[filter_key][1].header['PIXAR_A2']), 3) * size
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

    def forward(self, 
                save_folder:str='/data/scratch/bariskurtkaya/dataset/torch_dataset', 
                pid:str='test',
                instrume:str='NIRCAM') -> None:
        
        self.prepare_dataset(save_folder, pid, instrume)

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
        
        # Override
        max_pixel_distance = 50 if max_pixel_distance > 50 else max_pixel_distance

        del ra, dec, u_ra, u_dec, coord, width, height, query_results, min_idx, target_star, distance, one_pix_side_length_arcsec
        return torch.from_numpy(np.floor(max_pixel_distance.value.astype(np.float32))), torch.from_numpy(np.ceil(min_pixel_distance.value.astype(np.float32)))

class PSFDatasetGPU_Injection(nn.Module):
    def __init__(self, flux_coefficients=[1e-3, 1e-9], save_folder='/data/scratch/bariskurtkaya/dataset/torch_dataset_injection', device='cuda:0'):
        super().__init__()

        self.flux_coef = flux_coefficients
        self.save_folder = save_folder
        self.DEVICE = device

        os.makedirs(self.save_folder, exist_ok=True)

    def __sample_flux(self, num_flux):
        flux_tensor = torch.randint(num_flux, (num_flux,)).to(self.DEVICE) / num_flux
        flux_tensor = flux_tensor * (self.flux_coef[0] - self.flux_coef[1]) + self.flux_coef[1]
        return flux_tensor.view(-1, 1, 1)

    def __injection_item_GPU(self, psf_dict, batch, num_injection, sub_batch_idx):
        psf_gen = psf_dict['generated_psf']
        psfs = psf_dict['psfs'][sub_batch_idx*batch:(sub_batch_idx+1)*batch].to(self.DEVICE)

        # Will be refactored [crop , [center_x, center_y]]
        generated_psfs, locations = list(zip(*[self.modified_crop(psf_gen) for _ in range(num_injection*batch)])) # NI*B x H x W - (x, y)
        generated_psfs = torch.cat(generated_psfs, dim=0).to(self.DEVICE)

        flux_vector = self.__sample_flux(num_injection*batch) # NI*B x 1 x 1
        flux_tensor = flux_vector.repeat(1, generated_psfs.shape[1], generated_psfs.shape[2]) # NI*B x H x W

        psfs = psfs.repeat_interleave(num_injection, dim=0) # NI*B x Hp x Wp
        psf_integral = torch.sum(psfs, dim=(1,2)).view(-1, 1, 1).repeat(1, generated_psfs.shape[1], generated_psfs.shape[2]) # NI*B

        generated_psfs = generated_psfs * flux_tensor * psf_integral # NI*B x Hg x Wg
        
        pad_h = generated_psfs.shape[1] - psfs.shape[1]
        pad_w = generated_psfs.shape[2] - psfs.shape[2]

        psfs = torch.nn.functional.pad(psfs, pad=(pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='replicate') # NI*B x Hg x Wg

        injections = psfs + generated_psfs

        torch.save((injections.cpu(), torch.stack(locations).cpu(), flux_vector.view(-1).cpu()), f"{self.save_folder}/injection_{psf_dict['filter_key']}_{sub_batch_idx}.pth")

    @timing
    def injection_GPU(self, psf_paths, num_injection=10, max_size=None):
        psf_dicts = [torch_psf for psf_path in psf_paths for torch_psf in torch.load(psf_path)]

        old_height = 0
        for psf_dict in tqdm(psf_dicts):

            if max_size != None:
                center_crop = v2.CenterCrop(max_size)
                psf_dict['psfs'] = center_crop(psf_dict['psfs'])
                center_crop = v2.CenterCrop(max_size*2)
                psf_dict['generated_psf'] = center_crop(psf_dict['generated_psf'])


            batch, _, _ = psf_dict['psfs'].shape
            height, _ = psf_dict['generated_psf'].shape

            if old_height != height:
                self.modified_crop =  ModifiedRandomCrop(height//2)
            
            # VRAM_Consumption Variable Chosen for RTX A6000
            vram_consumption = int(batch * num_injection * height * height // (2*1e+9))
            if (vram_consumption > 1):
                sub_batch_size = int(np.ceil(batch // vram_consumption))
                for idx in range(vram_consumption):
                    self.__injection_item_GPU(psf_dict, sub_batch_size, num_injection, sub_batch_idx=idx)
            else:
                self.__injection_item_GPU(psf_dict, batch, num_injection, sub_batch_idx=0)

            old_height = height

            torch.save(psf_dict['psfs'].cpu(),  f"{self.save_folder}/base_{psf_dict['filter_key']}.pth")
        
    def forward(self, psf_paths, num_injection=10, max_size=None):
        self.injection_GPU(psf_paths, num_injection, max_size)


@timing
def main():
    seed_everything(42)
    
    data_path = '/data/scratch/bariskurtkaya/dataset/NIRCAM'
    save_folder = '/data/scratch/bariskurtkaya/dataset/torch_dataset'
    from glob import glob
    pids = [path.split('/')[-1] for path in glob(f'{data_path}/*')]
    
    instrume: str = 'NIRCAM'

    datasetGPU_base = PSFDatasetGPU_Base(psf_directory='')

    for pid in pids:
        psf_directory = f'/data/scratch/bariskurtkaya/dataset/{instrume}/{pid}/mastDownload/JWST/'

        try:
            datasetGPU_base.psf_directory = psf_directory
            datasetGPU_base.prepare_dataset(
                save_folder=save_folder,
                pid=pid,
                instrume=instrume
            )

        except Exception as err:
            print(f'Error: {err}')
            raise Exception

def main_v2():
    seed_everything(42)

    psf_directory = f'/data/scratch/bariskurtkaya/dataset/torch_dataset'
    psf_paths = glob(f'{psf_directory}/*.pth')

    injection_GPU = PSFDatasetGPU_Injection(flux_coefficients=[1e-2, 1e-5])

    try:
        injection_GPU.injection_GPU(psf_paths, num_injection=10, max_size=256)
    except Exception as err:
        print(err)

if __name__ == '__main__':
    if FIRST_STEP == True:
        main()
    else:
        main_v2()