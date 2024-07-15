import os
import glob
from src.utils import DATA_MAIN_DIR


def get_mast_token():
    main_dir = get_main_dir()
    secret_file = open(main_dir + "/src/secrets/mast_token.txt", "r")
    mast_token = secret_file.readline()
    secret_file.close()

    del secret_file, main_dir
    return mast_token

def get_dataset_dir():

    main_dir = get_main_dir()
    dataset_dir  = DATA_MAIN_DIR

    del main_dir
    return dataset_dir

def get_util_main_dir():
    return os.path.realpath(__file__)

def get_main_dir():
    current_dir = get_util_main_dir()
    current_dir = current_dir.split('/')
    main_dir = '/'.join(current_dir[:-3])

    del current_dir
    print(main_dir)
    return main_dir

def get_filename_from_dir(fits_dir):
    return '_'.join(fits_dir.split('/')[-1].split('_')[:-1])

def get_stage3_products(suffix, directory):
    return glob.glob(os.path.join(directory, f'*{suffix}.fits'))

if __name__ == "__main__":

    # Tests
    get_mast_token()
    get_main_dir()