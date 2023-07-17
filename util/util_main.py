import os


def get_mast_token():
    main_dir = get_main_dir()
    secret_file = open(main_dir + "/secrets/mast_token.txt", "r")
    mast_token = secret_file.readline()
    secret_file.close()

    del secret_file, main_dir
    return mast_token


def get_dataset_dir():
    main_dir = get_main_dir()
    dataset_dir = main_dir + "/dataset"

    dataset_dir = '/data/scratch/bariskurtkaya/dataset'

    del main_dir
    return dataset_dir


def get_util_main_dir():
    return os.path.realpath(__file__)


def get_main_dir():
    current_dir = get_util_main_dir()
    current_dir = current_dir.split('/')
    main_dir = '/'.join(current_dir[:-2])

    del current_dir
    return main_dir


def get_filename_from_dir(fits_dir):
    return '_'.join(fits_dir.split('/')[-1].split('_')[:-1])


if __name__ == "__main__":

    # Tests
    get_mast_token()
    get_main_dir()
