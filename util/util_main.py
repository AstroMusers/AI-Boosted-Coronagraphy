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

    del main_dir
    return dataset_dir

def get_current_dir():
     return os.path.realpath(__file__)

def get_main_dir():
    current_dir = get_current_dir()
    current_dir = current_dir.split('/')
    main_dir = '/'.join(current_dir[:-2])

    del current_dir
    return main_dir

def get_scripts_dir():
    main_dir = get_main_dir()
    scripts_dir = main_dir + "/scripts"

    del main_dir
    return scripts_dir

if __name__ == "__main__":

    # Tests
    get_mast_token()
    get_download_dir()
    get_main_dir()