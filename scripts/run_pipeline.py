import os
from jwst.pipeline import Image2Pipeline, Coron3Pipeline
from glob import glob
import astropy.io.fits as fits
import time
from jwst.associations.mkpool import mkpool
from jwst.associations import AssociationRegistry
from jwst.associations.mkpool import mkpool
from jwst.associations import generate



INSTRUME = 'NIRCAM'

os.environ["CRDS_PATH"] = '/home/sarperyn/crds_cache/jwst_ops'
os.environ["CRDS_SERVER_URL"] = 'https://jwst-crds.stsci.edu'

def get_exptypes(fits_):
    
    for file in range(len(fits_)):
        
        f = fits.open(fits_[file])
        exp_type = f[0].header['EXP_TYPE']
        print(exp_type)
        
def runimg2(filename, output_dir):
    
    img2 = Image2Pipeline()
    img2.output_dir = output_dir
    img2.save_results = True
    img2(filename)

def t_path(partial_path):
    __file__ = '/home/sarperyn/.conda/envs/jwst-dev/lib/python3.9/site-packages/jwst/associations/lib/rules_level3.py'
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, partial_path)


def runcoron(filename, output_dir):

    coron = Coron3Pipeline()
    coron.output_dir = output_dir
    coron.save_results = True
    coron.process(filename)

def get_stage3_products(asns,directory):
    
    for t in range(len(asns)):

        asn_dict = {}
        for i,j in zip(asns[t].keys(),asns[t].values()):
            print(i,j)
            asn_dict[i] = j
            
        runcoron(asn_dict,directory)


def process_products(programs:list):

    for program in programs:

        directory = f'/data/scratch/sarperyurtseven/dataset/{INSTRUME}/{program}/mastDownload/JWST/'
        rateints_files = glob(os.path.join(directory, '*/*rateints.fits'))
        batch_size = 4

        for i in range(0,len(rateints_files),batch_size):
        
            for f in rateints_files[i:i+batch_size]:
                
                output_dir = '/'.join(f.split('/')[:-1]) + '/' 
                runimg2(f,output_dir)
            time.sleep(1)
            
            
        calints_data = glob(os.path.join(directory, '**/**calints.fits'))
        print(len(calints_data))
        pool = mkpool(calints_data)
        pool_df = pool.to_pandas()
        pool_df.to_csv(f'calints_{INSTRUME}_{program}_pool.csv',index=False)

        t_path_r = '/home/sarperyn/.conda/envs/jwst-dev/lib/python3.9/site-packages/jwst/associations/lib/rules_level3.py'
        registry = AssociationRegistry([t_path(f'{t_path_r}')], include_default=False)
        asns = generate(pool,registry)
        print(len(asns))
        print(f'Association file for {program}:{len(asns)}')
    
        get_stage3_products(asns,directory)


programs = ['1386']

process_products(programs)

