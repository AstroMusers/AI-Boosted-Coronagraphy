import os
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
import jwst.associations
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from glob import glob
import astropy.io.fits as fits
import numpy as np
import time
import pandas as pd
import json
import pickle
from json import JSONEncoder
from collections import defaultdict
from CoronPipeline import MyCoron3Pipeline
from jwst.associations.mkpool import mkpool
from jwst.associations.lib.rules_level3 import Asn_Lv3Coron
from jwst.associations import AssociationPool, AssociationRegistry
from jwst.associations.mkpool import from_cmdline, mkpool
from jwst.associations import generate
from astropy.table import Table


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
    __file__ = '/home/sarperyn/sarperyurtseven/ProjectFiles/notebooks/asn_coron_rule.py'
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, partial_path)


def runcoron(filename, output_dir):

    coron = MyCoron3Pipeline()
    coron.output_dir = output_dir
    coron.save_results = True
    coron.process(filename)

def get_stage3_products(asns,directory):
    
    for t in range(len(asns)):

        asn_dict = {}
        for i,j in zip(asns[t].keys(),asns[t].values()):
            asn_dict[i] = j
            
        runcoron(asn_dict,directory)


def process_products(programs:list):

    for program in programs:

        directory = f'/home/sarperyn/sarperyurtseven/ProjectFiles/dataset/{INSTRUME}/{program}/mastDownload/JWST/'
        rateints_files = glob(os.path.join(directory, '*/*rateints.fits'))
        batch_size = 4
        for i in range(0,len(rateints_files),batch_size):
        
            for f in rateints_files[i:i+batch_size]:
                
                output_dir = '/'.join(f.split('/')[:-1]) + '/' 
                runimg2(f,output_dir)
            time.sleep(1)
    
        calints_data = glob(os.path.join(directory, '**/**calints.fits'))
    
        pool = mkpool(calints_data)
        pool_df = pool.to_pandas()
        pool_df.to_csv(f'calints_{INSTRUME}_{program}_pool.csv',index=False)
    
        registry = AssociationRegistry([t_path('rule_level3.py')], include_default=False)
        asns = generate(pool,registry)
        print(f'Association file for {program}:{len(asns)}')
    
        get_stage3_products(asns,directory)


programs = ['1537','4454']
process_products(programs)