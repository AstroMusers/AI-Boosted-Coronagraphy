from astropy.io import fits
from astropy.table import unique, vstack
from astropy.table import Table
from astropy.time import Time
from astroquery.mast import Mast,Observations

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from glob import glob
import json


def save_products(df,programs):

    for program in programs:
        
        jw = df[(df['proposal_id']==program)  & (df['productType'] == 'SCIENCE') & (df['calib_level'] == 2) & (df['productSubGroupDescription'] == 'RATEINTS')]
        products = Table.from_pandas(jw)
        Observations.download_products(products,mrp_only=False,download_dir=f'/data/scratch/bariskurtkaya/dataset/NIRCAM/{program}/mastDownload/JWST/')
        print(f'{program} is done.')
        

proposal_obs = Observations.query_criteria(obstype='all',obs_collection='JWST',instrument_name='NIRCAM/CORON')
#print("Number of observations:",len(proposal_obs))
print(proposal_obs)

batch_size = 10
batches = [proposal_obs[i:i+batch_size] for i in range(0, len(proposal_obs), batch_size)]
t = [Observations.get_product_list(obs) for obs in batches]
products = unique(vstack(t), keys='productFilename')
filtered_products = Observations.filter_products(products)

prd_df = filtered_products.to_pandas()
prd_df['proposal_id'].unique()

df = prd_df.loc[prd_df['dataRights'] != 'EXCLUSIVE_ACCESS']

programs = ['1075']

save_products(df=df,programs=programs)



/data/scratch/bariskurtkaya/dataset/NIRCAM/1075/mastDownload/JWST/