import os
import glob
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import unique, vstack
from astropy.table import Table
from astropy.time import Time
from astroquery.mast import Mast,Observations

import jwst

sys.path.insert(0, '../util')

from util_type import SERVICE, query_keywords, mast_request_params, PSGD
from util_main import get_mast_token, get_dataset_dir


class Query():
    def __init__(self, service:SERVICE, keywords: query_keywords, proposal_id: str, psgd: PSGD):
        print(f"Query.py file activated. Current JWST Version: {jwst.__version__}")

        # query param initialization
        self.service: SERVICE = service
        self.keywords: query_keywords = keywords
        self.proposal_id: str = proposal_id
        self.psgd: PSGD = psgd
        
        # Date formatting
        self.set_mjd_range()

        # Filter param creation
        self.filters = self.set_params()

        self.params: mast_request_params = {
                                            'columns': '*',
                                            'filters': self.filters
                                            }

    def set_params(self):
        return [{'paramName' : p, 'values' : v} for p, v in self.keywords.items()]

    def set_mjd_range(self):
        '''Set time range in MJD given limits expressed as ISO-8601 dates'''
        minimum = self.keywords['date_obs_mjd'][0]
        maximum = self.keywords['date_obs_mjd'][1]

        self.keywords['date_obs_mjd'] = [{
            "min": Time(minimum, format='isot').mjd, 
            "max": Time(maximum, format='isot').mjd
            }] # e.g. ['2022-01-01','2022-12-01'] -> ISO-8601 Format
        del minimum, maximum

    def get_proposal_products(self):
        try:
            t = Mast.service_request(self.service, self.params)
            fn = list(set(t['filename']))
            ids = list(set(['_'.join(x.split('_')[:-1]) for x in fn]))
            
            matched_obs = Observations.query_criteria(instrument_name= self.keywords['instrume'][0], #e.g.: 'Nircam',
                                                obs_id=ids,
                                                )
            batch_size = 10
            batches = [matched_obs[i:i+batch_size] for i in range(0, len(matched_obs), batch_size)]
            t = [Observations.get_product_list(obs) for obs in batches]
            self.products = unique(vstack(t), keys='productFilename')
            self.filtered_products = Observations.filter_products(self.products,
                                                        #dataproduct_type='image',
                                                        #productType = ['SCIENCE'],
                                                        proposal_id = [f'{self.proposal_id}'],
                                                        productSubGroupDescription = self.psgd
                                                        )

        except:
            raise Exception('Mast Service Connection Failure!')
        
        del t, fn, ids, matched_obs, batch_size, batches
        return self.products, self.filtered_products
    
    def determine_and_count(self, column:str, table):    
        prop_list = []
        counter = {}
        for i in range(len(table['proposal_id'])):

            if table[f'{column}'][i] not in prop_list:

                prop_list.append(table[f'{column}'][i])
                try:
                    counter[table[f'{column}'][i]] = 1
                except:
                    continue
            else:
                try:
                    counter[table[f'{column}'][i]] += 1
                except:
                    continue
                
        print(counter)

    def get_json_table(self, table,suffix):
        testing_df = table.to_pandas()
        product_files = testing_df.loc[testing_df['productSubGroupDescription'] == f'{suffix}']['obs_id']
        obs_ids = []
        for i in product_files:
            obs_ids.append(i)
            
        products_df = testing_df.loc[testing_df['obs_id'].isin(obs_ids)]
        info_files = products_df.loc[products_df['productType']=='INFO']
        info_table = Table.from_pandas(info_files)
        
        return info_table
    
    def get_rateints_files(self, table):
        rateints_df = table.to_pandas()
        rateints_df = rateints_df.loc[rateints_df['productSubGroupDescription'] == 'RATEINTS']
        rateints_table = Table.from_pandas(rateints_df)
        
        return rateints_table
    
    def download_files(self):
        mast_token: str = get_mast_token()
        Observations.login(mast_token)
        dataset_dir = get_dataset_dir()
        dataset_dir = dataset_dir + f"/{self.keywords['instrume'][0]}/{self.proposal_id}"
        try:
            print(f"Observations are downloading from Mast Server. \n Download Info: \n Instrume: {self.keywords['instrume'][0]} \n Proposal_id: {self.proposal_id} \n Dir: {dataset_dir}")
            self.coron_files = Observations.download_products(self.filtered_products, download_dir = dataset_dir)
        except:
            raise Exception("Download failed! This problem can occur due to internet issues.")

        del dataset_dir, mast_token

if __name__ == "__main__":

    # Example Inputs
    service: SERVICE = 'Mast.Jwst.Filtered.Nircam'

    keywords: query_keywords = {
                'exp_type':['NRC_CORON'],
                'instrume':['NIRCAM'],
                'date_obs_mjd': ['2022-01-01','2022-12-01'],
                }

    proposal_id: str = '1537'

    psgd: PSGD = ['RATEINTS', 'ASN']

    # Class init
    query = Query(service=service, keywords=keywords, proposal_id=proposal_id, psgd = psgd)

    # Get proposal products
    #products, filtered_products = query.get_proposal_products()
    _, _ = query.get_proposal_products()

    # ASN and Rateints count control
    #query.determine_and_count('productSubGroupDescription',filtered_products)

    #download files
    query.download_files()





