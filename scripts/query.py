import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from astropy.table import unique, vstack
from astroquery.mast import Observations
import jwst
from src.utils import get_mast_token, get_util_main_dir, get_main_dir
from src.utils import List



class Query:
    def __init__(self):
        print(
            f"Query.py file activated. Current JWST Version: {jwst.__version__}")

        mast_token: str = get_mast_token()
        Observations.login(mast_token)


        #query param initialization
        self.instrume: str = None
        self.proposal_id: str = None
        self.psgd = None
        self.instrument_names = None

    def get_proposal_products(self, instrume: str, proposal_id: str, psgd, instrument_names):
        self.instrume: str = instrume
        self.proposal_id: str = proposal_id
        self.psgd = psgd
        self.instrument_names = instrument_names

        try:
            obs_res = Observations.query_criteria(
                proposal_id=self.proposal_id, instrument_name=self.instrument_names)
            print('obs_res:', len(obs_res))

            batch_size = 10
            batches = [obs_res[i:i+batch_size]
                       for i in range(0, len(obs_res), batch_size)]
            tables_list = [Observations.get_product_list(
                obs) for obs in batches]
            print('tables_list:', len(tables_list))

            self.products = unique(vstack(tables_list), keys='productFilename')
            print('products:', len(self.products))

            self.filtered_products = Observations.filter_products(self.products,
                                                                  # dataproduct_type='image',
                                                                  # productType = ['SCIENCE'],
                                                                  proposal_id=f'{self.proposal_id}',
                                                                  productSubGroupDescription=self.psgd
                                                                  )
            print('filtered_products:', len(self.filtered_products))

        except:
            raise Exception('Error raised while querying the MAST server.')

        del tables_list, obs_res, batch_size, batches
        #return self.products, self.filtered_products

    def download_files(self, main_dataset_dir: str):
        # if main_dataset_dir is None:
        #     main_dataset_dir = get_dataset_dir()
        
        dataset_dir = main_dataset_dir
        dataset_dir = dataset_dir + f"/{self.instrume}/{self.proposal_id}"
        try:
            print(
                f"Observations are downloading from Mast Server. \n Download Info: \n Instrume: {self.instrume} \n Proposal_id: {self.proposal_id} \n Dir: {dataset_dir}")
            self.coron_files = Observations.download_products(
                self.filtered_products, download_dir=dataset_dir)
        except:
            raise Exception(
                "Download failed! This problem can occur due to internet issues.")

        del dataset_dir


if __name__ == "__main__":


    proposal_ids : List[str] = ['1194']
    instrume: str = 'NIRCAM'
    instrument_names= ['NIRCAM/CORON', 'NIRCAM/TARGACQ']

    psgd= ['RATEINTS']

    main_dataset_dir: str = '/data/scratch/sarperyurtseven/'

    # Class init
    query = Query()

    for idx in range(len(proposal_ids)):
        proposal_id = proposal_ids[idx]
        print(f"Proposal ID: {proposal_id}")

        # Get and download proposal products
        query.get_proposal_products(instrume=instrume, proposal_id=proposal_id, psgd=psgd, instrument_names=instrument_names)
        query.download_files(main_dataset_dir=main_dataset_dir)
