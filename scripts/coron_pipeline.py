import os
import glob
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import astropy.io.fits as fits
from astropy.table import Table

import jwst
from jwst.pipeline import Image2Pipeline
from jwst.associations.lib.rules_level3 import Asn_Lv3Coron
from jwst.associations import AssociationRegistry
from jwst.associations.mkpool import mkpool
from jwst.associations import generate
from jwst.pipeline.calwebb_coron3 import Coron3Pipeline

from src.utils import get_dataset_dir, INSTRUME, List



class CoronPipeline():
    def __init__(self, proposal_id: str, instrume: INSTRUME):
        print(
            f"coron_pipeline.py script activated. Current JWST Version: {jwst.__version__}")
        os.environ["CRDS_PATH"] = '/home/sarperyn/crds_cache/jwst_ops'
        os.environ["CRDS_SERVER_URL"] = 'https://jwst-crds.stsci.edu'

        self.instrume = instrume
        self.proposal_id = proposal_id

        self.dataset_dir = get_dataset_dir()
        self.dataset_dir = self.dataset_dir + \
            f'/{self.instrume}/{self.proposal_id}/mastDownload/JWST/'
        print(self.dataset_dir)

    def pipeline_stage2(self, rateints_check: bool = False):

        self.detector = 'nrcalong'

        rateints_files = glob.glob(os.path.join(
            self.dataset_dir, f'**/*rateints.fits'))
        if rateints_files == []:
            self.detector = 'nrca2'
            rateints_files = glob.glob(os.path.join(
                self.dataset_dir, f'**/*{self.detector}_rateints.fits'))
        
        if rateints_check:
            self.rateints_check(rateints_files=rateints_files)

        batch_size = 4
        img2pipeline = Image2Pipeline()

        for index in range(0, len(rateints_files), batch_size):
            for rateints_files_batch in rateints_files[index:index+batch_size]:
                output_dir = '/'.join(rateints_files_batch.split('/')
                                      [:-1]) + '/'
                
                #img2pipeline.output_dir = output_dir
                #img2pipeline.save_results = True
                img2pipeline.call(rateints_files_batch, save_results=True, output_dir=output_dir)

        del rateints_files, img2pipeline, batch_size

    def pipeline_stage3(self, asn_check: bool = False, calints_check: bool = False):
        calints_files = glob.glob(os.path.join(
            self.dataset_dir, f'**/*calints.fits'))

        if calints_check:
            self.calints_check(calints_files=calints_files)

        asn_files = self.asn_file_creation(calints_files=calints_files)

        if asn_check:
            self.asn_check(asn_files=asn_files)

        coron3_pipeline = Coron3Pipeline()
        coron3_pipeline.output_dir = self.dataset_dir
        coron3_pipeline.save_results = True
        print(asn_files)
        for idx, asn in enumerate(asn_files):
            asn_dict = {}
            for key, value in zip(asn.keys(), asn.values()):
                asn_dict[key] = value

            coron3_pipeline.process(asn_dict)

        del asn_files, calints_files, asn_dict, coron3_pipeline

    def calints_check(self, calints_files):
        print(
            f'Calints control: \n Total Calints: {len(calints_files)} \n First Calints info:')
        calints_sample = fits.open(calints_files[0])
        calints_sample.info()

    def asn_check(self, asn_files):
        print(
            f'ASN control: \n Total ASN Files: {len(asn_files)} \n First ASN info: {asn_files[0]}')

    def rateints_check(self, rateints_files):
        print(
            f'Rateints control: \n Total Rateints: {len(rateints_files)} \n First Rateints info: {rateints_files[0]}')

    def asn_file_creation(self, calints_files):
        pool = mkpool(calints_files)
        pool_df = pool.to_pandas()
        pool_df.to_csv(
            f'calints_{self.instrume}_{self.proposal_id}_pool.csv', index=False)

        asn_coron_rule_dir = '/home/sarperyn/.conda/envs/jwst-dev/lib/python3.9/site-packages/jwst/associations/lib/rules_level3.py'
        registry = AssociationRegistry(
            [asn_coron_rule_dir], include_default=False)

        asn_files = generate(pool, registry)

        return asn_files


if __name__ == "__main__":

    instrume: str       = 'NIRCAM'
    pids    : List[str] = ['1194']


    for pid in pids:
        try:
            proposal_id: str = pid.split('/')[-1]
            print("STARTING FOR PID:",proposal_id)
            coron = CoronPipeline(proposal_id=proposal_id, instrume=instrume)
            coron.pipeline_stage2()
            coron.pipeline_stage3()

        except:
            print(f"Somethings wrong with {pid}")
