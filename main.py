from scripts.query import Query
from scripts.coron_pipeline import CoronPipeline


def test_query():

    # Example Inputs
    service: SERVICE = 'Mast.Jwst.Filtered.Nircam'

    keywords: query_keywords = {
                'exp_type':['NRC_CORON'],
                'instrume':['NIRCAM'],
                'date_obs_mjd': ['2022-01-01','2022-12-01'],
                }

    proposal_id: str = '1441'

    psgd: PSGD = ['RATEINTS', 'ASN']

    query = Query(service=service, keywords=keywords, proposal_id=proposal_id, psgd = psgd)

    _, _ = query.get_proposal_products()

    query.download_files()


def test_coron_pipeline():

    instrume: INSTRUME = 'NIRCAM'
    proposal_id: str = '1441'

    coron = CoronPipeline(proposal_id=proposal_id, instrume=instrume)

    coron.pipeline_stage2()
    coron.pipeline_stage3()

if __name__ == '__main__':
    test_query()
    test_coron_pipeline()
    