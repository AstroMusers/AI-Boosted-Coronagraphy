from typing import Literal, List, TypedDict, Any

EXP_TYPE = Literal['NRC_CORON', 'MIR_IMAGE']
INSTRUME = Literal['NIRCAM', 'MIRI']
SERVICE = Literal['Mast.Jwst.Filtered.Nircam', 'Mast.Jwst.Filtered.Miri']

ProductSubGroupDescription = Literal['UNCAL', 'RATEINTS', 'CALINTS', 'ASN']
PSGD = List[ProductSubGroupDescription]

class query_keywords(TypedDict):
    exp_type: List[EXP_TYPE]
    instrume: List[INSTRUME]
    date_obs_mjd: Any

class mast_request_params(TypedDict):
    columns: Any
    filters: query_keywords

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__