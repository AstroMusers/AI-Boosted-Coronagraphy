import logging
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.associations.registry import RegistryMarker
from jwst.associations.lib.dms_base import (Constraint_TargetAcq, Constraint_TSO, nrsfss_valid_detector, nrsifu_valid_detector)
from jwst.associations.lib.process_list import ListCategory
from jwst.associations.lib.rules_level3_base import *
from jwst.associations.lib.rules_level3_base import (
    dms_product_name_sources, dms_product_name_noopt,
    format_product
)

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


__all__ = [
    'Asn_Lv3Coron',
    'Asn_Lv3ACQ_Reprocess',
]

@RegistryMarker.rule
class Asn_Lv3Coron(AsnMixin_Science):
    """Level 3 Coronagraphy Association
    Characteristics:
        - Association type: ``coron3``
        - Pipeline: ``calwebb_coron3``
        - Gather science and related PSF exposures
    Notes
    -----
    Coronagraphy is nearly completely defined by the association candidates
    produced by APT.
    Tracking Issues:
        - `github #311 <https://github.com/STScI-JWST/jwst/issues/311>`_
    """

    def __init__(self, *args, **kwargs):

        # Setup for checking.
        self.constraints = Constraint(
            [
                Constraint_Optical_Path(),
                DMSAttrConstraint(
                    name='exp_type',
                    sources=['exp_type'],
                    value=(
                        'nrc_coron'
                        '|mir_lyot'
                        '|mir_4qpm'
                    ),
                ),
                DMSAttrConstraint(
                    name='target',
                    sources=['targetid'],
                    onlyif=lambda item: self.get_exposure_type(item) == 'science',
                    force_reprocess=ListCategory.EXISTING,
                    only_on_match=True,
                ),
                Constraint(
                    [DMSAttrConstraint(
                        name='bkgdtarg',
                        sources=['bkgdtarg'],
                        force_unique=False,
                    )],
                    reduce=Constraint.notany
                ),
            ],
            name='asn_coron'
        )

        # PSF is required
        self.validity.update({
            'has_psf': {
                'validated': False,
                'check': lambda entry: entry['exptype'] == 'psf'
            }
        })

        # Check and continue initialization.
        super(Asn_Lv3Coron, self).__init__(*args, **kwargs)

    def _init_hook(self, item):
        """Post-check and pre-add initialization"""

        self.data['asn_type'] = 'coron3'
        super(Asn_Lv3Coron, self)._init_hook(item)

        
@RegistryMarker.rule
class Asn_Lv3ACQ_Reprocess(DMS_Level3_Base):
    """Level 3 Gather Target Acquisitions
    Characteristics:
        - Association type: Not applicable
        - Pipeline: Not applicable
        - Used to populate other related associations
    Notes
    -----
    For first loop, simply send acquisitions and confirms back.
    """

    def __init__(self, *args, **kwargs):

        # Setup for checking.
        self.constraints = Constraint([
            Constraint_TargetAcq(),
            SimpleConstraint(
                name='force_fail',
                test=lambda x, y: False,
                value='anything but None',
                reprocess_on_fail=True,
                work_over=ListCategory.NONSCIENCE,
                reprocess_rules=[]
            )
        ])

        super(Asn_Lv3ACQ_Reprocess, self).__init__(*args, **kwargs)