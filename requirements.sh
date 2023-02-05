#!/bin/bash

##############################
# This bash script coded for #
# downloading the required   #
# packages for jwst pipeline #
# and creates the necessary  #
# folders.                   #
##############################

# Downloads the requirement text.
pip install -r requirements.txt

# Downloads the required packages latest version.
pip install --upgrade numpy
pip install --upgrade git+https://github.com/spacetelescope/jwst
pip uninstall --yes crds
pip install --upgrade  git+https://github.com/spacetelescope/crds.git#egg=crds["submission","dev","test","docs"]


# Creates cache folder for crds.
mkdir /home/bariskurtkaya/crds_cache
mkdir /home/bariskurtkaya/crds_cache/jwst_ops

# Creates the secrets folder and mask_token text.
mkdir secrets
touch mast_token.txt