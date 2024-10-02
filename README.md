# AI-Boosted-Coronagraphy


# Introduction

## Requirements

## Environment Configurations

To use scripts in this repository, you must add the following environment variables in bashrc file.

To open your environment.
`$ nano ~/.bashrc`

Environment variables that you need to add in bashrc.
`export CRDS_PATH=$HOME/crds_cache/jwst_ops`
`export CRDS_SERVER_URL=https://jwst-crds.stsci.edu/`

Reinitialization of environment variables.
`$ source ~/.bashrc`


## Folders Configurations

### CRDS and JWST folder

To use pipeline you need to initialize the command line codes in section below.

`$ mkdir /home/bariskurtkaya/crds_cache`
`$ mkdir /home/bariskurtkaya/crds_cache/jwst_ops`

### Secrets Folder

Due to security of api tokens, this repo have scripts that gets the api tokens from secrets folder instead collecting in script. Thus, the steps must be done before activating the scripts which shown below.

1. Create a secrets folder in main repository directory. 
`$ mkdir secrets`

2. Create txt files, which are mandatory, in secrets folder. 
`$ touch mast_token.txt`

3. Then you must add your token without any addition in mast_token txt file

Further information for mast token: [Mast Authentication Website](https://auth.mast.stsci.edu/info "Mast Authentication Website")