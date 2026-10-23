# AI-Boosted-Coronagraphy


Coronagraphy asks us to find faint signals beside stars bright enough to drown them out. We use AI methods and JWST coronagraphic data workflows to explore that contrast problem. This guide covers the reference files and MAST access needed by the scripts.

## Environment setup

Set `AI_BOOSTED_CORONAGRAPHY_PATH` to the checkout directory when the repository is not imported from its default location. If it is unset, the utilities use the checkout containing `src/utils/variable_utils.py`. Dataset inputs live under the ignored `data/` directory, and generated figures belong under the ignored `visuals/` directory.

Most JWST workflows need files from the Calibration Reference Data System (CRDS). These settings tell JWST where to cache them and which server to query. Add them to your Bash startup file if you want them in new shells, or export them directly in the shell you are using.

```bash
mkdir -p "$HOME/crds_cache/jwst_ops"
export CRDS_PATH="$HOME/crds_cache/jwst_ops"
export CRDS_SERVER_URL="https://jwst-crds.stsci.edu/"
```

To keep these settings in Bash, add the exports to `~/.bashrc` and reload it with `source ~/.bashrc`. `scripts/coron_pipeline.py` respects existing `CRDS_PATH` and `CRDS_SERVER_URL` values, and otherwise uses the defaults above.

## MAST access

Some scripts query the Mikulski Archive for Space Telescopes (MAST). They read your token from `src/secrets/mast_token.txt`. Keep the token out of source code and never commit it. The `src/secrets/` directory is ignored by Git. Create the file and add your token as its first line:

```bash
mkdir -p src/secrets
touch src/secrets/mast_token.txt
```

For MAST authentication details, see the [MAST authentication website](https://auth.stsci.edu/info).