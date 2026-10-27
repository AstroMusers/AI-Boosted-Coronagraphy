# AI-Boosted-Coronagraphy


Coronagraphy asks us to find faint signals beside stars bright enough to drown them out. We use AI methods with JWST coronagraphic data to explore that contrast problem.

## Environment setup

Install the dependencies listed in `requirements.txt` with `python -m pip install -r requirements.txt`. The pinned TDpy dependency provides shared repository-path helpers.

Set `AI_BOOSTED_CORONAGRAPHY_PATH` to the checkout directory when the repository is not imported from its default location. If it is unset, the utilities use the checkout containing `src/utils/variable_utils.py`. Dataset inputs live under the ignored `data/` directory, and generated figures belong under the ignored `visuals/` directory.

The injection scripts use `data/webbpsf-data/` as the default WebbPSF cache. Set `WEBBPSF_PATH` to use a different WebbPSF reference-data directory. Pipeline progress and file-path messages are silent by default. Set `TDPY_VERBOSITY=1` to display them.

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

## Processing workflows

- `scripts/query.py` queries MAST for a proposal's coronagraph products and downloads them below `data/`.
- `scripts/coron_pipeline.py` runs JWST `Image2Pipeline` on rateints products, builds level-3 associations from calints products, and runs `Coron3Pipeline`.
- `scripts/injection.py` and `scripts/gpu_injection.py` create CPU- and GPU-based PSF injection datasets from local PSF stacks and WebbPSF models. Array and tensor products are written below `data/`.
- `scripts/train_classifier.py` selects injected and non-injected arrays for `src/models/exo_classifier.py`. `scripts/test_classifier.py` evaluates a saved classifier and writes plots and a CSV under `visuals/`.
- `scripts/test_SSL.py` prepares FITS-derived image tensors and contains a ViT-MAE pretraining workflow with repository-local data and checkpoint paths.

The notebooks in [`notebooks/`](notebooks/) support focused exploration of base and injected PSF products, MAE reconstruction, and latent-space plots. The scripts provide the main workflow entry points.

## Data and validation limits

The JWST workflows depend on local archive products, CRDS reference files, WebbPSF data, and MAST credentials. GPU scripts also require a compatible CUDA setup. The unit tests cover mocked components and synthetic helper inputs. Full archive reductions, injection-recovery rates, and classifier performance require evaluation with the relevant data and runtime setup. Interpret each output with the input products and run configuration that produced it.

## Interactive feature correlations

The numeric feature columns can also be compared with an interactive Pearson-correlation matrix. The Bokeh version uses the same pandas correlation calculation and writes a standalone HTML figure:

```python
from src.utils import plot_correlation_matrix_bokeh

plot_correlation_matrix_bokeh(features, "visuals/feature_correlation.html")
```

Hover over a cell to inspect its feature pair and Pearson coefficient. Each
coefficient summarizes the linear association between two numeric features.
