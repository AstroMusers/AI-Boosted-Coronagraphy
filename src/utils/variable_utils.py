import os
from pathlib import Path

from tdpy.paths import RepositoryPaths

INSTRUME = "NIRCAM"
PATH_ENV_VAR = "AI_BOOSTED_CORONAGRAPHY_PATH"
DEFAULT_REPOSITORY_PATH = Path(__file__).resolve().parents[2]

_REPOSITORY_PATHS = RepositoryPaths(PATH_ENV_VAR, DEFAULT_REPOSITORY_PATH)
get_repository_path = _REPOSITORY_PATHS.get_repository_path
get_data_path = _REPOSITORY_PATHS.get_data_path
get_visuals_path = _REPOSITORY_PATHS.get_visuals_path


REPOSITORY_PATH = get_repository_path()
DATA_MAIN_DIR = str(get_data_path()) + os.sep

TRAIN_DIR = os.path.join(DATA_MAIN_DIR, 'train')
TEST_DIR  = os.path.join(DATA_MAIN_DIR, 'test')

NIRCAM_DATA = str(get_data_path() / "NIRCAM") + os.sep

PLOT_SAVE_PATH = str(get_visuals_path()) + os.sep