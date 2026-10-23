import os
from pathlib import Path

INSTRUME = "NIRCAM"
PATH_ENV_VAR = "AI_BOOSTED_CORONAGRAPHY_PATH"
DEFAULT_REPOSITORY_PATH = Path(__file__).resolve().parents[2]


def get_repository_path():
	path_value = os.environ.get(PATH_ENV_VAR)
	if not path_value or not path_value.strip():
		path_value = str(DEFAULT_REPOSITORY_PATH)
		os.environ[PATH_ENV_VAR] = path_value
	return Path(path_value).expanduser().resolve()


def get_data_path():
	return get_repository_path() / "data"


def get_visuals_path():
	return get_repository_path() / "visuals"


REPOSITORY_PATH = get_repository_path()
DATA_MAIN_DIR = str(get_data_path()) + os.sep

TRAIN_DIR = os.path.join(DATA_MAIN_DIR, 'train')
TEST_DIR  = os.path.join(DATA_MAIN_DIR, 'test')

NIRCAM_DATA = str(get_data_path() / "NIRCAM") + os.sep

PLOT_SAVE_PATH = str(get_visuals_path()) + os.sep