import importlib.util
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VARIABLE_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "variable_utils.py"


def load_variable_utils():
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_variable_utils", VARIABLE_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_repository_path_defaults_to_checkout(monkeypatch):
    monkeypatch.delenv("AI_BOOSTED_CORONAGRAPHY_PATH", raising=False)
    paths = load_variable_utils()

    assert os.environ["AI_BOOSTED_CORONAGRAPHY_PATH"] == str(REPOSITORY_ROOT)
    assert paths.get_repository_path() == REPOSITORY_ROOT
    assert paths.get_data_path() == REPOSITORY_ROOT / "data"
    assert paths.get_visuals_path() == REPOSITORY_ROOT / "visuals"


def test_repository_path_honors_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_BOOSTED_CORONAGRAPHY_PATH", str(tmp_path))
    paths = load_variable_utils()

    assert paths.get_repository_path() == tmp_path.resolve()


def test_public_utils_export_repository_paths():
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from src.utils import get_data_path, get_repository_path, get_visuals_path

    assert get_data_path() == get_repository_path() / "data"
    assert get_visuals_path() == get_repository_path() / "visuals"