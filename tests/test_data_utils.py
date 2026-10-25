import importlib.util
from pathlib import Path

from tdpy.verbosity import print


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "data_utils.py"
CONFIG_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "config_utils.py"
VIZ_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "viz_utils.py"
MODEL_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "model_utils.py"
LOGGING_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "logging_utils.py"
CORON_PIPELINE_PATH = REPOSITORY_ROOT / "scripts" / "coron_pipeline.py"
QUERY_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "query.py"
INJECTION_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "injection.py"
GPU_INJECTION_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "gpu_injection.py"
EXO_CLASSIFIER_PATH = REPOSITORY_ROOT / "src" / "models" / "exo_classifier.py"
DATALOADERS_PATH = REPOSITORY_ROOT / "src" / "data" / "dataloaders.py"
TRAIN_CLASSIFIER_PATH = REPOSITORY_ROOT / "scripts" / "train_classifier.py"
TRAIN_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "train.py"
TEST_CLASSIFIER_PATH = REPOSITORY_ROOT / "scripts" / "test_classifier.py"
TEST_SSL_PATH = REPOSITORY_ROOT / "scripts" / "test_SSL.py"
TIMING_UTILS_PATH = REPOSITORY_ROOT / "src" / "utils" / "time.py"


def load_data_utils():
    print(f"Reading from {DATA_UTILS_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_data_utils", DATA_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_config_utils():
    print(f"Reading from {CONFIG_UTILS_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_config_utils", CONFIG_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_viz_utils():
    print(f"Reading from {VIZ_UTILS_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_viz_utils", VIZ_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_model_utils():
    print(f"Reading from {MODEL_UTILS_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_model_utils", MODEL_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_logging_utils():
    print(f"Reading from {LOGGING_UTILS_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_logging_utils", LOGGING_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_coron_pipeline(monkeypatch):
    import sys
    import types

    print(f"Reading from {CORON_PIPELINE_PATH}...")
    module_names = (
        "jwst",
        "jwst.pipeline",
        "jwst.pipeline.calwebb_coron3",
        "jwst.associations",
        "jwst.associations.lib",
        "jwst.associations.lib.rules_level3",
        "jwst.associations.mkpool",
    )
    stubs = {name: types.ModuleType(name) for name in module_names}
    stubs["jwst"].__version__ = "test"
    stubs["jwst.pipeline"].Image2Pipeline = object
    stubs["jwst.pipeline.calwebb_coron3"].Coron3Pipeline = object
    stubs["jwst.associations"].AssociationRegistry = object
    stubs["jwst.associations"].generate = object
    stubs["jwst.associations.lib.rules_level3"].Asn_Lv3Coron = object
    stubs["jwst.associations.mkpool"].mkpool = object
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)

    utils = types.ModuleType("src.utils")
    utils.get_dataset_dir = lambda: str(REPOSITORY_ROOT / "data")
    utils.INSTRUME = str
    utils.List = list
    source = types.ModuleType("src")
    source.__path__ = []
    source.utils = utils
    monkeypatch.setitem(sys.modules, "src", source)
    monkeypatch.setitem(sys.modules, "src.utils", utils)

    specification = importlib.util.spec_from_file_location(
        "ai_boosted_coron_pipeline", CORON_PIPELINE_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_query_script(monkeypatch):
    import sys
    import types

    print(f"Reading from {QUERY_SCRIPT_PATH}...")
    jwst = types.ModuleType("jwst")
    jwst.__version__ = "test"
    astroquery = types.ModuleType("astroquery")
    astroquery.__path__ = []
    mast = types.ModuleType("astroquery.mast")
    mast.Observations = types.SimpleNamespace(
        login=lambda token: None,
        download_products=lambda products, download_dir: (products, download_dir),
    )
    utils = types.ModuleType("src.utils")
    utils.get_mast_token = lambda: "unused-test-token"
    utils.get_util_main_dir = lambda: "unused"
    utils.get_main_dir = lambda: "unused"
    utils.get_dataset_dir = lambda: "unused-data"
    utils.List = list
    source = types.ModuleType("src")
    source.__path__ = []
    source.utils = utils
    for name, stub in (
        ("jwst", jwst),
        ("astroquery", astroquery),
        ("astroquery.mast", mast),
        ("src", source),
        ("src.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, stub)

    specification = importlib.util.spec_from_file_location(
        "ai_boosted_query_script", QUERY_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_injection_script(monkeypatch):
    import sys
    import types
    import warnings

    print(f"Reading from {INJECTION_SCRIPT_PATH}...")
    module_names = (
        "webbpsf",
        "astroquery",
        "astroquery.gaia",
        "astroquery.ipac",
        "astroquery.ipac.nexsci",
        "astroquery.ipac.nexsci.nasa_exoplanet_archive",
    )
    stubs = {name: types.ModuleType(name) for name in module_names}
    stubs["astroquery"].__path__ = []
    stubs["astroquery.ipac"].__path__ = []
    stubs["astroquery.ipac.nexsci"].__path__ = []
    stubs["astroquery.gaia"].Gaia = object()
    stubs["astroquery.ipac.nexsci.nasa_exoplanet_archive"].NasaExoplanetArchive = object()
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)

    utils = types.ModuleType("src.utils")
    utils.get_filename_from_dir = lambda path: Path(path).stem
    utils.get_dataset_dir = lambda: str(REPOSITORY_ROOT / "data")
    utils.get_stage3_products = lambda **kwargs: []
    utils.Augmentation = object
    utils.List = list
    source = types.ModuleType("src")
    source.__path__ = []
    source.utils = utils
    monkeypatch.setitem(sys.modules, "src", source)
    monkeypatch.setitem(sys.modules, "src.utils", utils)
    monkeypatch.setattr(warnings, "simplefilter", lambda *args, **kwargs: None)

    specification = importlib.util.spec_from_file_location(
        "ai_boosted_injection_script", INJECTION_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_gpu_injection_script(monkeypatch):
    import sys
    import types
    import warnings

    load_injection_script(monkeypatch)
    print(f"Reading from {GPU_INJECTION_SCRIPT_PATH}...")
    monkeypatch.setattr(warnings, "simplefilter", lambda *args, **kwargs: None)

    class RandomCrop:
        pass

    torchvision = types.ModuleType("torchvision")
    torchvision.__path__ = []
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__path__ = []
    v2 = types.ModuleType("torchvision.transforms.v2")
    v2.__path__ = []
    v2.RandomCrop = RandomCrop
    v2.CenterCrop = object
    transforms.v2 = v2
    v2_utils = types.ModuleType("torchvision.transforms.v2._utils")
    v2_utils.query_size = lambda *args, **kwargs: None
    for name, stub in (
        ("torchvision", torchvision),
        ("torchvision.transforms", transforms),
        ("torchvision.transforms.v2", v2),
        ("torchvision.transforms.v2._utils", v2_utils),
    ):
        monkeypatch.setitem(sys.modules, name, stub)

    seed = types.ModuleType("src.utils.seed")
    seed.seed_everything = lambda *args, **kwargs: None
    timing = types.ModuleType("src.utils.time")
    timing.timing = lambda function: function
    monkeypatch.setitem(sys.modules, "src.utils.seed", seed)
    monkeypatch.setitem(sys.modules, "src.utils.time", timing)

    specification = importlib.util.spec_from_file_location(
        "ai_boosted_gpu_injection_script", GPU_INJECTION_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_exo_classifier(monkeypatch):
    import sys
    import types

    print(f"Reading from {EXO_CLASSIFIER_PATH}...")
    source = types.ModuleType("src")
    source.__path__ = []
    models = types.ModuleType("src.models")
    models.__path__ = []
    utils = types.ModuleType("src.utils")
    utils.__path__ = []
    data_utils = types.ModuleType("src.utils.data_utils")
    variable_utils = types.ModuleType("src.utils.variable_utils")
    variable_utils.PLOT_SAVE_PATH = "unused"
    viz_utils = types.ModuleType("src.utils.viz_utils")
    viz_utils.save_dict_as_yaml = lambda *args, **kwargs: None
    viz_utils.plot_inputs_classifier = lambda *args, **kwargs: None
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = object()
    for name, stub in (
        ("src", source),
        ("src.models", models),
        ("src.utils", utils),
        ("src.utils.data_utils", data_utils),
        ("src.utils.variable_utils", variable_utils),
        ("src.utils.viz_utils", viz_utils),
        ("torchvision", torchvision),
    ):
        monkeypatch.setitem(sys.modules, name, stub)

    specification = importlib.util.spec_from_file_location(
        "src.models.exo_classifier", EXO_CLASSIFIER_PATH
    )
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, "src.models.exo_classifier", module)
    specification.loader.exec_module(module)
    return module


def load_dataloaders(monkeypatch, dataset_root=None):
    import sys
    import types
    import numpy as np
    import torch

    print(f"Reading from {DATALOADERS_PATH}...")

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, values):
            for transform in self.transforms:
                values = transform(values)
            return values

    class ToTensor:
        def __call__(self, values):
            return torch.as_tensor(np.asarray(values)).unsqueeze(0)

    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = types.SimpleNamespace(Compose=Compose, ToTensor=ToTensor)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    source = types.ModuleType("src")
    source.__path__ = []
    utils = types.ModuleType("src.utils")
    utils.get_dataset_dir = lambda: str(
        dataset_root or REPOSITORY_ROOT / "data"
    )
    source.utils = utils
    monkeypatch.setitem(sys.modules, "src", source)
    monkeypatch.setitem(sys.modules, "src.utils", utils)
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_dataloaders", DATALOADERS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_train_classifier(monkeypatch):
    import sys
    import types

    print(f"Reading from {TRAIN_CLASSIFIER_PATH}...")
    source = types.ModuleType("src")
    source.__path__ = []
    utils = types.ModuleType("src.utils")
    utils.__path__ = []
    data_utils = types.ModuleType("src.utils.data_utils")
    model_utils = types.ModuleType("src.utils.model_utils")
    variable_utils = types.ModuleType("src.utils.variable_utils")
    variable_utils.NIRCAM_DATA = "/mock/NIRCAM"
    viz_utils = types.ModuleType("src.utils.viz_utils")
    data_package = types.ModuleType("src.data")
    data_package.__path__ = []
    dataloaders = types.ModuleType("src.data.dataloaders")
    dataloaders.SynDatasetLabel = object
    models = types.ModuleType("src.models")
    models.__path__ = []
    classifier = types.ModuleType("src.models.exo_classifier")
    classifier.ExoClassifier = object
    module_stubs = (
        ("src", source),
        ("src.utils", utils),
        ("src.utils.data_utils", data_utils),
        ("src.utils.model_utils", model_utils),
        ("src.utils.variable_utils", variable_utils),
        ("src.utils.viz_utils", viz_utils),
        ("src.data", data_package),
        ("src.data.dataloaders", dataloaders),
        ("src.models", models),
        ("src.models.exo_classifier", classifier),
    )
    for name, stub in module_stubs:
        monkeypatch.setitem(sys.modules, name, stub)

    specification = importlib.util.spec_from_file_location(
        "ai_boosted_train_classifier", TRAIN_CLASSIFIER_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_mlp_training_script(monkeypatch, dataset_root):
    import sys
    import types

    print(f"Reading from {TRAIN_SCRIPT_PATH}...")
    source = types.ModuleType("src")
    source.__path__ = []
    data_package = types.ModuleType("src.data")
    data_package.__path__ = []
    data_loader = types.ModuleType("src.data.dataloaders")

    class SyntheticDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

    data_loader.SynthDatasetv2 = SyntheticDataset
    models = types.ModuleType("src.models")
    models.__path__ = []
    mlp = types.ModuleType("src.models.mlp_network")

    class MLPNetwork:
        def parameters(self):
            return []

    mlp.MLPNetwork = MLPNetwork
    utils = types.ModuleType("src.utils")
    utils.__path__ = []
    utils.get_dataset_dir = lambda: str(dataset_root)
    seed = types.ModuleType("src.utils.seed")
    seed.seed_everything = lambda value: None
    time_utils = types.ModuleType("src.utils.time")
    time_utils.timing = lambda function: function

    class Accelerator:
        device = "cpu"
        num_processes = 1

        def prepare(self, *items):
            return items

        def backward(self, loss):
            pass

    accelerate = types.ModuleType("accelerate")
    accelerate.Accelerator = Accelerator
    for name, stub in (
        ("src", source),
        ("src.data", data_package),
        ("src.data.dataloaders", data_loader),
        ("src.models", models),
        ("src.models.mlp_network", mlp),
        ("src.utils", utils),
        ("src.utils.seed", seed),
        ("src.utils.time", time_utils),
        ("accelerate", accelerate),
    ):
        monkeypatch.setitem(sys.modules, name, stub)

    specification = importlib.util.spec_from_file_location(
        "ai_boosted_mlp_training", TRAIN_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_csv_helpers_are_quiet_by_default_and_narrate_paths(monkeypatch, capsys):
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    data_utils = load_data_utils()
    read_paths = []
    write_calls = []
    expected_data = object()

    monkeypatch.setattr(
        data_utils.pd, "read_csv", lambda path: read_paths.append(path) or expected_data
    )

    class DataFrame:
        def to_csv(self, path, index):
            write_calls.append((path, index))

    assert data_utils.load_csv("catalog.csv") is expected_data
    data_utils.save_csv(DataFrame(), "summary.csv")
    data_utils.Augmentation()

    assert capsys.readouterr().out == ""
    assert read_paths == ["catalog.csv"]
    assert write_calls == [("summary.csv", False)]

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    data_utils.load_csv("catalog.csv")
    data_utils.save_csv(DataFrame(), "summary.csv")

    assert capsys.readouterr().out == (
        "Reading from catalog.csv...\n"
        "Writing to summary.csv...\n"
    )


def test_augmentation_normalize_sanity_checks_are_optional(monkeypatch):
    import numpy as np

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    augmentation = load_data_utils().Augmentation()
    values = np.array([2.0, 4.0])

    np.testing.assert_allclose(
        augmentation.normalize(values, sanity_checks=True), [0.0, 1.0]
    )
    np.testing.assert_allclose(augmentation.normalize(values), [0.0, 1.0])
    try:
        augmentation.normalize(np.ones(2), sanity_checks=True)
    except ValueError as error:
        assert "nonzero range" in str(error)
    else:
        raise AssertionError("Expected constant image to fail sanity checks")
    try:
        augmentation.normalize(np.array([1.0, np.nan]), sanity_checks=True)
    except ValueError as error:
        assert "finite values" in str(error)
    else:
        raise AssertionError("Expected non-finite image to fail sanity checks")


def test_cpu_injection_forwards_optional_sanity_checks():
    import ast

    print(f"Reading from {INJECTION_SCRIPT_PATH}...")
    source = INJECTION_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INJECTION_SCRIPT_PATH))
    injection_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Injection"
    )
    methods = {
        node.name: node for node in injection_class.body
        if isinstance(node, ast.FunctionDef)
    }

    injection_calls = [
        node for node in ast.walk(methods["apply_injection"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__injection"
    ]
    assert len(injection_calls) == 2
    assert all(
        any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in call.keywords
        )
        for call in injection_calls
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__nan_elimination"
        and any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(methods["apply_injection"])
    )
    for method in (methods["apply_injection"], methods["__injection"]):
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "normalize"
            and any(
                keyword.arg == "sanity_checks"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "sanity_checks"
                for keyword in node.keywords
            )
            for node in ast.walk(method)
        )


def test_cpu_injection_checks_input_psf_before_nan_elimination(monkeypatch):
    import numpy as np

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_injection_script(monkeypatch)
    injection = injection_module.Injection.__new__(injection_module.Injection)
    invalid_psf = np.array([[1.0, np.nan]])

    try:
        injection._Injection__nan_elimination(invalid_psf, sanity_checks=True)
    except ValueError as error:
        assert "finite values" in str(error)
    else:
        raise AssertionError("Expected non-finite input PSF to fail sanity checks")

    np.testing.assert_array_equal(
        injection._Injection__nan_elimination(invalid_psf), [[1.0, 0.0]]
    )

    constant_psf = np.ones((2, 2))
    try:
        injection._Injection__nan_elimination(constant_psf, sanity_checks=True)
    except ValueError as error:
        assert "nonzero range" in str(error)
    else:
        raise AssertionError("Expected constant input PSF to fail sanity checks")

    np.testing.assert_array_equal(
        injection._Injection__nan_elimination(constant_psf), constant_psf
    )


def test_cpu_injection_checks_generated_psf_when_enabled(monkeypatch):
    import numpy as np

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_injection_script(monkeypatch)
    injection = injection_module.Injection.__new__(injection_module.Injection)
    injection.augmentation = load_data_utils().Augmentation()
    injection.flux_coef = (0.1, 0.01)
    injection.is_save_injected = False

    injection._Injection__injection(
        psf=np.ones((4, 4)),
        generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
        max_pixel_distance=4,
        min_pixel_distance=1,
        max_pix=8,
        num_rand_flux=10,
        injection_count=1,
        sanity_checks=True,
    )

    sample_flux = injection._Injection__sample_flux
    try:
        injection._Injection__sample_flux = lambda _: float(np.float32(0.1))
        injection._Injection__injection(
            psf=np.ones((4, 4)),
            generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            max_pixel_distance=4,
            min_pixel_distance=1,
            max_pix=8,
            num_rand_flux=10,
            injection_count=1,
            sanity_checks=True,
        )
    finally:
        injection._Injection__sample_flux = sample_flux

    try:
        injection._Injection__injection(
            psf=np.ones((4, 4)),
            generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            max_pixel_distance=4,
            min_pixel_distance=1,
            max_pix=8,
            num_rand_flux=0,
            injection_count=1,
            sanity_checks=True,
        )
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("Expected invalid CPU flux sample count to be rejected")

    sample_flux = injection._Injection__sample_flux
    injection._Injection__sample_flux = lambda _: float("inf")
    try:
        injection._Injection__injection(
            psf=np.ones((4, 4)),
            generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            max_pixel_distance=4,
            min_pixel_distance=1,
            max_pix=8,
            num_rand_flux=10,
            injection_count=1,
            sanity_checks=True,
        )
    except ValueError as error:
        assert "sampled flux" in str(error)
    else:
        raise AssertionError("Expected invalid sampled flux to be rejected")
    finally:
        injection._Injection__sample_flux = sample_flux

    injection.flux_coef = (0.01, 0.1)
    try:
        injection._Injection__injection(
            psf=np.ones((4, 4)),
            generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            max_pixel_distance=4,
            min_pixel_distance=1,
            max_pix=8,
            injection_count=1,
            sanity_checks=True,
        )
    except ValueError as error:
        assert "maximum, minimum" in str(error)
    else:
        raise AssertionError("Expected reversed CPU flux bounds to be rejected")

    injection._Injection__injection(
        psf=np.ones((4, 4)),
        generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
        max_pixel_distance=4,
        min_pixel_distance=1,
        max_pix=8,
        injection_count=1,
    )

    injection.flux_coef = (0.1, 0.01)
    try:
        injection._Injection__injection(
            psf=np.ones((4, 4)),
            generated_psf=np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            max_pixel_distance=5,
            min_pixel_distance=4,
            max_pix=8,
            injection_count=1,
            sanity_checks=True,
        )
    except ValueError as error:
        assert "nonempty" in str(error)
    else:
        raise AssertionError("Expected an empty CPU placement interval to be rejected")

    try:
        injection._Injection__injection(
            psf=np.array([[1.0]]),
            generated_psf=np.ones((2, 2)),
            max_pixel_distance=1,
            min_pixel_distance=1,
            max_pix=4,
            injection_count=1,
            sanity_checks=True,
        )
    except ValueError as error:
        assert "nonzero range" in str(error)
    else:
        raise AssertionError("Expected generated PSF to fail normalization checks")

    try:
        injection._Injection__injection(
            psf=np.array([[1.0]]),
            generated_psf=np.array([[0.0, 1.0], [2.0, 3.0]]),
            max_pixel_distance=1,
            min_pixel_distance=1,
            max_pix=4,
            injection_count=2,
            sanity_checks=True,
        )
    except ValueError as error:
        assert "one source per PSF" in str(error)
    else:
        raise AssertionError("Expected unsupported injection count to be rejected")


def test_config_loader_is_quiet_by_default_and_narrates_input(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    config_utils = load_config_utils()
    config_path = tmp_path / "config.yaml"
    print(f"Writing to {config_path}...")
    config_path.write_text("target: TOI-1233\n", encoding="utf-8")

    assert config_utils.load_config(config_path) == {"target": "TOI-1233"}
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert config_utils.load_config(config_path) == {"target": "TOI-1233"}
    assert capsys.readouterr().out == f"Reading from {config_path}...\n"


def test_visualization_helpers_narrate_writes_and_default_to_quiet(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    monkeypatch.chdir(tmp_path)
    viz_utils = load_viz_utils()
    viz_utils.save_dict_as_yaml({"target": "TOI-1233"}, "settings.yaml")
    viz_utils.get_test_dirs(["first", "second"])
    assert viz_utils.calculate_conv_dims(4, [0], [3], [1], [0]) == [4, 2]
    assert viz_utils.calculate_convtrans_dim(4, [0], [3], [1]) == [4, 6]

    print("Reading from settings.yaml...")
    assert viz_utils.yaml.safe_load(Path("settings.yaml").read_text()) == {
        "target": "TOI-1233"
    }
    print("Reading from text_dirs.txt...")
    assert Path("text_dirs.txt").read_text().splitlines() == ["first", "second"]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    viz_utils.save_dict_as_yaml({"target": "TOI-1233"}, "settings.yaml")
    viz_utils.get_test_dirs(["third"])
    viz_utils.calculate_conv_dims(4, [0], [3], [1], [0])
    viz_utils.calculate_convtrans_dim(4, [0], [3], [1])
    assert capsys.readouterr().out == (
        "Writing to settings.yaml...\n"
        "Writing to text_dirs.txt...\n"
        "Done\n"
        "[4, 2]\n"
        "[4, 6]\n"
    )


def test_bokeh_correlation_plot_writes_interactive_html(tmp_path, monkeypatch, capsys):
    import pandas as pd

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    viz_utils = load_viz_utils()
    data = pd.DataFrame(
        {"flux": [1.0, 2.0, 3.0], "radius": [3.0, 2.0, 1.0], "noise": [1.0, 1.0, 2.0]}
    )
    quiet_path = tmp_path / "quiet_correlation.html"
    viz_utils.plot_correlation_matrix_bokeh(data, quiet_path)
    assert capsys.readouterr().out == ""
    print(f"Reading from {quiet_path}...")
    html = quiet_path.read_text(encoding="utf-8")
    assert "Feature correlation (Pearson r)" in html
    assert "flux" in html and "radius" in html and "noise" in html
    assert "Feature pair" in html and "Pearson r" in html

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    verbose_path = tmp_path / "verbose_correlation.html"
    viz_utils.plot_correlation_matrix_bokeh(data, verbose_path)
    assert capsys.readouterr().out == f"Writing to {verbose_path}...\n"


def test_classifier_input_plot_logs_and_writes_jpeg(tmp_path, monkeypatch, capsys):
    import numpy as np
    from PIL import Image

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    viz_utils = load_viz_utils()
    quiet_directory = tmp_path / "quiet"
    quiet_directory.mkdir()
    image_data = np.zeros((8, 3, 16, 16))
    viz_utils.plot_inputs_classifier(image_data, quiet_directory, 1, 2)
    assert capsys.readouterr().out == ""

    output_path = quiet_directory / "fig_1_2.jpg"
    print(f"Reading from {output_path}...")
    with Image.open(output_path) as image:
        assert image.format == "JPEG"
        assert image.width > 0 and image.height > 0

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    verbose_directory = tmp_path / "verbose"
    verbose_directory.mkdir()
    viz_utils.plot_inputs_classifier(image_data, verbose_directory, 1, 2)
    assert capsys.readouterr().out == (
        f"Writing to {verbose_directory / 'fig_1_2.jpg'}...\n"
    )


def test_mast_token_reader_narrates_path_without_exposing_token(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    from src.utils import query_utils

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    monkeypatch.setattr(query_utils, "get_main_dir", lambda: str(tmp_path))
    secret_path = tmp_path / "src" / "secrets" / "mast_token.txt"
    secret_path.parent.mkdir(parents=True)
    print(f"Writing to {secret_path}...")
    secret_path.write_text("temporary-test-token\n", encoding="utf-8")

    assert query_utils.get_mast_token() == "temporary-test-token\n"
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert query_utils.get_mast_token() == "temporary-test-token\n"
    assert capsys.readouterr().out == f"Reading from {secret_path}...\n"


def test_stage3_product_discovery_narrates_search_pattern(
    tmp_path, monkeypatch, capsys
):
    import os

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    from src.utils import query_utils

    expected_pattern = os.path.join(str(tmp_path), "*psfstack.fits")
    expected_files = [str(tmp_path / "one_psfstack.fits")]
    searched = []

    def fake_glob(pattern):
        searched.append(pattern)
        return expected_files

    monkeypatch.setattr(query_utils.glob, "glob", fake_glob)
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert query_utils.get_stage3_products("psfstack", str(tmp_path)) == expected_files
    assert searched == [expected_pattern]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert query_utils.get_stage3_products("psfstack", str(tmp_path)) == expected_files
    assert searched == [expected_pattern, expected_pattern]
    assert capsys.readouterr().out == f"Reading from {expected_pattern}...\n"


def test_model_checkpoint_io_narrates_paths_and_round_trips_tensor(
    tmp_path, monkeypatch, capsys
):
    import torch

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    model_utils = load_model_utils()
    state = {"weights": torch.tensor([1.0, 2.0])}
    quiet_path = tmp_path / "quiet.pt"
    model_utils.save_model(state, quiet_path)
    quiet_result = model_utils.load_model(quiet_path)

    assert torch.equal(quiet_result["weights"], state["weights"])
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    verbose_path = tmp_path / "verbose.pt"
    model_utils.save_model(state, verbose_path)
    verbose_result = model_utils.load_model(verbose_path)

    assert torch.equal(verbose_result["weights"], state["weights"])
    assert capsys.readouterr().out == (
        f"Writing to {verbose_path}...\n"
        f"Reading from {verbose_path}...\n"
    )


def test_logging_setup_narrates_destination_only_when_verbose(
    monkeypatch, capsys
):
    logging_utils = load_logging_utils()
    configured = []
    monkeypatch.setattr(
        logging_utils.logging,
        "basicConfig",
        lambda **kwargs: configured.append(kwargs),
    )
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    logging_utils.setup_logging("quiet.log")
    assert capsys.readouterr().out == ""
    assert configured[-1]["filename"] == "quiet.log"

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    logging_utils.setup_logging("verbose.log")
    assert capsys.readouterr().out == "Writing to verbose.log...\n"
    assert configured[-1]["filename"] == "verbose.log"


def test_coron_pipeline_stage2_narrates_input_and_output(monkeypatch, capsys, tmp_path):
    pipeline_module = load_coron_pipeline(monkeypatch)
    input_path = tmp_path / "input_rateints.fits"
    calls = []
    glob_patterns = []

    class Pipeline:
        def call(self, filepath, save_results, output_dir):
            calls.append((filepath, save_results, output_dir))

    def fake_glob(pattern):
        glob_patterns.append(pattern)
        return [str(input_path)]

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    monkeypatch.setattr(pipeline_module, "Image2Pipeline", Pipeline)
    monkeypatch.setattr(pipeline_module.glob, "glob", fake_glob)
    pipeline = pipeline_module.CoronPipeline("1234", "NIRCAM")
    pipeline.pipeline_stage2()

    output_directory = f"{input_path.parent}/"
    scan_pattern = f"{pipeline.dataset_dir}**/*rateints.fits"
    assert calls == [(str(input_path), True, output_directory)]
    assert glob_patterns == [scan_pattern]
    assert capsys.readouterr().out == ""

    glob_patterns.clear()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    pipeline.pipeline_stage2()
    assert glob_patterns == [scan_pattern]
    assert capsys.readouterr().out == (
        f"Reading from {scan_pattern}...\n"
        f"Reading from {input_path}...\n"
        f"Writing to {output_directory}...\n"
    )


def test_coron_pipeline_stage3_narrates_calints_scan(monkeypatch, capsys, tmp_path):
    pipeline_module = load_coron_pipeline(monkeypatch)
    pipeline = pipeline_module.CoronPipeline.__new__(pipeline_module.CoronPipeline)
    pipeline.dataset_dir = f"{tmp_path}/"
    scan_patterns = []
    processed = []
    association = {"asn_id": "mock"}

    class Pipeline:
        def process(self, asn_dict):
            processed.append(asn_dict)

    def fake_glob(pattern):
        scan_patterns.append(pattern)
        return ["calints.fits"]

    pipeline.asn_file_creation = lambda calints_files: [association]
    monkeypatch.setattr(pipeline_module, "Coron3Pipeline", Pipeline)
    monkeypatch.setattr(pipeline_module.glob, "glob", fake_glob)
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    pipeline.pipeline_stage3()

    scan_pattern = f"{pipeline.dataset_dir}**/*calints.fits"
    assert scan_patterns == [scan_pattern]
    assert processed == [association]
    assert capsys.readouterr().out == ""

    scan_patterns.clear()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    pipeline.pipeline_stage3()
    assert scan_patterns == [scan_pattern]
    assert capsys.readouterr().out == (
        f"Reading from {scan_pattern}...\n"
        f"[{association}]\n"
        f"Writing to {pipeline.dataset_dir}...\n"
    )
    assert processed == [association, association]


def test_coron_pipeline_association_export_narrates_inputs_and_csv(
    monkeypatch, capsys, tmp_path
):
    pipeline_module = load_coron_pipeline(monkeypatch)
    monkeypatch.chdir(tmp_path)
    calints_paths = [str(tmp_path / "first_calints.fits")]
    pool_calls = []
    csv_calls = []
    registry_calls = []
    generated_associations = [{"asn_id": "mock"}]

    class DataFrame:
        def to_csv(self, path, index):
            csv_calls.append((path, index))

    class Pool:
        def to_pandas(self):
            return DataFrame()

    pipeline = pipeline_module.CoronPipeline.__new__(
        pipeline_module.CoronPipeline
    )
    pipeline.instrume = "NIRCAM"
    pipeline.proposal_id = "1234"
    monkeypatch.setattr(
        pipeline_module, "mkpool", lambda paths: pool_calls.append(paths) or Pool()
    )
    monkeypatch.setattr(
        pipeline_module.inspect, "getsourcefile", lambda _: "rules_level3.py"
    )
    monkeypatch.setattr(
        pipeline_module,
        "AssociationRegistry",
        lambda paths, include_default: registry_calls.append((paths, include_default)),
    )
    monkeypatch.setattr(
        pipeline_module, "generate", lambda pool, registry: generated_associations
    )

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert pipeline.asn_file_creation(calints_paths) == generated_associations
    assert pool_calls == [calints_paths]
    assert csv_calls == [("calints_NIRCAM_1234_pool.csv", False)]
    assert registry_calls == [(["rules_level3.py"], False)]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    pipeline.asn_file_creation(calints_paths)
    assert capsys.readouterr().out == (
        f"Reading from {calints_paths[0]}...\n"
        "Writing to calints_NIRCAM_1234_pool.csv...\n"
    )


def test_query_download_narrates_destination_without_network(
    monkeypatch, capsys, tmp_path
):
    query_module = load_query_script(monkeypatch)
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    query = query_module.Query()
    query.instrume = "NIRCAM"
    query.proposal_id = "1234"
    query.filtered_products = ["mock-product"]
    download_calls = []
    monkeypatch.setattr(
        query_module.Observations,
        "download_products",
        lambda products, download_dir: download_calls.append((products, download_dir)),
    )

    query.download_files(str(tmp_path))
    destination = f"{tmp_path}/NIRCAM/1234"
    assert download_calls == [(["mock-product"], destination)]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    query.download_files(str(tmp_path))
    assert capsys.readouterr().out == (
        f"Writing to {destination}...\n"
        "Observations are downloading from Mast Server. \n"
        " Download Info: \n Instrume: NIRCAM \n Proposal_id: 1234 \n"
        f" Dir: {destination}\n"
    )


def test_query_main_uses_repository_data_path_without_network(
    monkeypatch, capsys, tmp_path
):
    query_module = load_query_script(monkeypatch)
    data_path = tmp_path / "data"
    calls = []

    class MockQuery:
        def get_proposal_products(self, **kwargs):
            calls.append(("query", kwargs))

        def download_files(self, main_dataset_dir):
            calls.append(("download", main_dataset_dir))

    monkeypatch.setattr(query_module, "Query", MockQuery)
    monkeypatch.setattr(query_module, "get_dataset_dir", lambda: str(data_path))
    monkeypatch.setenv("TDPY_VERBOSITY", "0")

    query_module.main()

    assert calls == [
        (
            "query",
            {
                "instrume": "NIRCAM",
                "proposal_id": "1194",
                "psgd": ["RATEINTS"],
                "instrument_names": ["NIRCAM/CORON", "NIRCAM/TARGACQ"],
            },
        ),
        ("download", str(data_path)),
    ]
    assert capsys.readouterr().out == ""


def test_injection_array_writer_narrates_and_preserves_array(
    tmp_path, monkeypatch, capsys
):
    import numpy as np

    injection_module = load_injection_script(monkeypatch)
    injection = injection_module.Injection.__new__(injection_module.Injection)
    values = np.arange(9).reshape(3, 3)
    quiet_path = tmp_path / "quiet.npy"

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection._Injection__save_psf_to_npy(str(quiet_path), values)
    assert capsys.readouterr().out == ""
    print(f"Reading from {quiet_path}...")
    np.testing.assert_array_equal(np.load(quiet_path), values)

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    verbose_path = tmp_path / "verbose.npy"
    injection._Injection__save_psf_to_npy(str(verbose_path), values)
    assert capsys.readouterr().out == f"Writing to {verbose_path}...\n"
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    print(f"Reading from {verbose_path}...")
    np.testing.assert_array_equal(np.load(verbose_path), values)


def test_cpu_injection_psf_directory_uses_repository_data_path(
    monkeypatch, tmp_path
):
    injection_module = load_injection_script(monkeypatch)
    data_path = tmp_path / "data"
    monkeypatch.setattr(injection_module, "get_dataset_dir", lambda: str(data_path))

    assert injection_module._get_psf_directory("NIRCAM", "train", "1194") == str(
        data_path / "NIRCAM" / "train" / "1194" / "mastDownload" / "JWST"
    )


def test_webbpsf_path_defaults_to_repo_data_and_honors_override(
    monkeypatch, tmp_path
):
    import os

    monkeypatch.delenv("WEBBPSF_PATH", raising=False)
    load_injection_script(monkeypatch)
    assert os.environ["WEBBPSF_PATH"] == str(REPOSITORY_ROOT / "data" / "webbpsf-data")

    override = tmp_path / "webbpsf-cache"
    monkeypatch.setenv("WEBBPSF_PATH", str(override))
    load_injection_script(monkeypatch)
    assert os.environ["WEBBPSF_PATH"] == str(override)


def test_cached_psf_reader_narrates_fits_path(monkeypatch, capsys, tmp_path):
    injection_module = load_injection_script(monkeypatch)
    injection = injection_module.Injection.__new__(injection_module.Injection)
    expected_path = tmp_path / "PSF_SAMPLES" / "NRCA5-F444W__-3.0.fits"
    opened_paths = []
    hdus = object()
    monkeypatch.setattr(injection_module, "get_dataset_dir", lambda: str(tmp_path))
    monkeypatch.setattr(injection_module, "glob", lambda pattern: [pattern])
    monkeypatch.setattr(
        injection_module.fits,
        "open",
        lambda path: opened_paths.append(path) or hdus,
    )

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert injection._Injection__generate_psf_model("NRCALONG", "F444W", 3.0) is hdus
    assert capsys.readouterr().out == ""
    assert opened_paths == [str(expected_path)]

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    injection._Injection__generate_psf_model("NRCALONG", "F444W", 3.0)
    assert capsys.readouterr().out == (
        f"Reading from {expected_path}...\n"
        f"Reading from {expected_path}...\n"
        "NRCA5-F444W PSF with 3.0 arcsec fov collected from cache.\n"
    )


def test_psf_cache_miss_narrates_search_path(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_injection_script(monkeypatch)
    injection = injection_module.Injection.__new__(injection_module.Injection)
    expected_path = tmp_path / "PSF_SAMPLES" / "NRCA5-F444W__-3.0.fits"
    searched_paths = []
    output = object()

    class NIRCam:
        def calc_psf(self, **kwargs):
            assert kwargs == {"fov_arcsec": 3.0, "oversample": 2}
            return output

    times = iter((0, 1))
    monkeypatch.setattr(injection_module, "get_dataset_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        injection_module, "glob", lambda path: searched_paths.append(path) or []
    )
    monkeypatch.setattr(injection_module, "webbpsf", SimpleNamespace(NIRCam=NIRCam))
    monkeypatch.setattr(injection_module, "time", lambda: next(times))

    assert injection._Injection__generate_psf_model(
        "NRCALONG", "F444W", 3.0, save=False
    ) is output
    assert searched_paths == [str(expected_path)]
    assert capsys.readouterr().out == ""

    times = iter((0, 1))
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert injection._Injection__generate_psf_model(
        "NRCALONG", "F444W", 3.0, save=False
    ) is output
    assert searched_paths == [str(expected_path), str(expected_path)]
    assert capsys.readouterr().out == (
        f"Reading from {expected_path}...\n"
        "NRCA5-F444W PSF generated respect to 3.0 arcsec fov in 1 seconds.\n"
    )


def test_gpu_injection_psf_stack_reader_narrates_fits_paths(
    monkeypatch, capsys, tmp_path
):
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Base.__new__(
        injection_module.PSFDatasetGPU_Base
    )
    input_path = tmp_path / "stack_psfstack.fits"
    hdus = object()
    opened_paths = []
    dataset.psfstacks_nircam_dirs = [str(input_path)]
    dataset.psfstacks = {}
    monkeypatch.setattr(
        injection_module, "get_filename_from_dir", lambda path: Path(path).stem
    )
    monkeypatch.setattr(
        injection_module.fits,
        "open",
        lambda path: opened_paths.append(path) or hdus,
    )

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    dataset._PSFDatasetGPU_Base__create_psfstacks_dict()
    assert dataset.psfstacks == {"stack_psfstack": hdus}
    assert opened_paths == [str(input_path)]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    dataset.psfstacks = {}
    dataset._PSFDatasetGPU_Base__create_psfstacks_dict()
    assert capsys.readouterr().out == f"Reading from {input_path}...\n"


def test_gpu_psf_cache_miss_narrates_search_path(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Base.__new__(
        injection_module.PSFDatasetGPU_Base
    )
    expected_path = tmp_path / "PSF_SAMPLES" / "NRCA5-F444W-3.0--.fits"
    searched_paths = []
    output = object()

    class NIRCam:
        def calc_psf(self, **kwargs):
            assert kwargs == {"fov_arcsec": 3.0, "oversample": 2}
            return output

    monkeypatch.setattr(injection_module, "get_dataset_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        injection_module, "glob", lambda path: searched_paths.append(path) or []
    )
    monkeypatch.setattr(injection_module, "webbpsf", SimpleNamespace(NIRCam=NIRCam))

    assert dataset._PSFDatasetGPU_Base__generate_psf_model(
        "NRCALONG", "F444W", "", "", 3.0, save=False
    ) is output
    assert searched_paths == [str(expected_path)]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert dataset._PSFDatasetGPU_Base__generate_psf_model(
        "NRCALONG", "F444W", "", "", 3.0, save=False
    ) is output
    assert searched_paths == [str(expected_path), str(expected_path)]
    assert capsys.readouterr().out == (
        f"Reading from {expected_path}...\n"
        "NRCA5-F444W PSF generated respect to 3.0 arcsec fov\n"
    )


def test_gpu_cached_psf_reader_narrates_fits_path(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Base.__new__(
        injection_module.PSFDatasetGPU_Base
    )
    expected_path = tmp_path / "PSF_SAMPLES" / "NRCA5-F444W-3.0--.fits"
    opened_paths = []
    hdus = object()
    monkeypatch.setattr(injection_module, "get_dataset_dir", lambda: str(tmp_path))
    monkeypatch.setattr(injection_module, "glob", lambda path: [path])
    monkeypatch.setattr(
        injection_module.fits,
        "open",
        lambda path: opened_paths.append(path) or hdus,
    )

    assert dataset._PSFDatasetGPU_Base__generate_psf_model(
        "NRCALONG", "F444W", "", "", 3.0
    ) is hdus
    assert opened_paths == [str(expected_path)]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    dataset._PSFDatasetGPU_Base__generate_psf_model(
        "NRCALONG", "F444W", "", "", 3.0
    )
    assert capsys.readouterr().out == (
        f"Reading from {expected_path}...\n"
        f"Reading from {expected_path}...\n"
        "NRCA5-F444W PSF with 3.0 arcsec fov collected from cache.\n"
    )


def test_gpu_injection_sanity_checks_reject_invalid_psfs(monkeypatch):
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Injection.__new__(
        injection_module.PSFDatasetGPU_Injection
    )
    dataset.flux_coef = [1e-3, 1e-9]
    valid_generated_psf = injection_module.torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    for psfs, message in (
        (injection_module.torch.ones((2, 2)), "psfs must have 3 dimensions"),
        (injection_module.torch.tensor([[[1.0, float("nan")], [2.0, 3.0]]]), "finite values"),
        (injection_module.torch.zeros((1, 2, 2)), "nonzero range"),
        (
            injection_module.torch.stack((
                injection_module.torch.ones((2, 2)),
                injection_module.torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            )),
            "nonzero range",
        ),
    ):
        monkeypatch.setattr(
            injection_module.torch,
            "load",
            lambda path, psfs=psfs: [{"psfs": psfs, "generated_psf": valid_generated_psf}],
        )
        try:
            dataset.injection_GPU(["psf-input.pth"], sanity_checks=True)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Expected invalid PSFs to fail sanity checks: {message}")

    monkeypatch.setattr(
        injection_module.torch,
        "load",
        lambda path: [{
            "psfs": injection_module.torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
            "generated_psf": injection_module.torch.ones((1, 2, 2)),
        }],
    )
    try:
        dataset.injection_GPU(["psf-input.pth"], sanity_checks=True)
    except ValueError as error:
        assert "generated_psf must have 2 dimensions" in str(error)
    else:
        raise AssertionError("Expected malformed generated PSF dimensions to be rejected")


def test_gpu_base_prepare_forwards_optional_sanity_checks():
    import ast

    print(f"Reading from {GPU_INJECTION_SCRIPT_PATH}...")
    source = GPU_INJECTION_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GPU_INJECTION_SCRIPT_PATH))
    base_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PSFDatasetGPU_Base"
    )
    methods = {
        node.name: node for node in base_class.body
        if isinstance(node, ast.FunctionDef)
    }

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__nan_elimination"
        and any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(methods["prepare_dataset"])
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prepare_dataset"
        and any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(methods["forward"])
    )


def test_gpu_base_checks_psf_before_nan_elimination(monkeypatch):
    import torch

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Base.__new__(
        injection_module.PSFDatasetGPU_Base
    )
    invalid_psf = torch.tensor([[1.0, float("nan")]])

    try:
        dataset._PSFDatasetGPU_Base__nan_elimination(
            invalid_psf, sanity_checks=True
        )
    except ValueError as error:
        assert "finite values" in str(error)
    else:
        raise AssertionError("Expected non-finite PSF stack to fail sanity checks")

    assert torch.equal(
        dataset._PSFDatasetGPU_Base__nan_elimination(invalid_psf),
        torch.tensor([[1.0, 0.0]]),
    )

    mixed_batch = torch.stack((
        torch.ones((2, 2)),
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
    ))
    try:
        dataset._PSFDatasetGPU_Base__nan_elimination(
            mixed_batch, sanity_checks=True
        )
    except ValueError as error:
        assert "nonzero range" in str(error)
    else:
        raise AssertionError("Expected constant PSF in the batch to fail checks")


def test_gpu_base_checks_generated_psf_when_enabled(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import torch

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Base.__new__(
        injection_module.PSFDatasetGPU_Base
    )
    dataset.psf_directory = "synthetic-psf-directory"
    dataset.DEVICE = "cpu"
    stack = [
        SimpleNamespace(
            header={"DETECTOR": "NRCALONG", "FILTER": "F444W", "CHANNEL": "SHORT"}
        ),
        SimpleNamespace(
            data=np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
            header={"PIXAR_A2": 1.0},
        ),
    ]
    generated_hdu = SimpleNamespace(
        data=np.array([[np.nan, 0.0], [1.0, 2.0]], dtype=np.float32)
    )
    saved = []
    monkeypatch.setattr(
        injection_module,
        "get_stage3_products",
        lambda **kwargs: ["synthetic-stack"],
    )
    monkeypatch.setattr(
        dataset,
        "_PSFDatasetGPU_Base__create_psfstacks_dict",
        lambda: setattr(dataset, "psfstacks", {"F444W": stack}),
    )
    monkeypatch.setattr(
        dataset,
        "_PSFDatasetGPU_Base__get_max_min_pixel_distance",
        lambda psfstack: (1, 0),
    )
    monkeypatch.setattr(
        dataset,
        "_PSFDatasetGPU_Base__generate_psf_model",
        lambda **kwargs: [generated_hdu],
    )
    monkeypatch.setattr(
        injection_module.torch,
        "save",
        lambda value, path: saved.append((value, path)),
    )

    try:
        dataset.prepare_dataset(save_folder=tmp_path, sanity_checks=True)
    except ValueError as error:
        assert "finite values" in str(error)
    else:
        raise AssertionError("Expected non-finite generated PSF to fail sanity checks")

    assert saved == []
    generated_hdu.data = np.ones((2, 2), dtype=np.float32)
    try:
        dataset.prepare_dataset(save_folder=tmp_path, sanity_checks=True)
    except ValueError as error:
        assert "nonzero range" in str(error)
    else:
        raise AssertionError("Expected constant generated PSF to fail checks")

    generated_hdu.data = np.array([[np.nan, 0.0], [1.0, 2.0]], dtype=np.float32)
    dataset.prepare_dataset(save_folder=tmp_path)
    assert len(saved) == 1
    assert torch.isnan(saved[0][0][0]["generated_psf"]).any()


def test_gpu_injection_rejects_nonpositive_count_when_checked(monkeypatch):
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Injection.__new__(
        injection_module.PSFDatasetGPU_Injection
    )

    dataset.flux_coef = [0.1, 0.01]
    dataset.injection_GPU([], num_injection=1, sanity_checks=True)

    for invalid_count in (0, 1.5, True):
        try:
            dataset.injection_GPU(
                [], num_injection=invalid_count, sanity_checks=True
            )
        except ValueError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError("Expected invalid injection count to be rejected")


def test_gpu_injection_checks_flux_bounds_only_when_enabled(monkeypatch):
    import torch

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Injection.__new__(
        injection_module.PSFDatasetGPU_Injection
    )
    dataset.flux_coef = [0.1, 0.01]
    dataset.injection_GPU([], sanity_checks=True)

    dataset.flux_coef = [0.01, 0.1]
    try:
        dataset.injection_GPU([], sanity_checks=True)
    except ValueError as error:
        assert "maximum, minimum" in str(error)
    else:
        raise AssertionError("Expected reversed flux bounds to be rejected")

    dataset.flux_coef = [float("nan"), 0.01]
    try:
        dataset.injection_GPU([], sanity_checks=True)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("Expected non-finite flux bounds to be rejected")

    dataset.flux_coef = [-0.1, 0.01]
    try:
        dataset.injection_GPU([], sanity_checks=True)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("Expected negative flux bounds to be rejected")

    dataset.flux_coef = [float("nan"), 0.01]
    dataset.injection_GPU([])


def test_gpu_injection_sanity_checks_are_opt_in(monkeypatch, tmp_path):
    import torch

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    injection_module = load_gpu_injection_script(monkeypatch)
    dataset = injection_module.PSFDatasetGPU_Injection.__new__(
        injection_module.PSFDatasetGPU_Injection
    )
    dataset.save_folder = str(tmp_path)
    psfs = torch.tensor([[[1.0, float("nan")], [2.0, 3.0]]])
    generated_psf = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    saved = []
    monkeypatch.setattr(
        injection_module.torch,
        "load",
        lambda path: [{
            "filter_key": "F444W",
            "psfs": psfs,
            "generated_psf": generated_psf,
        }],
    )
    monkeypatch.setattr(
        injection_module.torch, "save", lambda value, path: saved.append((value, path))
    )
    monkeypatch.setattr(injection_module, "ModifiedRandomCrop", lambda size: object())
    monkeypatch.setattr(
        dataset,
        "_PSFDatasetGPU_Injection__injection_item_GPU",
        lambda *args, **kwargs: None,
    )

    dataset.injection_GPU(["synthetic-input.pth"])

    assert len(saved) == 1
    assert torch.isnan(saved[0][0]).any()
    assert saved[0][1] == str(tmp_path / "base_F444W.pth")


def test_gpu_file_scans_use_narrated_glob_helper():
    import ast

    print(f"Reading from {GPU_INJECTION_SCRIPT_PATH}...")
    source = GPU_INJECTION_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GPU_INJECTION_SCRIPT_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    base_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PSFDatasetGPU_Base"
    )
    cache_method = next(
        node for node in base_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__generate_psf_model"
    )

    for function in (cache_method, functions["main"], functions["main_v2"]):
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_glob_paths"
            for node in ast.walk(function)
        )

    direct_glob_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "glob"
    ]
    assert len(direct_glob_calls) == 1
    assert direct_glob_calls[0] in ast.walk(functions["_glob_paths"])


def test_gpu_injection_default_output_uses_repository_data_path(
    monkeypatch, tmp_path
):
    injection_module = load_gpu_injection_script(monkeypatch)
    monkeypatch.setattr(
        injection_module, "get_dataset_dir", lambda: str(tmp_path / "data")
    )

    injection = injection_module.PSFDatasetGPU_Injection(device="cpu")

    expected_path = tmp_path / "data" / "torch_dataset_injection" / "train"
    assert injection.save_folder == str(expected_path)
    assert expected_path.is_dir()


def test_gpu_injection_output_matches_dataset_loader_default(
    monkeypatch, tmp_path, capsys
):
    import torch

    data_root = tmp_path / "data"
    dataloaders = load_dataloaders(monkeypatch, data_root)
    injection_module = load_gpu_injection_script(monkeypatch)
    monkeypatch.setattr(injection_module, "get_dataset_dir", lambda: str(data_root))
    injection = injection_module.PSFDatasetGPU_Injection(device="cpu")
    expected_directory = data_root / "torch_dataset_injection" / "train"
    assert injection.save_folder == str(expected_directory)

    patterns = []
    monkeypatch.setattr(
        dataloaders.glob,
        "glob",
        lambda pattern: patterns.append(pattern) or [str(expected_directory / "sample.pth")],
    )
    monkeypatch.setattr(
        dataloaders.torch, "load", lambda *args, **kwargs: torch.ones((1, 2, 2))
    )

    dataset = dataloaders.SynthDatasetv2(num_injection=1)

    assert patterns == [str(expected_directory / "*.pth")]
    assert dataset.psf_dicts.shape == (1, 2, 2)
    assert capsys.readouterr().out == ""

    patterns.clear()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    dataset = dataloaders.SynthDatasetv2(num_injection=1)
    assert patterns == [str(expected_directory / "*.pth")]
    assert capsys.readouterr().out == (
        f"Reading from {expected_directory}/*.pth...\n"
        f"Reading from {expected_directory}/sample.pth...\n"
    )


def test_classifier_training_is_quiet_and_narrates_checkpoint(
    monkeypatch, capsys, tmp_path
):
    import torch
    from types import SimpleNamespace

    classifier_module = load_exo_classifier(monkeypatch)
    monkeypatch.setattr(classifier_module, "PLOT_SAVE_PATH", str(tmp_path))
    saved_paths = []

    def save_model(model, path):
        saved_paths.append(path)
        Path(path).write_bytes(b"mock checkpoint")

    monkeypatch.setattr(classifier_module.torch, "save", save_model)
    args = SimpleNamespace(
        idx="case",
        device="cpu",
        lr=0.01,
        optim="sgd",
        epoch=1,
        apply_lowpass=False,
    )
    classifier = classifier_module.ExoClassifier.__new__(
        classifier_module.ExoClassifier
    )
    torch.nn.Module.__init__(classifier)
    classifier.args = args

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            return torch.log_softmax(self.linear(inputs), dim=1)

    train_dataloader = [
        (
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([0, 1]),
            None,
            None,
        )
    ]
    model = TinyModel()
    model_path = tmp_path / "training_results" / "case" / "models" / "model.pt"

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    classifier.train_model(model, train_dataloader)
    assert capsys.readouterr().out == ""
    assert saved_paths == [str(model_path)]
    assert model_path.read_bytes() == b"mock checkpoint"

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    classifier.train_model(model, train_dataloader)
    output = capsys.readouterr().out
    assert output.startswith(f"Writing to {model_path}...\nBatch_loss:")
    assert "Total_loss:" in output


def test_classifier_training_rejects_nonpositive_epoch_count(monkeypatch, tmp_path):
    import pytest
    import torch
    from types import SimpleNamespace

    classifier_module = load_exo_classifier(monkeypatch)
    monkeypatch.setattr(classifier_module, "PLOT_SAVE_PATH", str(tmp_path))
    classifier = classifier_module.ExoClassifier.__new__(
        classifier_module.ExoClassifier
    )
    torch.nn.Module.__init__(classifier)
    classifier.args = SimpleNamespace(
        idx="case",
        device="cpu",
        lr=0.01,
        optim="sgd",
        epoch=0,
        apply_lowpass=False,
    )

    with pytest.raises(ValueError, match="epoch must be positive"):
        classifier.train_model(torch.nn.Linear(2, 2), [])


def test_dataset_readers_narrate_numpy_and_torch_paths(
    monkeypatch, capsys, tmp_path
):
    import numpy as np
    import pytest
    import torch
    from types import SimpleNamespace

    dataloaders = load_dataloaders(monkeypatch)
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    image_path = tmp_path / "example-fc.npy"
    image_values = np.arange(16, dtype=np.float32).reshape(4, 4)
    print(f"Writing to {image_path}...")
    np.save(image_path, image_values)
    image_dataset = dataloaders.SynDatasetLabel(
        [str(image_path)], SimpleNamespace(apply_lowpass=False)
    )
    image, label, _, _ = image_dataset[0]
    assert torch.equal(image, torch.as_tensor(image_values).unsqueeze(0))
    assert label.item() == 1
    assert capsys.readouterr().out == ""

    invalid_image_path = tmp_path / "invalid-image.npy"
    print(f"Writing to {invalid_image_path}...")
    np.save(invalid_image_path, np.array([[np.nan, 1.0]], dtype=np.float32))
    default_dataset = dataloaders.SynDatasetLabel(
        [str(invalid_image_path)], SimpleNamespace(apply_lowpass=False)
    )
    assert torch.isnan(default_dataset[0][0]).any()
    checked_dataset = dataloaders.SynDatasetLabel(
        [str(invalid_image_path)],
        SimpleNamespace(apply_lowpass=False, sanity_checks=True),
    )
    with pytest.raises(ValueError, match="finite values"):
        checked_dataset[0]

    constant_image_path = tmp_path / "constant-image.npy"
    print(f"Writing to {constant_image_path}...")
    np.save(constant_image_path, np.ones((2, 2), dtype=np.float32))
    checked_constant_dataset = dataloaders.SynDatasetLabel(
        [str(constant_image_path)],
        SimpleNamespace(apply_lowpass=False, sanity_checks=True),
    )
    with pytest.raises(ValueError, match="nonzero range"):
        checked_constant_dataset[0]

    transformed_dataset = dataloaders.SynDatasetLabel(
        [str(image_path)],
        SimpleNamespace(apply_lowpass=False, sanity_checks=True),
    )
    transformed_dataset.transform = lambda _: torch.full((1, 4, 4), float("nan"))
    with pytest.raises(ValueError, match="Transformed classifier image"):
        transformed_dataset[0]

    lowpass_dataset = dataloaders.SynDatasetLabel(
        [str(image_path)],
        SimpleNamespace(apply_lowpass=True, sanity_checks=True),
    )
    lowpass_dataset.apply_low_pass = lambda values: values
    transform_call_count = 0

    def transform_with_invalid_lowpass(values):
        nonlocal transform_call_count
        transform_call_count += 1
        if transform_call_count == 2:
            return torch.full((1, 4, 4), float("inf"))
        return torch.as_tensor(values).unsqueeze(0)

    lowpass_dataset.transform = transform_with_invalid_lowpass
    with pytest.raises(ValueError, match="Low-pass classifier image"):
        lowpass_dataset[0]

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    image_dataset[0]
    assert capsys.readouterr().out == f"Reading from {image_path}...\n"

    checkpoint_path = tmp_path / "train" / "base.pth"
    checkpoint_path.parent.mkdir()
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    print(f"Writing to {checkpoint_path}...")
    torch.save(torch.ones((2, 3, 3)), checkpoint_path)
    tensor_dataset = dataloaders.SynthDatasetv2(str(tmp_path), "train", 1)
    assert tensor_dataset.psf_dicts.shape == (2, 3, 3)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    dataloaders.SynthDatasetv2(str(tmp_path), "train", 1)
    assert capsys.readouterr().out == (
        f"Reading from {tmp_path / 'train' / '*.pth'}...\n"
        f"Reading from {checkpoint_path}...\n"
    )


def test_training_selector_handles_one_proposal_and_multiple_filters(
    monkeypatch, capsys
):
    from types import SimpleNamespace

    training_module = load_train_classifier(monkeypatch)
    patterns = {
        "*F200W*fc*.npy": ["F200W-injected.npy"],
        "*F200W*[!fc].npy": ["F200W-control.npy"],
        "*F444W*fc*.npy": ["F444W-injected.npy"],
        "*F444W*[!fc].npy": ["F444W-control.npy"],
    }
    scanned_patterns = []
    monkeypatch.setattr(
        training_module.glob,
        "glob",
        lambda pattern: scanned_patterns.append(pattern) or patterns[next(
            key for key in patterns if key in pattern
        )],
    )
    monkeypatch.setattr(training_module.random, "shuffle", lambda paths: None)
    args = SimpleNamespace(
        train_pids="1234",
        train_filters="F200W F444W",
        mode="train",
    )

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    selected_paths = training_module.get_paths(args)

    assert set(selected_paths) == set(sum(patterns.values(), []))
    assert len(scanned_patterns) == 4
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    training_module.get_paths(args)
    verbose_output = capsys.readouterr().out
    assert "Reading from /mock/NIRCAM/train/1234/injections/*F200W*fc*.npy..." in verbose_output
    assert "Reading from /mock/NIRCAM/train/1234/injections/*F444W*[!fc].npy..." in verbose_output
    assert "INJECTED: 2" in verbose_output
    assert "NOT INJECTED: 2" in verbose_output


def test_mlp_training_uses_repository_checkpoint_path_and_quiet_output(
    monkeypatch, capsys, tmp_path
):
    import torch

    training = load_mlp_training_script(monkeypatch, tmp_path / "data")
    saved_paths = []
    monkeypatch.setattr(
        training.torch.optim,
        "AdamW",
        lambda parameters, lr: object(),
    )
    monkeypatch.setattr(
        training, "train_epoch", lambda *args, **kwargs: 0.5
    )
    monkeypatch.setattr(training.torch, "save", lambda model, path: saved_paths.append(path))

    class Progress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    monkeypatch.setattr(training, "tqdm", Progress)
    checkpoint_path = tmp_path / "data" / "models" / "model.ckpt"

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    training.main(num_epochs=1, batch_size=1)
    assert saved_paths == [str(checkpoint_path)]
    assert checkpoint_path.parent.is_dir()
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    training.main(num_epochs=1, batch_size=1)
    assert capsys.readouterr().out == f"Writing to {checkpoint_path}...\n"


def test_mlp_batch_sanity_checks_are_optional_and_enforce_binary_labels():
    import ast
    import torch
    import pytest

    print(f"Reading from {TRAIN_SCRIPT_PATH}...")
    source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_SCRIPT_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_training_batch"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(TRAIN_SCRIPT_PATH), "exec"),
        namespace,
    )
    validate = namespace["validate_training_batch"]

    invalid_batch = torch.tensor([[[float("nan")]]])
    invalid_labels = torch.tensor([2])
    valid_batch = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 3.0, 4.0]]])
    validate(invalid_batch, invalid_labels, enabled=False)
    validate(valid_batch, torch.tensor([0, 1]), enabled=True)
    with pytest.raises(ValueError, match="3-D"):
        validate(torch.ones((2, 3)), torch.tensor([0, 1]), enabled=True)
    with pytest.raises(ValueError, match="3-D"):
        validate(torch.empty((0, 1, 2)), torch.tensor([]), enabled=True)
    with pytest.raises(ValueError, match="finite"):
        validate(invalid_batch, torch.tensor([0]), enabled=True)
    with pytest.raises(ValueError, match="binary class indices"):
        validate(valid_batch[:1], invalid_labels, enabled=True)
    with pytest.raises(ValueError, match="finite values"):
        validate(valid_batch[:1], torch.tensor([float("nan")]), enabled=True)
    with pytest.raises(ValueError, match="integer class indices"):
        validate(valid_batch[:1], torch.tensor([0.5]), enabled=True)
    with pytest.raises(ValueError, match="match the batch size"):
        validate(valid_batch, torch.tensor([0]), enabled=True)
    with pytest.raises(ValueError, match="nonzero range"):
        validate(torch.ones((1, 1, 4)), torch.tensor([0]), enabled=True)


def test_mlp_loss_sanity_check_is_optional_and_wired_into_training():
    import ast
    import torch
    import pytest

    print(f"Reading from {TRAIN_SCRIPT_PATH}...")
    source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_SCRIPT_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    validator_module = ast.Module(
        body=[functions["validate_finite_loss"]], type_ignores=[]
    )
    namespace = {"torch": torch, "FloatingPointError": FloatingPointError}
    exec(compile(validator_module, str(TRAIN_SCRIPT_PATH), "exec"), namespace)
    validate = namespace["validate_finite_loss"]

    validate(torch.tensor(float("nan")), enabled=False)
    validate(torch.tensor(1.0), enabled=True)
    with pytest.raises(FloatingPointError, match="must be finite"):
        validate(torch.tensor(float("nan")), enabled=True)

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_finite_loss"
        and any(
            keyword.arg == "enabled"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(functions["train_epoch"])
    )


def test_training_parameter_sanity_checks_are_optional_and_wired():
    import ast
    import math
    from numbers import Integral, Real

    import pytest

    print(f"Reading from {TRAIN_SCRIPT_PATH}...")
    source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_SCRIPT_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    namespace = {"math": math, "Integral": Integral, "Real": Real}
    validator = functions["validate_training_parameters"]
    exec(
        compile(ast.Module(body=[validator], type_ignores=[]), str(TRAIN_SCRIPT_PATH), "exec"),
        namespace,
    )
    validate = namespace["validate_training_parameters"]
    validate(1, 8, 1e-3, enabled=True)
    validate(0, 0, float("nan"), enabled=False)
    for invalid_epochs, invalid_batch_size in ((0, 8), (1, 0), (1.5, 8)):
        with pytest.raises(ValueError, match="positive integer"):
            validate(invalid_epochs, invalid_batch_size, 1e-3, enabled=True)
    for invalid_learning_rate in (0, -1e-3, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            validate(1, 8, invalid_learning_rate, enabled=True)

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_training_parameters"
        and any(
            keyword.arg == "enabled"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(functions["main"])
    )


def test_evaluation_batch_loader_narrates_numpy_inputs(
    monkeypatch, capsys, tmp_path
):
    import ast
    import numpy as np
    from tdpy.verbosity import print as verbosity_print

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_batch"
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "print": verbosity_print}
    exec(compile(function_module, str(TEST_CLASSIFIER_PATH), "exec"), namespace)

    paths = [tmp_path / "first.npy", tmp_path / "second.npy"]
    arrays = [np.full((2, 2), value, dtype=np.float32) for value in (1, 2)]
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    for path, array in zip(paths, arrays):
        print(f"Writing to {path}...")
        np.save(path, array)

    batch = namespace["get_batch"]([str(path) for path in paths])
    np.testing.assert_array_equal(batch, np.stack(arrays))
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    namespace["get_batch"]([str(path) for path in paths])
    assert capsys.readouterr().out == "".join(
        f"Reading from {path}...\n" for path in paths
    )


def test_evaluation_batch_sanity_checks_are_optional(monkeypatch, tmp_path):
    import ast

    import numpy as np
    import pytest

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    tree = ast.parse(TEST_CLASSIFIER_PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_batch"
    )
    namespace = {"np": np, "print": lambda message: None}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(TEST_CLASSIFIER_PATH), "exec"),
        namespace,
    )
    invalid_path = tmp_path / "invalid.npy"
    constant_path = tmp_path / "constant.npy"
    one_dimensional_path = tmp_path / "one-dimensional.npy"
    print(f"Writing to {invalid_path}...")
    np.save(invalid_path, np.array([[np.nan, 1.0]], dtype=np.float32))
    print(f"Writing to {constant_path}...")
    np.save(constant_path, np.ones((2, 2), dtype=np.float32))
    print(f"Writing to {one_dimensional_path}...")
    np.save(one_dimensional_path, np.array([0.0, 1.0], dtype=np.float32))

    assert np.isnan(namespace["get_batch"]([str(invalid_path)])[0, 0, 0])
    assert namespace["get_batch"]([]).shape == (0,)
    assert namespace["get_batch"]([str(one_dimensional_path)]).shape == (1, 2)
    with pytest.raises(ValueError, match="at least one image"):
        namespace["get_batch"]([], sanity_checks=True)
    with pytest.raises(ValueError, match="finite values"):
        namespace["get_batch"]([str(invalid_path)], sanity_checks=True)
    with pytest.raises(ValueError, match="nonzero range"):
        namespace["get_batch"]([str(constant_path)], sanity_checks=True)
    with pytest.raises(ValueError, match="two-dimensional"):
        namespace["get_batch"]([str(one_dimensional_path)], sanity_checks=True)


def test_classifier_output_sanity_checks_are_optional_and_wired():
    import ast

    import pytest
    import torch

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    namespace = {
        "torch": torch,
        "FloatingPointError": FloatingPointError,
        "ValueError": ValueError,
    }
    exec(
        compile(
            ast.Module(body=[functions["validate_classifier_outputs"]], type_ignores=[]),
            str(TEST_CLASSIFIER_PATH),
            "exec",
        ),
        namespace,
    )
    validate = namespace["validate_classifier_outputs"]
    validate(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.7]]), enabled=True)
    validate(torch.tensor([[float("nan"), 1.0]]), torch.tensor([[0.7]]))
    validate(torch.tensor([0.0, 1.0]), torch.tensor([0.7]))
    with pytest.raises(ValueError, match=r"shape \(batch, 2\)"):
        validate(torch.tensor([[0.0, 1.0, 2.0]]), torch.tensor([[0.7]]), enabled=True)
    with pytest.raises(ValueError, match=r"shape \(batch, 1\)"):
        validate(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.4, 0.7]]), enabled=True)
    with pytest.raises(FloatingPointError, match="logits must be finite"):
        validate(torch.tensor([[float("nan"), 1.0]]), torch.tensor([[0.7]]), enabled=True)
    with pytest.raises(ValueError, match="must be finite"):
        validate(torch.tensor([[0.0, 1.0]]), torch.tensor([[float("nan")]]), enabled=True)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.2]]), enabled=True)

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_classifier_outputs"
        and any(keyword.arg == "enabled" for keyword in node.keywords)
        for node in ast.walk(functions["test_model"])
    )

    exec(
        compile(
            ast.Module(body=[functions["validate_classifier_targets"]], type_ignores=[]),
            str(TEST_CLASSIFIER_PATH),
            "exec",
        ),
        namespace,
    )
    validate_targets = namespace["validate_classifier_targets"]
    validate_targets(torch.tensor([0.0, 1.0]), batch_size=2, enabled=True)
    validate_targets(torch.tensor([0.0, float("nan")]), batch_size=2)
    with pytest.raises(ValueError, match="match the batch size"):
        validate_targets(torch.tensor([0.0]), batch_size=2, enabled=True)
    with pytest.raises(ValueError, match="must be finite"):
        validate_targets(torch.tensor([0.0, float("nan")]), batch_size=2, enabled=True)
    with pytest.raises(ValueError, match="integer class indices"):
        validate_targets(torch.tensor([0.0, 0.5]), batch_size=2, enabled=True)
    with pytest.raises(ValueError, match="binary class indices"):
        validate_targets(torch.tensor([0, 2]), batch_size=2, enabled=True)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_classifier_targets"
        and any(
            keyword.arg == "enabled"
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Name)
            and keyword.value.func.id == "getattr"
            for keyword in node.keywords
        )
        for node in ast.walk(functions["test_model"])
    )


def test_evaluation_test_paths_delegate_to_canonical_selector(monkeypatch):
    import ast

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_test_paths"
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    received = []
    expected = ["canonical-test-input.npy"]
    namespace = {"get_paths": lambda args: received.append(args) or expected}
    exec(compile(function_module, str(TEST_CLASSIFIER_PATH), "exec"), namespace)
    arguments = object()

    assert namespace["get_test_paths"](arguments) is expected
    assert received == [arguments]


def test_evaluation_coordinates_preserve_pixel_values():
    import ast

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_exo_coords"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(TEST_CLASSIFIER_PATH), "exec"),
        namespace,
    )

    assert namespace["get_exo_coords"]("plain.npy") == (0, 0)
    assert namespace["get_exo_coords"]("candidate-x12-y34.npy") == (12, 34)


def test_evaluation_pixel_distance_preserves_euclidean_value():
    import ast
    import math

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_distance"
    )
    namespace = {"math": math}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(TEST_CLASSIFIER_PATH), "exec"),
        namespace,
    )

    assert namespace["calculate_distance"](0, 0, 3, 4) == 5


def test_evaluation_file_discovery_narrates_glob_paths(monkeypatch, capsys):
    import ast
    import os
    import random
    from types import SimpleNamespace
    from tdpy.verbosity import print as verbosity_print

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_glob_paths", "get_paths"}
    ]
    patterns = []

    def fake_glob(pattern):
        patterns.append(pattern)
        if "fc*.npy" in pattern:
            return ["/repo/data/test/1386/injections/injected_fc5.npy"]
        return ["/repo/data/test/1386/injections/plain.npy"]

    namespace = {
        "glob": SimpleNamespace(glob=fake_glob),
        "NIRCAM_DATA": "/repo/data",
        "os": os,
        "random": random,
        "print": verbosity_print,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(TEST_CLASSIFIER_PATH), "exec"),
        namespace,
    )
    arguments = SimpleNamespace(train_pids="1386", train_filters="F200W", mode="test")
    expected_patterns = [
        "/repo/data/test/1386/injections/*F200W*fc*.npy",
        "/repo/data/test/1386/injections/*F200W*[!fc].npy",
    ]
    expected_paths = {
        "/repo/data/test/1386/injections/injected_fc5.npy",
        "/repo/data/test/1386/injections/plain.npy",
    }

    assert set(namespace["get_paths"](arguments)) == expected_paths
    assert patterns == expected_patterns
    assert capsys.readouterr().out == ""

    patterns.clear()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert set(namespace["get_paths"](arguments)) == expected_paths
    assert capsys.readouterr().out == "".join(
        f"Reading from {pattern}...\n" for pattern in expected_patterns
    ) + "INJECTED: 1\nNOT INJECTED: 1\n#Samples: 2\n"
    assert patterns == expected_patterns


def test_evaluation_workflow_runs_only_under_main_guard():
    import ast

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    main_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_guard = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    module_level_calls = [
        node.func.id
        for statement in tree.body
        if not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    guarded_calls = {
        node.func.id for statement in main_guard.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert main_function.name == "main"
    assert "main" in guarded_calls
    assert not {"inference_arg_parser", "get_model", "test_model"}.intersection(
        module_level_calls
    )
def test_evaluation_psf_info_uses_repo_path_and_closes_fits(monkeypatch, capsys):
    import ast
    import contextlib
    import os
    from tdpy.verbosity import print as verbosity_print

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_psf_info_bs1"
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    hdus = [object(), object()]
    opened = []
    closed = []

    @contextlib.contextmanager
    def open_fits(path):
        opened.append(path)
        try:
            yield hdus
        finally:
            closed.append(path)

    helpers = {
        "get_dataset_dir": lambda: "/repo/data",
        "get_wcs": lambda _: "wcs",
        "get_ra_dec": lambda _: (12.0, -30.0),
        "get_skycoord": lambda ra, dec: (ra, dec),
        "skycoord_to_pixel": lambda wcs, coord: (4.0, 5.0),
        "print": verbosity_print,
        "os": os,
        "fits": type("Fits", (), {"open": staticmethod(open_fits)}),
    }
    namespace = dict(helpers)
    exec(compile(function_module, str(TEST_CLASSIFIER_PATH), "exec"), namespace)
    injection_path = "/temporary/F444W-candidate-x1-y2-fc3.npy"
    expected_path = "/repo/data/NIRCAM/train/1386/mastDownload/JWST/F444W-candidate-x1-y2_psfstack.fits"

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert namespace["get_psf_info_bs1"](injection_path, "1386") == [(4.0, 5.0)]
    assert opened == [expected_path] and closed == [expected_path]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    namespace["get_psf_info_bs1"](injection_path, "1386")
    assert capsys.readouterr().out == f"Reading from {expected_path}...\n"


def test_evaluation_pixel_scale_reader_narrates_and_closes_fits(
    monkeypatch, capsys
):
    import ast
    import contextlib
    import numpy as np
    from types import SimpleNamespace
    from tdpy.verbosity import print as verbosity_print

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_arcsec_per_pixel"
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    opened = []
    closed = []
    hdus = [None, SimpleNamespace(header={"PIXAR_A2": 0.09})]

    @contextlib.contextmanager
    def open_fits(path):
        opened.append(path)
        try:
            yield hdus
        finally:
            closed.append(path)

    namespace = {"np": np, "fits": type("Fits", (), {"open": staticmethod(open_fits)}), "print": verbosity_print}
    exec(compile(function_module, str(TEST_CLASSIFIER_PATH), "exec"), namespace)
    psf_path = "/repo/data/NIRCAM/test/psfstack.fits"

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert namespace["get_arcsec_per_pixel"](psf_path) == 0.3
    assert opened == [psf_path] and closed == [psf_path]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert namespace["get_arcsec_per_pixel"](psf_path) == 0.3
    assert capsys.readouterr().out == f"Reading from {psf_path}...\n"


def test_evaluation_model_path_defaults_to_repo_visuals_and_logs_read(
    monkeypatch, capsys, tmp_path
):
    import argparse
    import ast
    import os
    import sys
    from types import SimpleNamespace
    from tdpy.verbosity import print as verbosity_print

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    parser_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "inference_arg_parser"
    )
    parser_module = ast.Module(body=[parser_function], type_ignores=[])
    visuals_path = tmp_path / "visuals"
    namespace = {"argparse": argparse, "os": os, "PLOT_SAVE_PATH": str(visuals_path)}
    exec(compile(parser_module, str(TEST_CLASSIFIER_PATH), "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["test_classifier"])
    args = namespace["inference_arg_parser"]()
    assert args.sanity_checks is False

    assert args.model_path == os.path.join(
        visuals_path, "training_results", "dospid", "2", "models", "model.pt"
    )
    visualize_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "visualize_results"
    )
    main_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_batch"
        and any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(visualize_function)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "visualize_results"
        and any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "sanity_checks"
            for keyword in node.keywords
        )
        for node in ast.walk(main_function)
    )

    model_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model"
    )
    model_module = ast.Module(body=[model_function], type_ignores=[])
    model_path = args.model_path
    model = object()
    loads = []
    model_namespace = {
        "get_arc_params": lambda: ([1], [1], [1]),
        "ExoClassifier": lambda **kwargs: object(),
        "args": SimpleNamespace(),
        "torch": SimpleNamespace(
            load=lambda path, map_location: loads.append((path, map_location)) or model
        ),
        "print": verbosity_print,
    }
    exec(compile(model_module, str(TEST_CLASSIFIER_PATH), "exec"), model_namespace)

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert model_namespace["get_model"](model_path, "cpu") is model
    assert loads == [(model_path, "cpu")]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    model_namespace["get_model"](model_path, "cpu")
    assert capsys.readouterr().out == f"Reading from {model_path}...\n"


def test_evaluation_figure_writer_narrates_path_and_forwards_options(
    monkeypatch, capsys
):
    import ast
    from types import SimpleNamespace
    from tdpy.verbosity import print as verbosity_print

    print(f"Reading from {TEST_CLASSIFIER_PATH}...")
    source = TEST_CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_CLASSIFIER_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_save_figure"
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    save_calls = []
    namespace = {
        "print": verbosity_print,
        "plt": SimpleNamespace(
            savefig=lambda path, **kwargs: save_calls.append((path, kwargs))
        ),
    }
    exec(compile(function_module, str(TEST_CLASSIFIER_PATH), "exec"), namespace)
    figure_path = "visuals/result.jpg"
    save_options = {"format": "jpg", "dpi": 200, "bbox_inches": "tight"}

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    namespace["_save_figure"](figure_path, **save_options)
    assert capsys.readouterr().out == ""
    assert save_calls == [(figure_path, save_options)]

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    namespace["_save_figure"](figure_path, **save_options)
    assert capsys.readouterr().out == f"Writing to {figure_path}...\n"
    assert save_calls[-1] == (figure_path, save_options)


def test_ssl_data_and_checkpoint_paths_use_repo_data_root(capsys):
    import ast
    import os

    print(f"Reading from {TEST_SSL_PATH}...")
    source = TEST_SSL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_SSL_PATH))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_mae_dataset_dir", "get_mae_checkpoint_dir"}
    ]
    assert {function.name for function in functions} == {
        "get_mae_dataset_dir", "get_mae_checkpoint_dir"
    }
    namespace = {
        "os": os,
        "get_dataset_dir": lambda: "/repo/data",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TEST_SSL_PATH), "exec"), namespace)

    assert namespace["get_mae_dataset_dir"]() == "/repo/data/mae_dataset"
    assert namespace["get_mae_checkpoint_dir"]() == "/repo/data/mae_checkpoints"
    assert capsys.readouterr().out == ""


def test_ssl_file_discovery_uses_narrated_glob_helper(monkeypatch, capsys):
    import ast
    from types import SimpleNamespace
    from tdpy.verbosity import print as verbosity_print

    print(f"Reading from {TEST_SSL_PATH}...")
    source = TEST_SSL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_SSL_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    mae_data = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MAEData"
    )
    initializer = next(
        node for node in mae_data.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    glob_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "glob"
        and node.func.attr == "glob"
    ]

    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_glob_paths"
        for node in ast.walk(functions["save_dataset"])
    ) == 3
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_glob_paths"
        for node in ast.walk(initializer)
    ) == 1
    helper = functions["_glob_paths"]
    assert len(glob_calls) == 1 and glob_calls[0] in ast.walk(helper)

    searched = []
    namespace = {
        "glob": SimpleNamespace(glob=lambda pattern: searched.append(pattern) or [pattern]),
        "print": verbosity_print,
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(TEST_SSL_PATH), "exec"),
        namespace,
    )
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert namespace["_glob_paths"]("data/**/*.fits") == ["data/**/*.fits"]
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert namespace["_glob_paths"]("data/**/*.fits") == ["data/**/*.fits"]
    assert searched == ["data/**/*.fits", "data/**/*.fits"]
    assert capsys.readouterr().out == "Reading from data/**/*.fits...\n"


def test_ssl_finite_loss_check_is_optional_and_detects_nan():
    import ast
    import torch
    import pytest

    print(f"Reading from {TEST_SSL_PATH}...")
    source = TEST_SSL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_SSL_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_finite_loss"
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch, "FloatingPointError": FloatingPointError}
    exec(compile(function_module, str(TEST_SSL_PATH), "exec"), namespace)
    validate = namespace["validate_finite_loss"]

    validate(torch.tensor(float("nan")), enabled=False)
    validate(torch.tensor(1.0), enabled=True)
    with pytest.raises(FloatingPointError, match="must be finite"):
        validate(torch.tensor(float("nan")), enabled=True)


def test_ssl_cached_observation_checks_are_optional(monkeypatch):
    import ast
    import os

    import pytest
    import torch

    print(f"Reading from {TEST_SSL_PATH}...")
    source = TEST_SSL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_SSL_PATH))
    mae_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MAEData"
    )
    main_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MAEData"
        and any(
            keyword.arg == "sanity_checks"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "sanity_checks_enabled"
            for keyword in node.keywords
        )
        for node in ast.walk(main_function)
    )

    invalid_observation = torch.tensor([[float("nan"), 1.0]])
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: invalid_observation.clone()
    )
    namespace = {
        "Dataset": object,
        "_glob_paths": lambda pattern: ["synthetic-observation.ckpt"],
        "get_mae_dataset_dir": lambda: "synthetic-cache",
        "os": os,
        "print": lambda message: None,
        "torch": torch,
    }
    exec(compile(ast.Module(body=[mae_class], type_ignores=[]), str(TEST_SSL_PATH), "exec"), namespace)

    with pytest.raises(ValueError, match="finite values"):
        namespace["MAEData"]("synthetic-cache", sanity_checks=True)
    dataset = namespace["MAEData"]("synthetic-cache")
    assert torch.equal(dataset.obs[0], torch.tensor([[0.0, 1.0]]))

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: torch.ones((2, 2)))
    with pytest.raises(ValueError, match="1, height, width"):
        namespace["MAEData"]("synthetic-cache", sanity_checks=True)

    valid_observation = torch.ones((1, 2, 2))
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: valid_observation.clone()
    )
    checked_dataset = namespace["MAEData"]("synthetic-cache", sanity_checks=True)
    assert checked_dataset[0].shape == (1, 2, 2)

    namespace["_glob_paths"] = lambda pattern: []
    empty_dataset = namespace["MAEData"]("synthetic-cache")
    assert len(empty_dataset) == 0
    with pytest.raises(ValueError, match="No cached SSL observations"):
        namespace["MAEData"]("synthetic-cache", sanity_checks=True)


def test_ssl_fits_dataset_checks_finite_observations_when_enabled(
    monkeypatch, tmp_path
):
    import ast
    import os
    from types import SimpleNamespace

    import numpy as np
    import pytest
    import torch

    print(f"Reading from {TEST_SSL_PATH}...")
    source = TEST_SSL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_SSL_PATH))
    save_dataset = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_dataset"
    )
    invalid_data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)

    class FakeHDUList:
        def __enter__(self):
            return [None, SimpleNamespace(data=invalid_data)]

        def __exit__(self, *args):
            return False

    interpolation_calls = []
    saved = []
    namespace = {
        "F_torch": SimpleNamespace(
            interpolate=lambda observation, size: (
                interpolation_calls.append((observation, size)) or observation
            )
        ),
        "_glob_paths": lambda pattern: ["synthetic-observation.fits"],
        "fits": SimpleNamespace(open=lambda path: FakeHDUList()),
        "get_dataset_dir": lambda: str(tmp_path),
        "get_mae_dataset_dir": lambda: str(tmp_path / "mae_dataset"),
        "np": np,
        "os": os,
        "print": lambda message: None,
        "torch": torch,
    }
    monkeypatch.setattr(
        torch, "save", lambda value, path: saved.append((value, path))
    )
    exec(
        compile(ast.Module(body=[save_dataset], type_ignores=[]), str(TEST_SSL_PATH), "exec"),
        namespace,
    )

    namespace["_glob_paths"] = lambda pattern: []
    with pytest.raises(ValueError, match="No FITS observations"):
        namespace["save_dataset"](sanity_checks=True)
    assert interpolation_calls == []
    assert saved == []

    namespace["_glob_paths"] = lambda pattern: ["synthetic-observation.fits"]
    with pytest.raises(ValueError, match="finite values"):
        namespace["save_dataset"](sanity_checks=True)
    assert interpolation_calls == []
    assert saved == []

    invalid_data = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D image or 3-D image stack"):
        namespace["save_dataset"](sanity_checks=True)
    assert interpolation_calls == []
    assert saved == []

    invalid_data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    namespace["save_dataset"]()
    assert len(saved) == 3
    assert torch.isnan(saved[0][0]).any()


def test_ssl_training_runs_only_from_script_entrypoint():
    import ast

    print(f"Reading from {TEST_SSL_PATH}...")
    source = TEST_SSL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_SSL_PATH))
    main_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    entry_guard = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "main"
        for statement in entry_guard.body
        for node in ast.walk(statement)
    )
    assert "MAEData()" not in ast.unparse(tree).split("def main():", 1)[0]
    assert "ViTMAEForPreTraining" not in ast.unparse(tree).split("def main():", 1)[0]
    assert main_function.name == "main"


def test_timing_decorator_is_quiet_by_default_and_verbose_on_request(
    monkeypatch, capsys
):
    import importlib.util

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    print(f"Reading from {TIMING_UTILS_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_timing_utils", TIMING_UTILS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    timestamps = iter((10.0, 12.5, 20.0, 21.25))
    monkeypatch.setattr(module, "time", lambda: next(timestamps))

    @module.timing
    def sample(value):
        return value + 1

    assert sample(2) == 3
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert sample(4) == 5
    assert capsys.readouterr().out == (
        "func:'sample' args:[(4,), {}] took: 1.2500 sec\n"
    )