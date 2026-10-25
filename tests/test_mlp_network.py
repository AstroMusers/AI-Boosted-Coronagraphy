import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as functional
from tdpy.verbosity import print


MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "mlp_network.py"


def load_mlp_network():
    print(f"Reading from {MODEL_PATH}...")
    specification = importlib.util.spec_from_file_location(
        "ai_boosted_mlp_network", MODEL_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.MLPNetwork


def test_loss_uses_logits_while_forward_returns_probabilities():
    model_type = load_mlp_network()
    model = model_type(in_dim=2)
    inputs = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    labels = torch.tensor([0, 1])

    logits = model._logits(inputs)
    probabilities = model(inputs)

    torch.testing.assert_close(model.get_loss(inputs, labels), functional.cross_entropy(logits, labels))
    torch.testing.assert_close(probabilities, functional.softmax(logits, dim=1))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))