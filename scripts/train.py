import math
from numbers import Integral, Real

import torch
from torch.utils.data import DataLoader

from src.data.dataloaders import SynthDatasetv2
import src.models.mlp_network as mlp
import src.utils.seed as sd
from src.utils import get_dataset_dir

from src.utils.time import timing
from tdpy.verbosity import print, tqdm

from accelerate import Accelerator

import os


def validate_training_parameters(
    num_epochs, batch_size, learning_rate, enabled=False
):
    if not enabled:
        return
    for name, value in (("num_epochs", num_epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")


def validate_training_batch(batch, labels, enabled=False):
    if not enabled:
        return
    if batch.ndim != 3 or any(size == 0 for size in batch.shape):
        raise ValueError(
            "Training batches must be nonempty 3-D (batch, height, width) arrays"
        )
    if not bool(torch.isfinite(batch).all()):
        raise ValueError("Training batch must contain only finite values")
    images = batch.flatten(start_dim=1)
    if bool((images.amax(dim=1) <= images.amin(dim=1)).any()):
        raise ValueError("Each training image must have a nonzero range")
    if labels.ndim != 1 or labels.shape[0] != batch.shape[0]:
        raise ValueError("Training labels must match the batch size")
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("Training labels must contain only finite values")
    if labels.numel() and (
        bool((labels < 0).any()) or bool((labels >= 2).any())
    ):
        raise ValueError("Training labels must be binary class indices")
    if labels.is_floating_point() and not bool((labels == labels.round()).all()):
        raise ValueError("Training labels must be integer class indices")


def validate_finite_loss(loss, enabled=False):
    if enabled and not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Training loss must be finite")


def train_epoch(model, train_loader, optimizer, accelerator, sanity_checks=False):
    model.train()

    batch_loss = 0

    for batch, label in train_loader:
        validate_training_batch(batch, label, enabled=sanity_checks)
        optimizer.zero_grad()

        loss = model.module.get_loss(batch, label)
        validate_finite_loss(loss, enabled=sanity_checks)
        accelerator.backward(loss)
        optimizer.step()

        batch_loss += loss.item()
    
    return batch_loss / len(train_loader)

@timing
def main(num_epochs=1, batch_size=2**8, learning_rate=1e-3, sanity_checks=False):
    validate_training_parameters(
        num_epochs, batch_size, learning_rate, enabled=sanity_checks
    )
    sd.seed_everything(42)

    accelerator = Accelerator()
    device = accelerator.device

    model = mlp.MLPNetwork() 

    # https://huggingface.co/docs/accelerate/concept_guides/performance
    learning_rate *= accelerator.num_processes

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = SynthDatasetv2()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    with tqdm(range(num_epochs)) as epochs:
        for _ in epochs:
            loss = train_epoch(
                model, train_loader, optimizer, accelerator,
                sanity_checks=sanity_checks,
            )
            epochs.set_postfix(loss=loss)
        
        checkpoint_directory = os.path.join(get_dataset_dir(), "models")
        os.makedirs(checkpoint_directory, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_directory, "model.ckpt")
        print(f"Writing to {checkpoint_path}...")
        torch.save(model, checkpoint_path)

