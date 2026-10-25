import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import Dataset, DataLoader

import transformers

import torchvision.transforms.v2 as v2

import numpy as np

from astropy.io import fits

import glob
import os
import sys

from accelerate import Accelerator
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils import get_dataset_dir
from tdpy.verbosity import print, tqdm

is_accelerator = False
gradient_accumulation = True
sanity_checks_enabled = os.environ.get("AIC_SANITY_CHECKS", "0").strip().lower() in {
    "1", "true", "yes"
}


def _glob_paths(pattern):
    print(f"Reading from {pattern}...")
    return glob.glob(pattern)


def validate_finite_loss(loss, enabled=False):
    if enabled and not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Training loss must be finite")


def get_mae_dataset_dir():
    return os.path.join(get_dataset_dir(), "mae_dataset")


def get_mae_checkpoint_dir():
    return os.path.join(get_dataset_dir(), "mae_checkpoints")


def save_dataset(sanity_checks=False):
    main_dataset_path = os.path.join(get_dataset_dir(), "NIRCAM")

    i2d_dataset = _glob_paths(os.path.join(main_dataset_path, '**', '*i2d.fits'))
    psf_dataset = _glob_paths(os.path.join(main_dataset_path, '**', '*psfstack.fits'))
    rateints_dataset = _glob_paths(os.path.join(main_dataset_path, '**', '*rateints.fits'))

    dataset_paths = i2d_dataset + psf_dataset + rateints_dataset
    if sanity_checks and not dataset_paths:
        raise ValueError("No FITS observations were found for the SSL dataset")

    observations = []

    for path in dataset_paths:
        print(f"Reading from {path}...")
        with fits.open(path) as hdus:
            obs = torch.from_numpy(np.array(hdus[1].data, dtype=np.float32))
        if sanity_checks:
            if obs.ndim not in (2, 3) or any(size == 0 for size in obs.shape):
                raise ValueError(
                    "Observed FITS data must be a nonempty 2-D image or 3-D image stack"
                )
            if not bool(torch.isfinite(obs).all()):
                raise ValueError("Observed FITS image must contain only finite values")
        observations.append(obs.unsqueeze(0) if len(obs.shape)==2 else obs)

    observations_interpolated = []

    for idx, obs in enumerate(observations):
        observations_interpolated.append(F_torch.interpolate(obs.unsqueeze(1), 512))

    observations_interpolated = torch.cat(observations_interpolated, dim=0)

    output_directory = get_mae_dataset_dir()
    os.makedirs(output_directory, exist_ok=True)
    for idx, obs in enumerate(observations_interpolated):
        output_path = os.path.join(output_directory, f"{idx}.ckpt")
        print(f"Writing to {output_path}...")
        torch.save(obs.cpu(), output_path)

class MAEData(Dataset):
    def __init__(self, obs_paths=None, sanity_checks=False):
        if obs_paths is None:
            obs_paths = get_mae_dataset_dir()
        self.obs_paths = _glob_paths(os.path.join(obs_paths, '*.ckpt'))
        if sanity_checks and not self.obs_paths:
            raise ValueError("No cached SSL observations were found")

        self.obs = []
        for obs_path in self.obs_paths:
            print(f"Reading from {obs_path}...")
            if str(torch.__version__).split('.')[0] == '2':
                obs = torch.load(obs_path, weights_only=False, map_location=torch.device('cpu'))
            else:
                obs = torch.load(obs_path, map_location=torch.device('cpu'))
            if sanity_checks and not bool(torch.isfinite(obs).all()):
                raise ValueError("Cached observation tensor must contain only finite values")
            if sanity_checks and (
                obs.ndim != 3 or obs.shape[0] != 1 or any(size == 0 for size in obs.shape)
            ):
                raise ValueError(
                    "Cached observations must have nonempty (1, height, width) shape"
                )
            self.obs.append(torch.nan_to_num(obs))

    def __len__(self):
        return len(self.obs)
        
    def __getitem__(self, idx):
        return self.obs[idx]


def main():
    if is_accelerator:
        accelerator = Accelerator()
        device = accelerator.device
        lr = 2.5e-4
        gradient_accumulation_steps = 1
    else:
        accelerator = None
        device = 'cuda:1'
        lr = 1e-3
        gradient_accumulation_steps = 16

    save_checkpoints = 100
    num_epoch = 1000
    batch_size = 256

    mae_dataset = MAEData(sanity_checks=sanity_checks_enabled)
    train_loader = DataLoader(mae_dataset, batch_size=batch_size, shuffle=True)
    mae_config = transformers.ViTMAEConfig(
        num_hidden_layers=12,
        num_attention_heads=12,
        patch_size=16,
        image_size=224,
        num_channels=1,
        mask_ratio=0.75,
        attn_implementation="eager"
    )
    mae = transformers.ViTMAEForPreTraining(mae_config)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr)
    lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=5e-6, end_factor=1e-3, total_iters=40
    )
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10000
    )

    try:
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [lr_scheduler1, lr_scheduler2], optimizer
        )
    except:
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [lr_scheduler1, lr_scheduler2]
        )

    loss_list = []
    checkpoint_directory = get_mae_checkpoint_dir()
    os.makedirs(checkpoint_directory, exist_ok=True)
    torch.nn.utils.clip_grad_norm_(mae.parameters(), 1)

    if is_accelerator:
        mae, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            mae, optimizer, train_loader, lr_scheduler
        )
        lr *= accelerator.num_processes
        batch_size *= accelerator.num_processes
    else:
        mae = mae.to(device)

    transforms = v2.Compose([
        v2.RandomCrop(size=(224, 224)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
    ])

    print('Training is starting!')
    optimizer.zero_grad()
    with tqdm(range(num_epoch)) as pbar:
        for epoch in pbar:
            loss = 0
            for idx, batch in enumerate(train_loader):
                if not is_accelerator:
                    batch = batch.to(device)
                batch = transforms(batch)

                if is_accelerator:
                    out = mae.module.forward(batch)
                    accelerator.backward(out[0])
                else:
                    out = mae.forward(batch)
                    out[0].backward()

                validate_finite_loss(out[0], enabled=sanity_checks_enabled)

                if (
                    idx % gradient_accumulation_steps
                    == gradient_accumulation_steps - 1
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=out[0].item())
                loss_list.append(out[0].item())
                loss += out[0].item() / batch.shape[0]

            pbar.set_postfix(e_loss=loss / len(train_loader))
            lr_scheduler.step()

            if epoch % save_checkpoints == save_checkpoints - 1:
                checkpoint_path = os.path.join(
                    checkpoint_directory, f"maev2_{epoch}.ckpt"
                )
                loss_path = os.path.join(
                    checkpoint_directory, f"loss_list_{epoch}.ckpt"
                )
                print(f"Writing to {checkpoint_path}...")
                torch.save(mae, checkpoint_path)
                print(f"Writing to {loss_path}...")
                torch.save(torch.from_numpy(np.array(loss_list)), loss_path)


if __name__ == "__main__":
    main()





