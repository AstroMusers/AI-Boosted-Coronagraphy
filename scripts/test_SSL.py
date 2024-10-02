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

import tqdm

from accelerate import Accelerator

is_accelerator = False
gradient_accumulation = True

def save_dataset():
    main_dataset_path = '/data/scratch/bariskurtkaya/dataset/NIRCAM'

    i2d_dataset = glob.glob(os.path.join(main_dataset_path, '**', '*i2d.fits'), recursive=True)
    psf_dataset = glob.glob(os.path.join(main_dataset_path, '**', '*psfstack.fits'), recursive=True)
    rateints_dataset = glob.glob(os.path.join(main_dataset_path, '**', '*rateints.fits'), recursive=True)

    dataset_paths = i2d_dataset + psf_dataset + rateints_dataset

    observations = []

    for path in dataset_paths:
        obs = torch.from_numpy(np.array(fits.open(path)[1].data, dtype=np.float32))
        observations.append(obs.unsqueeze(0) if len(obs.shape)==2 else obs)

    observations_interpolated = []

    for idx, obs in enumerate(observations):
        observations_interpolated.append(F_torch.interpolate(obs.unsqueeze(1), 512))

    observations_interpolated = torch.cat(observations_interpolated, dim=0)

    for idx, obs in enumerate(observations_interpolated):
        torch.save(obs.cpu(), f'/data/scratch/bariskurtkaya/dataset/mae_dataset/{idx}.ckpt')

class MAEData(Dataset):
    def __init__(self, obs_paths='/data/scratch/bariskurtkaya/dataset/mae_dataset'):
        self.obs_paths = glob.glob(os.path.join(obs_paths, '*.ckpt'))

        self.obs = []
        for obs_path in self.obs_paths:
            if str(torch.__version__).split('.')[0] == '2':
                self.obs.append(torch.nan_to_num(torch.load(obs_path, weights_only=False, map_location=torch.device('cpu'))))
            else:
                self.obs.append(torch.nan_to_num(torch.load(obs_path, map_location=torch.device('cpu'))))

    def __len__(self):
        return len(self.obs)
        
    def __getitem__(self, idx):
        return self.obs[idx]


if is_accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    lr = 2.5e-4
else:
    device = 'cuda:1'
    lr = 1e-3
    gradient_accumulation = 16

save_checkpoints = 100

num_epoch = 1000
batch_size = 256


mae_dataset = MAEData()
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

lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, end_factor=1e-3, total_iters=40)
lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

try:
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([lr_scheduler1, lr_scheduler2], optimizer)
except:
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([lr_scheduler1, lr_scheduler2])

loss_list = []

torch.nn.utils.clip_grad_norm_(mae.parameters(), 1)

if is_accelerator:
    mae, optimizer, train_loader, lr_scheduler = accelerator.prepare(mae, optimizer, train_loader, lr_scheduler)

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
with tqdm.trange(num_epoch) as pbar:
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

            assert out[0] != float('nan')

            if (gradient_accumulation and idx%gradient_accumulation==gradient_accumulation-1) or not gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=out[0].item())
            loss_list.append(out[0].item())
            loss += out[0].item() / batch.shape[0]
        
        
        pbar.set_postfix(e_loss=loss/len(train_loader))
        lr_scheduler.step()

        if epoch % save_checkpoints == save_checkpoints-1:
            torch.save(mae, f'checkpoints/maev2_{epoch}.ckpt')
            torch.save(torch.from_numpy(np.array(loss_list)), f'loss_list_{epoch}.ckpt')





