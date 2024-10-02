import torch
from torch.utils.data import DataLoader

from src.data.dataloaders import SynthDataset
import src.models.mlp_network as mlp
import src.utils.seed as sd

from src.utils.time import timing

from accelerate import Accelerator

import tqdm
import os


def train_epoch(model, train_loader, optimizer, accelerator):
    model.train()

    batch_loss = 0

    for batch, label in train_loader:
        optimizer.zero_grad()

        loss = model.module.get_loss(batch, label)
        
        accelerator.backward(loss)
        optimizer.step()

        batch_loss += loss.item()
    
    return batch_loss / len(train_loader)

@timing
def main(num_epochs=1, batch_size=2**8, learning_rate=1e-3):
    sd.seed_everything(42)

    accelerator = Accelerator()
    device = accelerator.device

    model = mlp.MLPNetwork() 

    # https://huggingface.co/docs/accelerate/concept_guides/performance
    learning_rate *= accelerator.num_processes

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = SynthDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    with tqdm.trange(num_epochs) as epochs:
        for _ in epochs:
            loss = train_epoch(model, train_loader, optimizer, accelerator)
            epochs.set_postfix(loss=loss)
        
        os.makedirs('models', exist_ok=True)
        torch.save(model, 'models/model.ckpt')

