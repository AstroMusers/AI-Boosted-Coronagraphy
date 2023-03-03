import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F

import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms

from tqdm.notebook import tqdm

import optuna

#import wandb

import pickle5 as pickle

from warnings import filterwarnings
filterwarnings('ignore')

torch.manual_seed(42)

"""
def wandb_init(num_epochs):
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-test-project",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": "Optuna lr",
        "architecture": "CNN AE",
        "dataset": "MNIST",
        "epochs": num_epochs,
        }
    )
"""

def fetch_mnist():
    train_dataset = torchvision.datasets.MNIST('./dataset/', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST('./dataset/', train=False, download=True)

    return train_dataset, test_dataset

def create_loader(train, test, batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_dataset.transform = transform
    test_dataset.transform = transform

    total_train_size = len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(total_train_size-total_train_size*0.2), int(total_train_size*0.2)])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    print("Batches in Train Loader: {}".format(len(train_loader)))
    print("Batches in Valid Loader: {}".format(len(valid_loader)))
    print("Batches in Test Loader: {}".format(len(test_loader)))

    print("Examples in Train Loader: {}".format(len(train_loader.sampler)))
    print("Examples in Valid Loader: {}".format(len(valid_loader.sampler)))
    print("Examples in Test Loader: {}".format(len(test_loader.sampler)))

    return train_loader, valid_loader, test_loader


def set_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('Total GPU count:', torch.cuda.device_count())
        print('Selected GPU index:', torch.cuda.current_device())
        current_device = torch.cuda.current_device()
        print('Selected GPU Name:', torch.cuda.get_device_name(current_device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB')
        print('Max Memmory Cached:', round(torch.cuda.max_memory_cached(current_device)/1024**3,1), 'GB')
    
    return device

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


def objective(trial):

    device = set_device()
    num_epochs= 20


    loss_fn = torch.nn.MSELoss()

    fc2_input_dim = trial.suggest_int("fc2_input_dim", 32, 128,step=32)
    encoded_space_dim = trial.suggest_int("encoded_space_dim", 32, 128,step=32)
    # Generate the model.
    encoder = Encoder(encoded_space_dim, fc2_input_dim).to(device)
    decoder = Decoder(encoded_space_dim, fc2_input_dim).to(device)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    # Generate the optimizers.

    # try RMSprop and SGD
    '''
    optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,momentum=momentum)
    '''
    
    #try Adam, AdaDelta adn Adagrad
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta","Adagrad"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1,log=True)
    optimizer = getattr(optim, optimizer_name)(params_to_optimize, lr=lr)
    batch_size=  256 #trial.suggest_int("batch_size", 64, 256,step=64)
    
    # Training of the model.
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
       
        for batch_idx, (images, _) in enumerate(train_loader):
            # Limiting training images for faster epochs.
            #if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
            #    break

            images = images.to(device)

            # Encode data
            encoded_data = encoder(images)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Evaluate loss
            loss = loss_fn(decoded_data, images)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Validation of the model.
        encoder.eval()
        decoder.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(valid_loader):
                # Limiting validation images.
               # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                #    break
                images = images.to(device)
                # Encode data
                encoded_data = encoder(images)
                # Decode data
                decoded_data = decoder(encoded_data)

                val_loss = loss_fn(decoded_data, images)
                        
        trial.report(val_loss.data, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    with open("{}_enc.pickle".format(trial.number), "wb") as fout:
        pickle.dump(encoder, fout)
    with open("{}_dec.pickle".format(trial.number), "wb") as fout:
        pickle.dump(decoder, fout)

    
    return loss



batch_size = 256

train_dataset, test_dataset = fetch_mnist()
train_loader, valid_loader, test_loader = create_loader(train_dataset, test_dataset, batch_size)

if __name__ == "__main__":
    #num_epochs = 10

    #wandb_init(num_epochs)

    study = optuna.create_study(direction='minimize')
    
    study.optimize(objective, n_trials=20)

    trial = study.best_trial

    print('Loss: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    wandb.log({"best": trial.value})
    wandb.log({"params": trial.params})
    
    wandb.finish()



    """
    # Load the best model.
    with open("{}_enc.pickle".format(study.best_trial.number), "rb") as fin:
        best_enc = pickle.load(fin)
    with open("{}_dec.pickle".format(study.best_trial.number), "rb") as fin:
        best_enc = pickle.load(fin)
    """


    #print(accuracy_score(y_valid, best_clf.predict(X_valid)))











