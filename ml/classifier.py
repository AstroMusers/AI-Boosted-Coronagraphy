import torch
from torch.utils.data import DataLoader
from glob import glob
import os
import wandb
import random
import data


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # input 120, 120
        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=7, stride=1, padding=3) # 120,120 -> 120,120
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3) # 120,120 -> 120,120
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3) # 120,120 -> 120,120
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3) # 120,120 -> 120,120
        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3) # 120,120 -> 120,120

        self.fc = torch.nn.Linear(128*120*120, 1) # 120,120 -> 1,1

        self.activation = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

        self.BN1 = torch.nn.BatchNorm2d(32)
        self.BN2 = torch.nn.BatchNorm2d(64)

        self.dropout = torch.nn.Dropout2d(p=0.4)

    def forward(self, x):

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1, 128*120*120)
        x = self.fc(x)

        x = self.tanh(x)

        x = torch.where(x < 0.5, x, 1)
        x = torch.where(x > -0.5, x, -1)
        x = torch.where((-0.5 < x) == (x < 0.5), 0, x)

        return x

def train_with_validation(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.type(torch.float32).squeeze(1), target.type(torch.float32))
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(epoch, epochs, i, len(train_loader), loss.item()))
                wandb.log({"Training Loss": loss.item()})
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                output = model(data)
                loss = criterion(output.type(torch.float32).squeeze(1), target.type(torch.float32))
                val_loss += loss.item()

                val_correct += (output == target).sum().item()
                val_total += len(target)

        print('Epoch: {}/{}, Validation Loss: {}, Validation Accuracy: {}'.format(epoch, epochs, val_loss/len(val_loader), val_correct/val_total))
        wandb.log({"Validation Loss": val_loss/len(val_loader), "Validation Accuracy": val_correct/val_total, "Learning Rate": scheduler.get_last_lr()[0]})

        if epoch % 30 == 0 and epoch != 0:
            for g in optimizer.param_groups:
                g['lr'] = 0.005
        scheduler.step()

        if (epoch % 10 == 0 and epoch != 0) or epoch == epochs-1:
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/model_{}.pth'.format(epoch))

def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        # 1 for exoplanet and -1 for non-exoplanet
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output.type(torch.float32).squeeze(1), target.type(torch.float32))
            test_loss += loss.item()

            test_correct += (output == target).sum().item()
            test_total += len(target)

    print('Test Loss: {}, Test Accuracy: {}'.format(test_loss/len(test_loader), test_correct/test_total))
    wandb.log({"Test Loss": test_loss/len(test_loader), "Test Accuracy": test_correct/test_total})

def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        return output


if __name__ == '__main__':
    wandb.init(
        # set the wandb project where this run will be logged
        project="CORON_CNN",
        name='CNN_1386_25102023',
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "ConvNet",
        "epochs": 300,
        }
)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = '/home/bariskurtkaya/github/AI-Boosted-Coronagraphy/ml/models/model_99.pth'

    train_path = '/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/injections_24102023/train'
    test_path = '/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/injections_24102023/test'

    train_dataset = glob(os.path.join(train_path, '*.npy'))
    test_dataset = glob(os.path.join(test_path, '*.npy'))

    random.shuffle(train_dataset)

    val_dataset = train_dataset[:int(0.1*len(train_dataset))]
    train_dataset = train_dataset[int(0.1*len(train_dataset)):]


    nircam_train = data.NIRCamDataset(train_dataset, device)
    nircam_val = data.NIRCamDataset(val_dataset, device)
    nircam_test = data.NIRCamDataset(test_dataset, device)

    nircam_train_loader = DataLoader(nircam_train, batch_size=512, shuffle=True)
    nircam_val_loader = DataLoader(nircam_val, batch_size=256, shuffle=True)
    nircam_test_loader = DataLoader(nircam_test, batch_size=128, shuffle=True)

    os.makedirs('./models', exist_ok=True)
    if os.path.exists(model_name):
        model = ConvNet()
        model.load_state_dict(torch.load(model_name))
        model.to(device)
    else:
        model = ConvNet()
        model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    train_with_validation(model, criterion, optimizer, scheduler, nircam_train_loader, nircam_val_loader, epochs=300)

    test(model, criterion, nircam_test_loader)

    wandb.finish()