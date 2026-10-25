import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    def __init__(self, in_dim=256*256, out_class=2):
        super().__init__()

        self.MLP1 = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU()
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU()
        )

        self.MLP3 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU()
        )

        self.classifier = nn.Linear(128, out_class)
    
    def _logits(self, x: torch.Tensor):
        batch, width, height = x.shape

        x = x.view(batch, -1)

        x = self.MLP1(x)
        x = self.MLP2(x)
        x = self.MLP3(x)
        x = self.classifier(x)

        return x

    def forward(self, x: torch.Tensor):
        return F.softmax(self._logits(x), dim=1)
    
    def get_loss(self, x, gt):
        return F.cross_entropy(self._logits(x), gt)

    
    def get_accuracy(self, x, gt):
        pred = self.forward(x)
        acc = (pred.argmax(1) == gt).type(torch.float).sum().item()
        return acc

