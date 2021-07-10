import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchutils import BaseModel
from torchutils.utils import *
from torchutils.callbacks.callbacks import ModelCheckpoint
from torchutils.losses.losses import cross_entropy_focal_loss
from torchutils.metrics.metrics import accuraty

dataset = MNIST(root='data/', download=True, transform=ToTensor())
val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

labels = [y for _, y in train_ds]

batch_size = 128

train_loader = DataLoader(train_ds,
                          # batch_size,
                          batch_sampler=RandomBalancedSampler(list(range(len(labels))), labels, batch_size=batch_size),
                          # shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

for x in train_loader:
    break


class MnistModel(BaseModel):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out


input_size = 784
hidden_size = 32
num_classes = 10
model = MnistModel(input_size, hidden_size, num_classes)

optim = torch.optim.Adam(model.parameters(), 0.001)

callbacks = [
    ModelCheckpoint('model.pth', monitor='acc', mode='max', verbose=True)
]

model.compile(loss=lambda y_pred, y_true: F.cross_entropy(y_pred, y_true.view(-1)),
              optimizer=optim,
              metrics={'acc': accuraty},
              callbacks=callbacks)

if __name__ == '__main__':
    model.fit(train_loader,
              epochs=3,
              val_loader=val_loader)
