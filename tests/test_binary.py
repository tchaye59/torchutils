import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import torchutils as U
from torchutils.callbacks.callbacks import ModelCheckpoint
from torchutils.metrics.metrics import binary_accuraty
from torchutils.models import BaseModel

dataset = MNIST(root='data', download=True, transform=ToTensor(),
                target_transform=T.Lambda(lambda y: torch.tensor([float(y == 8), ])), )

val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=0, pin_memory=True)


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
        return torch.sigmoid(out)


input_size = 784
hidden_size = 32
num_classes = 1
model = MnistModel(input_size, hidden_size, num_classes)

optim = torch.optim.Adam(model.parameters(), 0.0001)

callbacks = [
    ModelCheckpoint('model.pth', monitor='loss', mode='min', verbose=True)
]

model = model.compile(loss=F.binary_cross_entropy,
                      optimizer=optim,
                      metrics={'acc': U.metrics.binary_accuraty},
                      callbacks=callbacks)

model.fit(train_loader,
          epochs=2,
          val_loader=val_loader)

model.evaluate(train_loader)
