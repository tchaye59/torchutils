import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms as T

from callbacks import ModelCheckpoint
from losses import binary_cross_entropy_focal_loss, cross_entropy_focal_loss, binary_cross_entropy_weighted_focal_loss
from metrics import accuraty, binary_accuraty
from models.base import BaseModel
from models.mlhydra import MLHydra

dataset = MNIST(root='data/', download=True, transform=ToTensor(), )

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
        self.out_features_size = hidden_size
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
        pred = self.linear2(out)
        return torch.softmax(pred, -1), out


class MnistModelHead(BaseModel):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size, ):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, 32)
        # output layer
        self.linear2 = nn.Linear(32, 1)

    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return torch.sigmoid(out)


input_size = 784
num_classes = 10
backbone = MnistModel(input_size, 32, num_classes)

optim = torch.optim.Adam(backbone.parameters(), 0.001)

callbacks = [
    ModelCheckpoint('model.pth', monitor='loss', mode='min', verbose=True)
]

backbone.compile(loss=lambda y_pred, y_true: cross_entropy_focal_loss(y_pred[0], y_true),
                 optimizer=optim,
                 metrics={'acc': lambda y_pred, y_true: accuraty(y_pred[0], y_true)},
                 callbacks=callbacks,
                 )


def get_head(idx, input_size):
    model = MnistModelHead(input_size)
    optim = torch.optim.Adam(model.parameters(), 0.001)
    callbacks = [
        ModelCheckpoint(f'head{idx + 1}_model.pth', monitor=f'acc{idx + 1}', mode='max', verbose=True)
    ]

    model.compile(
        loss=binary_cross_entropy_weighted_focal_loss,
        optimizer=optim,
        metrics={f'acc': binary_accuraty}, callbacks=callbacks
    )
    return model


heads = [get_head(i, backbone.out_features_size) for i in range(num_classes)]

hydra = MLHydra(backbone, heads)

# hydra.fit_backbone(train_loader,
#                    epochs=2,
#                    val_loader=val_loader)

hydra.fit_heads(train_loader,
                epochs=2,
                val_loader=val_loader, )

# for X, _ in train_loader:
#     p = hydra.predict(X)
#     print(p.shape)
