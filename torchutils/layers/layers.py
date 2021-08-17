import torch


class LambdaModule(torch.nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        return self.loss_fn(pred, target)