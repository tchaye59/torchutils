from typing import List

import torch
import torch.nn as nn

from callbacks import Callback


def to_device(data, device=None):
    """Move tensor(s) to chosen device"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def history_to_string(history):
    return ' - '.join([f'{key}: {history[key][-1].item():.2f}' for key in history])


class BaseModel(nn.Module):
    """ """

    def __init__(self, ):
        super().__init__()
        self.loss_fn = None
        self.optimizer = None
        self.metrics_fn = {}
        self.callbacks: List[Callback] = []

    def forward(self, X):
        assert False, 'forward not defined'

    def compile(self, metrics={}, loss=None, optimizer=None, callbacks: List[Callback] = []):
        self.metrics_fn = metrics
        self.loss_fn = loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        [callback.set_model(self) for callback in callbacks]

    def fit(self, train_loader, epochs=1, val_loader=None):
        assert self.optimizer, "Optimizer not defined"
        assert self.loss_fn, "loss function not defined"

        history = {}
        history_sum = {}
        train_steps = len(train_loader)
        # Training
        [callback.on_train_begin() for callback in self.callbacks]
        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1}/{epochs}:')
            [callback.on_epoch_begin(epoch) for callback in self.callbacks]
            for batch_idx, batch in enumerate(train_loader):
                info = self.training_step(batch)
                self.update_history(history, history_sum, info, batch_idx, train_steps)
                if batch_idx % 1 == 0:
                    print(f'{batch_idx + 1}/{len(train_loader)}  {history_to_string(history)}\r', end='')
            [callback.on_epoch_end(epoch, history) for callback in self.callbacks]
            print()
            history_sum = {}

        [callback.on_train_end(history) for callback in self.callbacks]

        return history

    def update_history(self, history, history_sum, info, batch_idx,train_steps):
        for key in info:
            ss = history_sum.get(key, 0) + info[key]
            history_sum[key] = ss
            if batch_idx+1 == train_steps:
                data = history.get(key, [])
                data.append(ss / (batch_idx + 1))
                history[key] = data

    def training_step(self, batch):
        X, y_true = batch
        y_true = y_true.detach()
        self.optimizer.zero_grad()
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y_true)
        if type(loss) == dict:
            for key in loss:
                loss[key].backward()
        else:
            loss.backward()
        self.optimizer.step()

        # build metrics
        history = {}
        with torch.no_grad():
            for key in self.metrics_fn:
                history[f'val_{key}'] = self.metrics_fn[key](y_true, y_pred)
            if type(loss) == dict:
                history.update(loss)
            else:
                history['val_loss'] = loss
        return history

    def validation_step(self, batch):
        with torch.no_grad():
            X, y_true = batch
            y_pred = self(X)
            loss = self.loss_fn(y_pred, y_true)

            # build metrics
            history = {}
            for key in self.metrics_fn:
                history[key] = self.metrics_fn[key](y_true, y_pred)
            if type(loss) == dict:
                history.update(loss)
            else:
                history['loss'] = loss
            return history

    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
