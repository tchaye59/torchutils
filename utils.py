import sys
from time import sleep
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
    return ' - '.join([f'{key}: {history[key][-1].item():.3f}' for key in history])


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
        val_steps = len(val_loader) if val_loader is not None else 0

        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1}/{epochs}:')
            [callback.on_epoch_begin(epoch) for callback in self.callbacks]

            # Training
            [callback.on_train_begin() for callback in self.callbacks]
            for batch_idx, batch in enumerate(train_loader):
                info = self.training_step(batch)
                self.update_history(history, history_sum, info, batch_idx, train_steps)
                if batch_idx % 1 == 0:
                    print(f'{batch_idx + 1}/{train_steps}  {history_to_string(history)}', end='\r', file=sys.stdout,
                          flush=True)
                    sys.stdout.flush()
            [callback.on_train_end(history) for callback in self.callbacks]

            # Validation
            if val_loader is not None:
                [callback.on_test_begin(epoch, ) for callback in self.callbacks]
                for batch_idx, batch in enumerate(val_loader):
                    info = self.validation_step(batch)
                    self.update_history(history, history_sum, info, batch_idx, val_steps)
                [callback.on_test_end(history) for callback in self.callbacks]
            print(f'{train_steps}/{train_steps}  {history_to_string(history)}', end='\r', file=sys.stdout,flush=True)

            [callback.on_epoch_end(epoch, history) for callback in self.callbacks]
            print()
            history_sum = {}

        return history

    def update_history(self, history, history_sum, info, batch_idx, train_steps):
        for key in info:
            ss = history_sum.get(key, 0) + info[key]
            history_sum[key] = ss
            if batch_idx + 1 == train_steps:
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
                history[key] = self.metrics_fn[key](y_pred, y_true)
            if type(loss) == dict:
                history.update(loss)
            else:
                history['loss'] = loss
        return history

    def validation_step(self, batch):
        with torch.no_grad():
            X, y_true = batch
            y_pred = self(X)
            loss = self.loss_fn(y_pred, y_true)

            # build metrics
            history = {}
            for key in self.metrics_fn:
                history[f'val_{key}'] = self.metrics_fn[key](y_pred, y_true)
            if type(loss) == dict:
                history.update(loss)
            else:
                history['val_loss'] = loss
            return history
