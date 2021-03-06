import sys
from typing import List

import dill
import torch
import torch.nn as nn
from torchutils.callbacks import Callback

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_device(data, device=None):
    """Move tensor(s) to chosen device"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def detach(data, ):
    """Detach Tensor from grad"""
    if isinstance(data, (list, tuple)):
        return [detach(x) for x in data]
    return data.detach()


def load_model(path, device=None, pickle_module=dill):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(path, pickle_module=pickle_module, map_location=torch.device(device))


def epoch_info_to_string(info, n_steps):
    return ' - '.join([f'{key}: {info[key].item() / n_steps:.3f}' for key in info])


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

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = to_device(X)
            return self(X)

    def compile(self, metrics={}, loss=None, optimizer=None, callbacks: List[Callback] = [], device=None):
        self.metrics_fn = metrics
        self.loss_fn = loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        [callback.set_model(self) for callback in callbacks]
        return to_device(self, device)

    def fit(self, train_loader, epochs=1, val_loader=None, accum=1, accum_mode=1):

        assert self.optimizer, "Optimizer not defined"
        assert self.loss_fn, "loss function not defined"

        history = {}

        train_steps = len(train_loader)
        val_steps = len(val_loader) if val_loader is not None else 0
        self.train(mode=True)

        for epoch in range(epochs):
            epoch_info_sum = {}
            losses = {}

            print(f'Epoch: {epoch + 1}/{epochs}:')
            [callback.on_epoch_begin(epoch) for callback in self.callbacks]
            # Training
            [callback.on_train_begin() for callback in self.callbacks]

            max_accum = accum
            for batch_idx, batch in enumerate(train_loader):
                # is_last_step = (batch_idx + 1) >= train_steps
                accum_step = batch_idx % accum
                if accum_step == 0:
                    max_accum = accum if batch_idx + accum < train_steps else train_steps - batch_idx

                batch = to_device(batch)
                losses, info = self.training_step(batch, accum_step, max_accum, losses=losses, accum_mode=accum_mode)

                self.update_history(history, epoch_info_sum, info, n_steps=batch_idx + 1)

                print(f'Training: {batch_idx + 1}/{train_steps}  {epoch_info_to_string(epoch_info_sum, batch_idx + 1)}',
                      end='\r', file=sys.stdout, flush=True)
            self.update_history(history, epoch_info_sum, info, n_steps=batch_idx + 1, epoch_end=True)
            [callback.on_train_end(history) for callback in self.callbacks]

            # Validation
            if val_loader is not None:
                epoch_info_sum = {}
                print()
                [callback.on_test_begin(epoch, ) for callback in self.callbacks]
                for batch_idx, batch in enumerate(val_loader):
                    # is_last_step = (batch_idx + 1) >= train_steps
                    batch = to_device(batch)
                    info = self.validation_step(batch)
                    self.update_history(history, epoch_info_sum, info, n_steps=batch_idx + 1)
                    print(
                        f'Validation: {batch_idx + 1}/{val_steps}  {epoch_info_to_string(epoch_info_sum, batch_idx + 1)}',
                        end='\r',
                        file=sys.stdout, flush=True)
                self.update_history(history, epoch_info_sum, info, n_steps=batch_idx + 1, epoch_end=True)
                [callback.on_test_end(history) for callback in self.callbacks]

            [callback.on_epoch_end(epoch, history) for callback in self.callbacks]
            print()

        return history

    def update_history(self, history, epoch_info_sum, info, n_steps=1, epoch_end=False):
        if history is not None and epoch_end:
            for key in info:
                ss = epoch_info_sum.get(key, 0)
                ss = float((ss / n_steps).cpu().numpy())
                data = history.get(key, [])
                data.append(ss)
                history[key] = data
        else:
            for key in info:
                # if val.numel() > 1:
                #    val = val.mean()
                ss = epoch_info_sum.get(key, 0) + info[key].detach()
                epoch_info_sum[key] = ss

    def training_step(self, batch, accum_step, accum, losses={}, accum_mode=1):
        X, y_true = batch
        y_true = detach(y_true)
        X = detach(X)

        if accum_step == 0:
            self.optimizer.zero_grad()
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y_true)

        # accumulate losses
        if type(loss) == dict:
            for key in loss:
                arr = losses.get(key, [])
                arr.append(loss[key])
                losses[key] = arr
        else:
            arr = losses.get('loss', [])
            arr.append(loss)
            losses['loss'] = arr

        if accum_mode != 0:
            # fill gradients
            if type(loss) == dict:
                [(loss[key] / accum).backward() for key in loss]
            else:
                (loss / accum).backward()

        if accum_step == (accum - 1):
            # fill gradients
            if accum_mode == 0:
                for key in losses:
                    l = sum(losses[key]) / len(losses[key])
                    l.backward()
            # update weights
            self.optimizer.step()
            losses = {}

        # build metrics
        history = {}
        with torch.no_grad():
            for key in self.metrics_fn:
                history[key] = self.metrics_fn[key](y_pred, y_true)
            if type(loss) == dict:
                history.update(loss)
            else:
                history['loss'] = loss
        return losses, history

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

    def evaluate(self, data_loader, ):
        history = {}
        history_sum = {}
        steps = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            is_last_step = (batch_idx + 1) >= steps
            batch = to_device(batch)
            info = self.validation_step(batch)
            self.update_history(history, history_sum, info, batch_idx + 1, epoch_end=is_last_step)
            if batch_idx % 1 == 0:
                print(f'Evaluate: {batch_idx + 1}/{steps}  {epoch_info_to_string(history_sum, batch_idx + 1)}',
                      end='\r',
                      file=sys.stdout,
                      flush=True)
        print()
        return history
