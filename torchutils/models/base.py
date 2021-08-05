import sys
from typing import List

import dill
import torch
import torch.nn as nn
from torchutils.callbacks.callbacks import Callback

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


# deprecated
def load_model(path, device=None, pickle_module=dill):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(path, pickle_module=pickle_module, map_location=torch.device(device))


def load(path, device=None, pickle_module=dill):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(path, pickle_module=pickle_module, map_location=torch.device(device))


def metrics_to_string(metrics):
    return ' - '.join([f'{key}: {metrics[key].get() :.5f}' for key in metrics])


class Metric:

    def __init__(self):
        self.count = 0
        self.sum = 0

    def update(self, value):
        self.count += torch.numel(value)
        self.sum += value.sum()

    def get(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0


class BaseModel(nn.Module):
    """ """

    def __init__(self, ):
        super().__init__()
        self.loss_fn = None
        self.optimizer = None
        self.metrics_fn = {}
        self.metrics = {}
        self.history = {}
        self.callbacks: List[Callback] = []
        self.device = None

    def forward(self, X):
        assert False, 'forward not defined'

    def predict(self, X, eval=True, no_grad=True):
        if eval:
            self.eval()
        if not no_grad:
            return self(X)
        with torch.no_grad():
            X = to_device(X)
            return self(X)

    def compile(self, metrics={}, loss=None, optimizer=None, callbacks: List[Callback] = [], device=None):
        self.metrics_fn = metrics
        self.loss_fn = loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.device = device
        [callback.set_model(self) for callback in callbacks]
        return to_device(self, device)

    def fit(self, train_loader, epochs=1, val_loader=None, accum=1, accum_mode=1):

        assert self.optimizer, "Optimizer not defined"
        assert self.loss_fn, "loss function not defined"

        self.history = {}

        train_steps = len(train_loader)
        val_steps = len(val_loader) if val_loader is not None else 0
        self.train(mode=True)

        for epoch in range(epochs):
            losses = {}

            print(f'Epoch: {epoch + 1}/{epochs}:')
            self.reset_metrics()
            [callback.on_epoch_begin(epoch) for callback in self.callbacks]
            # Training
            [callback.on_train_begin() for callback in self.callbacks]

            max_accum = accum
            for batch_idx, batch in enumerate(train_loader):
                # is_last_step = (batch_idx + 1) >= train_steps
                accum_step = batch_idx % accum
                if accum_step == 0:
                    max_accum = accum if batch_idx + accum < train_steps else train_steps - batch_idx

                batch = to_device(batch, device=self.device)
                losses, info = self.training_step(batch, accum_step, max_accum, losses=losses, accum_mode=accum_mode)
                self.update_metrics(info)

                sys.stdout.write(f'\rTraining: {batch_idx + 1}/{train_steps}  {metrics_to_string(self.metrics)}', )
                sys.stdout.flush()
            self.update_history()
            self.reset_metrics()
            [callback.on_train_end(self.history) for callback in self.callbacks]

            # Validation
            if val_loader is not None:
                epoch_info_sum = {}
                print()
                [callback.on_test_begin(epoch, ) for callback in self.callbacks]
                for batch_idx, batch in enumerate(val_loader):
                    # is_last_step = (batch_idx + 1) >= train_steps
                    batch = to_device(batch)
                    info = self.validation_step(batch)
                    self.update_metrics(info)
                    sys.stdout.write(f'\rValidation: {batch_idx + 1}/{val_steps}  {metrics_to_string(self.metrics)}')
                    sys.stdout.flush()

                [callback.on_test_end(self.history) for callback in self.callbacks]

            self.update_history()
            self.reset_metrics()
            [callback.on_epoch_end(epoch, self.history) for callback in self.callbacks]

            print()

        return self.history

    def update_history(self):
        for key in self.metrics:
            arr = self.history.get(key, [])
            arr.append(self.metrics[key].get().cpu().detach().numpy())
            self.history[key] = arr

    def training_step(self, batch, accum_step, accum, losses={}, accum_mode=1):
        X, y_true = batch
        y_true = detach(y_true)
        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, 1)
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
                [(loss[key] / accum).sum().backward() for key in loss]
            else:
                (loss / accum).sum().backward()

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

    def update_metrics(self, logs):
        for key in logs:
            metric = self.metrics.get(key, Metric())
            metric.update(logs[key])
            self.metrics[key] = metric

    def reset_metrics(self, ):
        self.metrics = {}

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
        steps = len(data_loader)
        self.reset_metrics()
        for batch_idx, batch in enumerate(data_loader):
            batch = to_device(batch)
            info = self.validation_step(batch)
            self.update_metrics(info)
            self.update_history()
            sys.stdout.write(f'\rEvaluate: {batch_idx + 1}/{steps}  {metrics_to_string(self.metrics)}')
            sys.stdout.flush()
        print()
        return self.history


class ModelWrapper(BaseModel):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)

    def wrapped_model(self):
        return self.model
