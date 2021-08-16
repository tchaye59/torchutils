import copy
import sys
from typing import Optional

import dill
import pytorch_lightning as pl
import torch
from torchmetrics import Metric

from torchutils.metrics import MeanMetric, LambdaMetric

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
    return ' - '.join([f'{key}: {metrics[key].compute() :.5f}' for key in metrics])


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.val_metrics = {}
        self.losses = {}
        self.optimizer = None
        self.history = {}
        self.trackers = {}

    def predict(self, X):
        with torch.no_grad():
            X = to_device(X)
            return self(X)

    def compile(self, metrics={}, loss=None, optimizer=None):
        # Losses
        if type(loss) == dict:
            self.losses.update(loss)
        else:
            self.losses['loss'] = loss

        # optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), 0.001)
        # Metrics
        if metrics:
            for key in metrics:
                metric = metrics[key]
                if not isinstance(metric, Metric):
                    metric = LambdaMetric(metric)
                self.metrics[key] = metric
        self.val_metrics = copy.deepcopy(self.metrics)
        return self

    def update_trackers(self, name, value):
        if name not in self.trackers:
            self.trackers[name] = MeanMetric()
        tracker = self.trackers[name]
        # tracker.update(value)
        tracker(value)

    def update_history(self, name, value):
        if name not in self.history:
            self.history[name] = []
        value = value.cpu().detach().numpy()
        self.history[name].append(float(value))

    def on_fit_start(self) -> None:
        self.reset_state()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        y_pred = self(x)
        # Compute the loss
        losses = []
        for name in self.losses:
            value = self.losses[name](y_pred, y)
            losses.append(value)
            self.log(name, value)
            self.update_trackers(name, value)

        loss = torch.stack(losses).mean()
        # Logging to TensorBoard by default
        # Log losses
        if len(self.losses) > 1:
            self.log("loss", loss)
            self.update_trackers("loss", loss)
        # Log metrics
        for name in self.metrics:
            self.log(name, self.metrics[name](y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        # def training_epoch_end(self, outs):
        # log epoch metric
        for name in self.metrics:
            metric = self.metrics[name]
            value = metric.compute()
            self.log(name, value)
            # update history
            self.update_history(name, value)
            metric.reset()
        # losses
        for name in self.losses:
            tracker = self.trackers[name]
            value = tracker.compute()
            self.log(name, value)
            # update history
            self.update_history(name, value)
            tracker.reset()
        print("Train end")

    def reset_state(self):
        self.history = {}
        for _, item in self.metrics.items():
            item.reset()
        for _, item in self.val_metrics.items():
            item.reset()
        for _, item in self.trackers.items():
            item.reset()

    def get_history(self):
        min_size = min([len(self.history[key]) for key in self.history])
        history = [(key, item[-min_size:]) for key, item in self.history.items()]
        return dict(history)

    def on_epoch_start(self):
        print('\n')

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        y_pred = self(x)
        # Compute the loss
        losses = []
        for name in self.losses:
            value = self.losses[name](y_pred, y)
            losses.append(value)
            self.log(f"val_{name}", value, on_epoch=True, prog_bar=True, sync_dist=True)
            self.update_trackers(f"val_{name}", value)

        loss = torch.stack(losses).mean()
        # Logging to TensorBoard by default
        # Log losses
        if len(self.losses) > 1:
            self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.update_trackers("val_loss", loss)
        # Log metrics
        for name in self.val_metrics:
            self.log(f"val_{name}", self.val_metrics[name](y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        self.reset_state()
        self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        # def training_epoch_end(self, outs):
        # log epoch metric
        for name in self.val_metrics:
            metric = self.val_metrics[name]
            value = metric.compute()
            name = f"val_{name}"
            self.log(name, value)
            # update history
            self.update_history(name, value)
            metric.reset()
        # losses
        for name in self.losses:
            name = f"val_{name}"
            tracker = self.trackers[name]
            value = tracker.compute()
            self.log(name, value)
            # update history
            self.update_history(name, value)
            tracker.reset()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        return self.optimizer


class ModelWrapper(BaseModel):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)

    def wrapped_model(self):
        return self.model
