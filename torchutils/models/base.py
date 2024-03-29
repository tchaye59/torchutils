import copy
from typing import Optional
import dill
import pytorch_lightning as pl
import torch
from torchmetrics import Metric, StatScores
from torchutils.layers import Lambda
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
        self.metrics = torch.nn.ModuleDict()
        self.val_metrics = torch.nn.ModuleDict()
        self.losses = torch.nn.ModuleDict()
        self.optimizer = None
        self.history = {}
        self.trackers = torch.nn.ModuleDict()
        self.logs = {}

    def predict(self, X):
        with torch.no_grad():
            X = to_device(X)
            return self(X)

    def compile(self, metrics={}, loss=None, optimizer=None):
        # Losses
        if type(loss) == dict:
            for key, fn in loss.items():
                if not isinstance(fn, torch.nn.Module):
                    loss[key] = Lambda(fn)
            self.losses.update(loss)
        else:
            if not isinstance(loss, torch.nn.Module):
                loss = Lambda(loss)
            self.losses['loss'] = loss

        # optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), 0.001)
        # Metrics
        tmp_metrics = {}
        if metrics:
            for key in metrics:
                metric = metrics[key]
                if not isinstance(metric, Metric):
                    metric = LambdaMetric(metric)
                tmp_metrics[key] = metric
        self.metrics.update(tmp_metrics)
        self.val_metrics.update(copy.deepcopy(tmp_metrics))
        return self

    def update_trackers(self, name, value):
        if name not in self.trackers:
            self.trackers[name] = MeanMetric()
        tracker = self.trackers[name]
        tracker.update(value)
        return tracker.compute()

    def update_history(self, name, value):
        if name not in self.history:
            self.history[name] = []
        value = value.cpu().detach().numpy()
        self.history[name].append(float(value))

    def on_fit_start(self) -> None:
        self.reset_state()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        logs = {}
        x, y = batch
        y_pred = self(x)
        # Compute the loss
        losses = []
        for name in self.losses:
            value = self.losses[name](y_pred, y)
            losses.append(value)
            value = self.update_trackers(name, value)
            self.log(name, value)
            logs[name] = value
        loss = torch.stack(losses).mean()
        # Logging to TensorBoard by default
        # Log losses
        if len(self.losses) > 1:
            value = self.update_trackers("loss", loss)
            self.log("loss", value)
            logs["loss"] = value
        # Log metrics
        for name in self.metrics:
            metric = self.metrics[name]
            metric.update(y_pred, y)
            value = metric.compute()
            if type(metric) == StatScores:
                for tmp_name, val in zip(['tp', 'fp', 'tn', 'fn'], value):
                    self.log(tmp_name, val, on_epoch=True, prog_bar=True, sync_dist=True)
                    logs[tmp_name] = val
            else:
                self.log(name, value, on_epoch=True, prog_bar=True, sync_dist=True)
                logs[name] = value
        self.logs = logs
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        # def training_epoch_end(self, outs):
        # log epoch metric
        for name in self.metrics:
            metric = self.metrics[name]
            value = metric.compute()

            if type(metric) == StatScores:
                for tmp_name, val in zip(['tp', 'fp', 'tn', 'fn'], value):
                    self.log(f"{tmp_name}", val, on_epoch=True, prog_bar=True, sync_dist=True)
                    self.update_history(f"{tmp_name}", val)
            else:
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
        logs = {}
        x, y = batch
        y_pred = self(x)
        # Compute the loss
        losses = []
        for name in self.losses:
            value = self.losses[name](y_pred, y)
            value = self.update_trackers(f"val_{name}", value)
            losses.append(value)
            self.log(f"val_{name}", value, on_epoch=True, prog_bar=True, sync_dist=True)
            logs[f"val_{name}"] = value

        loss = torch.stack(losses).mean()
        # Logging to TensorBoard by default
        # Log losses
        if len(self.losses) > 1:
            loss = self.update_trackers("val_loss", loss)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            logs["val_loss"] = loss
        # Log metrics
        for name in self.val_metrics:
            metric = self.val_metrics[name]
            metric.update(y_pred, y)
            value = metric.compute()
            if type(metric) == StatScores:
                for tmp_name, val in zip(['tp', 'fp', 'tn', 'fn'], value):
                    self.log(f"val_{tmp_name}", val, on_epoch=True, prog_bar=True, sync_dist=True)
                    logs[f"val_{tmp_name}"] = val
            else:
                self.log(f"val_{name}", value, on_epoch=True, prog_bar=True, sync_dist=True)
                logs[f"val_{name}"] = value
        self.logs = logs

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

            if type(metric) == StatScores:
                for tmp_name, val in zip(['tp', 'fp', 'tn', 'fn'], value):
                    name = f"val_{tmp_name}"
                    self.log(name, val, on_epoch=True, prog_bar=True, sync_dist=True)
                    self.update_history(name, val)
            else:
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
