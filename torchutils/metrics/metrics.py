import torch
from torchmetrics.metric import Metric


class LambdaMetric(Metric):
    def __init__(self, metric_fn, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.metric_fn = metric_fn

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        value = self.metric_fn(preds, target)

        self.correct = value if self.correct == 0 else self.correct + value
        self.total = value.numel() if self.total == 0 else self.total + value.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


class MeanMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value):
        self.total_sum += value
        self.count += value.numel()

    def compute(self):
        # compute final result
        return self.total_sum.float() / self.count


def binary_accuracy(y_pred, y_true, treshold=0.5):
    with torch.no_grad():
        y_pred = y_pred > treshold
        return (y_pred.view(-1) == y_true.view(-1)).sum() / y_true.numel()


def accuracy(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.argmax(y_pred, -1)
        return (y_true.view(-1) == y_pred.view(-1)).sum() / y_pred.numel()
