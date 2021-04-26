import torch


class Mean:
    def __init__(self, ):
        self.sum = 0.
        self.count = 0

    def getValue(self, ):
        return self.sum / self.count

    def update_state(self, x):
        self.sum += x
        self.count += 1


class Means:
    def __init__(self, ):
        self.means = {}

    def update(self, key, val):
        if key not in val:
            self.means[key] = Mean()
        self.means[key].update_state(val)

    def update_from_dict(self, values):
        for key in values:
            self.update(key, values[key])

    def to_dict(self):
        return dict([(k, self.means[k].getValue()) for k in self.means])


def binary_accuraty(y_pred, y_true, treshold=0.5, mask=None):
    with torch.no_grad():
        y_pred = y_pred > treshold
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask
        return (y_pred == y_true).sum() / torch.numel(y_true)


def accuraty(y_pred, y_true, mask=None):
    with torch.no_grad():
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask
        _, y_pred = torch.max(y_pred, -1)
        return (y_true == y_pred).sum() / y_pred.size()[0]
