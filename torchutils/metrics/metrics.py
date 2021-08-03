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


def binary_accuraty(y_pred, y_true, treshold=0.5):
    with torch.no_grad():
        y_pred = y_pred > treshold
        return (y_pred.view(-1) == y_true.view(-1)).sum() / y_true.numel()


def accuraty(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.argmax(y_pred, -1)
        return (y_true.view(-1) == y_pred.view(-1)).sum() / y_pred.numel()
