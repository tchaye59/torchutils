import torch


def binary_accuraty(y_true, y_pred, treshold=0.5, mask=None):
    with torch.no_grad():
        y_pred = y_pred > treshold
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask
        return (y_pred == y_true).sum() / y_true.size()


def accuraty(y_true, y_pred, mask=None):
    with torch.no_grad():
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask
        _, y_pred = torch.max(y_pred, -1)
        return (y_true == y_pred).sum() / y_pred.size()[0]
