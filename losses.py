import torch
from torch.nn import functional as F


def cross_entropy_focal_loss(y_pred, y_true, alpha=1, gamma=2, mask=None):
    # important to add reduction='none' to keep per-batch-item loss
    ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
    return focal_loss


def binary_cross_entropy_focal_loss(y_pred, y_true, alpha=1, gamma=2, mask=None):
    # important to add reduction='none' to keep per-batch-item loss
    ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
    return focal_loss


def bce_focal_loss(alpha=1, gamma=2):
    def fn(y_pred, y_true, mask=None):
        return binary_cross_entropy_focal_loss(y_pred, y_true, alpha, gamma, mask=mask)
    return fn


def ce_focal_loss(alpha=1, gamma=2):
    def fn(y_pred, y_true, mask=None):
        return cross_entropy_focal_loss(y_pred, y_true, alpha, gamma, mask=mask)
    return fn
