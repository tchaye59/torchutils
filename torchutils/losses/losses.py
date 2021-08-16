import torch
from torch import nn
from torch.nn import functional as F

from torchutils import to_device


class FocalLoss(nn.Module):
    """weighted version of Focal Loss"""

    def __init__(self, alpha=.25, gamma=2, device=None):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.alpha = to_device(self.alpha, device=device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        targets = targets.long()
        at = self.alpha.gather(0, targets.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def binary_cross_entropy_weighted_focal_loss(y_pred, y_true, alpha=0.25, gamma=6, mask=None):
    return FocalLoss(alpha=alpha, gamma=gamma, )(y_pred, y_true)


def cross_entropy_focal_loss(y_pred, y_true, weight=None, alpha=0.25, gamma=6, mask=None):
    # important to add reduction='none' to keep per-batch-item loss
    ce_loss = F.cross_entropy(y_pred, y_true, reduction='none', weight=weight)
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
    return focal_loss


def binary_cross_entropy_focal_loss___(y_pred, y_true, alpha=0.25, gamma=6, mask=None):
    # important to add reduction='none' to keep per-batch-item loss
    ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
    return focal_loss


def bce_focal_loss(alpha=0.25, gamma=6):
    def fn(y_pred, y_true, mask=None):
        return binary_cross_entropy_focal_loss___(y_pred, y_true, alpha, gamma, mask=mask)

    return fn


def ce_focal_loss(alpha=0.25, gamma=6):
    def fn(y_pred, y_true, mask=None):
        return cross_entropy_focal_loss(y_pred, y_true, alpha, gamma, mask=mask)

    return fn
