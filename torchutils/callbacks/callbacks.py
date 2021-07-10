import torch
import dill

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchutils.models import BaseModel


class Callback(object):
    """Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.model: 'BaseModel'

    def set_model(self, model: 'BaseModel'):
        self.model = model

    def on_epoch_begin(self, epoch):
        """Called at the start of an epoch.
        """

    def on_epoch_end(self, epoch, logs):
        """Called at the end of an epoch.
        """

    def on_train_begin(self, ):
        """Called at the beginning of training.
        """

    def on_train_end(self, logs):
        """Called at the end of training.
        """

    def on_test_begin(self, epoch):
        """Called at the beginning of evaluation or validation.
        """

    def on_test_end(self, epoch, logs=None):
        """Called at the end of evaluation or validation.
        """


class ModelCheckpoint(Callback):
    """Callback to save the model or model weights."""

    def __init__(self, filepath, monitor='val_loss', verbose=True, save_weights_only=True, mode=''):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        assert mode in ['min', 'max'], 'Invalid mode'
        self.mode = mode
        self.last_value = float('-inf') if mode == 'max' else float('inf')

    def compare(self, prev_val, val):
        if self.mode == 'max':
            return val > prev_val
        else:
            return val < prev_val

    def on_epoch_end(self, epoch, logs=None):
        if not self.monitor in logs:
            return
        prev_val = self.last_value
        val = logs[self.monitor][-1]
        comp = self.compare(prev_val, val)

        if self.verbose:
            if not comp:
                print(f'\nEpoch {epoch + 1}: {self.monitor} did not improve from {prev_val:.5f}')
            else:
                print(f'\nEpoch {epoch + 1}: {self.monitor} improved from {prev_val:.5f} to {val:.5f}, saving model to {self.filepath}')
        # save the model
        if comp:
            self.last_value = val
            if self.save_weights_only:
                torch.save(self.model.state_dict(), self.filepath, pickle_module=dill)
            else:
                torch.save(self.model, self.filepath, pickle_module=dill)
