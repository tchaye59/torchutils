import torch
import dill

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from torchutils.models import BaseModel


from tqdm.auto import tqdm

import torch
from pytorch_lightning.callbacks import Callback


