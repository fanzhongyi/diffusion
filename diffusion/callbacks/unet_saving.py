# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Unet Saving for inference."""

from typing import List, Optional

import torch
from composer import Callback, Logger, State
from composer.core import TimeUnit, get_precision_context
from torch.nn.parallel import DistributedDataParallel


class UNetSaving(Callback):
    """Save Unet torch model for inference.

    """

    def __init__(self,):
        ...

    def eval_batch_end(self, state: State, logger: Logger):
        ...
