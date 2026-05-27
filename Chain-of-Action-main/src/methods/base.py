# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn
from typing import Union, Tuple, Any


BatchedActionSequence = Union[torch.Tensor, np.ndarray]
Metrics = dict[str, np.ndarray]

class BaseMethod(nn.Module, ABC):
    """
    Base class for all methods
    """
    
    def __init__(self, accelerator=None, *args, **kwargs):
        super().__init__()
        self.accelerator = accelerator
        if accelerator is not None:
            self.prepare_accelerator()
        
    @abstractmethod
    def forward(self, observations: dict[str, torch.Tensor], training: bool = True) -> Union[BatchedActionSequence, Tuple[Any, ...]]:
        pass

    def prepare_accelerator(self) -> None:
        if self.accelerator is None:
            return
        for k, v in self.__dict__.items():
            if isinstance(v, nn.Module):
                setattr(self, k, self.accelerator.prepare_model(v))
            elif isinstance(v, torch.optim.Optimizer):
                setattr(self, k, self.accelerator.prepare_optimizer(v))
            elif isinstance(v, torch.optim.lr_scheduler.LRScheduler):
                setattr(self, k, self.accelerator.prepare_scheduler(v))
