from typing import NamedTuple, Optional

import torch
from torch import nn


class InterFeatState(NamedTuple):
    y: torch.Tensor
    alpha: torch.Tensor


class IntermediateFeatureNormalizerBase(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        index: int,
        rot_index: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> InterFeatState:
        raise NotImplementedError


class NullIntermediateFeatureNormalizer(IntermediateFeatureNormalizerBase):
    _instances = {}

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(1, dtype=dtype, device=device))

    @classmethod
    def get_instance(cls, dtype: torch.dtype, device: torch.device):
        key = (dtype, device)
        if key not in cls._instances:
            cls._instances[key] = cls(dtype, device)
        return cls._instances[key]

    def forward(
        self,
        x: torch.Tensor,
        index: int,
        rot_index: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> InterFeatState:
        return InterFeatState(x, self.alpha)
