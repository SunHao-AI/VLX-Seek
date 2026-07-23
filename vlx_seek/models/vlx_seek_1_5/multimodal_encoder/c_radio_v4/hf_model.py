from collections import namedtuple
from typing import Callable, List, Optional, Union

import torch
from timm.models import VisionTransformer
from transformers import PretrainedConfig, PreTrainedModel

from .common import DEFAULT_VERSION, RESOURCE_MAP
from .input_conditioner import InputConditioner, get_default_conditioner
from .radio_model import RADIOModel as RADIOModelBase
from .radio_model import Resolution, create_model_from_args
from . import extra_timm_models  # Registers vit_so400m_patch16_224 with timm.


class RADIOConfig(PretrainedConfig):
    """Configuration for VLX-Seek's C-RADIOv4-SO400M inference backbone."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: str = DEFAULT_VERSION,
        patch_size: Optional[int] = None,
        max_resolution: Optional[int] = None,
        preferred_resolution: Optional[Union[Resolution, List[int]]] = None,
        summary_idxs: Optional[List[int]] = None,
        **kwargs,
    ):
        if version != DEFAULT_VERSION:
            raise ValueError(
                "Only C-RADIOv4-SO400M is supported; "
                f"received version={version!r}."
            )
        resource = RESOURCE_MAP[version]
        self.args = args or {}
        self.version = version
        self.patch_size = patch_size or resource.patch_size
        self.max_resolution = max_resolution or resource.max_resolution
        self.preferred_resolution = preferred_resolution or resource.preferred_resolution
        self.summary_idxs = summary_idxs or [0, 1]
        super().__init__(**kwargs)


class RADIOModel(PreTrainedModel):
    """Hugging Face wrapper for VLX-Seek's minimal C-RADIOv4 inference path."""

    config_class = RADIOConfig

    def __init__(self, config: RADIOConfig):
        super().__init__(config)
        args = namedtuple("RADIOArgs", config.args.keys())(**config.args)
        model = create_model_from_args(args)
        # Keep conditioner buffers off the parent loader's meta-device
        # initialization path for the same reason as the timm backbone.
        with torch.device("cpu"):
            input_conditioner: InputConditioner = get_default_conditioner()

        dtype = getattr(torch, args.dtype)
        model.to(dtype=dtype)
        input_conditioner.dtype = dtype

        preferred_resolution = Resolution(*config.preferred_resolution)
        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            summary_idxs=torch.tensor(config.summary_idxs, dtype=torch.int64),
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            preferred_resolution=preferred_resolution,
        )

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> InputConditioner:
        return self.radio_model.input_conditioner

    @property
    def patch_size(self) -> int:
        return self.radio_model.patch_size

    @property
    def max_resolution(self) -> int:
        return self.radio_model.max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self.radio_model.preferred_resolution

    @property
    def min_resolution_step(self) -> int:
        return self.radio_model.min_resolution_step

    def get_nearest_supported_resolution(
        self, height: int, width: int
    ) -> Resolution:
        return self.radio_model.get_nearest_supported_resolution(height, width)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError(
            "VLX-Seek only consumes C-RADIO intermediate features."
        )
