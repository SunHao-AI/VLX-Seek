from typing import Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn
from timm.models import VisionTransformer, PretrainedCfg, create_model

from .enable_cpe_support import enable_cpe
from .input_conditioner import InputConditioner


class Resolution(NamedTuple):
    height: int
    width: int


class RADIOModel(nn.Module):
    def __init__(
        self,
        model: VisionTransformer,
        input_conditioner: InputConditioner,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        summary_idxs: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.input_conditioner = input_conditioner
        self.register_buffer(
            "summary_idxs",
            summary_idxs
            if summary_idxs is not None
            else torch.empty(0, dtype=torch.int64),
        )
        self._patch_size = patch_size
        self._max_resolution = max_resolution
        self._preferred_resolution = preferred_resolution

    @property
    def blocks(self) -> Iterable[nn.Module]:
        return self.model.blocks

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self._preferred_resolution

    @property
    def min_resolution_step(self) -> int:
        return self.patch_size

    def get_nearest_supported_resolution(
        self, height: int, width: int
    ) -> Resolution:
        step = self.min_resolution_step
        return Resolution(
            height=max(int(round(height / step) * step), step),
            width=max(int(round(width / step) * step), step),
        )

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
    ):
        x = self.input_conditioner(x)
        return self.model.forward_intermediates(
            x,
            indices=indices,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            stop_early=stop_early,
            output_fmt=output_fmt,
            intermediates_only=intermediates_only,
        )


def create_model_from_args(args) -> VisionTransformer:
    """Build the fixed SO400M CPE ViT architecture from explicit config."""
    if args.model != "vit_so400m_patch16_224":
        raise ValueError(
            "Only C-RADIOv4-SO400M is supported; "
            f"received model={args.model!r}."
        )

    with torch.device("cpu"):
        model = create_model(
            args.model,
            pretrained=False,
            pretrained_cfg=PretrainedCfg(),
            in_chans=3,
            num_classes=None,
            weight_init="skip",
        )
        model.norm = nn.Identity()
        model.head = nn.Identity()
        enable_cpe(
            model,
            args.cpe_max_size,
            num_cls_tokens=args.num_cls_tokens,
            register_multiple=args.register_multiple,
            num_registers=args.cpe_num_registers,
        )
    return model
