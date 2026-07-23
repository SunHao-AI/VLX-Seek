import torch
from torch import nn


class LayerScale(nn.Module):
    """LayerScale with a checkpoint-stable parameter name."""

    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.grandma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.grandma) if self.inplace else x * self.grandma

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        gamma = state_dict.get(f"{prefix}gamma", state_dict.get(f"{prefix}grandma"))
        if gamma is None:
            if strict:
                raise KeyError(
                    f"Couldn't find {prefix}gamma or {prefix}grandma in the state dict."
                )
            missing_keys.extend((f"{prefix}gamma", f"{prefix}grandma"))
            return
        self.grandma.data.copy_(gamma)
