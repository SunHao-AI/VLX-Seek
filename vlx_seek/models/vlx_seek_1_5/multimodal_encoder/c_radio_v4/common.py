from dataclasses import dataclass
from .radio_model import Resolution


@dataclass(frozen=True)
class RadioResource:
    patch_size: int
    max_resolution: int
    preferred_resolution: Resolution


DEFAULT_VERSION = "c-radio_v4-so400m"

RESOURCE_MAP = {
    DEFAULT_VERSION: RadioResource(
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
    )
}
