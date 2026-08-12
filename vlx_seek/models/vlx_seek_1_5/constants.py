import importlib.util
import platform

# Windows 不支持 flash_attn，使用 PyTorch 内置的 sdpa；Linux 服务器使用 flash_attention_2，
# 未安装 flash-attn 时自动回退 sdpa（避免 from_pretrained 时抛 ImportError）
if platform.system() == "Windows":
    ATTN_IMPLEMENTATION = "sdpa"
elif importlib.util.find_spec("flash_attn") is not None:
    ATTN_IMPLEMENTATION = "flash_attention_2"
else:
    ATTN_IMPLEMENTATION = "sdpa"

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

DEFAULT_IM_START_TOKEN = "<|vision_start|>"
DEFAULT_IM_END_TOKEN = "<|vision_end|>"

VLX_SEEK_1_5_IMAGE_TOKEN = "<|image_pad|>"
VLX_SEEK_1_5_IMAGE_TOKEN_INDEX = 248056
VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX = 248181

DEFAULT_OBJECT_TOKEN = "<obj<i>>"
DEFAULT_OBJECT_FEATURE_TOKEN = "<objfeat>"
DEFAULT_OBJECT_INDEX = -300
