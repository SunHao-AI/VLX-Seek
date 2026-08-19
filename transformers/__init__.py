"""Minimal stub for transformers to satisfy imports in tests.
Only provides AutoTokenizer, AutoConfig, and AutoModelForCausalLM classes with from_pretrained returning self.
Also provides stubs for specific model configurations like Qwen3_5Config.
"""


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class AutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


# Stub for specific model imports that might be needed during testing
class Qwen3_5Config:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


# Make the module structure appear to exist for transformers.models.qwen3_5.configuration_qwen3_5
import sys
from types import ModuleType

# Create the module hierarchy: transformers.models.qwen3_5.configuration_qwen3_5
transformers_models_qwen3_5_configuration_qwen3_5 = ModuleType("transformers.models.qwen3_5.configuration_qwen3_5")
transformers_models_qwen3_5_configuration_qwen3_5.Qwen3_5Config = Qwen3_5Config

# Register the modules in sys.modules so imports work
sys.modules["transformers"] = sys.modules.get("transformers", ModuleType("transformers"))
sys.modules["transformers.models"] = sys.modules.get("transformers.models", ModuleType("transformers.models"))
sys.modules["transformers.models.qwen3_5"] = sys.modules.get("transformers.models.qwen3_5", ModuleType("transformers.models.qwen3_5"))
sys.modules["transformers.models.qwen3_5.configuration_qwen3_5"] = transformers_models_qwen3_5_configuration_qwen3_5

# Also attach to the transformers module if it exists
if "transformers" in sys.modules:
    if not hasattr(sys.modules["transformers"], "models"):
        sys.modules["transformers"].models = sys.modules["transformers.models"]
    if not hasattr(sys.modules["transformers"].models, "qwen3_5"):
        sys.modules["transformers"].models.qwen3_5 = sys.modules["transformers.models.qwen3_5"]
    if not hasattr(sys.modules["transformers"].models.qwen3_5, "configuration_qwen3_5"):
        sys.modules["transformers"].models.qwen3_5.configuration_qwen3_5 = transformers_models_qwen3_5_configuration_qwen3_5
