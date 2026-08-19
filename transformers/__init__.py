"""Minimal stub for transformers to satisfy imports in tests.
Only provides AutoTokenizer, AutoConfig, and AutoModelForCausalLM classes with from_pretrained returning self.
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
