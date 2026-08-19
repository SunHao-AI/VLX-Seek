"""Minimal stub for transformers to satisfy imports in tests.
Only provides AutoTokenizer class with from_pretrained returning self.
"""

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
