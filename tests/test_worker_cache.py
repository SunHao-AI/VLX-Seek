"""验证 VLXSeekWorker 图片缓存分支：命中缓存时跳过图像预处理、返回 elapsed、缓存可清除。

用 __new__ 绕过 __init__（避免加载真实模型），只 stub 出 predict() 依赖的最小接口。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

import vlx_seek_worker as vw
from vlx_seek_worker import VLXSeekWorker


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return type("T", (), {"input_ids": [1]})()

    def decode(self, ids, **kwargs):
        return ""


class FakeModel:
    def __init__(self):
        self.cleared = False

    def generate(self, **kwargs):
        # 返回 (1, prompt_len + 3)，模拟解码了 3 个 token
        n = kwargs["inputs"].shape[1]
        return torch.arange(n + 3).unsqueeze(0)

    def clear_cached_image(self):
        self.cleared = True


def make_worker(prep_counter: dict) -> VLXSeekWorker:
    w = VLXSeekWorker.__new__(VLXSeekWorker)
    w.device = torch.device("cpu")
    w.tokenizer = FakeTokenizer()
    w.model = FakeModel()
    w._cached_inputs = None
    w.log_timing = False

    def fake_prepare(image, boxes):
        prep_counter["n"] += 1
        return ["img"], ["thw"], ["aux"]

    w._prepare_image_inputs = fake_prepare
    w._expand_multimodal_tokens = lambda raw, thws: torch.arange(raw.shape[1])
    return w


def test_cache_skips_prepare_and_returns_elapsed():
    vw.tokenizer_image_token = lambda prompt, tok, return_tensors="pt": torch.tensor([[5, 6, 7]])
    img = Image.new("RGB", (64, 64))

    prep_counter = {"n": 0}
    w = make_worker(prep_counter)

    # 无缓存：调用 _prepare_image_inputs
    r1 = w.predict(img, "q")
    assert prep_counter["n"] == 1, "未命中缓存时应调用图像预处理"
    assert "elapsed" in r1 and r1["elapsed"] >= 0

    # 命中缓存：不再调用 _prepare_image_inputs
    w._cached_inputs = ("img", "thw", "aux")
    r2 = w.predict(img, "q")
    assert prep_counter["n"] == 1, "命中缓存时不应再次调用图像预处理"
    assert "elapsed" in r2

    # clear 后缓存与模型缓存同时清空
    w.clear_image_cache()
    assert w._cached_inputs is None and w.model.cleared


if __name__ == "__main__":
    test_cache_skips_prepare_and_returns_elapsed()
    print("test_worker_cache OK")
