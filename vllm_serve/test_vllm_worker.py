# -*- coding: utf-8 -*-
"""Task 3 冒烟测试：VLXSeekVLLMWorker 接口（detect / detect_multi_prompt）。

运行：
    python -m vllm_serve.test_vllm_worker --model-path resources/VLX-Seek-1.5-10B
"""
from __future__ import annotations

import argparse

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    from vllm_serve.vlx_seek_vllm_worker import VLXSeekVLLMWorker

    worker = VLXSeekVLLMWorker(args.model_path, gpu_memory_utilization=0.7)
    worker.log_timing = True

    img = Image.new("RGB", (448, 448), "red")
    boxes = [[50, 50, 200, 200], [200, 200, 400, 400]]

    print("=" * 60)
    print("1) detect_multi_prompt（2 批类别，批量提交）")
    result = worker.detect_multi_prompt(
        img,
        boxes,
        [["person", "car"], ["dog", "cat"]],
        lang="zh",
        max_new_tokens=64,
        temperature=0.0,
    )
    print(f"  answer: {result['answer']!r}")
    print(f"  result_bbox_list: {len(result['result_bbox_list'])} items")
    print(f"  prompt_tokens: {result['prompt_tokens']}")

    print("=" * 60)
    print("2) detect（单批）")
    result2 = worker.detect(
        img, boxes, ["person", "car"], lang="zh", max_new_tokens=64, temperature=0.0
    )
    print(f"  answer: {result2['answer']!r}")
    print(f"  result_bbox_list: {len(result2['result_bbox_list'])} items")
    print(f"  prompt_tokens: {result2['prompt_tokens']}")

    print("=" * 60)
    print("3) predict_batch（混合请求批量）")
    results = worker.predict_batch(
        [
            {"image": img, "question": "描述图片", "bbox_list": boxes},
            {"image": img, "question": "图片里有什么"},
        ],
        max_new_tokens=64,
        temperature=0.0,
    )
    for r in results:
        print(f"  answer: {r['answer']!r} | bbox: {len(r['result_bbox_list'])}")
    print("=" * 60)

    worker.clear_image_cache()
    print("OK: VLXSeekVLLMWorker smoke test passed")


if __name__ == "__main__":
    main()
