#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM 自定义注册冒烟测试（服务器运行，仓库根目录执行）。

    python -m vllm_serve.test_vllm --model-path resources/VLX-Seek-1.5-10B [--image a.jpg]

输出：
  1. text-only 生成（验证 注册+config+权重加载+引擎 全链路）
  2. 图像生成（验证 embed_multimodal 图像路径）
"""
import argparse
import time

import vllm_serve.plugin  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    vllm_serve.plugin.init()  # 手动注册（未安装 entry point 时）

    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.7)
    print(f"[加载耗时] {time.perf_counter() - t0:.1f}s")

    # 1) text-only
    print("=" * 60)
    print("1) text-only 生成")
    t0 = time.perf_counter()
    outs = llm.generate(["你好，请简单介绍一下你自己。"], sampling_params=SamplingParams(max_tokens=32, temperature=0.0))
    print(f"[生成耗时] {time.perf_counter() - t0:.1f}s")
    for o in outs:
        print(f"  output: {o.outputs[0].text!r}")

    # 2) 图像
    if args.image:
        print("=" * 60)
        print("2) 图像生成")
        from PIL import Image
        from vlx_seek.models.vlx_seek_1_5.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        )

        pil = Image.open(args.image).convert("RGB")
        prompt = (
            f"<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
            "检测一下图片里的物体。<|im_end|>\n<|im_start|>assistant\n"
        )
        t0 = time.perf_counter()
        outs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": pil}}],
            sampling_params=SamplingParams(max_tokens=64, temperature=0.0),
        )
        print(f"[生成耗时] {time.perf_counter() - t0:.1f}s")
        for o in outs:
            print(f"  output: {o.outputs[0].text!r}")


if __name__ == "__main__":
    main()
