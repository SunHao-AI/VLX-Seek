# -*- coding: utf-8 -*-
"""Task 2 基准：prefix caching 命中验证。

构造 N 个共享前缀请求（同图 + 同 objects + 不同类别后缀），
对比 enable_prefix_caching on/off 的端到端耗时。

运行（需分别跑两次对比）：
    # APC 开启
    python -m vllm_serve.bench_prefix_cache --model-path resources/VLX-Seek-1.5-10B --enable-prefix-cache
    # APC 关闭（对照基线）
    python -m vllm_serve.bench_prefix_cache --model-path resources/VLX-Seek-1.5-10B
"""
from __future__ import annotations

import argparse
import time

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=14,
        help="共享前缀请求数（模拟同 crop 14 组 prompt）",
    )
    parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help="开启 automatic prefix caching（默认关）",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    import vllm_serve.plugin
    vllm_serve.plugin.init()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=8192,
        enable_prefix_caching=args.enable_prefix_cache,
    )

    # aux processor（C-RADIOv4 硬编码配置）
    from transformers import CLIPImageProcessor

    aux_processor = CLIPImageProcessor(
        do_resize=False,
        do_center_crop=False,
        do_rescale=True,
        do_normalize=False,
        do_convert_rgb=True,
        resample=3,
    )

    img = Image.new("RGB", (448, 448), "red")
    bbox_list = torch.tensor(
        [[[50, 50, 200, 200], [200, 200, 400, 400]]], dtype=torch.float32
    )
    aux_data = aux_processor.preprocess(img, return_tensors="pt")
    images_aux = aux_data["pixel_values"]

    # 类别文本（可变后缀），其余前缀完全一致
    categories = [
        "描述图片中的目标",
        "图片中有哪些物体",
        "描述图中的物体及其位置",
        "找出图片中的目标",
        "图片里有什么",
        "详细描述图中的物体",
        "列出图中的所有目标",
        "图中目标的位置在哪里",
        "描述图中物体的特征",
        "图中有什么物体",
        "指出图中的目标",
        "描述图中物体",
        "图片中的物体是什么",
        "图中目标的类别",
    ][: args.num_prompts]

    requests = []
    for cat in categories:
        prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "<obj0><objfeat><obj1><objfeat>\n"
            f"{cat}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        requests.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img},
                "mm_processor_kwargs": {
                    "bbox_list": bbox_list,
                    "images_aux": images_aux,
                },
            }
        )

    print("=" * 60)
    print(
        f"prefix caching benchmark: APC={'ON' if args.enable_prefix_cache else 'OFF'}, "
        f"{len(requests)} requests, shared prefix"
    )
    t0 = time.perf_counter()
    outs = llm.generate(
        requests, SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    )
    elapsed = time.perf_counter() - t0
    print(f"[总耗时] {elapsed:.1f}s ({len(requests)} requests)")
    for i, o in enumerate(outs):
        print(f"  [{i}] {o.outputs[0].text!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
