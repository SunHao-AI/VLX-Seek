# -*- coding: utf-8 -*-
"""1b-2 测试：object features 推理（图像 + bbox_list + images_aux）。

运行：
    python -m vllm_serve.test_object_features --model-path resources/VLX-Seek-1.5-10B
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    model_path = args.model_path

    # 1. 初始化 vLLM
    import vllm_serve.plugin
    vllm_serve.plugin.init()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=8192,
    )

    # 2. aux image processor = CLIPImageProcessor（C-RADIOv4 硬编码配置）
    from transformers import CLIPImageProcessor

    aux_processor = CLIPImageProcessor(
        do_resize=False,
        do_center_crop=False,
        do_rescale=True,
        do_normalize=False,
        do_convert_rgb=True,
        resample=3,
    )

    # 3. 构建测试数据：纯红色 448x448 图片 + 2 个 bbox
    img = Image.new("RGB", (448, 448), "red")

    # bbox_list: [num_images, num_bbox, 4]（带 batch 维，与字段配置 batched 对齐）
    bbox_list = torch.tensor(
        [[[50, 50, 200, 200], [200, 200, 400, 400]]], dtype=torch.float32
    )

    # images_aux: CLIPImageProcessor 输出 [1, C, H, W]（带 batch 维）
    aux_data = aux_processor.preprocess(img, return_tensors="pt")
    images_aux = aux_data["pixel_values"]

    # 4. 构建 prompt（<objfeat> 会被 processor 替换为 248181）
    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "<obj0><objfeat><obj1><objfeat>\n"
        "描述图片中的目标<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("=" * 60)
    print("2) object features 生成（图像 + 2 个 bbox）")
    print(f"prompt: {prompt!r}")
    print(f"bbox_list: {bbox_list.shape} -> {bbox_list}")
    print(f"images_aux shape: {images_aux.shape}")

    t0 = time.perf_counter()
    outs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {"image": img},
            "mm_processor_kwargs": {
                "bbox_list": bbox_list,
                "images_aux": images_aux,
            },
        }],
        SamplingParams(max_tokens=64, temperature=0.0),
    )
    elapsed = time.perf_counter() - t0
    print(f"[生成耗时] {elapsed:.1f}s")
    print(f"  output: {outs[0].outputs[0].text!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
