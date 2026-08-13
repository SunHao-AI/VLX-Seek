#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0 可行性尖峰脚本——必须在 GPU 服务器上运行（本机无 CUDA）。

目标（对应计划 Task 0）：
1. 输出环境版本矩阵（torch / vllm / transformers / flash_attn / CUDA / GPU 计算能力），
   与计划中的版本兼容性表对照，确认 vLLM 0.17.0 + torch 2.10.0 + transformers 5.13.0 是否可加载。
2. 尝试让 vLLM 直接加载 VLXSeek1_5 模型（text-only 路径先跑通 LLM 主干）。
   - 若直接加载失败（预期：未知 architecture 名，vLLM 不认识 VLXSeek1_5ForCausalLM），
     打印确切报错并记录到报告——这就是 Task 1 需要做自定义注册的证据。
3. 用 HF 基线（VLXSeekWorker.predict）跑同一 prompt，输出作为 vLLM 输出的一致性对照基准。

用法（服务器，仓库根目录）：
    python vllm_serve/minimal_spike.py --model-path <model_dir> [--prompt "检测一下图片里的猫" --image /path/to/test.png]

注意：
- 本脚本只做"环境探测 + 直接加载尝试 + HF 基线"，不含 vLLM 自定义模型注册
  （那是 Task 1 的 vllm_serve/vlx_seek_vlm.py 工作，gate 通过后开始）。
- vLLM 版本 API 以实际安装版本为准（`pip show vllm` 查看）。
"""
from __future__ import annotations

import argparse
import platform
import sys

# ---------------------------------------------------------------------------
# Step 1: 环境版本矩阵
# ---------------------------------------------------------------------------


def check_env() -> None:
    print("=" * 64)
    print("1) 环境版本矩阵（对照计划中的版本兼容性表）")
    print("=" * 64)
    print(f"  平台            : {platform.platform()}")
    print(f"  Python          : {sys.version.split()[0]}")
    try:
        import torch

        print(f"  torch           : {torch.__version__}  (cuda={torch.version.cuda})")
        print(f"  CUDA 可用       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU             : {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            print(f"  计算能力        : {cap[0]}.{cap[1]}")
    except ImportError as exc:
        print(f"  [错误] torch 导入失败: {exc}", file=sys.stderr)
    try:
        import transformers

        print(f"  transformers    : {transformers.__version__}")
    except ImportError as exc:
        print(f"  [错误] transformers 导入失败: {exc}", file=sys.stderr)
    try:
        import vllm

        print(f"  vllm            : {vllm.__version__}")
    except ImportError as exc:
        print(f"  [警告] vllm 未安装或导入失败: {exc}", file=sys.stderr)
    try:
        import flash_attn

        print(f"  flash_attn      : {flash_attn.__version__}")
    except ImportError as exc:
        print(f"  [警告] flash_attn 未安装或导入失败: {exc}", file=sys.stderr)
    print()


# ---------------------------------------------------------------------------
# Step 2: vLLM 直接加载尝试（text-only，不涉及多模态）
# ---------------------------------------------------------------------------


def try_direct_load(model_path: str) -> None:
    print("=" * 64)
    print("2) vLLM 直接加载尝试（text-only）")
    print("=" * 64)
    try:
        from vllm import LLM

        # 预期结果：VLXSeek1_5ForCausalLM 不在 vLLM 支持列表，直接加载大概率失败。
        # 失败信息即为 Task 1 自定义注册的输入。若意外成功，说明 vLLM 已内置 Qwen3.5
        # 同架构映射，则 Task 1 工作量大幅缩小。
        llm = LLM(model=model_path, trust_remote_code=True)
        out = llm.generate("你好，请简单介绍一下你自己。", sampling_params={"max_tokens": 32})
        print(f"  [成功] 直接加载并生成: {out[0].outputs[0].text[:100]!r}")
    except Exception as exc:  # noqa: BLE001 - 尖峰脚本需要捕获任意加载错误并记录
        print(f"  [失败] 直接加载报错（预期，见计划 Task 0 Step 2）:")
        print(f"    {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  → Task 1 需做自定义注册：ModelRegistry.register_model("
              "\"VLXSeek1_5ForCausalLM\", \"vllm_serve.vlx_seek_vlm:VLXSeek1_5ForCausalLM\")")
    print()


# ---------------------------------------------------------------------------
# Step 3: HF 基线输出（一致性对照）
# ---------------------------------------------------------------------------


def hf_baseline(model_path: str, prompt: str, image_path: str | None) -> None:
    print("=" * 64)
    print("3) HF 基线输出（一致性对照基准）")
    print("=" * 64)
    try:
        from vlx_seek_worker import VLXSeekWorker

        worker = VLXSeekWorker(model_path, device="cuda")
        if image_path:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (512, 512))  # 纯文本时给一张空图占位
        result = worker.predict(image, prompt, max_new_tokens=128, temperature=0.0)
        print(f"  [HF] answer : {result['answer']!r}")
        print(f"  [HF] boxes  : {result['result_bbox_list'][:3]}")
        print("  → vLLM 路径跑通后，同 prompt 同 seed 的输出须与此逐字一致（见计划 Task 3 Step 3）。")
    except Exception as exc:  # noqa: BLE001
        print(f"  [失败] HF 基线执行报错: {type(exc).__name__}: {exc}", file=sys.stderr)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="M0 vLLM 可行性尖峰（服务器运行）")
    parser.add_argument("--model-path", required=True, help="模型目录")
    parser.add_argument("--prompt", default="检测一下图片里的所有物体。")
    parser.add_argument("--image", default=None, help="测试图片路径（可选）")
    args = parser.parse_args()

    check_env()
    try_direct_load(args.model_path)
    hf_baseline(args.model_path, args.prompt, args.image)


if __name__ == "__main__":
    main()
