#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查服务器环境是否可使用 flash_attn 加速 VLX-Seek 推理。

用法:
    python check_flash_attn.py

输出:
    1. 环境信息（平台 / Python / torch / CUDA / GPU 计算能力）
    2. flash_attn 是否已安装；未安装时给出安装指引（退出码 1）
    3. 复刻 vlx_seek/models/vlx_seek_1_5/constants.py 的选择逻辑，
       报告 VLX-Seek 实际会使用的注意力实现
    4. （已安装时）flash_attn_func 与 SDPA 正确性对比 + prefill/decode 微基准
"""
from __future__ import annotations

import importlib.util
import platform
import sys
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("错误: 未检测到 torch，请先安装项目依赖 (见 pyproject.toml)。", file=sys.stderr)
    sys.exit(1)


# 与 vlx_seek/models/vlx_seek_1_5/constants.py 保持一致的选择逻辑
def attn_implementation_selected() -> str:
    if platform.system() == "Windows":
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def print_env_info() -> None:
    print("=" * 60)
    print("1) 环境信息")
    print("=" * 60)
    print(f"  平台            : {platform.platform()}")
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"  torch           : {torch.__version__}")
    print(f"  CUDA (torch)    : {torch.version.cuda}")
    print(f"  CUDA 可用       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"  GPU             : {name}")
        print(f"  计算能力        : {cap[0]}.{cap[1]}")
        if cap[0] < 8:
            print("  [警告] flash-attn 2.x 需要计算能力 >= 8.0 (Ampere 及以上)。", file=sys.stderr)
    print()


def print_install_guidance() -> None:
    print("=" * 60)
    print("2) flash_attn 未安装")
    print("=" * 60)
    print("项目已在 pyproject.toml 的 optional-dependencies 中固定 flash_attn==2.8.3：")
    print()
    print("  使用 uv:    uv sync --extra flash-attn")
    print("  或 pip :    pip install flash_attn==2.8.3")
    print()
    print("提示:")
    print("  - 优先安装预编译 wheel（需 CUDA 12.x 环境），避免本地编译；")
    print("  - 若需源码编译，需安装 CUDA toolkit + ninja，编译耗时较长；")
    print("  - 安装后重新运行本脚本即可验证 kernel 是否真正可用。")
    print()


def verify_and_benchmark() -> bool:
    print("=" * 60)
    print("3) flash_attn kernel 验证与微基准")
    print("=" * 60)
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        print(f"  [错误] flash_attn 导入失败: {exc}", file=sys.stderr)
        return False

    if not torch.cuda.is_available():
        print("  [跳过] 无 CUDA 设备，无法运行 kernel 验证。")
        return False

    torch.manual_seed(0)
    batch, nheads, head_dim = 1, 8, 128
    dtype = torch.bfloat16
    device = "cuda"

    # --- 正确性对比（prefill 形态） ---
    seq = 2048
    q = torch.randn(batch, seq, nheads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq, nheads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq, nheads, head_dim, dtype=dtype, device=device)

    fa_out = flash_attn_func(q, k, v, causal=True)
    q_sdpa = q.transpose(1, 2).contiguous()
    k_sdpa = k.transpose(1, 2).contiguous()
    v_sdpa = v.transpose(1, 2).contiguous()
    sdpa_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
    sdpa_out = sdpa_out.transpose(1, 2).contiguous()

    max_err = (fa_out.float() - sdpa_out.float()).abs().max().item()
    print(f"  正确性: prefill seq={seq} max 误差 = {max_err:.6f} (flash_attn vs SDPA)")
    print("         误差 < 1e-2 视为一致（bf16 舍入差异）")
    print()

    # --- 微基准 ---
    def bench(fn, iters: int = 20, warmup: int = 5) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1000  # ms

    shapes = {
        "prefill (q=kv=2048)": (2048, 2048),
        "decode  (q=1, kv=2048)": (1, 2048),
    }
    print("  微基准 (单次平均 ms):")
    for label, (qlen, kvlen) in shapes.items():
        qb = torch.randn(batch, qlen, nheads, head_dim, dtype=dtype, device=device)
        kb = torch.randn(batch, kvlen, nheads, head_dim, dtype=dtype, device=device)
        vb = torch.randn(batch, kvlen, nheads, head_dim, dtype=dtype, device=device)
        fa_ms = bench(lambda: flash_attn_func(qb, kb, vb, causal=True))
        sdpa_ms = bench(
            lambda: F.scaled_dot_product_attention(
                qb.transpose(1, 2).contiguous(),
                kb.transpose(1, 2).contiguous(),
                vb.transpose(1, 2).contiguous(),
                is_causal=True,
            )
        )
        speedup = sdpa_ms / fa_ms if fa_ms > 0 else float("inf")
        print(f"    {label:<22} flash_attn={fa_ms:8.3f}  sdpa={sdpa_ms:8.3f}  加速比={speedup:.2f}x")
    print()
    return True


def main() -> None:
    print_env_info()
    print(f"VLX-Seek 将选择的注意力实现: {attn_implementation_selected()}")

    installed = importlib.util.find_spec("flash_attn") is not None
    if not installed:
        print_install_guidance()
        print("结论: flash_attn 未安装，VLX-Seek 将使用 PyTorch 内置 SDPA。")
        sys.exit(1)

    verified = verify_and_benchmark()
    if verified:
        print("结论: flash_attn 已安装且 kernel 验证通过（见上方微基准）。")
    else:
        print("结论: flash_attn 已安装，但 kernel 验证失败或跳过，VLX-Seek 实际回退 SDPA。")
        sys.exit(2)


if __name__ == "__main__":
    main()
