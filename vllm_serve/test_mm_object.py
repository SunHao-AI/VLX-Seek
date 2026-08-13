# -*- coding: utf-8 -*-
"""1b-2 诊断：确认 <objfeat> tokenizer 行为 + vLLM 自定义数据传递机制。

运行：
    python -m vllm_serve.test_mm_object --model-path resources/VLX-Seek-1.5-10B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    model_path = Path(args.model_path)

    # 1. 确认 <objfeat> 在 tokenizer 中的表现
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    objfeat_str = "<objfeat>"
    tokens = tokenizer.encode(objfeat_str, add_special_tokens=False)
    print(f"[1] <objfeat> encode: {tokens} (len={len(tokens)})")

    # 检查 248181 是否在 vocab 中
    vocab = tokenizer.get_vocab()
    objfeat_token_id = 248181
    reverse_vocab = {v: k for k, v in vocab.items()}
    print(f"[1] token 248181 in vocab: {objfeat_token_id in vocab.values()}")
    if objfeat_token_id in vocab.values():
        print(f"[1] token 248181 -> {reverse_vocab[objfeat_token_id]!r}")

    # 检查 <objfeat> 是否能被 tokenizer 直接识别
    if "<objfeat>" in vocab:
        print(f"[1] '<objfeat>' in vocab: id={vocab['<objfeat>']}")
    else:
        print("[1] '<objfeat>' NOT in vocab (will be split into subwords)")

    # 2. 确认 <obj0> <obj1> 等 object token 的 tokenizer 行为
    for s in ["<obj0>", "<obj1>", "<obj2>"]:
        t = tokenizer.encode(s, add_special_tokens=False)
        print(f"[2] {s!r} encode: {t} (len={len(t)})")

    # 3. 确认 <|image_pad|> 的 token id
    image_pad_tokens = tokenizer.encode("<|image_pad|>", add_special_tokens=False)
    print(f"[3] <|image_pad|> encode: {image_pad_tokens}")

    # 4. 构建 VLX-Seek 风格的 prompt，看 tokenizer 如何处理
    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><image><|vision_end|>\n"
        "<obj0><objfeat><obj1><objfeat>\n"
        "描述图片中的目标<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"[4] full prompt token count: {len(prompt_tokens)}")

    # 检查 prompt 中是否有 248056 / 248181
    has_248056 = 248056 in prompt_tokens
    has_248181 = 248181 in prompt_tokens
    print(f"[4] has 248056 (image_pad): {has_248056}")
    print(f"[4] has 248181 (objfeat): {has_248181}")

    # 5. 确认 vLLM multi_modal_data 是否支持自定义 key
    print("\n[5] vLLM multi_modal_data 自定义 key 测试：")
    try:
        from vllm.multimodal.parse import MultiModalDataItems
        # 尝试解析含自定义 key 的数据
        test_data = {
            "image": "placeholder",
            "bbox_list": [[0.1, 0.2, 0.3, 0.4]],
        }
        # MultiModalDataItems.from_data 是入口
        print(f"  MultiModalDataItems methods: {[m for m in dir(MultiModalDataItems) if not m.startswith('_')]}")
    except Exception as e:
        print(f"  Error: {e}")

    # 6. 检查 preprocessor_config_aux.json
    aux_config_path = model_path / "preprocessor_config_aux.json"
    if aux_config_path.exists():
        cfg = json.loads(aux_config_path.read_text())
        print(f"\n[6] preprocessor_config_aux.json exists")
        print(f"  image_processor_type: {cfg.get('image_processor_type', 'N/A')}")
    else:
        print(f"\n[6] preprocessor_config_aux.json NOT found at {aux_config_path}")
        # 搜索其他可能的 aux processor 配置
        for p in model_path.glob("*aux*"):
            print(f"  found: {p.name}")

    print("\n诊断完成。")


if __name__ == "__main__":
    main()
