# -*- coding: utf-8 -*-
"""Task 3 Step 3 一致性回归：HF vs vLLM 同图同输入对比检测输出。

用法（服务器，两端各跑一次，用同一张真实图片 / 同一类别 / 同一 boxes）：
    # HF（.venv 环境）
    python -m vllm_serve.test_consistency \
        --backend hf \
        --model-path resources/VLX-Seek-1.5-10B \
        --image data/images/xxx.jpg \
        --categories "人群密集; 道路施工区域或施工场景; 水面浑浊、不洁的水体; 水中游泳的人" \
        --out results/hf.json

    # vLLM（.venv-vllm 环境）
    python -m vllm_serve.test_consistency \
        --backend vllm \
        --model-path resources/VLX-Seek-1.5-10B \
        --image data/images/xxx.jpg \
        --categories "人群密集; 道路施工区域或施工场景; 水面浑浊、不洁的水体; 水中游泳的人" \
        --out results/vllm.json

两端均用 temperature=0（贪心解码，确定性输出），固定图片/类别/boxes，
对比 detect / detect_multi_prompt 的 answer 与 result_bbox_list。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_boxes(s: str | None) -> list[list[float]] | None:
    """解析 'x1,y1,x2,y2;x1,y1,x2,y2' 为 bbox 列表；None/空串返回 None。"""
    if not s:
        return None
    boxes = []
    for part in s.split(";"):
        coords = [float(v) for v in part.split(",")]
        if len(coords) != 4:
            raise ValueError(f"bbox 需要 4 个坐标，got {part!r}")
        boxes.append(coords)
    return boxes


def split_categories(categories: list[str], batch_size: int) -> list[list[str]]:
    """与 distill/generate_pseudo_labels._split_categories 一致。"""
    if batch_size <= 0 or len(categories) <= batch_size:
        return [categories]
    return [categories[i : i + batch_size] for i in range(0, len(categories), batch_size)]


def build_worker(backend: str, model_path: str):
    if backend == "hf":
        from vlx_seek_worker import VLXSeekWorker

        return VLXSeekWorker(model_path, device="cuda")
    from vllm_serve.vlx_seek_vllm_worker import VLXSeekVLLMWorker

    return VLXSeekVLLMWorker(model_path, gpu_memory_utilization=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF vs vLLM 一致性回归")
    parser.add_argument("--backend", choices=("hf", "vllm"), required=True)
    parser.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    parser.add_argument("--image", required=True, help="真实图片路径")
    parser.add_argument(
        "--categories", required=True, help="检测类别（与 distill 相同的 prompt），分号分隔"
    )
    parser.add_argument(
        "--boxes",
        default=None,
        help='候选框（原图像素坐标）"x1,y1,x2,y2;x1,y1,x2,y2"，脚本自动等比换算到缩放后坐标'
        "（不传则纯图检测）",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="多批类别每批个数，<=0 全部一批")
    parser.add_argument("--lang", choices=("en", "zh"), default="zh")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="等比缩图到最长边不超过该值（默认 1024，模拟 distill 的 crop 视觉规模；"
        "整张大图直出会因视觉 token 超 max_model_len 失败）。0=不缩放。",
    )
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    original_size = list(image.size)
    if args.max_side > 0 and max(image.size) > args.max_side:
        image.thumbnail((args.max_side, args.max_side))
    resized_size = list(image.size)
    if original_size != resized_size:
        print(f"[resize] {original_size} -> {resized_size}")
    boxes = parse_boxes(args.boxes)
    # --boxes 用原图坐标，等比换算到缩放后坐标（与传给 worker 的图一致）
    if boxes and original_size != resized_size:
        scale_x = resized_size[0] / original_size[0]
        scale_y = resized_size[1] / original_size[1]
        boxes = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for x1, y1, x2, y2 in boxes
        ]
        print(f"[boxes] scaled -> {boxes}")
    categories = [c.strip() for c in args.categories.split(";") if c.strip()]

    if args.backend == "hf":
        print("[env] HF 后端请使用项目原始环境（勿在 .venv-vllm 中跑，缺 accelerate）")
    worker = build_worker(args.backend, args.model_path)
    worker.log_timing = True

    common = dict(lang=args.lang, max_new_tokens=args.max_new_tokens, temperature=0.0)

    print("=" * 60)
    print("1) detect（单批全类别）")
    r1 = worker.detect(image, boxes, categories, **common)
    print(f"  answer: {r1['answer']!r}")
    print(f"  result_bbox_list: {len(r1['result_bbox_list'])} items")

    print("=" * 60)
    print("2) detect_multi_prompt（按 batch_size 分批批量提交）")
    batches = split_categories(categories, args.batch_size)
    print(f"  batches: {len(batches)}")
    r2 = worker.detect_multi_prompt(image, boxes, batches, **common)
    print(f"  answer: {r2['answer']!r}")
    print(f"  result_bbox_list: {len(r2['result_bbox_list'])} items")
    print("=" * 60)

    result = {
        "backend": args.backend,
        "image": args.image,
        "original_size": original_size,
        "resized_size": resized_size,
        "categories": categories,
        "boxes": boxes,
        "lang": args.lang,
        "max_new_tokens": args.max_new_tokens,
        "detect": r1,
        "detect_multi_prompt": r2,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
