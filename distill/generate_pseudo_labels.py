"""用 VLX-Seek 批量生成 COCO 格式伪标签，用于蒸馏训练 YOLO-World。

流程：
    1. 对每张图用 WeDetect 生成候选区域（proposals）。
    2. 调用 VLX-Seek 开放词汇检测，得到 {label, bbox} 结果。
    3. 过滤（面积、类别匹配）后写入 COCO JSON。

用法示例：
    python distill/generate_pseudo_labels.py \
        --image-dir data/images \
        --categories "person; car; dog" \
        --output data/pseudo_labels.json \
        --model-path resources/VLX-Seek-1.5-10B \
        --device cuda

支持 --resume 断点续跑：已出现在输出文件中的图像会被跳过。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# 将项目根目录加入 sys.path，以便导入 vlx_seek_worker / inference
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coco_utils import save_coco, xyxy_to_xywh  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLX-Seek 生成 COCO 伪标签")
    parser.add_argument("--image-dir", required=True, help="输入图像目录")
    parser.add_argument(
        "--categories",
        required=True,
        help="检测类别，分号分隔，如 'person; car; dog'。同时作为 COCO categories。",
    )
    parser.add_argument("--output", default="pseudo_labels.json", help="输出 COCO JSON 路径")
    parser.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    parser.add_argument(
        "--detector-checkpoint",
        default="resources/wedetect_base_uni.pth",
        help="WeDetect 候选区域检测器权重；缺失时自动从 Hugging Face 下载。",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-area", type=float, default=0.0, help="过滤小于该像素面积的框")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已存在的图像")
    parser.add_argument("--start-index", type=int, default=0, help="从第几张图开始（分片用）")
    parser.add_argument("--end-index", type=int, default=None, help="处理到第几张图（不含，分片用）")
    return parser.parse_args()


def load_proposals(image: Image.Image, detector_checkpoint: str) -> list[list[float]]:
    """复用 inference.py 的 WeDetect proposal 生成逻辑。"""
    from inference import load_wedetect_proposals

    return load_wedetect_proposals(image, detector_checkpoint)


def collect_image_paths(image_dir: str) -> list[Path]:
    paths = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"目录中没有图片: {image_dir}")
    return paths


def main() -> None:
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(";") if c.strip()]
    if not categories:
        raise ValueError("--categories 不能为空")

    # label(小写) -> category_id 映射
    cat_id_map = {name.lower(): idx for idx, name in enumerate(categories)}
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(categories)],
    }

    # 断点续跑：读取已有输出中的 file_name
    done_names: set[str] = set()
    if args.resume and Path(args.output).is_file():
        import json

        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_names = {img["file_name"] for img in existing["images"]}
        coco = existing
        print(f"断点续跑：跳过 {len(done_names)} 张已处理图像", file=sys.stderr)

    image_paths = collect_image_paths(args.image_dir)
    image_paths = image_paths[args.start_index : args.end_index]

    from vlx_seek_worker import VLXSeekWorker

    worker = VLXSeekWorker(args.model_path, device=args.device)

    next_image_id = max((img["id"] for img in coco["images"]), default=-1) + 1
    next_ann_id = max((ann["id"] for ann in coco["annotations"]), default=-1) + 1

    total = len(image_paths)
    for i, img_path in enumerate(image_paths):
        if img_path.name in done_names:
            continue

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        try:
            boxes = load_proposals(image, args.detector_checkpoint)
            result = worker.detect(
                image,
                boxes,
                categories,
                lang=args.lang,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # 单张失败不中断整体
            print(f"[{i + 1}/{total}] 失败 {img_path.name}: {exc}", file=sys.stderr)
            continue

        image_id = next_image_id
        next_image_id += 1
        coco["images"].append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )

        for rb in result.get("result_bbox_list", []):
            label = rb["label"].strip().lower()
            if label not in cat_id_map:
                continue
            x1, y1, x2, y2 = rb["xmin"], rb["ymin"], rb["xmax"], rb["ymax"]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or bw * bh < args.min_area:
                continue
            coco["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cat_id_map[label],
                    "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            save_coco(coco, args.output)
            print(
                f"[{i + 1}/{total}] {img_path.name} 完成，"
                f"累计 {len(coco['images'])} 图 / {len(coco['annotations'])} 框",
                file=sys.stderr,
            )

    save_coco(coco, args.output)
    print(f"伪标签已保存到 {args.output}")
    print(f"图像数: {len(coco['images'])}，标注数: {len(coco['annotations'])}")


if __name__ == "__main__":
    main()
