"""用 VLX-Seek 批量生成 COCO 格式伪标签，用于蒸馏训练 YOLO-World。

流程：
    1. 对每张图用 WeDetect 生成候选区域（proposals）。
    2. 调用 VLX-Seek 开放词汇检测，得到 {label, bbox} 结果。
    3. 过滤（面积、类别匹配）后写入 COCO JSON。

用法示例（单卡）：
    python distill/generate_pseudo_labels.py \
        --image-dir data/images \
        --categories "person; car; dog" \
        --output data/pseudo_labels.json \
        --model-path resources/VLX-Seek-1.5-10B \
        --device cuda

用法示例（多卡）：
    python distill/generate_pseudo_labels.py \
        --image-dir data/images \
        --categories "person; car; dog" \
        --output data/pseudo_labels.json \
        --model-path resources/VLX-Seek-1.5-10B \
        --gpu-ids 0,1,2

多卡说明：
    --gpu-ids 指定参与推理的 GPU 索引（逗号分隔）。脚本会为每张卡启动一个
    子进程，把图像按轮询方式分片，各子进程用 CUDA_VISIBLE_DEVICES 隔离到
    对应 GPU，分别写入 <output>.shard<i>.json，全部完成后合并为最终输出。
    分片文件保留，配合 --resume 可断点续跑。

裁剪推理说明：
    默认开启（--no-crop-inference 关闭）。对每张图用 cv_utils 的 CropImage
    做滑窗裁剪（默认 1000x1000，重叠 10%），对每个裁剪块分别跑 WeDetect +
    VLX-Seek，再合并回原图坐标。适合大图小目标场景。可用 --slice-width /
    --slice-height / --overlap-width-ratio / --overlap-height-ratio 调整。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已存在的图像")
    parser.add_argument("--start-index", type=int, default=0, help="从第几张图开始（分片用）")
    parser.add_argument("--end-index", type=int, default=None, help="处理到第几张图（不含，分片用）")
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="多卡推理：逗号分隔的 GPU 索引，如 '0,1,2'。设置后按卡分片并行处理。",
    )
    parser.add_argument(
        "--crop-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用滑窗裁剪推理（默认开启，--no-crop-inference 关闭）。",
    )
    parser.add_argument("--slice-width", type=int, default=1000, help="裁剪块宽度（像素）")
    parser.add_argument("--slice-height", type=int, default=1000, help="裁剪块高度（像素）")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.1, help="宽度方向重叠比例")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.1, help="高度方向重叠比例")
    return parser.parse_args()


# 进程内复用的 WeDetect proposal 生成器，避免每个裁剪块/每张图重复加载权重
_wedetect_generator = None
_wedetect_checkpoint: str | None = None


def get_wedetect_generator(detector_checkpoint: str):
    """返回进程内缓存的 WeDetect proposal 生成器（首次调用时构建）。"""
    global _wedetect_generator, _wedetect_checkpoint
    if _wedetect_generator is None or _wedetect_checkpoint != detector_checkpoint:
        from inference import WeDetectProposalGenerator

        _wedetect_generator = WeDetectProposalGenerator(detector_checkpoint)
        _wedetect_checkpoint = detector_checkpoint
    return _wedetect_generator


def load_proposals(image: Image.Image, detector_checkpoint: str) -> list[list[float]]:
    """复用 inference.py 的 WeDetect proposal 生成逻辑（进程内缓存模型）。"""
    return get_wedetect_generator(detector_checkpoint)(image)


def _add_annotations(
    coco: dict,
    image_id: int,
    detections: list[tuple],
    cat_id_map: dict[str, int],
    min_area: float,
    next_ann_id: int,
) -> int:
    """把 (label, x1, y1, x2, y2) 列表过滤后写入 COCO，返回新的 next_ann_id。"""
    for label, x1, y1, x2, y2 in detections:
        label = label.strip().lower()
        if label not in cat_id_map:
            continue
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0 or bw * bh < min_area:
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
    return next_ann_id


def detect_with_crop(
    image: Image.Image,
    worker,
    categories: list[str],
    args: argparse.Namespace,
    cat_id_map: dict[str, int],
) -> list[tuple]:
    """用 cv_utils 的 CropImage 做滑窗裁剪推理，返回原图坐标的 (label, x1, y1, x2, y2) 列表。"""
    from cv_utils.inference import CropImage

    generator = get_wedetect_generator(args.detector_checkpoint)

    def callback(slices) -> None:
        for slc in slices:
            crop = slc.image
            try:
                boxes = generator(crop)
                result = worker.detect(
                    crop,
                    boxes,
                    categories,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:  # 单个裁剪块失败不中断整体
                print(f"裁剪推理失败: {exc}", file=sys.stderr)
                continue

            shapes = []
            for rb in result.get("result_bbox_list", []):
                label = rb["label"].strip().lower()
                if label not in cat_id_map:
                    continue
                x1, y1, x2, y2 = rb["xmin"], rb["ymin"], rb["xmax"], rb["ymax"]
                if x2 <= x1 or y2 <= y1:
                    continue
                shapes.append(
                    {
                        "label": label,
                        "points": [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
                        "shape_type": "rectangle",
                    }
                )
            slc.labelme = {"version": "1.0.0", "flags": {}, "shapes": shapes}

    slicer = CropImage(
        slice_width=args.slice_width,
        slice_height=args.slice_height,
        overlap_width_ratio=args.overlap_width_ratio,
        overlap_height_ratio=args.overlap_height_ratio,
        output_type="rectangle",
    )
    labelme = slicer(image, callback)

    detections = []
    for shape in labelme.get("shapes", []):
        x1, y1, x2, y2 = shape["xyxy"]
        detections.append((shape["label"], x1, y1, x2, y2))
    return detections


def collect_image_paths(image_dir: str) -> list[Path]:
    paths = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"目录中没有图片: {image_dir}")
    return paths


def run_pipeline(args: argparse.Namespace, image_paths: list[Path] | None = None) -> None:
    """单进程单卡处理一份图像列表，写入 args.output。

    多卡模式下由子进程调用，image_paths 为分配给该卡的图像分片。
    """
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
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_names = {img["file_name"] for img in existing["images"]}
        coco = existing
        print(f"断点续跑：跳过 {len(done_names)} 张已处理图像", file=sys.stderr)

    if image_paths is None:
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
            if args.crop_inference:
                detections = detect_with_crop(
                    image, worker, categories, args, cat_id_map
                )
            else:
                boxes = load_proposals(image, args.detector_checkpoint)
                result = worker.detect(
                    image,
                    boxes,
                    categories,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                detections = [
                    (rb["label"], rb["xmin"], rb["ymin"], rb["xmax"], rb["ymax"])
                    for rb in result.get("result_bbox_list", [])
                ]
        except Exception as exc:  # 单张失败不中断整体
            print(f"[{i + 1}/{total}] 失败 {img_path.name}: {exc}", file=sys.stderr)
            continue

        image_id = next_image_id
        next_image_id += 1
        coco["images"].append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )

        next_ann_id = _add_annotations(
            coco, image_id, detections, cat_id_map, args.min_area, next_ann_id
        )

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


def _split_shards(paths: list[Path], num_shards: int) -> list[list[Path]]:
    """按轮询方式把图像列表均分到 num_shards 份，尽量均衡负载。"""
    shards: list[list[Path]] = [[] for _ in range(num_shards)]
    for idx, path in enumerate(paths):
        shards[idx % num_shards].append(path)
    return shards


def _worker_shard(
    args: argparse.Namespace, gpu_id: int, shard_paths: list[Path], output_path: str
) -> None:
    """子进程入口：用 CUDA_VISIBLE_DEVICES 隔离到指定 GPU 后处理一份分片。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shard_args = argparse.Namespace(**vars(args))
    shard_args.device = "cuda:0"  # 隔离后 cuda:0 即物理 GPU gpu_id
    shard_args.output = output_path
    run_pipeline(shard_args, image_paths=shard_paths)


def merge_shards(shard_paths: list[str], output_path: str) -> None:
    """把各 GPU 分片文件合并为最终 COCO JSON，并重排 image/annotation id。"""
    merged: dict = {"images": [], "annotations": [], "categories": None}
    next_image_id = 0
    next_ann_id = 0
    for shard_path in shard_paths:
        if not Path(shard_path).is_file():
            continue
        with open(shard_path, "r", encoding="utf-8") as f:
            shard = json.load(f)
        if merged["categories"] is None:
            merged["categories"] = shard["categories"]

        id_map: dict[int, int] = {}
        for img in shard["images"]:
            id_map[img["id"]] = next_image_id
            new_img = dict(img)
            new_img["id"] = next_image_id
            merged["images"].append(new_img)
            next_image_id += 1
        for ann in shard["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[ann["image_id"]]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

    save_coco(merged, output_path)


def run_multigpu(args: argparse.Namespace) -> None:
    """多卡入口：按 --gpu-ids 分片并行推理，最后合并结果。"""
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids 不能为空")

    image_paths = collect_image_paths(args.image_dir)
    shards = _split_shards(image_paths, len(gpu_ids))

    output = Path(args.output)
    shard_outputs = [
        str(output.with_name(f"{output.stem}.shard{i}.json"))
        for i in range(len(gpu_ids))
    ]

    ctx = mp.get_context("spawn")
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_shard,
            args=(args, gpu_id, shards[i], shard_outputs[i]),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p.exitcode for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"部分 GPU 分片失败，exit codes: {failed}")

    merge_shards(shard_outputs, args.output)
    with open(args.output, "r", encoding="utf-8") as f:
        merged = json.load(f)
    print(f"多卡伪标签已合并保存到 {args.output}")
    print(f"图像数: {len(merged['images'])}，标注数: {len(merged['annotations'])}")


def main() -> None:
    args = parse_args()
    if args.gpu_ids:
        run_multigpu(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
