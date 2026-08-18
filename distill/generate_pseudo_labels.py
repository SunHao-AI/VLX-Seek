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
    大裁剪块（如 --slice-width 2500）进入 VLX-Seek 前会先按 --letterbox-size
    （默认 1024）做 letterbox：长边缩放、居中补灰边，bbox 同步变换、结果自动
    还原回裁剪块坐标，显著降低视觉塔显存占用；WeDetect 仍在原分辨率裁剪块上
    提取 proposals，保证小目标召回。

类别名还原说明：
    --categories 传入的是推理 prompt（可能与真实中文类别名不同）。若
    --prompt-map 指向的 category_prompts.json（generate_prompts.py 输出）
    含 prompt_to_category 反向映射，输出 COCO 的 categories.name 会替换为
    真实中文类别名；文件缺失或缺少该字段时保持 prompt 原样。
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


def _split_categories(categories: list[str], batch_size: int) -> list[list[str]]:
    """将类别列表按 batch_size 分批。batch_size<=0 或类别数<=batch_size 时返回单批。"""
    if batch_size <= 0 or len(categories) <= batch_size:
        return [categories]
    return [categories[i : i + batch_size] for i in range(0, len(categories), batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLX-Seek 生成 COCO 伪标签")
    parser.add_argument("--image-dir", required=True, help="输入图像目录")
    parser.add_argument(
        "--categories",
        required=True,
        help="检测类别，分号分隔，如 'person; car; dog'。同时作为 COCO categories。",
    )
    parser.add_argument("--output", default="pseudo_labels.json", help="输出 COCO JSON 路径")
    parser.add_argument(
        "--prompt-map",
        default=str(Path(__file__).resolve().parent / "data" / "category_prompts.json"),
        help="类别 prompt 映射文件（generate_prompts.py 输出）。用于把 COCO categories.name" " 从推理 prompt 还原为真实中文类别名；文件不存在或缺少 prompt_to_category 时保持原样。",
    )
    parser.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    parser.add_argument(
        "--detector-checkpoint",
        default="resources/wedetect_base_uni.pth",
        help="WeDetect 候选区域检测器权重。默认优先在 --model-path 目录下查找" " wedetect_base_uni.pth，其次用 resources/wedetect_base_uni.pth；" "仍缺失时自动从 Hugging Face 下载。",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="推理后端：hf=HF 原始路径（默认，行为零变化）；vllm=vLLM 引擎" "（需在 .venv-vllm 环境运行，批量共享前缀 KV 加速）。",
    )
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="过滤面积小于该值的检测框（像素），默认 0 不过滤",
    )
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
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=0,
        help="每个子提示词包含的类别数上限。0 表示不拆分（默认）。设为 30 则每 30 个类别一组循环推理。",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=100,
        help="WeDetect 每个裁剪块保留的最大候选框数（proposals 已按分数降序）。调小可缩短 prompt 和解码。",
    )
    parser.add_argument(
        "--letterbox-size",
        type=int,
        default=1024,
        help="VLX-Seek 推理前对裁剪图/原图做 letterbox 的长边目标尺寸（像素）。"
        "大图（如 2500×2500 裁剪块）直接进入视觉塔会显著抬高显存（aux C-RADIOv4 "
        "不缩放、按原生分辨率提取 4 层特征图）；letterbox 后长边缩放、居中补灰边，"
        "bbox 同步变换、结果自动还原回原图坐标。0 表示关闭。",
    )
    parser.add_argument(
        "--log-timing",
        action="store_true",
        help="打印每次 generate 的耗时与 token 数（用于定位 prefill/decode 耗时分布）。",
    )
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


def _truncate_proposals(boxes: list[list[float]], max_proposals: int) -> list[list[float]]:
    """proposals 已按分数降序，截断到前 max_proposals 个（<=0 不过滤）。"""
    if max_proposals > 0 and len(boxes) > max_proposals:
        return boxes[:max_proposals]
    return boxes


def load_proposals(image: Image.Image, detector_checkpoint: str, max_proposals: int = 100) -> list[list[float]]:
    """复用 inference.py 的 WeDetect proposal 生成逻辑（进程内缓存模型）。"""
    return _truncate_proposals(get_wedetect_generator(detector_checkpoint)(image), max_proposals)


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
    use_batch = args.prompt_batch_size > 0 and len(categories) > args.prompt_batch_size
    # vLLM 后端两阶段化：阶段 1 先对所有切片跑 WeDetect 收集 proposals，
    # 阶段 2 再逐个切片推理（每切片一次 generate）。避免 WeDetect（快）与
    # VLX-Seek（慢）在 GPU 上交替等待，让 LLM 推理连续执行。
    precollect = args.backend == "vllm"
    proposals_by_slice: list[list[list[float]]] = []

    def callback(slices) -> None:
        nonlocal proposals_by_slice
        if precollect:
            proposals_by_slice = [_truncate_proposals(generator(slc.image), args.max_proposals) for slc in slices]
        for idx, slc in enumerate(slices):
            crop = slc.image
            try:
                boxes = proposals_by_slice[idx] if precollect else _truncate_proposals(generator(crop), args.max_proposals)
                if use_batch:
                    category_batches = _split_categories(categories, args.prompt_batch_size)
                    if precollect:
                        # vLLM 下 encode_image_cache/clear_image_cache 为 no-op，跳过
                        result = worker.detect_multi_prompt(
                            crop,
                            boxes,
                            category_batches,
                            lang=args.lang,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                    else:
                        worker.encode_image_cache(crop, boxes)
                        result = worker.detect_multi_prompt(
                            crop,
                            boxes,
                            category_batches,
                            lang=args.lang,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                        worker.clear_image_cache()
                else:
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
                if not precollect:
                    worker.clear_image_cache()
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
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"目录中没有图片: {image_dir}")
    return paths


def load_prompt_to_category(prompt_map: str) -> dict[str, str]:
    """读取 category_prompts.json 的 prompt_to_category（推理 prompt -> 真实中文名）。

    文件缺失、解析失败或缺少该字段时返回空字典（categories.name 保持原样）。
    """
    path = Path(prompt_map)
    if not path.is_file():
        print(f"[warn] 未找到 prompt 映射文件 {path}，categories.name 保持 prompt 原样", file=sys.stderr)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] 读取 prompt 映射失败: {exc}，categories.name 保持 prompt 原样", file=sys.stderr)
        return {}
    mapping = data.get("prompt_to_category")
    return mapping if isinstance(mapping, dict) else {}


def run_pipeline(args: argparse.Namespace, image_paths: list[Path] | None = None) -> None:
    """单进程单卡处理一份图像列表，写入 args.output。

    多卡模式下由子进程调用，image_paths 为分配给该卡的图像分片。
    """
    categories = [c.strip() for c in args.categories.split(";") if c.strip()]
    if not categories:
        raise ValueError("--categories 不能为空")

    # label(小写) -> category_id 映射
    cat_id_map = {name.lower(): idx for idx, name in enumerate(categories)}
    # 反向映射：推理 prompt -> 真实中文类别名，替换 COCO categories.name
    prompt_to_category = load_prompt_to_category(args.prompt_map)

    def real_category_name(prompt: str) -> str:
        return prompt_to_category.get(prompt, prompt)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": real_category_name(name)} for idx, name in enumerate(categories)],
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

    if args.backend == "vllm":
        from vllm_serve.vlx_seek_vllm_worker import VLXSeekVLLMWorker

        worker = VLXSeekVLLMWorker(
            args.model_path,
            device=args.device,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            max_model_len=8192,
            letterbox_size=args.letterbox_size,
        )
    else:
        from vlx_seek_worker import VLXSeekWorker

        worker = VLXSeekWorker(
            args.model_path, device=args.device, letterbox_size=args.letterbox_size
        )

    worker.log_timing = args.log_timing

    next_image_id = max((img["id"] for img in coco["images"]), default=-1) + 1
    next_ann_id = max((ann["id"] for ann in coco["annotations"]), default=-1) + 1

    # 类别分批日志
    use_batch = args.prompt_batch_size > 0 and len(categories) > args.prompt_batch_size
    if use_batch:
        n_batches = len(_split_categories(categories, args.prompt_batch_size))
        print(
            f"类别分批: {len(categories)} 个类别 → {n_batches} 批，" f"每批 ≤{args.prompt_batch_size} 个",
            file=sys.stderr,
        )

    from tqdm import tqdm

    total = len(image_paths)
    pbar = tqdm(image_paths, desc="生成伪标签", unit="图", file=sys.stderr)
    for i, img_path in enumerate(pbar):
        if img_path.name in done_names:
            continue

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        try:
            if args.crop_inference:
                detections = detect_with_crop(image, worker, categories, args, cat_id_map)
            elif use_batch:
                category_batches = _split_categories(categories, args.prompt_batch_size)
                boxes = load_proposals(image, args.detector_checkpoint, args.max_proposals)
                worker.encode_image_cache(image, boxes)
                result = worker.detect_multi_prompt(
                    image,
                    boxes,
                    category_batches,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                worker.clear_image_cache()
                detections = [(rb["label"], rb["xmin"], rb["ymin"], rb["xmax"], rb["ymax"]) for rb in result.get("result_bbox_list", [])]
            else:
                boxes = load_proposals(image, args.detector_checkpoint, args.max_proposals)
                result = worker.detect(
                    image,
                    boxes,
                    categories,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                detections = [(rb["label"], rb["xmin"], rb["ymin"], rb["xmax"], rb["ymax"]) for rb in result.get("result_bbox_list", [])]
        except Exception as exc:  # 单张失败不中断整体
            pbar.write(f"[{i + 1}/{total}] 失败 {img_path.name}: {exc}")
            if use_batch:
                worker.clear_image_cache()
            continue

        image_id = next_image_id
        next_image_id += 1
        coco["images"].append({"id": image_id, "file_name": img_path.name, "width": width, "height": height})

        next_ann_id = _add_annotations(coco, image_id, detections, cat_id_map, args.min_area, next_ann_id)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            save_coco(coco, args.output)
        pbar.set_postfix_str(f"{len(coco['images'])} 图 / {len(coco['annotations'])} 框")

    save_coco(coco, args.output)
    print(f"伪标签已保存到 {args.output}")
    print(f"图像数: {len(coco['images'])}，标注数: {len(coco['annotations'])}")


def _split_shards(paths: list[Path], num_shards: int) -> list[list[Path]]:
    """按轮询方式把图像列表均分到 num_shards 份，尽量均衡负载。"""
    shards: list[list[Path]] = [[] for _ in range(num_shards)]
    for idx, path in enumerate(paths):
        shards[idx % num_shards].append(path)
    return shards


def _setup_logging(log_path: str) -> None:
    """把框架警告/日志写入 log_path（追加模式），终端只保留脚本自身的进度输出。"""
    import logging

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)

    # warnings.warn 触发的警告（torch/transformers 的 FutureWarning/UserWarning 等）
    # 统一走 py.warnings logger → 写入日志文件，不再刷终端
    logging.captureWarnings(True)


def _worker_shard(args: argparse.Namespace, gpu_id: int, queue: mp.Queue, output_path: str) -> None:
    """子进程入口：用 CUDA_VISIBLE_DEVICES 隔离到指定 GPU，从共享队列拉取图像批处理。"""
    _setup_logging(output_path + ".log")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shard_args = argparse.Namespace(**vars(args))
    shard_args.device = "cuda:0"  # 隔离后 cuda:0 即物理 GPU gpu_id
    shard_args.output = output_path

    # 从共享队列批量拉取图像，减少 run_pipeline 调用开销的同时保持负载均衡
    batch_size = 32
    batch: list[Path] = []
    while True:
        try:
            img_path = queue.get_nowait()
            batch.append(img_path)
            if len(batch) >= batch_size:
                run_pipeline(shard_args, image_paths=batch)
                batch.clear()
        except mp.queues.Empty:
            break
    if batch:
        run_pipeline(shard_args, image_paths=batch)


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


# Current flow: round-robin split of image_paths into shards, one shard per GPU.
# Each worker processes its shard independently and writes <output>.shard<i>.json.
def run_multigpu(args: argparse.Namespace) -> None:
    """多卡入口：创建共享队列，各 GPU 进程动态拉取图像，最后合并结果。"""
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids 不能为空")

    image_paths = collect_image_paths(args.image_dir)

    # 用共享队列实现动态负载均衡：快卡多拉，慢卡少拉，避免静态分片导致的部分卡空闲
    queue: mp.Queue = mp.Queue()
    for p in image_paths:
        queue.put(p)

    output = Path(args.output)
    shard_outputs = [str(output.with_name(f"{output.stem}.shard{i}.json")) for i in range(len(gpu_ids))]

    ctx = mp.get_context("spawn")
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_shard,
            args=(args, gpu_id, queue, shard_outputs[i]),
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


def _resolve_detector_checkpoint(args: argparse.Namespace) -> None:
    """未显式指定 --detector-checkpoint 时，自动在 --model-path 目录下查找。

    若 <model-path>/wedetect_base_uni.pth 存在则优先使用；否则保留默认的
    resources/wedetect_base_uni.pth（缺失时仍由下游触发联网下载）。
    """
    default = "resources/wedetect_base_uni.pth"
    if args.detector_checkpoint != default:
        return  # 用户已显式指定，尊重之
    candidate = Path(args.model_path) / "wedetect_base_uni.pth"
    if candidate.is_file():
        args.detector_checkpoint = str(candidate)


def main() -> None:
    args = parse_args()
    _setup_logging(args.output + ".log")
    _resolve_detector_checkpoint(args)
    if args.gpu_ids:
        run_multigpu(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
