"""用 COCO 伪标签微调官方 YOLO-World（ultralytics 框架）。

流程：
    1. 读取 COCO 伪标签，提取类别列表。
    2. 划分 train / val（或使用独立 val COCO）。
    3. 转成 YOLO txt 格式并生成 dataset.yaml。
    4. 调用 ultralytics YOLOWorld.train() 微调。

依赖：需要额外安装 ultralytics（pip install ultralytics）。

用法示例：
    python distill/finetune_yolo_world.py \
        --coco-json data/pseudo_labels.json \
        --image-dir data/images \
        --output-dir runs/yolo_world \
        --weights yolov8s-worldv2.pt \
        --epochs 50 --imgsz 640 --batch 16 --device 0

    启用 albumentations 随机裁剪在线增强（需 ultralytics>=8.4 + albumentations，
    通过官方 model.train(augmentations=...) 参数注入）：
    python distill/finetune_yolo_world.py \
        --coco-json distill/data/pseudo_labels.json \
        --image-dir distill/data/images \
        --weights yolov8s-worldv2.pt \
        --epochs 50 --imgsz 960 --batch 32 --device 0,1,2,3,5,6,7,8 \
        --crop-aug --crop-scale 0.5 1.0 --crop-erosion 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 coco_utils
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coco_utils import (  # noqa: E402
    category_names,
    coco_to_yolo_txt,
    load_coco,
    split_coco,
    write_dataset_yaml,
)


def build_crop_augmentation(
    imgsz: int,
    crop_scale: tuple[float, float],
    crop_erosion: float,
    crop_prob: float,
) -> list:
    """构造 albumentations 随机裁剪变换列表，供 ultralytics 官方 augmentations 参数使用。

    官方机制（ultralytics>=8.4 文档：integrations/albumentations）：
    model.train(augmentations=[...]) 会把自定义变换列表传入训练增强链中的
    Albumentations 实例，替换其默认的 Blur/CLAHE 等低强度增强，同时保留
    Mosaic/HSV/翻转等其他 YOLO 增强。空间变换（DualTransform）会被自动识别，
    bbox 以 yolo 归一化格式自动适配。

    这里使用 RandomSizedBBoxSafeCrop：随机裁剪一块区域并缩放回 imgsz×imgsz，
    保证至少保留一个目标框。
    """
    try:
        import albumentations as A
    except ImportError as exc:
        sys.exit(
            f"启用 --crop-aug 需要先安装依赖，请执行: uv pip install albumentations（缺少: {exc.name}）"
        )

    # RandomSizedBBoxSafeCrop 的 scale（裁剪面积占原图比例范围）参数在旧版存在、新版可能移除，按版本兼容
    crop_kwargs: dict = dict(height=imgsz, width=imgsz, erosion_rate=crop_erosion, p=crop_prob)
    try:
        import inspect

        if "scale" in inspect.signature(A.RandomSizedBBoxSafeCrop.__init__).parameters:
            crop_kwargs["scale"] = crop_scale
    except (TypeError, ValueError):
        pass

    return [A.RandomSizedBBoxSafeCrop(**crop_kwargs)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO 伪标签微调 YOLO-World")
    parser.add_argument("--coco-json", required=True, help="COCO 伪标签路径")
    parser.add_argument("--image-dir", required=True, help="图像目录（COCO file_name 相对此目录）")
    parser.add_argument("--output-dir", default="runs/yolo_world", help="训练输出目录")
    parser.add_argument("--weights", default="yolov8s-worldv2.pt", help="预训练权重")
    parser.add_argument("--val-coco-json", default=None, help="独立验证集 COCO；缺省时从训练集划分")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="从训练集划分验证集的比例")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=30,
        help="早停轮数：验证指标连续 N 轮无提升则提前结束（默认 30）",
    )
    parser.add_argument("--crop-aug", action="store_true", help="启用 albumentations 随机裁剪在线增强（需 ultralytics>=8.4 + albumentations）")
    parser.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0], metavar=("MIN", "MAX"), help="随机裁剪面积占原图比例范围（默认 0.5 1.0）")
    parser.add_argument("--crop-erosion", type=float, default=0.2, help="随机裁剪对目标框边缘的最大侵蚀比例 0~1（默认 0.2）")
    parser.add_argument("--crop-prob", type=float, default=1.0, help="每张训练图应用随机裁剪的概率（默认 1.0）")
    return parser.parse_args()


def prepare_dataset(
    coco: dict,
    image_dir: str,
    output_dir: str,
    val_coco: dict | None,
    val_ratio: float,
    seed: int,
) -> Path:
    """构建 ultralytics 数据集目录并返回 dataset.yaml 路径。"""
    output_dir = Path(output_dir)
    dataset_root = output_dir / "dataset"
    train_images = dataset_root / "images" / "train"
    train_labels = dataset_root / "labels" / "train"
    val_images = dataset_root / "images" / "val"
    val_labels = dataset_root / "labels" / "val"

    if val_coco is None:
        train_coco, val_coco = split_coco(coco, val_ratio=val_ratio, seed=seed)
    else:
        train_coco = coco

    coco_to_yolo_txt(train_coco, image_dir, train_images, train_labels)
    coco_to_yolo_txt(val_coco, image_dir, val_images, val_labels)

    names = category_names(coco)
    return write_dataset_yaml(dataset_root, names)


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLOWorld
    except ImportError:
        sys.exit("未安装 ultralytics，请先执行: pip install ultralytics")

    coco = load_coco(args.coco_json)
    if not coco["images"]:
        sys.exit("COCO 中没有图像，请先运行 generate_pseudo_labels.py 生成伪标签。")

    val_coco = load_coco(args.val_coco_json) if args.val_coco_json else None

    dataset_yaml = prepare_dataset(
        coco,
        args.image_dir,
        args.output_dir,
        val_coco,
        args.val_ratio,
        args.seed,
    )
    print(f"数据集已就绪: {dataset_yaml}")

    augmentations = None
    if args.crop_aug:
        augmentations = build_crop_augmentation(
            args.imgsz,
            tuple(args.crop_scale),
            args.crop_erosion,
            args.crop_prob,
        )
        print("已启用 albumentations 随机裁剪在线增强")

    model = YOLOWorld(args.weights)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(Path(args.output_dir)),
        name="yolo_world_finetune",
        augmentations=augmentations,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
