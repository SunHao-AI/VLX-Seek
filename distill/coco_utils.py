"""COCO 格式工具：读写、坐标转换、COCO→YOLO txt 转换。

供蒸馏流程中的伪标签生成脚本与微调脚本共用。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def xyxy_to_xywh(box: list[float]) -> list[float]:
    """[x1, y1, x2, y2] -> [x, y, w, h]（COCO 格式，左上角 + 宽高）。"""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    """[x, y, w, h] -> [x1, y1, x2, y2]。"""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def save_coco(coco: dict[str, Any], output_path: str | Path) -> Path:
    """保存 COCO 标注到 JSON 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return output_path


def load_coco(coco_path: str | Path) -> dict[str, Any]:
    """读取 COCO 标注 JSON。"""
    with open(coco_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sorted_categories(coco: dict[str, Any]) -> list[dict[str, Any]]:
    """按 category_id 升序返回类别列表。"""
    return sorted(coco["categories"], key=lambda c: c["id"])


def category_names(coco: dict[str, Any]) -> list[str]:
    """返回按 id 排序的类别名列表（0-indexed 顺序）。"""
    return [c["name"] for c in sorted_categories(coco)]


def split_coco(
    coco: dict[str, Any],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按图像随机划分 COCO 为 train / val 两份。

    返回 (train_coco, val_coco)，annotations 按 image_id 归属到对应子集。
    """
    rng = random.Random(seed)
    images = coco["images"]
    val_ids = set(rng.sample([img["id"] for img in images], int(len(images) * val_ratio)))

    train_images, val_images = [], []
    for img in images:
        (val_images if img["id"] in val_ids else train_images).append(img)

    train_anns, val_anns = [], []
    for ann in coco["annotations"]:
        (val_anns if ann["image_id"] in val_ids else train_anns).append(ann)

    base = {"categories": coco["categories"]}
    train_coco = {**base, "images": train_images, "annotations": train_anns}
    val_coco = {**base, "images": val_images, "annotations": val_anns}
    return train_coco, val_coco


def coco_to_yolo_txt(
    coco: dict[str, Any],
    image_dir: str | Path,
    out_images_dir: str | Path,
    out_labels_dir: str | Path,
) -> None:
    """把 COCO 标注转成 YOLO txt 格式，并复制/链接图像。

    - 每张图一个 ``<stem>.txt``，每行 ``class_id cx cy w h``（归一化）。
    - class_id 为按 category_id 排序后的 0-indexed 索引。
    - 图像复制到 ``out_images_dir``，标签写到 ``out_labels_dir``。
    """
    image_dir = Path(image_dir)
    out_images_dir = Path(out_images_dir)
    out_labels_dir = Path(out_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    cat_id_to_idx = {
        c["id"]: i for i, c in enumerate(sorted_categories(coco))
    }

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in coco["images"]:
        src = image_dir / img["file_name"]
        if not src.is_file():
            continue
        dst = out_images_dir / img["file_name"]
        if not dst.exists():
            shutil_copy(src, dst)

        lines = []
        for ann in anns_by_image.get(img["id"], []):
            if ann.get("iscrowd", False):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / img["width"]
            cy = (y + h / 2) / img["height"]
            nw = w / img["width"]
            nh = h / img["height"]
            lines.append(f"{cat_id_to_idx[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = out_labels_dir / f"{Path(img['file_name']).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")


def shutil_copy(src: Path, dst: Path) -> None:
    """复制文件（避免顶层 import shutil 造成循环依赖）。"""
    import shutil

    shutil.copy2(src, dst)


def write_dataset_yaml(
    dataset_root: str | Path,
    names: list[str],
    train_rel: str = "images/train",
    val_rel: str = "images/val",
) -> Path:
    """生成 ultralytics 训练所需的 dataset.yaml。"""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_root / "dataset.yaml"
    content = {
        "path": str(dataset_root.resolve()),
        "train": train_rel,
        "val": val_rel,
        "names": {i: name for i, name in enumerate(names)},
    }
    yaml_path.write_text(_dump_yaml(content), encoding="utf-8")
    return yaml_path


def _dump_yaml(data: dict) -> str:
    """手写 YAML 输出，避免依赖 PyYAML。"""
    lines = [f"path: {data['path']}", f"train: {data['train']}", f"val: {data['val']}", "names:"]
    for idx, name in data["names"].items():
        lines.append(f"  {idx}: {name}")
    return "\n".join(lines) + "\n"
