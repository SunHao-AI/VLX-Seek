"""COCO / YOLO / LabelMe 标注格式双向转换工具。

统一解析为 Image/Object 中间结构后导出目标格式。
复用 distill/coco_utils.py 的读写函数，不改动现有文件。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Object:
    """单个标注对象：类别名 + bbox 或/和多边形（均为像素坐标）。"""

    category_name: str
    category_id: int | None = None
    bbox_xywh: list[float] | None = None
    polygon: list[list[float]] | None = None


@dataclass
class Image:
    """一张图片及其标注。"""

    id: int
    file_name: str
    width: int
    height: int
    objects: list[Object] = field(default_factory=list)


class Warnings:
    """收集转换过程中的告警，结束时汇总打印。"""

    def __init__(self) -> None:
        self.items: list[str] = []

    def warn(self, msg: str) -> None:
        self.items.append(msg)

    def report(self) -> None:
        if not self.items:
            return
        print(f"[warn] 共 {len(self.items)} 条告警：")
        for msg in self.items:
            print(f"  - {msg}")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def polygon_to_bbox(points: list[list[float]]) -> list[float]:
    """多边形顶点 [[x, y], ...] -> COCO bbox [x, y, w, h]。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    return [x1, y1, max(xs) - x1, max(ys) - y1]


def bbox_to_rectangle(x: float, y: float, w: float, h: float) -> list[list[float]]:
    """bbox -> LabelMe rectangle 的 points [[x1, y1], [x2, y2]]。"""
    return [[x, y], [x + w, y + h]]


def parse_coco(coco: dict[str, Any], w: Warnings) -> list[Image]:
    """COCO dict -> list[Image]。segmentation 仅支持多边形列表，RLE 跳过。"""
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images: list[Image] = []
    for img in coco["images"]:
        objs: list[Object] = []
        for ann in anns_by_image.get(img["id"], []):
            if ann.get("iscrowd"):
                w.warn(f"图 {img['file_name']}: 跳过 iscrowd 标注 id={ann['id']}")
                continue
            name = cat_id_to_name.get(ann["category_id"], f"<cat_{ann['category_id']}>")
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                w.warn(f"图 {img['file_name']}: 跳过非法 bbox 标注 id={ann['id']}")
                continue
            obj = Object(category_name=name, category_id=ann["category_id"], bbox_xywh=[x, y, bw, bh])
            seg = ann.get("segmentation")
            if isinstance(seg, dict):
                w.warn(f"图 {img['file_name']}: 跳过 RLE 分割标注 id={ann['id']}")
            elif seg:
                pts = seg[0]
                obj.polygon = [[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)]
            objs.append(obj)
        images.append(Image(id=img["id"], file_name=img["file_name"],
                            width=img["width"], height=img["height"], objects=objs))
    return images


def export_coco(images: list[Image], w: Warnings) -> dict[str, Any]:
    """list[Image] -> COCO dict（含 categories / images / annotations）。"""
    name_to_id: dict[str, int] = {}
    cats: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_anns: list[dict[str, Any]] = []
    ann_id = 0
    for img in images:
        if not img.objects:
            continue
        coco_images.append({"id": img.id, "file_name": img.file_name,
                            "width": img.width, "height": img.height})
        for obj in img.objects:
            if obj.category_name not in name_to_id:
                cid = len(cats)
                name_to_id[obj.category_name] = cid
                cats.append({"id": cid, "name": obj.category_name})
            cid = name_to_id[obj.category_name]
            ann: dict[str, Any] = {"id": ann_id, "image_id": img.id,
                                   "category_id": cid, "iscrowd": 0}
            if obj.bbox_xywh:
                ann["bbox"] = [round(v, 2) for v in obj.bbox_xywh]
                ann["area"] = round(obj.bbox_xywh[2] * obj.bbox_xywh[3], 2)
            if obj.polygon:
                flat = [round(c, 2) for p in obj.polygon for c in p]
                ann["segmentation"] = [flat]
                if "bbox" not in ann:
                    b = polygon_to_bbox(obj.polygon)
                    ann["bbox"] = [round(v, 2) for v in b]
                    ann["area"] = round(b[2] * b[3], 2)
            coco_anns.append(ann)
            ann_id += 1
    return {"categories": cats, "images": coco_images, "annotations": coco_anns}


def load_names(names_path: str | Path) -> list[str]:
    """读取类别名：names.txt（每行一个，行号=class_id）或 data.yaml（names 映射）。"""
    text = Path(names_path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("names:") or "path:" in text.splitlines()[0]:
        # data.yaml 布局（含 path/train/val 等键）：只取 names: 之后 "数字: 类别名" 的行
        names: list[str] = []
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                continue
            if not in_names:
                continue
            if not stripped or ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            if not key.strip().isdigit():
                continue
            names.append(val.strip().strip("\"'"))
        return names
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def save_names(names: list[str], path: str | Path) -> None:
    Path(path).write_text("\n".join(names) + "\n", encoding="utf-8")


def _image_size(image_dir: Path, file_name: str, w: Warnings) -> tuple[int, int] | None:
    src = image_dir / file_name
    if not src.is_file():
        w.warn(f"缺图 {file_name}，跳过该图")
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(src) as im:
            return im.width, im.height
    except Exception as e:
        w.warn(f"读取图片尺寸失败 {file_name}: {e}")
        return None


def parse_yolo(image_dir: str | Path, label_dir: str | Path, names: list[str],
               w: Warnings) -> list[Image]:
    """YOLO labels 目录 -> list[Image]。每行检测 txt(5 列) 或 seg txt(>5 列)。"""
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    images: list[Image] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        img_file = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = image_dir / f"{stem}{ext}"
            if cand.is_file():
                img_file = cand.name
                break
        if img_file is None:
            size = _image_size(image_dir, f"{stem}.jpg", w)
            if size is None:
                continue
            img_file = f"{stem}.jpg"
            width, height = size
        else:
            size = _image_size(image_dir, img_file, w)
            if size is None:
                continue
            width, height = size
        objs: list[Object] = []
        for ln in label_path.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) < 5:
                w.warn(f"{label_path.name}: 非法行，跳过：{ln}")
                continue
            cls = int(float(parts[0]))
            if cls >= len(names):
                w.warn(f"{label_path.name}: class_id {cls} 超出 names 范围，跳过")
                continue
            if len(parts) == 5:
                cx, cy, nw, nh = (float(v) for v in parts[1:])
                x = (cx - nw / 2) * width
                y = (cy - nh / 2) * height
                objs.append(Object(category_name=names[cls], category_id=cls,
                                   bbox_xywh=[x, y, nw * width, nh * height]))
            else:
                if (len(parts) - 1) % 2 != 0:
                    w.warn(f"{label_path.name}: seg 行坐标数非法，跳过：{ln}")
                    continue
                vals = [float(v) for v in parts[1:]]
                points = [[vals[i] * width, vals[i + 1] * height]
                          for i in range(0, len(vals), 2)]
                objs.append(Object(category_name=names[cls], category_id=cls,
                                   polygon=points))
        images.append(Image(id=len(images), file_name=img_file,
                            width=width, height=height, objects=objs))
    return images


def export_yolo(images: list[Image], out_images_dir: str | Path,
                out_labels_dir: str | Path, copy_images: bool,
                w: Warnings, image_dir: str | Path | None = None) -> list[str]:
    """list[Image] -> YOLO labels txt +（可选）复制图片。返回类别名列表（顺序=class_id）。"""
    out_images_dir, out_labels_dir = Path(out_images_dir), Path(out_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    src_root = Path(image_dir) if image_dir else None

    names: list[str] = []
    name_to_idx: dict[str, int] = {}
    for img in images:
        for obj in img.objects:
            if obj.category_name not in name_to_idx:
                name_to_idx[obj.category_name] = len(names)
                names.append(obj.category_name)

    for img in images:
        src = (src_root / img.file_name) if src_root else Path(img.file_name)
        if copy_images and src.is_file():
            import shutil
            shutil.copy2(src, out_images_dir / src.name)
        lines: list[str] = []
        for obj in img.objects:
            idx = name_to_idx[obj.category_name]
            if obj.bbox_xywh:
                x, y, bw, bh = obj.bbox_xywh
                cx = _clamp((x + bw / 2) / img.width, 0.0, 1.0)
                cy = _clamp((y + bh / 2) / img.height, 0.0, 1.0)
                nw = _clamp(bw / img.width, 0.0, 1.0)
                nh = _clamp(bh / img.height, 0.0, 1.0)
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            elif obj.polygon:
                flat = [f"{_clamp(c / img.width if i % 2 == 0 else c / img.height, 0.0, 1.0):.6f}"
                        for i, c in enumerate(c for p in obj.polygon for c in p)]
                lines.append(f"{idx} " + " ".join(flat))
        (out_labels_dir / f"{Path(img.file_name).stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
    return names


def parse_labelme(labelme_dir: str | Path, w: Warnings) -> list[Image]:
    """LabelMe 标注目录（一图一 json）-> list[Image]。"""
    labelme_dir = Path(labelme_dir)
    images: list[Image] = []
    for json_path in sorted(labelme_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        objs: list[Object] = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            stype = shape.get("shape_type", "")
            points = [[float(p[0]), float(p[1])] for p in shape.get("points", [])]
            if stype == "rectangle":
                if len(points) != 2:
                    w.warn(f"{json_path.name}: rectangle 需要 2 点，跳过")
                    continue
                (x1, y1), (x2, y2) = points
                x, y = min(x1, x2), min(y1, y2)
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                if bw <= 0 or bh <= 0:
                    w.warn(f"{json_path.name}: 非法 rectangle，跳过")
                    continue
                objs.append(Object(category_name=label, bbox_xywh=[x, y, bw, bh]))
            elif stype == "polygon":
                if len(points) < 3:
                    w.warn(f"{json_path.name}: polygon 至少 3 点，跳过")
                    continue
                objs.append(Object(category_name=label, polygon=points))
            else:
                w.warn(f"{json_path.name}: 未知 shape_type={stype}，跳过")
        if not objs:
            w.warn(f"{json_path.name}: 无有效标注，跳过该图")
            continue
        images.append(Image(id=len(images),
                            file_name=data.get("imagePath", json_path.stem),
                            width=int(data.get("imageWidth", 0)),
                            height=int(data.get("imageHeight", 0)),
                            objects=objs))
    return images


def export_labelme(images: list[Image], out_dir: str | Path, w: Warnings) -> None:
    """list[Image] -> LabelMe JSON（每图一个，imageData 留空）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shapes = []
        for obj in img.objects:
            shape: dict[str, Any] = {"label": obj.category_name,
                                     "group_id": None, "flags": {}}
            if obj.bbox_xywh:
                x, y, bw, bh = obj.bbox_xywh
                shape["shape_type"] = "rectangle"
                shape["points"] = bbox_to_rectangle(x, y, bw, bh)
            elif obj.polygon:
                shape["shape_type"] = "polygon"
                shape["points"] = [[round(c, 2) for c in p] for p in obj.polygon]
            else:
                continue
            shapes.append(shape)
        if not shapes:
            w.warn(f"{img.file_name}: 无标注可导出，跳过")
            continue
        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": img.file_name,
            "imageData": "",
            "imageWidth": img.width,
            "imageHeight": img.height,
        }
        (out_dir / f"{Path(img.file_name).stem}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
