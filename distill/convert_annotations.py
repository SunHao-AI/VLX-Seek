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
