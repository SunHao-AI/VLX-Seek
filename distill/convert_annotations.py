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
