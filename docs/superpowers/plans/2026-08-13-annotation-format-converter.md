# COCO ↔ YOLO ↔ LabelMe 标注格式转换脚本实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `distill/convert_annotations.py`，通过统一 IR 实现 COCO / YOLO / LabelMe 三格式双向转换，CLI 为 6 个 argparse 子命令。

**Architecture:** 所有格式解析为统一 `Image`/`Object` 中间结构（parse_*），再导出为目标格式（export_*）。3 个解析器 + 3 个导出器覆盖 6 个方向。复用 `distill/coco_utils.py` 的 `load_coco/save_coco/xywh_to_xyxy/write_dataset_yaml`，不改现有文件。

**Tech Stack:** Python 3.12 标准库（argparse/json/pathlib/dataclasses），PIL（读图片尺寸，项目已有依赖）。

## Global Constraints

- 仅新建 `distill/convert_annotations.py`，不修改任何现有文件。
- 纯标准库 + PIL（`from PIL import Image` 仅用于读尺寸），不新增依赖。
- 支持 bbox 与多边形两类标注；RLE segmentation 跳过并告警。
- 边界数据一律"跳过 + 告警"，汇总打印，不抛错中断。
- YOLO 输出结构：`<out-dir>/images/` + `<out-dir>/labels/` + `<out-dir>/names.txt` + `<out-dir>/dataset.yaml`。
- 中文注释/help，风格与现有 distill 脚本一致（argparse + 中文 docstring）。
- 验证方式：手动命令行验证（不引入测试框架）。

---

### Task 1: IR 数据结构与公共工具函数

**Files:**
- Create: `distill/convert_annotations.py`（本任务只写 IR 与工具部分，后续任务追加）

**Interfaces:**
- Produces（后续任务依赖）:
  - `@dataclass class Object: category_name: str; category_id: int | None = None; bbox_xywh: list[float] | None = None; polygon: list[list[float]] | None = None`
  - `@dataclass class Image: id: int; file_name: str; width: int; height: int; objects: list[Object]`
  - `class Warnings: warn(msg: str) -> None; report() -> None`（收集并汇总打印告警）
  - `def polygon_to_bbox(points: list[list[float]]) -> list[float]` → `[x, y, w, h]`
  - `def bbox_to_rectangle(x: float, y: float, w: float, h: float) -> list[list[float]]` → `[[x, y], [x+w, y+h]]`
  - `def _clamp(v: float, lo: float, hi: float) -> float`

- [ ] **Step 1: 写 IR 与工具代码**

```python
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
```

- [ ] **Step 2: 验证工具函数**

Run:
```powershell
uv run python -c "import sys; sys.path.insert(0, 'distill'); from convert_annotations import polygon_to_bbox, bbox_to_rectangle; assert polygon_to_bbox([[1,2],[5,10]]) == [1, 2, 4, 8]; assert bbox_to_rectangle(1, 2, 4, 8) == [[1, 2], [5, 10]]; print('OK')"
```
Expected: 输出 `OK`（无 assert 失败）。

- [ ] **Step 3: Commit**

```bash
git add distill/convert_annotations.py
git commit -m "feat: 标注转换脚本 IR 数据结构与工具函数"
```

---

### Task 2: COCO 解析与导出

**Files:**
- Modify: `distill/convert_annotations.py`（追加函数）

**Interfaces:**
- Consumes: Task 1 的 `Image` / `Object` / `Warnings` / `polygon_to_bbox`；`coco_utils.load_coco` / `coco_utils.save_coco` / `coco_utils.xywh_to_xyxy`。
- Produces:
  - `def parse_coco(coco: dict[str, Any], w: Warnings) -> list[Image]`
  - `def export_coco(images: list[Image], w: Warnings) -> dict[str, Any]`（返回 COCO dict，写盘由 CLI 用 `coco_utils.save_coco`）

**规则：**
- 解析：`images[].width/height` 为像素基准；annotations 的 `bbox` → `Object.bbox_xywh`，`segmentation` 为 `[[x1,y1,x2,y2,...]]` 列表 → 拆成 `[[x,y],...]` 填入 `Object.polygon`；RLE（dict 类型）→ 跳过并告警；`iscrowd=1` 或 w/h ≤ 0 → 跳过并告警。
- 导出：类别 id 按首次出现顺序稳定分配；同一 Object 的 bbox 与 polygon 并存时同时写出；坐标取整保留 2 位小数。

- [ ] **Step 1: 追加 parse_coco / export_coco 实现**

```python
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
```

- [ ] **Step 2: 验证 parse/export 往返**

Run:
```powershell
uv run python -c "import sys, json; sys.path.insert(0, 'distill'); from coco_utils import load_coco; from convert_annotations import parse_coco, export_coco, Warnings; coco = load_coco('distill/examples/pseudo_labels.json'); w = Warnings(); imgs = parse_coco(coco, w); out = export_coco(imgs, w); assert len(out['images']) == 2 and len(out['annotations']) == 5, (len(out['images']), len(out['annotations'])); assert [c['name'] for c in out['categories']] == ['orange', 'apple']; print('OK')"
```
Expected: 输出 `OK`。

- [ ] **Step 3: Commit**

```bash
git add distill/convert_annotations.py
git commit -m "feat: COCO 解析与导出"
```

---

### Task 3: YOLO 解析与导出

**Files:**
- Modify: `distill/convert_annotations.py`（追加函数）

**Interfaces:**
- Consumes: Task 1 的 `Image` / `Object` / `Warnings` / `_clamp`；Task 2 无依赖；`coco_utils.write_dataset_yaml`（CLI 层用，本任务实现 names 读写）。
- Produces:
  - `def load_names(names_path: str | Path) -> list[str]`（names.txt 每行一个 或 data.yaml 的 names 映射）
  - `def save_names(names: list[str], path: str | Path) -> None`
  - `def parse_yolo(image_dir: str | Path, label_dir: str | Path, names: list[str], w: Warnings) -> list[Image]`
  - `def export_yolo(images: list[Image], out_images_dir: str | Path, out_labels_dir: str | Path, copy_images: bool, w: Warnings, image_dir: str | Path | None = None) -> list[str]`（返回类别名列表，顺序=class_id）

**规则：**
- YOLO 检测 txt（5 列）→ `Object.bbox_xywh`；seg txt（>5 列，列数-1 为偶数）→ `Object.polygon`。
- 归一化解析：`cx cy w h` 或 `x1 y1 x2 y2...` 均除以图片宽高反归一化为像素。
- 解析需图片尺寸：从 `image_dir` 用 `PIL.Image.open` 读取；图片缺失或读取失败 → 跳过该图并告警。
- 导出：类别 class_id 按类别名首次出现顺序分配（类别名保真）；坐标超 [0,1] 均 clamp；归一化保留 6 位小数。
- 无标注的图片也写出空 txt。
- `copy_images` 为真时，源图片路径：`image_dir` 提供则 `image_dir / file_name`，否则 `file_name`（相对 cwd）。

- [ ] **Step 1: 追加 YOLO 相关函数**

```python
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
```

- [ ] **Step 2: 验证 names 读写与 YOLO 解析**

Run:
```powershell
uv run python -c "import sys, tempfile; sys.path.insert(0, 'distill'); from pathlib import Path; from convert_annotations import load_names, save_names, parse_yolo, Warnings; d = Path(tempfile.mkdtemp()); (d/'n.txt').write_text('orange\napple\n', encoding='utf-8'); save_names(['orange','apple'], d/'n2.txt'); assert load_names(d/'n.txt') == ['orange','apple']; assert load_names(d/'n2.txt') == ['orange','apple']; print('OK')"
```
Expected: 输出 `OK`。

- [ ] **Step 3: Commit**

```bash
git add distill/convert_annotations.py
git commit -m "feat: YOLO 解析与导出（names 读写、检测/分割 txt）"
```

---

### Task 4: LabelMe 解析与导出

**Files:**
- Modify: `distill/convert_annotations.py`（追加函数）

**Interfaces:**
- Consumes: Task 1 的 `Image` / `Object` / `Warnings` / `polygon_to_bbox` / `bbox_to_rectangle`。
- Produces:
  - `def parse_labelme(labelme_dir: str | Path, w: Warnings) -> list[Image]`
  - `def export_labelme(images: list[Image], out_dir: str | Path, w: Warnings) -> None`

**规则：**
- LabelMe JSON 字段：`imagePath` / `imageWidth` / `imageHeight` / `shapes[].{label, shape_type, points, group_id}`。
- 解析：`rectangle` → bbox（points `[[x1,y1],[x2,y2]]` 的 w/h 取正数）；`polygon` → polygon；未知 shape_type → 跳过并告警；空 shapes → 跳过该图并告警。
- 导出：`rectangle` 用 `bbox_to_rectangle`；`polygon` 用顶点列表；`imageData` 置空字符串；每图一个 `<stem>.json`。

- [ ] **Step 1: 追加 LabelMe 相关函数**

```python
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
```

- [ ] **Step 2: 验证 LabelMe 往返（coco → labelme → coco）**

Run:
```powershell
uv run python -c "import sys, tempfile; sys.path.insert(0, 'distill'); from pathlib import Path; from coco_utils import load_coco; from convert_annotations import parse_coco, export_labelme, parse_labelme, export_coco, Warnings; coco = load_coco('distill/examples/pseudo_labels.json'); w = Warnings(); imgs = parse_coco(coco, w); d = Path(tempfile.mkdtemp()); export_labelme(imgs, d, w); imgs2 = parse_labelme(d, w); out = export_coco(imgs2, w); assert len(out['annotations']) == 5, len(out['annotations']); assert out['annotations'][0]['bbox'] == [120, 90, 90, 90]; print('OK')"
```
Expected: 输出 `OK`。

- [ ] **Step 3: Commit**

```bash
git add distill/convert_annotations.py
git commit -m "feat: LabelMe 解析与导出"
```

---

### Task 5: CLI 组装与端到端往返验证

**Files:**
- Modify: `distill/convert_annotations.py`（追加 argparse 主函数 + `if __name__ == "__main__"`）

**Interfaces:**
- Consumes: Task 1-4 全部函数；`coco_utils.load_coco / save_coco / write_dataset_yaml`。
- Produces: 可执行 CLI，6 个子命令。

**子命令参数（风格与现有脚本一致）：**

| 子命令 | 参数 |
|---|---|
| `coco2yolo` | `--coco-json`(必填), `--image-dir`(必填), `--out-dir`(必填), `--no-copy-images` |
| `yolo2coco` | `--image-dir`(必填), `--label-dir`(必填), `--names`(必填), `--out-json`(必填) |
| `coco2labelme` | `--coco-json`(必填), `--out-dir`(必填) |
| `labelme2coco` | `--labelme-dir`(必填), `--out-json`(必填) |
| `yolo2labelme` | `--image-dir`(必填), `--label-dir`(必填), `--names`(必填), `--out-dir`(必填) |
| `labelme2yolo` | `--labelme-dir`(必填), `--out-dir`(必填), `--image-dir`(可选), `--no-copy-images` |

**CLI 逻辑：**
- `coco2yolo`：`load_coco` → `parse_coco` → `export_yolo(out/images, out/labels, copy, w, image_dir=args.image_dir)` → `save_names` + `write_dataset_yaml(out, names)`。
- `yolo2coco`：`load_names` → `parse_yolo` → `export_coco` → `save_coco`。
- `coco2labelme`：`load_coco` → `parse_coco` → `export_labelme`。
- `labelme2coco`：`parse_labelme` → `export_coco` → `save_coco`。
- `yolo2labelme`：`load_names` → `parse_yolo` → `export_labelme`。
- `labelme2yolo`：`parse_labelme` → `export_yolo(out/images, out/labels, copy, w, image_dir=args.image_dir)` → `save_names` + `write_dataset_yaml(out, names)`。
- 每个子命令结束打印 `转换完成：N 张图` 并调用 `w.report()`。
- `--no-copy-images` 缺省为复制图片；`labelme2yolo` 未传 `--image-dir` 时图片源为相对 cwd 的 file_name。

- [ ] **Step 1: 追加 CLI 实现**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COCO / YOLO / LabelMe 标注格式双向转换")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common_yolo(p: argparse.ArgumentParser) -> None:
        p.add_argument("--image-dir", required=True, help="图像目录（YOLO 反归一化读尺寸/复制图片）")
        p.add_argument("--label-dir", required=True, help="YOLO labels 目录（*.txt）")
        p.add_argument("--names", required=True, help="类别名：names.txt（每行一个）或 data.yaml")

    def add_out_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out-dir", required=True, help="输出目录（自动生成 images/ labels/ names.txt dataset.yaml）")

    p = sub.add_parser("coco2yolo", help="COCO JSON -> YOLO txt")
    p.add_argument("--coco-json", required=True, help="COCO 标注 JSON")
    p.add_argument("--image-dir", required=True, help="COCO file_name 相对此目录的图片源目录")
    add_out_dir(p)
    p.add_argument("--no-copy-images", action="store_true", help="不复制图片到 out/images")

    p = sub.add_parser("yolo2coco", help="YOLO txt -> COCO JSON")
    add_common_yolo(p)
    p.add_argument("--out-json", required=True, help="输出 COCO JSON 路径")

    p = sub.add_parser("coco2labelme", help="COCO JSON -> LabelMe JSON")
    p.add_argument("--coco-json", required=True, help="COCO 标注 JSON")
    p.add_argument("--out-dir", required=True, help="LabelMe JSON 输出目录")

    p = sub.add_parser("labelme2coco", help="LabelMe JSON -> COCO JSON")
    p.add_argument("--labelme-dir", required=True, help="LabelMe 标注目录（一图一 json）")
    p.add_argument("--out-json", required=True, help="输出 COCO JSON 路径")

    p = sub.add_parser("yolo2labelme", help="YOLO txt -> LabelMe JSON")
    add_common_yolo(p)
    add_out_dir(p)

    p = sub.add_parser("labelme2yolo", help="LabelMe JSON -> YOLO txt")
    p.add_argument("--labelme-dir", required=True, help="LabelMe 标注目录（一图一 json）")
    p.add_argument("--image-dir", default=None, help="图片源目录（复制到 out/images；缺省时相对 cwd 查找）")
    add_out_dir(p)
    p.add_argument("--no-copy-images", action="store_true", help="不复制图片到 out/images")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    w = Warnings()
    from coco_utils import load_coco, save_coco, write_dataset_yaml

    out = Path(args.out_dir) if hasattr(args, "out_dir") and args.out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "coco2yolo":
        imgs = parse_coco(load_coco(args.coco_json), w)
        names = export_yolo(imgs, out / "images", out / "labels",
                            not args.no_copy_images, w, image_dir=args.image_dir)
        save_names(names, out / "names.txt")
        write_dataset_yaml(out, names)
        print(f"转换完成：{len(imgs)} 张图")
    elif args.cmd == "yolo2coco":
        imgs = parse_yolo(args.image_dir, args.label_dir, load_names(args.names), w)
        save_coco(export_coco(imgs, w), args.out_json)
        print(f"转换完成：{len(imgs)} 张图")
    elif args.cmd == "coco2labelme":
        imgs = parse_coco(load_coco(args.coco_json), w)
        export_labelme(imgs, args.out_dir, w)
        print(f"转换完成：{len(imgs)} 张图")
    elif args.cmd == "labelme2coco":
        imgs = parse_labelme(args.labelme_dir, w)
        save_coco(export_coco(imgs, w), args.out_json)
        print(f"转换完成：{len(imgs)} 张图")
    elif args.cmd == "yolo2labelme":
        imgs = parse_yolo(args.image_dir, args.label_dir, load_names(args.names), w)
        export_labelme(imgs, args.out_dir, w)
        print(f"转换完成：{len(imgs)} 张图")
    elif args.cmd == "labelme2yolo":
        imgs = parse_labelme(args.labelme_dir, w)
        names = export_yolo(imgs, out / "images", out / "labels",
                            not args.no_copy_images, w, image_dir=args.image_dir)
        save_names(names, out / "names.txt")
        write_dataset_yaml(out, names)
        print(f"转换完成：{len(imgs)} 张图")
    w.report()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 端到端验证（examples 往返 + shard 冒烟）**

Run（依次执行，全部成功即通过）:
```powershell
# 1. coco2yolo
uv run python distill/convert_annotations.py coco2yolo --coco-json distill/examples/pseudo_labels.json --image-dir distill/examples/images --out-dir distill/examples/runs/yolo
# 预期：runs/yolo 下出现 images/ labels/ names.txt dataset.yaml；2 张图 5 条标注

# 2. yolo2coco（回读验证）
uv run python distill/convert_annotations.py yolo2coco --image-dir distill/examples/runs/yolo/images --label-dir distill/examples/runs/yolo/labels --names distill/examples/runs/yolo/names.txt --out-json distill/examples/runs/back.json
# 预期：back.json 有 2 图 5 标注，bbox 与原始 [120,90,90,90] 等一致（±1px）

# 3. coco2labelme + labelme2coco
uv run python distill/convert_annotations.py coco2labelme --coco-json distill/examples/pseudo_labels.json --out-dir distill/examples/runs/labelme
uv run python distill/convert_annotations.py labelme2coco --labelme-dir distill/examples/runs/labelme --out-json distill/examples/runs/back2.json
# 预期：back2.json 同样 2 图 5 标注

# 4. shard0 冒烟（280+ 中文类别）
uv run python distill/convert_annotations.py coco2yolo --coco-json distill/data/pseudo_labels.shard0.json --image-dir distill/data/images --out-dir distill/data/yolo_out
# 预期：labels/ 与 images/ 数量一致，names.txt 为中文类别名，无 RLE 告警
```

- [ ] **Step 3: 检查 labels 内容**

Run:
```powershell
Get-Content distill/examples/runs/yolo/labels/demo_image.txt
```
Expected: 每行 5 列（`cls cx cy w h`），数值均在 [0,1]。

- [ ] **Step 4: Commit**

```bash
git add distill/convert_annotations.py
git commit -m "feat: 标注转换 CLI（6 个子命令）与端到端验证"
```

---

## 自审记录

- **Spec 覆盖**：IR 架构（T1）、COCO 解析/导出含 RLE 跳过（T2）、YOLO 解析/导出含 names 与 dataset.yaml（T3/T5）、LabelMe 解析/导出（T4）、6 子命令 CLI（T5）、边界跳过+告警（各 parse/export 内）、手动往返验证（T5 Step 2）。全部覆盖。
- **占位符**：无 TBD/TODO；所有步骤含具体代码与命令。
- **类型一致性**：`parse_coco` 返回 `list[Image]`、`export_coco` 返回 `dict`、`parse_yolo`/`parse_labelme` 返回 `list[Image]`、`export_yolo` 返回 `list[str]`、`export_labelme` 返回 `None`，各任务引用一致。
