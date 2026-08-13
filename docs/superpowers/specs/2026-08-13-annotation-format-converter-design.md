# COCO ↔ YOLO ↔ LabelMe 标注格式转换脚本设计

日期：2026-08-13
状态：已确认

## 背景

`distill/data/pseudo_labels.shard0.json` 是 VLX-Seek 生成的 COCO 格式伪标签（仅 bbox，无 segmentation），图片在 `distill/data/images/`。现有 `distill/coco_utils.py` 已含 COCO 读写、COCO→YOLO txt、dataset.yaml 生成。需要新建脚本，支持 COCO / YOLO / LabelMe 三种格式的双向转换，为后续人工标注（LabelMe）与训练（YOLO）流程打通。

## 需求

1. 支持 bbox 与多边形（segmentation/polygon）两类标注。
2. 6 个转换方向：COCO↔YOLO、COCO↔LabelMe、YOLO↔LabelMe。
3. 独立 CLI 脚本（argparse 子命令式），复用 `coco_utils.py`，不改现有文件。
4. YOLO 类别名需显式提供（`--names`）；输出 YOLO 时自动生成 names.txt 与 dataset.yaml。
5. 边界数据（RLE 分割、非法 bbox、空标注、图片缺失）跳过并告警。
6. 验证方式：用 `distill/examples/pseudo_labels.json` 手动往返验证。

## 架构：统一中间表示（IR）

所有格式先解析为统一 IR，再导出为目标格式：

```
COCO  ──parse──┐
YOLO  ──parse──┤→ IR (Image) →──export──→ COCO
LabelMe─parse──┘            ├──export──→ YOLO
                            └──export──→ LabelMe
```

核心数据结构：

```python
@dataclass
class Object:
    category_name: str
    category_id: int | None    # 仅 COCO 需要
    bbox_xywh: list[float] | None   # 像素坐标 [x, y, w, h]
    polygon: list[list[float]] | None  # 像素坐标 [[x, y], ...]

@dataclass
class Image:
    id: int
    file_name: str
    width: int
    height: int
    objects: list[Object]
```

实现 6 个函数：`parse_coco` / `parse_yolo` / `parse_labelme` 与 `export_coco` / `export_yolo` / `export_labelme`。

## 格式映射

| 源格式 | bbox | 多边形 |
|---|---|---|
| COCO | `bbox [x,y,w,h]` | `segmentation [[x1,y1,x2,y2,...]]` |
| YOLO | txt 行 `cls cx cy w h`（归一化） | seg txt 行 `cls x1 y1 x2 y2 ...`（归一化） |
| LabelMe | `shape_type=rectangle`，points `[[x1,y1],[x2,y2]]` | `shape_type=polygon`，points `[[x,y],...]` |

规则：

- 多边形→bbox：取点集 x/y 的 min/max。
- bbox→rectangle：`[[x, y], [x+w, y+h]]`。
- 归一化以图片 width/height 为准，输出 clamp 到 [0,1]；解析时反归一化回像素。
- COCO segmentation 仅支持多边形列表；RLE 跳过并告警。
- LabelMe 输出 `imageData` 置空字符串，避免 base64 撑爆 JSON。
- 图片尺寸来源：COCO 的 `images[].width/height`、LabelMe JSON 自带；YOLO→他格式需从 `--image-dir` 用 PIL 读取（缺图则跳过并告警）。

## CLI 设计

6 个子命令，公共参数风格与现有脚本一致（argparse + 中文 help）：

```bash
python distill/convert_annotations.py coco2yolo    --coco-json X --image-dir I --out-dir O
python distill/convert_annotations.py yolo2coco    --image-dir I --label-dir L --names N --out-json X
python distill/convert_annotations.py coco2labelme --coco-json X --out-dir O
python distill/convert_annotations.py labelme2coco --labelme-dir D --out-json X
python distill/convert_annotations.py yolo2labelme --image-dir I --label-dir L --names N --out-dir O
python distill/convert_annotations.py labelme2yolo --labelme-dir D --out-dir O
```

- YOLO 输出结构：`<out-dir>/images/` + `<out-dir>/labels/` + `<out-dir>/names.txt` + `<out-dir>/dataset.yaml`（复用 `coco_utils.write_dataset_yaml`）。
- 图片复制到 `images/` 默认开启，`--no-copy-images` 关闭。
- `--names` 接受：names.txt（每行一个类别名，行号=class_id）或 data.yaml（`names: {0: a, 1: b}`）。
- 输出 COCO 时类别名→id：YOLO→COCO 直接用 names.txt 的索引顺序（class_id 即索引）；LabelMe→COCO 按 label 首次出现顺序分配，保持 `category_id` 稳定。
- 类型：bbox 与 polygon 的区分。YOLO 解析时同一目录可能存在检测 txt（5 列）与 seg txt（>5 列），按列数自动识别。LabelMe 的 `shape_type` 决定 bbox 还是 polygon。

## 边界处理（跳过 + 告警）

统一走 `warn()` 收集告警，结束前汇总打印数量：

- COCO RLE segmentation → 跳过该标注。
- bbox w/h ≤ 0 或坐标非法 → 跳过该标注。
- 空 objects 的图片 → 跳过该图（LabelMe 空 shapes 同理）。
- 图片文件缺失（YOLO→他格式读尺寸）→ 跳过该图。
- 无法解析的类别名行 / 未知 shape_type → 跳过该标注。

## 代码位置

- 新建 `distill/convert_annotations.py`（单文件，纯标准库：argparse/json/pathlib/dataclasses）。
- 复用 `distill/coco_utils.py`：`load_coco` / `save_coco` / `xywh_to_xyxy` / `write_dataset_yaml`。
- 不改动现有文件。

## 验证

手动往返验证（用 `distill/examples/pseudo_labels.json`，orange/apple 两个类别）：

1. `coco2yolo` → 检查 labels 行数、归一化值在 [0,1]、names.txt/dataset.yaml 内容。
2. `yolo2coco` → 检查 category 名与 bbox 像素值与原始一致（±1px）。
3. `coco2labelme` → 检查 rectangle points 与 bbox 对应。
4. `labelme2coco` → 与原始伪标签对比。
5. `yolo2labelme` / `labelme2yolo` → 同上。
6. 另用 `pseudo_labels.shard0.json` 跑 `coco2yolo` 冒烟，确认中文类别名与 280+ 类别规模下正常。

## 非目标（YAGNI）

- 不做 RLE 分割解析/导出。
- 不做 LabelMe `imageData` 的读取/写入。
- 不做 keypoints、info/licenses 等 COCO 扩展字段。
- 不引入测试框架（手动验证即可）。
