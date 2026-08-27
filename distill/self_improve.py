# -*- coding: utf-8 -*-
"""步骤6: 自改进迭代循环(教师伪标签 → m0 → d1 → m1 → ... 自改进)。

Round 0:  d0(教师伪标签)        -> 训练 m0
Round 1..N:
    Bk.  m_{k-1} 整图推理 -> raw_d_k
    Ck.  VLM 清洗 raw_d_k    -> clean_d_k
    Dk.  clean 子集微调 m_{k-1}(热启动) -> m_k
    Ek.  固定 val 集上评估 m_k -> eval.json + summary.json

策略: student 推理 + 清洗(减法迭代) + 热启动微调; 不调 VLX-Seek 教师;
早停 mAP50 连续 N 轮无提升。用法详见 distill/README.md 步骤 6。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill.coco_utils import load_coco, save_coco, split_coco  # noqa: E402


def split_coco_by_image(
    coco: dict,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict, dict]:
    """按 image 切 train/val, annotations 同图跟随。

    内部委托 `coco_utils.split_coco`(已按 image 切), 这里补 annotation id 各自
    0..n 连续化(原 split_coco 不改 ann id,这里 self_improve 的入口稳定 API)。
    """
    train_coco, val_coco = split_coco(coco, val_ratio=val_ratio, seed=seed)
    for c in (train_coco, val_coco):
        for i, a in enumerate(c['annotations']):
            a['id'] = i
    return train_coco, val_coco


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='自改进迭代: m 整图推理 -> Qwen 清洗 -> 热启动微调(多轮自动)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--init-coco-json', required=True, help='d0 教师伪标签 COCO')
    p.add_argument('--image-dir', required=True, help='图像目录')
    p.add_argument('--category-map', required=True,
                   help='category_prompts.json 路径')
    p.add_argument('--init-weights', default='yolov8s-worldv2.pt')
    p.add_argument('--max-rounds', type=int, default=3)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf-thresh', type=float, default=0.30)
    p.add_argument('--nms-iou', type=float, default=0.50)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--optimizer', default='auto')
    p.add_argument('--lr0', type=float, default=None)
    p.add_argument('--train-device', default='0')
    p.add_argument('--infer-device', default='0')
    p.add_argument('--patience', type=int, default=30)

    # 清洗(透传 clean_pseudo_labels.parse_args)
    p.add_argument('--clean-base-url', default='http://127.0.0.1:8101/v1')
    p.add_argument('--model', default=None,
                   help='VLM served-model-name, e.g. qwen3.8-vllm')
    p.add_argument('--api-key', default=None)
    p.add_argument('--clean-concurrency', type=int, default=16)
    p.add_argument('--min-crop-size', type=int, default=640)
    p.add_argument('--max-side', type=int, default=960)
    p.add_argument('--box-color', default='red')

    # 调度
    p.add_argument('--run-dir', required=True)
    p.add_argument('--early-stop-no-improve', type=int, default=2)
    p.add_argument('--ap-drop-alert', type=float, default=0.20)
    p.add_argument('--ap-drop-window', type=int, default=2)
    p.add_argument('--skip-clean', action='store_true')
    p.add_argument('--model-provider', default=None,
                   help='测试后门: "module.ClassName"; 不传走 ultralytics')
    return p.parse_args(argv)


def load_category_map(path: str) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """加载 category_prompts.json, 返回 (names_list, train_to_cid, cn_to_train)。

    names_list 的 index = category_id = `coco["categories"]` 的 index(同 COCO 顺序)。
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    cats = data.get('categories', {})
    names_list: list[str] = []
    cn_to_train: dict[str, str] = {}
    train_to_cid: dict[str, int] = {}
    for cid, (cn, entry) in enumerate(cats.items()):
        names_list.append(cn)
        train = str(entry.get('train_name', '')).strip()
        if not train:
            continue
        cn_to_train.setdefault(cn, train)
        train_to_cid.setdefault(train, cid)
    return names_list, train_to_cid, cn_to_train


def infer_one_image(model, image_path, imgsz: int, conf: float,
                    iou: float) -> list[tuple[int, float, float, float, float]]:
    """对单张图推理, 返回 [(cls_idx, x, y, w, h), ...](原图像素)。

    letterbox 反算: ultralytics `model.predict` 出的 `boxes.xyxy` 在 letterbox
    域; 输入 src_w × src_h → scale = min(imgsz/w, imgsz/h, 1.0), 居中 pad:
    pad_x = (imgsz - src_w*scale)/2, pad_y = (imgsz - src_h*scale)/2,
    原图 (x-pgx)/scale。越界裁剪到 [0, src] 内, 宽高 < 1px 丢。
    """
    import PIL.Image as _PIL
    src_w, src_h = _PIL.open(image_path).size
    scale = min(imgsz / src_w, imgsz / src_h, 1.0)
    pad_x = (imgsz - src_w * scale) / 2
    pad_y = (imgsz - src_h * scale) / 2

    def _f(v) -> float:
        try:
            return float(v.item())
        except AttributeError:
            return float(v)

    result = model.predict(image_path, imgsz=imgsz, conf=conf, iou=iou,
                           device='0', verbose=False)
    boxes = result[0].boxes
    xyxy = getattr(boxes, 'xyxy', None)
    n = getattr(xyxy, 'shape', (None,))[0]
    if n is None:
        n = len(xyxy)
    if boxes is None or xyxy is None or n == 0:
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for xy, cls in zip(boxes.xyxy, boxes.cls):
        cid = int(_f(cls))
        x1 = _f(xy[0]); y1 = _f(xy[1]); x2 = _f(xy[2]); y2 = _f(xy[3])
        ox1 = max(0.0, (x1 - pad_x) / scale)
        oy1 = max(0.0, (y1 - pad_y) / scale)
        ox2 = min(src_w, (x2 - pad_x) / scale)
        oy2 = min(src_h, (y2 - pad_y) / scale)
        w = ox2 - ox1
        h = oy2 - oy1
        if w < 1 or h < 1:
            continue
        out.append((cid, ox1, oy1, w, h))
    return out


def build_round_coco(image_paths: list, preds_by_image: dict,
                     names_list: list[str]) -> dict:
    """纯函数: 图像组预测 → COCO。调用方负责 save_coco 落盘。"""
    coco: dict = {
        'images': [
            {'id': i, 'file_name': p.name} for i, p in enumerate(image_paths)
        ],
        'categories': [
            {'id': i, 'name': n} for i, n in enumerate(names_list)
        ],
        'annotations': [],
    }
    ann_id = 0
    for img_id, p in enumerate(image_paths):
        for cid, x, y, w, h in preds_by_image.get(p, []):
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cid,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1
    return coco


def _resolve_model_provider_args(args):
    """返回 callable(checkpoint) -> model 实例。

    生产不传 --model-provider → 走 `ultralytics.YOLOWorld`。
    测试传 'distill.tests.test_self_improve.FakeYOLOWorld' → importlib 注入。
    """
    if args.model_provider:
        import importlib
        mod_name, _, cls_name = args.model_provider.rpartition('.')
        if not mod_name or not cls_name:
            raise ValueError(
                f'--model-provider {args.model_provider!r} 必须为 "module.ClassName"'
            )
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return lambda ckpt: cls(ckpt)
    from ultralytics import YOLOWorld
    return YOLOWorld
