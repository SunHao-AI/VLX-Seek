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


# 以下为占位 stub,Task 4/5 会替换为真实实现;
# 占位仅为满足 tests/test_self_improve.py 顶层 import 的接口名。
def parse_args(argv=None):
    raise NotImplementedError('Task 4 实现')


def load_category_map(path):
    raise NotImplementedError('Task 4 实现')


def infer_one_image(model, image_path, imgsz, conf, iou):
    raise NotImplementedError('Task 4 实现')


def build_round_coco(image_paths, preds_by_image, names_list):
    raise NotImplementedError('Task 4 实现')
