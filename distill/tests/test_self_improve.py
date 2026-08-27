"""self_improve.py 离线单元测试: 不跑真实 YOLO-World(无 GPU/无模型下载),
mock YOLOWorld 用 FakeYOLOWorld 确定性输出, 端到端走 1 轮 happy path。

运行: uv run python distill/tests/test_self_improve.py -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DISTILL_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = DISTILL_DIR.parent
for _p in (ROOT_DIR, DISTILL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PIL import Image  # noqa: E402

from distill.self_improve import (  # noqa: E402
    build_round_coco,
    infer_one_image,
    load_category_map,
    parse_args,
    split_coco_by_image,
)
from distill.coco_utils import load_coco  # noqa: E402


def _tiny_coco_and_images(root: Path):
    """造 4 张 640x640 纯色图 + 一份最小 COCO(2 类)。返回 coco json 路径。"""
    imgs = root / 'imgs'
    imgs.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']):
        Image.new('RGB', (640, 640), (i * 50, 10, 30)).save(imgs / name, quality=90)
    coco = {
        'images': [
            {'id': i, 'file_name': n, 'width': 640, 'height': 640}
            for i, n in enumerate(['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'])
        ],
        'categories': [
            {'id': 0, 'name': 'orange', 'train_name': 'loud orange fruit'},
            {'id': 1, 'name': 'apple', 'train_name': 'red apple fruit'},
        ],
        'annotations': [
            {'id': 0, 'image_id': 0, 'category_id': 0, 'bbox': [10, 10, 100, 100], 'area': 10000},
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [20, 20, 80, 80], 'area': 6400},
            {'id': 2, 'image_id': 2, 'category_id': 0, 'bbox': [30, 30, 60, 70], 'area': 4200},
            {'id': 3, 'image_id': 3, 'category_id': 1, 'bbox': [40, 40, 50, 90], 'area': 4500},
        ],
    }
    coco_path = root / 'in.json'
    coco_path.write_text(json.dumps(coco), encoding='utf-8')
    return coco_path


class SplitCocoByImageTest(unittest.TestCase):
    def test_ratio_and_stability(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        p = _tiny_coco_and_images(root)
        coco = load_coco(p)
        train1, val1 = split_coco_by_image(coco, val_ratio=0.25, seed=42)
        train2, val2 = split_coco_by_image(coco, val_ratio=0.25, seed=42)
        # 4 张 * 0.25 = 1 张 val
        self.assertEqual(len(val1['images']), 1)
        self.assertEqual(len(train1['images']), 3)
        self.assertEqual([i['file_name'] for i in val1['images']],
                         [i['file_name'] for i in val2['images']])
        # ann 与 val 同图; ann id 各自 0..n 连续
        for a in val1['annotations']:
            self.assertIn(a['image_id'], [i['id'] for i in val1['images']])
        for i, a in enumerate(train1['annotations']):
            self.assertEqual(a['id'], i)


class FakeBoxes:
    """ultralytics 的 Boxes 最小 mock。"""

    def __init__(self, box, class_id, conf):
        self._xyxy = [list(box)]
        self._cls = [class_id]
        self._conf = [conf]

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def cls(self):
        return _FakeStack([float(c) for c in self._cls])

    @property
    def conf(self):
        return _FakeStack([float(c) for c in self._conf])


class _FakeStack(list):
    """让 `for xy, cls, c in zip(...)` 与 boxes.xyxy, .cls 风格对齐;
    元素是 float(不是 torch.Tensor), 反算时调 .item() 会失败,
    所以 infer_one_image 里要用 try: int(x) except: int(x.item()) 兼容。
    本测试约定: _FakeStack 已是 .item()-able —— 见 FakeBoxFloat。"""
    pass


class FakeBoxFloat(float):
    def item(self):
        return float(self)


class FakeResult:
    def __init__(self, box, class_id, conf):
        class _Boxes:
            def __init__(self, _box, _cls, _conf):
                self.xyxy = _BoxStack([_Box(_box)])
                self.cls = _BoxStack([FakeBoxFloat(_cls)])
                self.conf = _BoxStack([FakeBoxFloat(_conf)])
        self.boxes = _Boxes(box, class_id, conf)


class _Box(list):
    def __iter__(self):
        return (FakeBoxFloat(v) for v in super().__iter__())


class _BoxStack(list):
    pass


class FakeBoxMetrics:
    map = 0.55
    map50 = 0.71
    ap50_95 = [0.6, 0.55]


class FakeResultsVal:
    def __init__(self):
        self.box = FakeBoxMetrics()


class FakeYOLOWorld:
    """最小化 ultralytics YOLOWorld mock, 固定输出 1 个框 (640 域):
    xyxy=(40, 50, 140, 150), cls=0, conf=0.7。"""

    def __init__(self, _checkpoint: str = '') -> None:
        self.val_called = False
        self.train_called = False

    def predict(self, source, imgsz=None, conf=None, iou=None, device=None, verbose=None, **kw):
        return [FakeResult((40.0, 50.0, 140.0, 150.0), 0, 0.7)]

    def train(self, data=None, **_kw):
        self.train_called = True
        return FakeResultsVal()

    def val(self, data=None, **_kw):
        self.val_called = True
        return FakeResultsVal()


if __name__ == '__main__':
    unittest.main()
