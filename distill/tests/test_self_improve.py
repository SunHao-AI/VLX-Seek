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
        # mock: 写一个 best.pt(project/name/best.pt)
        import os
        project = _kw.get('project') or ''
        name = _kw.get('name') or 'yolo_world'
        pt_dir = os.path.join(project, name)
        os.makedirs(pt_dir, exist_ok=True)
        with open(os.path.join(pt_dir, 'best.pt'), 'wb') as fh:
            fh.write(b'FAKE_BEST')
        return FakeResultsVal()

    def val(self, data=None, **_kw):
        self.val_called = True
        return FakeResultsVal()


class LoadCategoryMapTest(unittest.TestCase):
    def test_three_indexes(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        p = root / 'category_prompts.json'
        p.write_text(json.dumps({
            'categories': {
                'orange': {'train_name': 'loud orange fruit'},
                'apple': {'train_name': 'red apple fruit'},
            }
        }), encoding='utf-8')
        names_list, train_to_cid, cn_to_train = load_category_map(p)
        self.assertEqual(names_list, ['orange', 'apple'])
        self.assertEqual(train_to_cid, {
            'loud orange fruit': 0,
            'red apple fruit': 1,
        })
        self.assertEqual(cn_to_train, {
            'orange': 'loud orange fruit',
            'apple': 'red apple fruit',
        })


class ParseArgsTest(unittest.TestCase):
    def _required(self, root: Path) -> list[str]:
        return [
            '--init-coco-json', str(root / 'in.json'),
            '--image-dir', str(root / 'imgs'),
            '--category-map', str(root / 'category_prompts.json'),
            '--run-dir', str(root / 'run'),
        ]

    def test_defaults(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        args = parse_args(self._required(root) + ['--model', 'mock-model'])
        self.assertEqual(args.max_rounds, 3)
        self.assertAlmostEqual(args.val_ratio, 0.1)
        self.assertEqual(args.imgsz, 640)
        self.assertAlmostEqual(args.conf_thresh, 0.30)
        self.assertAlmostEqual(args.nms_iou, 0.50)
        self.assertEqual(args.early_stop_no_improve, 2)
        self.assertEqual(args.init_weights, 'yolov8s-worldv2.pt')
        self.assertFalse(args.skip_clean)
        self.assertEqual(args.box_color, 'red')
        self.assertEqual(args.min_crop_size, 640)


class InferOneImageTest(unittest.TestCase):
    def test_letterbox_no_scale(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        _tiny_coco_and_images(root)
        img = root / 'imgs' / 'a.jpg'
        model = FakeYOLOWorld()
        preds = infer_one_image(model, img, imgsz=640, conf=0.1, iou=0.5)
        self.assertEqual(preds, [(0, 40.0, 50.0, 100.0, 100.0)])

    def test_letterbox_downscale(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        (root / 'imgs').mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (1280, 640), (1, 2, 3)).save(root / 'imgs' / 'big.jpg',
                                                       quality=90)
        img = root / 'imgs' / 'big.jpg'

        # letterbox 域 640x640 正方形, 内容 640x320 (scale=0.5) 居中 →
        # pad_x=0, pad_y=(640-640*0.5)/2=160, 内容 y 范围 [160, 480]。
        class _PatchModel:
            """只覆写 predict, 返回指定 box(覆盖 FakeYOLOWorld 的固定 box)。"""
            def __init__(self, box):
                self._box = box
            def predict(self, source, imgsz=None, conf=None, iou=None,
                        device=None, verbose=None, **kw):
                return [FakeResult(self._box, 0, 0.7)]

        # ① box 落在 letterbox 内容区(200,300 都 >= 160): 反算 should 得到原图像素
        #   ox=(x-0)/0.5: (40..140)/0.5 = 80..280 (w=200)
        #   oy=(y-160)/0.5: (200..300)/… = (40..140)/0.5 = 80..280 (h=200)
        preds = infer_one_image(_PatchModel((40.0, 200.0, 140.0, 300.0)),
                                img, imgsz=640, conf=0.1, iou=0.5)
        self.assertEqual(preds, [(0, 80.0, 80.0, 200.0, 200.0)])

        # ② box 完全落在 pad 区(40..150 都 < 160): 反算 oy2<0,h<1 → 丢
        preds2 = infer_one_image(_PatchModel((40.0, 50.0, 140.0, 150.0)),
                                 img, imgsz=640, conf=0.1, iou=0.5)
        self.assertEqual(preds2, [])


class EndToEndMockTest(unittest.TestCase):
    """完整 mock 1 轮: d0 → 训 m0(mock) → 推理 d1(mock) → 清洗(skip) →
    训 m1(mock) → 评估(固定 0.71)+ summary 链路。"""

    def test_one_round_happy_path(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        coco_path = _tiny_coco_and_images(root)
        cat_map_path = root / 'category_prompts.json'
        cat_map_path.write_text(json.dumps({
            'categories': {
                'orange': {'train_name': 'loud orange fruit'},
                'apple': {'train_name': 'red apple fruit'},
            }
        }), encoding='utf-8')
        run_dir = root / 'run'
        argv = [
            '--init-coco-json', str(coco_path),
            '--image-dir', str(root / 'imgs'),
            '--category-map', str(cat_map_path),
            '--run-dir', str(run_dir),
            '--max-rounds', '1',
            '--val-ratio', '0.25',
            '--epochs', '1',
            '--batch', '1',
            '--train-device', 'cpu',
            '--model', 'mock-model',
            '--clean-base-url', 'http://127.0.0.1:9/v1',
            '--skip-clean',
            '--model-provider', 'distill.tests.test_self_improve.FakeYOLOWorld',
        ]
        from distill.self_improve import main
        main(argv)
        # 验证产物
        self.assertTrue((run_dir / 'config.json').is_file())
        split = json.loads((run_dir / 'split.json').read_text())
        self.assertEqual(len(split['val_file_names']), 1)
        self.assertTrue((run_dir / 'round_0' / 'm0.pt').is_file())
        self.assertTrue((run_dir / 'round_0' / 'eval.json').is_file())
        self.assertTrue((run_dir / 'round_1' / 'raw_d1.json').is_file())
        self.assertTrue((run_dir / 'round_1' / 'clean_d1.json').is_file())
        self.assertTrue((run_dir / 'round_1' / 'm1.pt').is_file())
        summary = json.loads((run_dir / 'summary.json').read_text())
        self.assertEqual(len(summary['rounds']), 2)  # round0 + round1
        self.assertAlmostEqual(summary['rounds'][1]['mAP50'], 0.71, places=2)
        self.assertEqual(summary['rounds'][1]['delta_map50'],
                         round(0.71 - summary['rounds'][0]['mAP50'], 5))
        # round_1 的 raw_d1 必须有 4 条 ann(FakeYOLOWorld 每图固定 1 框)
        raw = load_coco(run_dir / 'round_1' / 'raw_d1.json')
        self.assertEqual(len(raw['annotations']), 4)


class ResumeStateTest(unittest.TestCase):
    """续跑: 同一 run_dir 二次跑不产生新 round 目录。"""

    def test_resume_skips_completed_rounds(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        coco_path = _tiny_coco_and_images(root)
        cat_map_path = root / 'category_prompts.json'
        cat_map_path.write_text(json.dumps({
            'categories': {'orange': {'train_name': 'a'},
                           'apple': {'train_name': 'b'}}
        }), encoding='utf-8')
        run_dir = root / 'run'
        argv = [
            '--init-coco-json', str(coco_path),
            '--image-dir', str(root / 'imgs'),
            '--category-map', str(cat_map_path),
            '--run-dir', str(run_dir),
            '--max-rounds', '2',
            '--val-ratio', '0.25',
            '--skip-clean',
            '--model-provider', 'distill.tests.test_self_improve.FakeYOLOWorld',
        ]
        from distill.self_improve import main
        main(argv)
        self.assertTrue((run_dir / 'round_0' / 'm0.pt').is_file())
        self.assertTrue((run_dir / 'round_1' / 'm1.pt').is_file())
        self.assertTrue((run_dir / 'round_2' / 'm2.pt').is_file())
        s1 = json.loads((run_dir / 'summary.json').read_text())
        n1 = len(s1['rounds'])
        main(argv)  # 第二次同 run_dir → last_done=2, range(3,3) 空
        s2 = json.loads((run_dir / 'summary.json').read_text())
        self.assertEqual(len(s2['rounds']), n1)
        self.assertFalse((run_dir / 'round_3').is_dir())


class EarlyStopRuleTest(unittest.TestCase):
    """连续 N 轮 mAP50 无提升 → 停在 max_rounds 前。"""

    def test_two_consecutive_no_improve_stops_at_round_2(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        coco_path = _tiny_coco_and_images(root)
        cat_map_path = root / 'category_prompts.json'
        cat_map_path.write_text(json.dumps({
            'categories': {'orange': {'train_name': 'a'},
                           'apple': {'train_name': 'b'}}
        }), encoding='utf-8')
        run_dir = root / 'run'
        argv = [
            '--init-coco-json', str(coco_path),
            '--image-dir', str(root / 'imgs'),
            '--category-map', str(cat_map_path),
            '--run-dir', str(run_dir),
            '--max-rounds', '5',
            '--val-ratio', '0.25',
            '--skip-clean',
            '--early-stop-no-improve', '2',
            '--model-provider', 'distill.tests.test_self_improve.FakeYOLOWorld',
        ]
        # FakeResultsVal 固定 map50=0.71 → 每轮 delta=0, 第 2 轮满足
        # "连续 2 轮 <=0" 即提前停(不等 max_rounds=5)
        from distill.self_improve import main
        main(argv)
        summary = json.loads((run_dir / 'summary.json').read_text())
        self.assertEqual(summary['rounds'][-1]['round'], 2)
        self.assertFalse((run_dir / 'round_3').is_dir())


class PerClassAPAlertTest(unittest.TestCase):
    """每类 AP 连续 2 轮跌 > 20% → stderr 警告(不终止)。"""

    def test_drop_warn_written_to_stderr(self):
        self.skipTest('需要可变 metrics 源; 现阶段由 summary 展示告警(人工看)')


if __name__ == '__main__':
    unittest.main()
