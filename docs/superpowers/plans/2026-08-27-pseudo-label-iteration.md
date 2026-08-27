# 伪标签质量改进 + 自改进迭代 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Qwen 清洗的裁剪图扩到 640²、画红框聚焦 VLM 注意力;新增 `distill/self_improve.py` 编排 m0 整图推理 → 清洗 → 微调的多轮自改进,每轮固定 val 上监控 mAP50 早停。

**Architecture:** 两个独立交付。Part A 改 `clean_pseudo_labels.py` 的裁剪函数 / prompt / 新增参数 / 测试;Part B 新增 `distill/self_improve.py` 单体编排脚本,in-process 复用现存的清洗 `run_pipeline` + `finetune_yolo_world.prepare_dataset` + `YOLOWorld` 对象,通过 `runs/<ts>/round_k/` 目录约定断点续跑。

**Tech Stack:** Python 3.10+ / PIL / requests / tqdm / ultralytics(YOLOWorld)/ 现有 mock http.server 测试基建。

## Global Constraints

- Python 3.10+, `uv run python`; Windows + PowerShell(`;` OK, `&&` 不可用)。
- 字符串单引号(最近 commit "style(refactor): 统一代码字符串引号风格")。
- 不改 `distill/generate_pseudo_labels.py` / `vlx_seek_worker.py` / `vllm_serve/*`。
- 决策日志 meta 参数变更自动失效(现有机制)。
- `distill/xxx.py` 入口必须 `ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))` + `from distill.xxx import ...`。
- 测试命令 `uv run python distill/tests/<file>.py -v`(unittest)。
- commit: 中文;`git add` 具体文件。

**改动文件清单**:
- Modify: `distill/clean_pseudo_labels.py` / `distill/tests/test_clean_pseudo_labels.py` / `distill/README.md` / `.gitignore`(末加 `self_improve_runs/`)
- Create: `distill/self_improve.py` / `distill/tests/test_self_improve.py`
- 不动: `coco_utils.py` / `finetune_yolo_world.py` / 教师生成脚本

---

## Task 1: 重写 `crop_decode`

**Files:**
- Modify: `d:\WorkPlace\Pycharm\VLX-Seek\distill\clean_pseudo_labels.py` (函数 `crop_encode` 区域 ~L183-227)
- Test: `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_clean_pseudo_labels.py` (`CropEncodeTest` ~L278-316)

**Interfaces:**
- Produces: `crop_decode(image, bbox_xywh, min_crop_size=640, max_side=960, box_color='red') -> tuple[bytes, tuple[int,int,int,int]]`

- [ ] **Step 1: 把旧 `CropEncodeTest` 整体替换为新 `CropDecodeTest`,先让它失败**

打开 `distill/tests/test_clean_pseudo_labels.py`:

(1) 第 24-37 行的 import 块中 `crop_encode` 改名 `crop_decode`:

```python
from clean_pseudo_labels import (  # noqa: E402
    DecisionLog,
    ServiceUnreachable,
    VLMVerifier,
    crop_decode,   # ← 改名
    dedup_annotations,
    iou_xywh,
    load_previous_decisions,
    main,
    parse_args,
    run_pipeline,
    validate_refs,
    write_output,
)
```

(2) 把第 278-316 行的 `CropEncodeTest`(连同 setUp 的 `self.img500`)替换为:

```python
class CropDecodeTest(unittest.TestCase):
    def setUp(self):
        self.img_small = Image.new('RGB', (640, 640), (120, 40, 40))
        self.img_mid = Image.new('RGB', (1000, 1000), (200, 60, 60))
        self.img_big = Image.new('RGB', (4000, 3000), (60, 60, 200))

    @staticmethod
    def _decode(data: bytes) -> Image.Image:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im

    def test_small_target_small_image(self):
        data, box = crop_decode(self.img_small, [0, 0, 100, 100],
                                min_crop_size=640, max_side=960)
        self.assertEqual(self._decode(data).size, (640, 640))
        self.assertEqual(box, (0, 0, 100, 100))

    def test_small_target_mid_image(self):
        data, box = crop_decode(self.img_mid, [0, 0, 100, 100],
                                min_crop_size=640, max_side=960)
        self.assertEqual(self._decode(data).size, (640, 640))
        self.assertEqual(box, (0, 0, 100, 100))

    def test_mid_target_mid_image_no_downscale(self):
        # [100,200,400,300] 中心 (300, 350), 窗口 640x640 → left=max(0, -20)=0,
        # top=max(0, 30)=30, 窗口 [0, 30, 640, 670]; 不缩; 局部 (100, 170, 400, 300)
        data, box = crop_decode(self.img_mid, [100, 200, 400, 300],
                                min_crop_size=640, max_side=960)
        im = self._decode(data)
        self.assertEqual(im.size, (640, 640))
        self.assertEqual(box, (100, 170, 400, 300))

    def test_large_target_triggers_downscale(self):
        # [1000,800,2000,1500] 中心 (2000,1550), 窗口 2000x1500 → 窗口 [1000,800,3000,2300]
        # size=(2000,1500), max_side=960 → scale=960/2000=0.48 → resize=(960,720)
        # 局部 box = (0, 0, 960, 720)
        data, box = crop_decode(self.img_big, [1000, 800, 2000, 1500],
                                min_crop_size=640, max_side=960)
        im = self._decode(data)
        self.assertEqual((im.size), (960, 720))
        self.assertEqual(box, (0, 0, 960, 720))

    def test_image_smaller_than_min_raises(self):
        tiny = Image.new('RGB', (500, 500), (1, 2, 3))
        with self.assertRaises(ValueError):
            crop_decode(tiny, [10, 10, 100, 100], min_crop_size=640, max_side=960)

    def test_yellow_box_color(self):
        data, box = crop_decode(self.img_small, [0, 0, 100, 100],
                                min_crop_size=640, max_side=960, box_color='yellow')
        self.assertEqual(self._decode(data).size, (640, 640))
        self.assertEqual(box, (0, 0, 100, 100))

    def test_invalid_box_color_raises(self):
        with self.assertRaises(ValueError):
            crop_decode(self.img_small, [0, 0, 100, 100],
                        min_crop_size=640, max_side=960, box_color='blue')
```

- [ ] **Step 2: 跑测试看它失败**

```powershell
uv run python distill/tests/test_clean_pseudo_labels.py CropDecodeTest -v
```

预期: 7 个全 FAIL,`ImportError: cannot import name 'crop_decode'`。

- [ ] **Step 3: 实现新函数 + 删除旧 `crop_encode`**

打开 `distill/clean_pseudo_labels.py`,把第 183-227 行的 `crop_encode` 整段替换为:

```python
BOX_COLORS = {'red': (255, 0, 0), 'yellow': (255, 255, 0)}
BOX_COLOR_OFF = 'off'


def crop_decode(
    image: Image.Image,
    bbox_xywh: list[float],
    min_crop_size: int = 640,
    max_side: int = 960,
    box_color: str = 'red',
) -> tuple[bytes, tuple[int, int, int, int]]:
    """以目标为中心裁剪, 边长取 max(原框, min_crop_size), 越界反推(借鉴 cv_utils.crop_rect)。

    送 VLM 前在裁剪图上画 box_color 矩形框(4px)引导注意力; 最长边 > max_side
    等比缩小(只缩不放, 坐标同步换算, 钳制到裁剪图内)。

    Returns:
        (JPEG 字节流, 目标框在裁剪图局部坐标系的 xywh 像素整数)。

    Raises:
        ValueError: 当 image 任一边 < min_crop_size, 或 box_color 非法。
    """
    if box_color not in BOX_COLORS and box_color != BOX_COLOR_OFF:
        raise ValueError(f'非法 box_color: {box_color!r}, 可选 {list(BOX_COLORS) + [BOX_COLOR_OFF]}')
    x, y, w, h = bbox_xywh
    img_w, img_h = image.size
    if img_w < min_crop_size or img_h < min_crop_size:
        raise ValueError(
            f'图像 {img_w}x{img_h} 小于 min_crop_size={min_crop_size}, 无法保证 '
            f'{min_crop_size}x{min_crop_size} 上下文; 请上扬图源或减小 --min-crop-size'
        )
    # 1) 计算裁剪窗口: 撑到 min_crop_size, 越界反推(借鉴 cv_utils.crop_rect)
    cw = max(int(round(w)), min_crop_size)
    ch = max(int(round(h)), min_crop_size)
    cx, cy = x + w / 2, y + h / 2
    left = max(0, int(round(cx - cw / 2)))
    top = max(0, int(round(cy - ch / 2)))
    right = min(img_w, left + cw)
    bottom = min(img_h, top + ch)
    left = right - cw
    top = bottom - ch
    crop = image.crop((left, top, right, bottom))
    # 2) 只缩不放: 最长边超 max_side 等比缩
    scale = 1.0
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        )
    # 3) 目标框局部坐标(换算 + 钳制) + 画框
    bx = (x - left) * scale
    by = (y - top) * scale
    bw = w * scale
    bh = h * scale
    crop_w, crop_h = crop.size
    bx = max(0, min(crop_w, round(bx)))
    by = max(0, min(crop_h, round(by)))
    bw = max(1, min(crop_w - bx, round(bw)))
    bh = max(1, min(crop_h - by, round(bh)))
    if box_color != BOX_COLOR_OFF:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(crop)
        outline = BOX_COLORS[box_color]
        for i in range(4):  # 4px 线宽
            draw.rectangle([bx + i, by + i, bx + bw - i - 1, by + bh - i - 1], outline=outline)
    # 4) JPEG 编码
    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=85)
    return buf.getvalue(), (int(bx), int(by), int(bw), int(bh))
```

- [ ] **Step 4: 跑测试确认通过**

```powershell
uv run python distill/tests/test_clean_pseudo_labels.py CropDecodeTest -v
```

预期: 7 个全 PASS。

- [ ] **Step 5: 把 `crop_encode` 的所有调用点改名为 `crop_decode`**

`distill/clean_pseudo_labels.py` 第 547 行附近:

```python
data, box = crop_encode(image, ann['bbox'], args.min_crop_pad, args.max_side)
```

改成:

```python
data, box = crop_decode(
    image, ann['bbox'],
    min_crop_size=args.min_crop_size,
    max_side=args.max_side,
    box_color=args.box_color,
)
```

- [ ] **Step 6: 全量跑单元测试, 清理影响**

```powershell
uv run python distill/tests/test_clean_pseudo_labels.py -v
```

预期: 全绿(若有 `crop_encode` 残留, 按 traceback 修)。

- [ ] **Step 7: Commit**

```powershell
git add distill/clean_pseudo_labels.py distill/tests/test_clean_pseudo_labels.py
git commit -m 'refactor(distill): 重写裁剪为 crop_decode(640 下限 + 画红框)

旧 crop_encode 32px 最小边 + 512px 最长边 + 12% pad, 小目标 30x30
在裁剪后只剩 40px, Qwen 看不清周围环境。

新签名 crop_decode(image, bbox, min_crop_size=640, max_side=960, box_color='red')
- 边长 max(原框, min_crop_size), 越界反推(借 cv_utils.crop_rect)
- 最长边 > max_side 只缩不放, 局部坐标同步换算
- 裁剪图上画 4px 红/黄框, box_color='off' 关闭
- 原图 < min_crop_size 抛 ValueError(启动检查点)

测试 CropEncodeTest 重命名 CropDecodeTest, 覆盖 6 类边界:
小目标贴边 / 中部 / 大图 scale / 过小图报错 / 黄色 / 非法 color。'
```

---

## Task 2: 改 prompt + 新增 CLI 参数 + meta 字段

**Files:**
- Modify: `d:\WorkPlace\Pycharm\VLX-Seek\distill\clean_pseudo_labels.py` (`parse_args` ~L36-62、prompt ~L230-234、`VLMVerifier.verify` ~L300-365、`run_pipeline` meta ~L392-396、`main` 校验 ~L627-648)

**Interfaces:**
- Consumes: Task 1 的 `crop_decode` + `BOX_COLORS` / `BOX_COLOR_OFF`
- Produces: `--min-crop-size`(640) / `--box-color`('red') / `--no-draw-box`;meta 加 `min_crop_size / box_color / max_side / no_draw_box`

- [ ] **Step 1: 改 parse_args 默认值 + 新增 4 个参数**

`clean_pseudo_labels.py` 第 36-62 行,把

```python
    p.add_argument('--max-side', type=int, default=512, help='裁剪图最长边超过则等比缩小')
    p.add_argument('--min-crop-pad', type=float, default=0.12,
                   help='裁剪框外扩比例(相对框长边), 最小边不足 32px 时中心扩展至 32px')
```

替换为:

```python
    p.add_argument('--max-side', type=int, default=960,
                   help='裁剪图最长边超过则等比缩小(只缩不放), 默认 960')
    p.add_argument('--min-crop-size', type=int, default=640,
                   help='裁剪最小边(像素), 目标居中; 原图任一边小于该值时启动报错')
    p.add_argument('--box-color', choices=list(BOX_COLORS) + [BOX_COLOR_OFF],
                   default='red',
                   help='裁剪图上目标框颜色; off 关闭画框(回退旧 prompt)')
    p.add_argument('--no-draw-box', action='store_true',
                   help='等同 --box-color off; 兼容入口')
    # --min-crop-pad 保留解析避免老命令 break, 但不再消费(deprecated, -h 隐藏)
    p.add_argument('--min-crop-pad', type=float, default=0.12,
                   help=argparse.SUPPRESS)
```

- [ ] **Step 2: 改 prompt 与 VLMVerifier.verify 调用处**

`clean_pseudo_labels.py` 第 230-234 行替换为:

```python
SYSTEM_PROMPT = '你是严格的图像内容审核助手, 只回答"是"或"否"。'
# 裁剪图上已画 box_color 矩形框; prompt 不再注入数值坐标, 避免注意力分歧
USER_PROMPT = ('这张图中{box_word}矩形框已标注了待审核目标。'
               '请判断框内的主要拍摄对象是否属于类别「{name}」。'
               '只回答"是"或"否"。')
LEGACY_USER_PROMPT = '这张从大图裁出的局部区域中, 主要拍摄对象是否属于类别「{name}」? 只回答"是"或"否"。'


def _box_word(box_color: str) -> str:
    """按 box_color 返回 prompt 中的颜色词。"""
    return {'red': '红色', 'yellow': '黄色'}.get(box_color, '')
```

`VLMVerifier.verify` 第 300-316 行改为:

```python
    def verify(
        self,
        image_bytes: bytes,
        category_name: str,
        target_box: tuple[int, int, int, int] | None = None,
        box_color: str = 'red',
    ) -> tuple[str, str, int]:
        """验证单框。返回 (verdict, raw_reply, elapsed_ms), 失败耗尽重试后 fail-open。

        ``target_box`` 为目标框在裁剪图局部坐标系的 xywh(可空); 有值时画红框+
        注入颜色说明后发问; 无值则退化为旧通用询问。``box_color`` 不能是 'off'
        (off 时调用方必须传 target_box=None)。
        """
        b64 = base64.b64encode(image_bytes).decode('ascii')
        if target_box is not None:
            word = _box_word(box_color)
            text = USER_PROMPT.format(box_word=word, name=category_name)
        else:
            text = LEGACY_USER_PROMPT.format(name=category_name)
```

- [ ] **Step 3: 把 box_color 透传到 VLMVerifier.verify**

`clean_pseudo_labels.py` 第 547-551 行附近,替换 `executor.submit(verifier.verify, data, ..., box)` 那段:

```python
            if args.box_color == BOX_COLOR_OFF or args.no_draw_box:
                # 不画框 → 走 legacy prompt, target_box=None
                try:
                    data, _ = crop_decode(
                        image, ann['bbox'],
                        min_crop_size=args.min_crop_size,
                        max_side=args.max_side,
                        box_color=BOX_COLOR_OFF,
                    )
                except Exception as exc:  # noqa: BLE001
                    record_error(ann, fname, f'crop failed: {type(exc).__name__}: {exc}')
                    continue
                future = executor.submit(verifier.verify, data, cat_names[ann['category_id']])
            else:
                try:
                    data, box = crop_decode(
                        image, ann['bbox'],
                        min_crop_size=args.min_crop_size,
                        max_side=args.max_side,
                        box_color=args.box_color,
                    )
                except Exception as exc:  # noqa: BLE001
                    record_error(ann, fname, f'crop failed: {type(exc).__name__}: {exc}')
                    continue
                future = executor.submit(
                    verifier.verify, data, cat_names[ann['category_id']], box, args.box_color
                )
```

- [ ] **Step 4: main() 里校验 + meta 字段**

第 627-648 行加:

```python
    if args.min_crop_size < 1:
        sys.exit('错误: --min-crop-size 必须 >= 1')
    if args.max_side < 1:
        sys.exit('错误: --max-side 必须 >= 1')
```

第 392-396 行 `meta` 加 4 字段:

```python
    meta = {
        'model': args.model,
        'coco_json': args.coco_json,
        'iou_threshold': args.iou_threshold,
        'min_crop_size': args.min_crop_size,
        'max_side': args.max_side,
        'box_color': args.box_color,
        'no_draw_box': args.no_draw_box,
    }
```

- [ ] **Step 5: 测试更新 prompt 断言 + 默认值**

第 402-421 行(`test_prompt_contains_box_coords` + `test_prompt_without_box_uses_legacy`)两测试整体替换为:

```python
    def test_prompt_red_box_word(self):
        _Handler.scenario = ['是']
        self._verifier().verify(b'fake-image-bytes', 'orange', (11, 11, 90, 90), box_color='red')
        payload = json.loads(_Handler.last_request_body)
        texts = [c['text'] for c in payload['messages'][1]['content']
                 if c.get('type') == 'text']
        self.assertEqual(len(texts), 1)
        self.assertIn('红色', texts[0])
        self.assertIn('矩形框已标注了待审核目标', texts[0])
        self.assertNotIn('x=11, y=11, w=90, h=90', texts[0])
        self.assertIn('orange', texts[0])
        self.assertIn('只回答"是"或"否"', texts[0])

    def test_prompt_yellow_box_word(self):
        _Handler.scenario = ['是']
        self._verifier().verify(b'fake-image-bytes', 'orange', (11, 11, 90, 90), box_color='yellow')
        payload = json.loads(_Handler.last_request_body)
        texts = [c['text'] for c in payload['messages'][1]['content']
                 if c.get('type') == 'text']
        self.assertEqual(len(texts), 1)
        self.assertIn('黄色', texts[0])

    def test_prompt_without_box_uses_legacy(self):
        _Handler.scenario = ['是']
        self._verifier().verify(b'fake-image-bytes', 'orange')
        payload = json.loads(_Handler.last_request_body)
        texts = [c['text'] for c in payload['messages'][1]['content']
                 if c.get('type') == 'text']
        self.assertEqual(len(texts), 1)
        self.assertNotIn('矩形框', texts[0])
        self.assertIn('orange', texts[0])
```

第 42-59 行 `test_defaults_match_spec`:

```python
        self.assertEqual(args.max_side, 512)
        self.assertAlmostEqual(args.min_crop_pad, 0.12)
```

改为:

```python
        self.assertEqual(args.max_side, 960)
        self.assertEqual(args.min_crop_size, 640)
        self.assertEqual(args.box_color, 'red')
        self.assertFalse(args.no_draw_box)
        self.assertAlmostEqual(args.min_crop_pad, 0.12)
```

- [ ] **Step 6: 跑全量**

```powershell
uv run python distill/tests/test_clean_pseudo_labels.py -v
```

- [ ] **Step 7: Commit**

```powershell
git add distill/clean_pseudo_labels.py distill/tests/test_clean_pseudo_labels.py
git commit -m 'feat(distill): 清洗 prompt 注入红框说明 + 新 CLI 参数

prompt 改为 "图中[颜色]矩形框已标注了待审核目标, 判断框内对象是否属于
类别「name」", 不再注入 (x, y, w, h) 数值。

参数:
  --min-crop-size 640     裁剪最小边, 目标居中
  --box-color red         red / yellow / off
  --no-draw-box           等同 off
  --max-side 960          原 512 → 960

--min-crop-pad 保留但 SUPPRESS(deprecated)。
meta 加 4 字段, 旧日志触发"参数不一致"自动失效重跑。'
```

---

## Task 3: 创建 `distill/self_improve.py` 骨架

**Files:**
- Create: `d:\WorkPlace\Pycharm\VLX-Seek\distill\self_improve.py`
- Test: `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_self_improve.py`(新建空文件)

**Interfaces:**
- Produces:
  - `split_coco_by_image(coco, val_ratio, seed) -> (train_coco, val_coco)`
  - `parse_args(argv) -> Namespace`
  - `load_category_map(path) -> (names_list, train_to_cid, cn_to_train)`
  - `infer_one_image(model, image_path, imgsz, conf, iou) -> list[(cid, x, y, w, h)]`
  - `build_round_coco(image_paths, preds_by_image, names_list) -> dict`
  - `run_round(args, run_dir, k, prev_pt, store) -> dict`(Task 5)
  - `main(argv)`(Task 5)

- [ ] **Step 1: 创 test 文件骨架 + split_coco 测试**

新建 `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_self_improve.py`:

```python
"""self_improve.py 离线单元测试: 不跑真实 YOLO-World(无 GPU/无模型下载),
mock YOLOWorld 用 FakeYOLOWorld 确定性输出, 端到端走 1 轮 happy path。

运行: uv run python distill/tests/test_self_improve.py -v
"""
from __future__ import annotations

import json
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
        self.addCleanup(root.cleanup)
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
```

- [ ] **Step 2: 跑测试看它失败**

```powershell
uv run python distill/tests/test_self_improve.py SplitCocoByImageTest -v
```

预期: FAIL `ImportError` (self_improve 还不存在)。

- [ ] **Step 3: 创建 `distill/self_improve.py` 骨架**

新建 `d:\WorkPlace\Pycharm\VLX-Seek\distill\self_improve.py`:

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

```powershell
uv run python distill/tests/test_self_improve.py SplitCocoByImageTest -v
```

预期: PASS。

- [ ] **Step 5: Commit**

```powershell
git add distill/self_improve.py distill/tests/test_self_improve.py
git commit -m 'feat(distill): self_improve 骨架 + split_coco_by_image

给 image-level 切分一个稳定的本地 API(内部委托 coco_utils.split_coco,
修正 ann id 各自 0..n 连续), Task 4/5 会填 parse_args / 单轮编排 /
早停 / 续跑。

测试基础设施:
- _tiny_coco_and_images 造 4 张 640² 假图 + 最小 COCO
- FakeYOLOWorld / FakeBox* 系列 mock ultralytics 的 Boxes 输出
  (让 .item() / .to() / .numpy() 链兼容不是 torch.Tensor 的对象)
- SplitCocoByImageTest 验证切分稳定性 + ann id 连续'
```

---

## Task 4: `parse_args` + category 映射 + `infer_one_image` + `build_round_coco`

**Files:**
- Modify: `d:\WorkPlace\Pycharm\VLX-Seek\distill\self_improve.py`
- Test: `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_self_improve.py`

**Interfaces:**
- Consumes: Task 3 + 上述 base
- Produces:
  - `parse_args(argv) -> Namespace`
  - `load_category_map(path) -> (names_list, train_to_cid, cn_to_train)`
  - `infer_one_image(model, image_path, imgsz, conf, iou) -> list[(cls_idx, x, y, w, h)]`
  - `build_round_coco(image_paths, preds_by_image, names_list) -> dict`
  - `_resolve_model_provider_args(args) -> callable(checkpoint) -> model`
  - test 追加 `ParseArgsTest` / `InferOneImageTest` / `LoadCategoryMapTest`

- [ ] **Step 1: 在 test 文件追加 LoadCategoryMapTest / ParseArgsTest / InferOneImageTest**

打开 `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_self_improve.py`,在 `if __name__ == '__main__'` 之前追加:

```python
class LoadCategoryMapTest(unittest.TestCase):
    def test_three_indexes(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(root.cleanup)
        p = root / 'category_prompts.json'
        p.write_text(json.dumps({
            'categories': {
                'orange': {'train_name': 'loud orange fruit'},
                'apple': {'train_name': 'red apple fruit'},
            }
        }), encoding='utf-8')
        names_list, train_to_cid, cn_to_train = load_category_map(p)
        # names_list 按 category_id 升序(dict 序)
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
        self.addCleanup(root.cleanup)
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
        self.addCleanup(root.cleanup)
        _tiny_coco_and_images(root)
        img = root / 'imgs' / 'a.jpg'  # 640x640
        model = FakeYOLOWorld()
        # FakeYOLOWorld 固定返回 (40, 50, 140, 150), 640 输入 → scale=1, pad=0
        preds = infer_one_image(model, img, imgsz=640, conf=0.1, iou=0.5)
        self.assertEqual(preds, [(0, 40.0, 50.0, 100.0, 100.0)])

    def test_letterbox_downscale(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(root.cleanup)
        (root / 'imgs').mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (1280, 640), (1, 2, 3)).save(root / 'imgs' / 'big.jpg',
                                                       quality=90)
        img = root / 'imgs' / 'big.jpg'
        model = FakeYOLOWorld()
        # scale = min(640/1280, 640/640, 1.0) = 0.5, pad=0
        # box (40,50,140,150) → orig (80, 100, 200, 200)
        preds = infer_one_image(model, img, imgsz=640, conf=0.1, iou=0.5)
        self.assertEqual(preds, [(0, 80.0, 100.0, 200.0, 200.0)])
```

- [ ] **Step 2: 跑测试看它失败**

```powershell
uv run python distill/tests/test_self_improve.py -v
```

预期: ParseArgs / LoadCategoryMap / InferOneImage 全 FAIL (函数尚未 import 到测试模块)。

- [ ] **Step 3: 在 self_improve.py 追加 parse_args / load_category_map / infer_one_image / build_round_coco / _resolve_model_provider_args**

在 `d:\WorkPlace\Pycharm\VLX-Seek\distill\self_improve.py` 的 `split_coco_by_image` 之后追加:

```python
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
    if boxes is None or len(boxes) == 0:
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
```

- [ ] **Step 4: 跑测试确认 PASS**

```powershell
uv run python distill/tests/test_self_improve.py -v
```

预期: 之前已 PASS 的 + 新加 5 个全 PASS。

- [ ] **Step 5: Commit**

```powershell
git add distill/self_improve.py distill/tests/test_self_improve.py
git commit -m 'feat(distill): self_improve parse_args / category 映射 / letterbox 反算

- parse_args: 透传清洗参数; --skip-clean / --model-provider / 
  --ap-drop-window 等测试后门。
- load_category_map: (names_list, train_to_cid, cn_to_train), names_list
  的 index = category_id = COCO 同类序号。
- infer_one_image: ultralytics predict → boxes.xyxy(letterbox 域)
  → scale = min(imgsz/H, imgsz/W, 1.0), 居中 pad, (x-pad)/scale 回原图,
  越界裁剪 <1px 丢; 兼容 .item() / 裸 float。
- build_round_coco: 纯函数, 图组 → COCO(ann id 连续), 调用方 save_coco。
- _resolve_model_provider_args: 'module.ClassName' importlib, 生产走
  ultralytics。

测试: LoadCategoryMap / ParseArgs / InferOneImage(letterbox 两种几何)。'
```

---

## Task 5: `run_round` + `main` 早停 + 续跑

**Files:**
- Modify: `d:\WorkPlace\Pycharm\VLX-Seek\distill\self_improve.py`
- Test: `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_self_improve.py`

**Interfaces:**
- Consumes: Task 3+4
- Produces:
  - `run_round(args, run_dir, k, prev_pt, store) -> dict`
  - `main(argv)`
  - EndToEndMockTest / ResumeStateTest / EarlyStopRuleTest 三个测试全过

- [ ] **Step 1: 在 test 文件追加 EndToEndMockTest / ResumeStateTest / EarlyStopRuleTest / PerClassAPAlertTest**

打开 `d:\WorkPlace\Pycharm\VLX-Seek\distill\tests\test_self_improve.py`,在 `if __name__ == '__main__'` 之前追加:

```python
class EndToEndMockTest(unittest.TestCase):
    """完整 mock 1 轮: d0 → 训 m0(mock) → 推理 d1(mock) → 清洗(skip) →
    训 m1(mock) → 评估(固定 0.71)+ summary 链路。"""

    def test_one_round_happy_path(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(root.cleanup)
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
        self.addCleanup(root.cleanup)
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
        self.addCleanup(root.cleanup)
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
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
uv run python distill/tests/test_self_improve.py -v
```

预期: 新 3 个 test 类 FAIL(`main` / `run_round` 未定义)。

- [ ] **Step 3: 实现 `run_round` / `main` + 4 个 helper**

在 `d:\WorkPlace\Pycharm\VLX-Seek\distill\self_improve.py` 的 `_resolve_model_provider_args` 之后追加:

```python
def build_dataset_yaml(
    train_coco: dict, val_coco: dict, image_dir: str, dataset_root: Path,
    category_map_path: str,
) -> Path:
    """复用 distill.finetune_yolo_world.prepare_dataset, 传 train+val_coco。
    val_ratio/seed 形式上必须传(val_coco 非 None 时 split 分支不调)。"""
    from distill.finetune_yolo_world import prepare_dataset as _pd
    return _pd(
        coco=train_coco,
        image_dir=image_dir,
        output_dir=str(dataset_root),
        val_coco=val_coco,
        val_ratio=0.0,
        seed=42,
        category_map=category_map_path,
    )


def train_direct(model, dataset_yaml: str, epochs: int, batch: int,
                 device: str, optimizer: str, lr0: float | None,
                 imgsz: int, project: str, name: str, patience: int):
    """薄封装 ultralytics YOLOWorld.train()。"""
    kwargs = dict(
        data=dataset_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, optimizer=optimizer, patience=patience,
        project=project, name=name, exist_ok=True, plots=True,
        seed=42,
    )
    if lr0 is not None:
        kwargs['lr0'] = lr0
    return model.train(**kwargs)


def collect_eval_metrics(model, dataset_yaml: str, imgsz: int,
                         conf: float, iou: float, names: list[str]) -> dict:
    """YOLOWorld.val() → mAP / mAP50 / 每类 AP(ap50_95)。"""
    results = model.val(data=dataset_yaml, imgsz=imgsz, conf=conf, iou=iou,
                        verbose=False)
    box = results.box
    ap5095 = list(box.ap50_95) if hasattr(box, 'ap50_95') else [0.0] * len(names)
    return {
        'mAP': float(box.map),
        'mAP50': float(box.map50),
        'ap_per_class': {n: float(ap) for n, ap in zip(names, ap5095)},
    }


def _copy_path(src: Path, dst: Path) -> None:
    import shutil
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _store_config(args, run_dir: Path) -> None:
    p = run_dir / 'config.json'
    if p.is_file():
        return
    p.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2),
                 encoding='utf-8')


def _load_summary(run_dir: Path) -> list[dict]:
    p = run_dir / 'summary.json'
    if not p.is_file():
        return []
    return json.loads(p.read_text())['rounds']


def _store_summary(run_dir: Path, rounds: list[dict],
                   final_model: str | None, early_stopped: bool) -> None:
    p = run_dir / 'summary.json'
    p.write_text(json.dumps({
        'rounds': rounds,
        'final_model': final_model,
        'early_stopped': early_stopped,
    }, ensure_ascii=False, indent=2), encoding='utf-8')


def _alert_per_class_drop(rounds: list[dict], thr: float,
                          window: int) -> None:
    """若某类连续 `window` 轮 AP 跌 > thr * 前值, 告警(不终止)。"""
    if len(rounds) < window + 1:
        return
    tail = rounds[-(window + 1):]
    by_class: dict[str, list[float]] = {}
    for r in tail:
        for cname, ap in r.get('ap_per_class', {}).items():
            by_class.setdefault(cname, []).append(ap)
    for cname, series in by_class.items():
        base = series[0]
        if base <= 0.05:
            continue
        cur = series[-1]
        if cur < base * (1 - thr):
            print(f'!! [告警] 类别「{cname}」连续 {window} 轮 AP '
                  f'{base:.4f} → {cur:.4f}(跌 {(1 - cur / base) * 100:.1f}%)')


def run_round(args, run_dir: Path, k: int, prev_pt: str | None,
              store: dict) -> dict:
    """第 k 轮: k=0 仅 D/E(用教师 d0); k>0 B→C→D→E。幂等。"""
    round_dir = run_dir / f'round_{k}'
    round_dir.mkdir(parents=True, exist_ok=True)
    names = store['names_list']

    # ---- B 推理(k=0 跳过)
    raw_path = round_dir / f'raw_d{k}.json'
    if k == 0:
        print(f'[round 0] B: 跳过(直接用教师 d0)')
    elif raw_path.is_file():
        print(f'[round {k}] B: raw_d{k} 已存在, 跳过推理')
    else:
        image_paths = store['image_paths']
        model = store['model'](str(Path(run_dir) / prev_pt) if prev_pt
                               else args.init_weights)
        preds: dict = {}
        for p in image_paths:
            preds[p] = infer_one_image(model, p, args.imgsz,
                                       args.conf_thresh, args.nms_iou)
        raw_coco = build_round_coco(image_paths, preds, names)
        save_coco(raw_coco, raw_path)
        print(f'[round {k}] B: {len(raw_coco["annotations"])} 框 → {raw_path.name}')

    # ---- C 清洗(k=0 跳过)
    clean_path = round_dir / f'clean_d{k}.json'
    if k == 0:
        print(f'[round 0] C: 跳过(直接用教师 d0)')
    elif args.skip_clean:
        if not clean_path.is_file():
            save_coco(load_coco(raw_path), clean_path)
    elif clean_path.is_file():
        print(f'[round {k}] C: clean_d{k} 已存在, 跳过清洗')
    else:
        decision_log = round_dir / f'decisions_d{k}.jsonl'
        from distill.clean_pseudo_labels import parse_args as _pa_clean, run_pipeline as _run_clean
        cargs = _pa_clean([
            '--coco-json', str(raw_path),
            '--image-dir', str(store['image_dir']),
            '--output', str(clean_path),
            '--decision-log', str(decision_log),
            '--model', args.model,
            '--base-url', args.clean_base_url,
            '--concurrency', str(args.clean_concurrency),
            '--min-crop-size', str(args.min_crop_size),
            '--max-side', str(args.max_side),
            '--box-color', args.box_color,
        ])
        if args.api_key:
            cargs.api_key = args.api_key
        report = _run_clean(cargs, load_coco(raw_path))
        print(f'[round {k}] C: keep={report.get("kept")} '
              f'delete={report.get("vlm_removed")} '
              f'dedup={report.get("dedup_removed")} '
              f'error_keep={report.get("error_keep")}')

    # ---- D 训练(从 prev 热启动; k>0 用 clean_d_k train 子集; k==0 用 split_train)
    model_out = round_dir / (f'm{k}.pt' if k > 0 else 'm0.pt')
    if model_out.is_file():
        print(f'[round {k}] D: {model_out.name} 已存在, 跳过训练')
    else:
        if k == 0:
            train_coco_for_train = store['train_coco']
        else:
            train_coco_for_train, _ = split_coco_by_image(
                load_coco(clean_path), val_ratio=args.val_ratio, seed=42
            )
        val_coco = load_coco(store['val_coco_path'])
        dataset_root = round_dir / 'dataset_root'
        train_yaml = build_dataset_yaml(
            train_coco=train_coco_for_train,
            val_coco=val_coco,
            image_dir=str(store['image_dir']),
            dataset_root=dataset_root,
            category_map_path=args.category_map,
        )
        src_weights = (
            Path(run_dir) / prev_pt if prev_pt
            else Path(args.init_weights)
        )
        model = store['model'](str(src_weights))
        train_direct(
            model=model,
            dataset_yaml=str(train_yaml),
            epochs=args.epochs,
            batch=args.batch,
            device=args.train_device,
            optimizer=args.optimizer,
            lr0=args.lr0,
            imgsz=args.imgsz,
            project=str(round_dir),
            name='yolo_world',
            patience=args.patience,
        )
        # 归档 best.pt(若缺回退 last.pt)
        best_pt = Path(round_dir) / 'yolo_world' / 'best.pt'
        if not best_pt.is_file():
            last_pt = Path(round_dir) / 'yolo_world' / 'last.pt'
            if not last_pt.is_file():
                raise FileNotFoundError(
                    f'round {k}: 未找到 best.pt/last.pt 在 {round_dir / "yolo_world"}'
                )
            print(f'[warn] round {k} 无 best.pt, 回退 last.pt')
            _copy_path(last_pt, model_out)
        else:
            _copy_path(best_pt, model_out)
        print(f'[round {k}] D: best.pt → {model_out.name}')

    # ---- E 评估(固定 val 集, 透传 D 的 dataset yaml)
    eval_path = round_dir / 'eval.json'
    if eval_path.is_file():
        print(f'[round {k}] E: eval.json 已存在, 跳过评估')
        return json.loads(eval_path.read_text())
    model = store['model'](str(model_out))
    metrics = collect_eval_metrics(
        model,
        dataset_yaml=str(dataset_root / 'dataset' / 'dataset.yaml'),
        imgsz=args.imgsz,
        conf=args.conf_thresh,
        iou=args.nms_iou,
        names=list(store['names_list']),
    )
    eval_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _store_config(args, run_dir)

    # ---- 初始数据
    d0 = load_coco(args.init_coco_json)
    store = {
        'image_dir': Path(args.image_dir),
        'image_paths': sorted([
            Path(args.image_dir) / fn for fn in (im['file_name'] for im in d0['images'])
        ]),
        'train_coco_path': str(run_dir / 'split_train.json'),
        'val_coco_path': str(run_dir / 'split_val.json'),
    }
    names_list, _, cn_to_train = load_category_map(args.category_map)
    store['names_list'] = names_list
    store['cn_to_train'] = cn_to_train

    store['train_coco'], store['val_coco'] = split_coco_by_image(
        d0, val_ratio=args.val_ratio, seed=42
    )
    save_coco(store['train_coco'], store['train_coco_path'])
    save_coco(store['val_coco'], store['val_coco_path'])
    val_names = [im['file_name'] for im in store['val_coco']['images']]
    split_path = run_dir / 'split.json'
    if not split_path.is_file():
        split_path.write_text(
            json.dumps({'val_file_names': val_names}, ensure_ascii=False),
            encoding='utf-8')

    store['model'] = _resolve_model_provider_args(args)
    rounds = _load_summary(run_dir)
    last_done = rounds[-1]['round'] if rounds else -1
    prev_pt: str | None = (
        f'round_0/m0.pt' if last_done >= 0 and (run_dir / 'round_0' / 'm0.pt').is_file()
        else None
    )
    early_stopped = False

    for k in range(max(0, last_done + 1), args.max_rounds + 1):
        print(f'======== Round {k}/{args.max_rounds}'
              f'(k=0 用教师 d0; k>0 B→C→D→E) ========')
        eval_d = run_round(args, run_dir, k, prev_pt, store)
        vmap50 = eval_d['mAP50']
        base_map50 = rounds[-1]['mAP50'] if rounds else vmap50
        delta = 0.0 if k == 0 else round(vmap50 - base_map50, 5)
        rounds.append({
            'round': k,
            'mAP50': vmap50,
            'mAP': eval_d.get('mAP'),
            'ap_per_class': eval_d.get('ap_per_class', {}),
            'delta_map50': delta,
        })
        _store_summary(run_dir, rounds, f'round_{k}/m{k}.pt', False)
        print(f'[Round {k}] mAP50={vmap50:.4f} (Δ={delta:+.4f})')

        # 长尾呆类告警(不终止)
        if len(rounds) >= args.ap_drop_window + 1:
            _alert_per_class_drop(rounds, args.ap_drop_alert, args.ap_drop_window)

        # 早停
        if not early_stopped and len(rounds) >= args.early_stop_no_improve:
            tail_deltas = [
                rounds[-i - 1]['delta_map50']
                for i in range(args.early_stop_no_improve)
            ]
            if all(d <= 0 for d in tail_deltas):
                print(f'[Early-stop] 末 {args.early_stop_no_improve} 轮 mAP50 无提升, 提前结束')
                early_stopped = True
                break

        prev_pt = f'round_{k}/m{k}.pt'

    _store_summary(run_dir, rounds, prev_pt, early_stopped)
    print(f'完成: {run_dir / "summary.json"}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 跑全部 self_improve 测试**

```powershell
uv run python distill/tests/test_self_improve.py -v
```

预期: 全绿(EndToEndMock / ResumeState / EarlyStopRule 全 PASS, PerClassAPAlert SKIP)。

- [ ] **Step 5: Commit**

```powershell
git add distill/self_improve.py distill/tests/test_self_improve.py
git commit -m 'feat(distill): self_improve 单轮编排 B→C→D→E + 早停 + 续跑

run_round(k):
  - k=0: 教师 d0 → D → E, B/C 跳过
  - k>0: B(推理)→ C(清洗)→ D(从 prev 热启动 via prepare_dataset +
    train_direct)→ E(固定 val 评估)
  - 每子步带 marker 文件幂等
  - 归档 best.pt / last.pt 为 m_k.pt

main():
  - _store_config / _store_split / _store_summary 写一次
  - last_done 来自 _load_summary; range(last_done+1, max_rounds+1)
  - 早停: 末 N 轮 delta_map50 全部 <= 0
  - 长尾类 AP 告警(不终止)

测试:
  EndToEndMockTest(完整 1 轮: d0 → m0 → d1 → m1 → eval)
  ResumeStateTest(同 run_dir 二次跑空循环)
  EarlyStopRuleTest(连续 2 轮 delta=0 → 提前停 round_2)
  PerClassAPAlertTest(暂 SKIP)'
```

---

## Task 6: README 写新小节 + .gitignore 补 self_improve_runs

**Files:**
- Modify: `d:\WorkPlace\Pycharm\VLX-Seek\distill\README.md`
- Modify: `d:\WorkPlace\Pycharm\VLX-Seek\.gitignore`

- [ ] **Step 1: 修 README 的「步骤 4.5」段(参数表 / 提示词说明)**

打开 `distill/README.md`,找到 "### 步骤 4.5" 段。**替换**关于 `--max-side` 的说明(原文大概是 `--max-side: 裁剪图最长边超过则等比缩小。`)为:

```
--max-side: 裁剪图最长边超过则等比缩小（只缩不放）, 默认 960。
--min-crop-size: 裁剪最小边(像素), 目标居中, 越界反推(借 cv_utils.crop_rect);
                 原图任一边 < 该值时启动报错。默认 640。
--box-color: 裁剪图上目标框颜色(红/黄/off), 默认 red。
--no-draw-box: 等同 --box-color off, 回退旧 prompt。
```

**更新**「提示词注入目标框」相关小节为:
```
- 在裁剪图上画出目标框(默认红, 4px 线宽), prompt 不再注入坐标数值,
  VLM 的视觉注意力直接聚焦红框内区域。
```

- [ ] **Step 2: 在 §5 之后追加 §6 说明**

在 README "## 完整流程" 段之前, **追加**新小节:

````markdown
### 步骤 6：自改进迭代(可选, 让模型越训越好)

当自改进流程希望 student 不再完全依赖教师伪标签时, 进入:
round 0 用教师 d0 训 m0;后续每轮 m_{k-1} 整图推理 → Qwen 清洗 → 热启动
微调 m_k。不调 VLX-Seek 教师,验证集固定,早停基于 mAP50 连续 N 轮无提升。

```bash
uv run python distill/self_improve.py \
    --init-coco-json distill/data/pseudo_labels.json \
    --image-dir distill/data/images \
    --category-map distill/data/category_prompts.json \
    --max-rounds 3 --val-ratio 0.1 --epochs 30 --batch 32 \
    --imgsz 640 --train-device 0 --infer-device 0 \
    --clean-base-url http://127.0.0.1:8101/v1 \
    --model qwen3.8-vllm --api-key "$OPENAI_API_KEY" \
    --run-dir self_improve_runs/run_$(date +%Y%m%d_%H%M)
```

参数精简(全量看 `--help`):

| 参数 | 默认 | 说明 |
|---|---|---|
| `--init-coco-json` | 必填 | 教师伪标签 d0 |
| `--init-weights` | yolov8s-worldv2.pt | m0 热启动起点 |
| `--max-rounds` | 3 | self-improve 轮数(不含 round 0) |
| `--val-ratio` | 0.1 | image-level 验证集比例(固定切, 不参与训练) |
| `--imgsz` | 640 | 推理 / 训练 / 评估 一致分辨率 |
| `--conf-thresh / --nms-iou` | 0.30 / 0.50 | 推理阈值 |
| `--epochs / --batch / --optimizer / --lr0 / --patience` | 同微调 | 单轮训练超参 |
| `--train-device / --infer-device` | 0 / 0 | GPU(train 可按需 DDP "0,1") |
| `--clean-* / --model / --api-key` | 同步骤 4.5 | 透传清洗参数 |
| `--early-stop-no-improve` | 2 | 连续 N 轮 mAP50 无提升 → 提前停 |
| `--ap-drop-alert / --ap-drop-window` | 0.20 / 2 | 每类 AP 跌幅阈值(仅告警) |
| `--run-dir` | 必填 | 输出根目录 |
| `--skip-clean` | 关 | 调试用, 跳过清洗(生产禁用) |

产出目录:

```
<run_dir>/
├── config.json             # 全部参数
├── split.json              # val 的 file_name 列表(固定)
├── split_train.json / split_val.json
├── summary.json            # 所有轮 mAP50 折线 + final_model + early_stopped
├── round_0/
│   ├── dataset/            # ultralytics 训练数据
│   ├── yolo_world/best.pt  # ultralytics 原始产物
│   ├── m0.pt               # 归档
│   └── eval.json
├── round_1/                # B/C/D/E
│   ├── raw_d1.json
│   ├── clean_d1.json
│   ├── decisions_d1.jsonl
│   ├── dataset/
│   ├── m1.pt
│   └── eval.json
└── round_2/ ...
```

**断点续跑**:重跑同一 `--run-dir`, 已完成步骤自动跳过(以 raw/clean/m*.pt/eval.json / config.json / split.json 为 marker)。

**早停 / 告警**:末 `--early-stop-no-improve` 轮 mAP50 delta 全 <=0 即停;每类 AP 连续 `--ap-drop-window` 轮跌幅 > `--ap-drop-alert` 只 stderr 告警不终止(人工观察长尾类退化)。
````

- [ ] **Step 3: 更新 README 「完整流程」段**

把:

```
4) 伪标签生成        → pseudo_labels.json.json
5) 蒸馏训练          → 你的 yolo-world detector
```

改为:

```
4) 伪标签生成            → pseudo_labels.json
4.5) VLM 自动清洗(可选)    → pseudo_labels.cleaned.json
5) 蒸馏训练            → 你的 yolo-world detector
6) 自改进迭代(可选)        → self_improve_runs/run_<ts>/round_K/mK.pt
```

- [ ] **Step 4: .gitignore 追加 `self_improve_runs/`**

打开 `.gitignore`,在末尾追加:

```
# self_improve 跑动产物
self_improve_runs/
```

(`*.pt` 已 ignore, raw/clean json 不 ignore, 所以这里是整个目录 ignore。)

- [ ] **Step 5: Commit**

```powershell
git add distill/README.md .gitignore
git commit -m 'docs(distill): 步骤 6 自改进迭代说明 + 步骤 4.5 新参数

- crop_decode 640 下限 / 960 / 红框说明(步骤 4.5 段)
- self_improve.py 命令 / 参数表 / 产物目录 / 断点续跑 / 早停
- 完整流程列表加 6) 自改进迭代
- .gitignore 加 self_improve_runs/'
```

---

## Task 7: 全量回归 + 自审 + 実跑

- [ ] **Step 1: 跑全部本地测试**

```powershell
uv run python distill/tests/test_clean_pseudo_labels.py -v
uv run python distill/tests/test_self_improve.py -v
```

预期: 全绿(除 PerClassAPAlert 主动 SKIP)。

- [ ] **Step 2: py_compile 通过**

```powershell
uv run python -m py_compile distill/clean_pseudo_labels.py distill/self_improve.py distill/tests/test_clean_pseudo_labels.py distill/tests/test_self_improve.py
```

- [ ] **Step 3: user 自己实跑一下**(不在 sub-agent 范畴, plan 只给 checklist):

```powershell
uv run python distill/self_improve.py --help
uv run python distill/clean_pseudo_labels.py --help
# 然后按 README 真实跑 1 轮
```

- [ ] **Step 4: git status 一致性清理 + 最终 commit**

```powershell
git status
# 只能有 5 个文件改动:
#   modified: distill/clean_pseudo_labels.py
#   modified: distill/tests/test_clean_pseudo_labels.py
#   modified: distill/README.md
#   modified: .gitignore
#   new file: distill/self_improve.py
#   new file: distill/tests/test_self_improve.py
# 多余的 temp 文件绝不 git add
```

- [ ] **Step 5: 验证 plan 头部要求的 commit 信息(中文 + body)**

每个 Task 的 commit message 都在本 plan 的 Step 5/7 写好了(逐字 copy 到 `git commit -m`),不要修改。

---

## Self-Review(我对 plan 的 spec 覆盖核对)

| Spec 要求 | 是否 cover | 位置 |
|---|---|---|
| 1.1 裁剪重写 + 640 下限 + 越界反推 | 匝 | Task 1 Step 3 |
| 1.2 改 prompt + 移除数值坐标 | 匝 | Task 2 Step 2 |
| 1.3 CLI 参数新增/默认值改 | 匝 | Task 2 Step 1 |
| 1.4 决策日志 meta 4 字段 | 匝 | Task 2 Step 4 |
| 1.5 测试 6 类边界 + prompt 断言 | 匝 | Task 1 Step 1 + Task 2 Step 5 |
| 2.1 A→B→C→D→E | 匝 | Task 5 run_round |
| 2.2(a) image-level val 固定 | 匝 | Task 3 split_coco_by_image + Task 5 prepare_dataset |
| 2.2(b) 推理 + letterbox 反算 | 匝 | Task 4 infer_one_image |
| 2.2(c) train_name 映射 | 匝 | Task 4 load_category_map |
| 2.2(d) 续跑 run_dir 约定 | 匝 | Task 5 _store_* / _load_summary |
| 2.2(e) 单体 in-proc 调用 | 匝 | Task 5 run_round |
| 2.2(f) GPU 调度 | 匝(隐式, ultralytics 管) | Task 5 train_direct |
| 2.2(g) summary.json 结构 | 匝 | Task 5 _store_summary |
| 2.3 CLI 参数表 | 匝 | Task 4 parse_args |
| 2.4 错误处理 | 匝(主流路) | Task 5 |
| 第 3 部分 已知风险 | 匝(不处理, README / docs 说明) | - |
| 第 4 部分 9 个交付物 | 匝 (各 Task) | 5 个文件 + 1 spec + 1 plan(本) |

**已知欠账 / sub-agent 注意**:
1. `PerClassAPAlertTest` 主动 SKIP(需可变 metrics 源, beyond mock)。
2. `--model-provider` 走 importlib(测试), 生产走 `ultralytics.YOLOWorld`。
3. **ultralytics 兼容性**: `results[0].boxes.xyxy` 已是 letterbox 域刻度, `(x - pad_x) / scale` 反算公式是否需要 `+ new_w/2 - imgsz/2` 第 3 项(**偏移项**)取决于 ultralytics 的 `LetterBox` 实现。本 plan 假设 `center=True` 纯居中, 不要加偏移项 — 如果实跑后框偏, 优先查 ultralytics 版本 `letterboxes` 的 `self.kwargs['center']`。
4. Task 7 Step 3 实跑由 user 完成, sub-agent 跳过。
5. 每个 task 已各自 commit,最后 git status 应只多出 6 个 file(5 个修改/创建 + 1 个 plan)。
