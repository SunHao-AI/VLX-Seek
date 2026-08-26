# VLM 伪标签清洗（clean_pseudo_labels.py）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可选的步骤4.5 脚本 `distill/clean_pseudo_labels.py`：对伪标签 COCO 先做本地 IoU-NMS 去重，再借助 OpenAI 兼容多模态模型（vLLM 部署）逐框裁剪验证，删除误检框，输出更干净的 COCO。

**Architecture:** 单脚本三阶段流水线（阶段1 本地去重 → 阶段2 线程池并发逐框验证 → 阶段3 写出与统计报告）。JSONL 决策日志即时落盘，`(file_name, ann_id)` 索引支持断点续跑；生产者逐图裁剪 + 有界 future 队列控制内存峰值；两级 tqdm 进度条走 stderr，ETA 反映剩余真实调用量。

**Tech Stack:** Python 3.10+；既有依赖 `requests` / `Pillow` / `tqdm`；测试用标准库 `unittest` + 内嵌 `http.server` mock 服务（无需真实 vLLM/GPU）。

## Global Constraints

以下约束适用于每一个任务：

- 本仓库**未初始化 git**：所有任务的 Commit 步骤一律省略；不要执行任何 git 命令。
- Windows PowerShell 环境：串联多条 shell 命令用 `;` 分隔，**禁止 `&&`**。
- 仅允许既有依赖：`requests`、`Pillow`、`tqdm`、Python 标准库；**禁止引入** httpx/aiohttp/pytest 等新依赖。
- 所有 tqdm 进度条必须 `file=sys.stderr`（仓库既有惯例，避免污染 stdout）。
- 脚本 import 惯例照 `distill/generate_pseudo_labels.py`：`ROOT = Path(__file__).resolve().parents[1]` 加入 `sys.path` 后 `from distill.coco_utils import ...`（`# noqa: E402`）；`requests` 在使用处延迟导入。
- 注释、docstring、终端文案、文档全部使用中文。
- fail-open 原则：除「首个请求因连接拒绝/域名解析失败快速退出」与「启动参数校验失败」两类情况外，任何单框失败（网络/超时/5xx/回复不可解析/图片缺失）都必须保守保留（`error_keep`），不得中断整体。
- 输出文件永不覆盖输入文件；决策日志首行 `_meta` 与当前参数不一致时告警并丢弃旧日志重来。
- 规格来源：`docs/superpowers/specs/2026-08-26-vlm-clean-pseudo-labels-design.md`（CLI 默认值以规格 §3 表为准）。

## File Structure

- Create: `distill/clean_pseudo_labels.py` — 唯一交付脚本（CLI、去重、裁剪编码、VLM 客户端、决策日志、主流程、报告全部在此文件，遵循仓库"一步一脚本"惯例）。
- Create: `distill/tests/test_clean_pseudo_labels.py` — 离线单元测试（新建 `tests/` 目录，标准库 unittest；mock 服务内嵌，完全离线可跑）。
- Modify: `distill/README.md` — 目录树、环境要求、新增步骤4.5 小节、步骤5 与注意事项交叉引用。

任务间接口契约（后续任务实现者只看得到自己的任务，靠此对齐签名）：

| 符号 | 签名 | 定义于 |
|---|---|---|
| `parse_args` | `(argv: list[str] \| None = None) -> argparse.Namespace` | Task 1 |
| `validate_refs` | `(coco: dict) -> None`（失败 `sys.exit`） | Task 1 |
| `DecisionLog` | `(path: Path, meta: dict)`；`.append(record: dict) -> None`；`.close() -> None` | Task 2 |
| `load_previous_decisions` | `(path: Path, meta: dict) -> tuple[dict[tuple[str, int], dict], bool]` | Task 2 |
| `iou_xywh` | `(a: list[float], b: list[float]) -> float` | Task 3 |
| `dedup_annotations` | `(coco: dict, threshold: float) -> tuple[list[dict], list[dict]]` | Task 3 |
| `crop_encode` | `(image: Image.Image, bbox_xywh: list[float], min_crop_pad: float = 0.12, max_side: int = 512) -> bytes` | Task 4 |
| `ServiceUnreachable` | `class ServiceUnreachable(Exception)` | Task 5 |
| `VLMVerifier` | `(base_url: str, model: str, api_key: str \| None = None, max_retries: int = 3, timeout: int = 120, backoff_base: float = 2.0)`；`.verify(image_bytes: bytes, category_name: str) -> tuple[str, str, int]`；属性 `.calls: int`、`.failures: int` | Task 5 |
| `write_output` | `(coco: dict, kept_anns: list[dict], output_path: Path) -> None` | Task 6 |
| `run_pipeline` | `(args: argparse.Namespace, coco: dict) -> dict`（返回统计报告） | Task 6 |

---

### Task 1: 脚手架、CLI 解析与输入校验

**Files:**
- Create: `distill/clean_pseudo_labels.py`
- Create: `distill/tests/test_clean_pseudo_labels.py`

**Interfaces:**
- Consumes: `distill/coco_utils.load_coco(path) -> dict`
- Produces: `parse_args`、`validate_refs`（见上表）；`main(argv=None) -> None`（本任务先做到"校验 + 打印加载统计"，Task 6 接入 `run_pipeline`）

- [ ] **Step 1: 写失败测试**

创建 `distill/tests/test_clean_pseudo_labels.py`：

```python
"""clean_pseudo_labels.py 离线单元测试：无需 GPU/vLLM，网络交互走内嵌 mock 服务。

运行: python distill/tests/test_clean_pseudo_labels.py -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image

DISTILL_DIR = Path(__file__).resolve().parents[1]
if str(DISTILL_DIR) not in sys.path:
    sys.path.insert(0, str(DISTILL_DIR))

from clean_pseudo_labels import main, parse_args, validate_refs  # noqa: E402
from coco_utils import load_coco  # noqa: E402


class ParseArgsTest(unittest.TestCase):
    def test_defaults_match_spec(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = parse_args([
                "--coco-json", "in.json",
                "--image-dir", "imgs",
                "--model", "qwen3-vl-8b",
            ])
        self.assertEqual(args.base_url, "http://localhost:8000/v1")
        self.assertEqual(args.concurrency, 16)
        self.assertEqual(args.iou_threshold, 0.55)
        self.assertFalse(args.no_dedup)
        self.assertEqual(args.max_side, 512)
        self.assertAlmostEqual(args.min_crop_pad, 0.12)
        self.assertIsNone(args.output)
        self.assertIsNone(args.decision_log)
        self.assertIsNone(args.report)
        self.assertEqual(args.max_retries, 3)
        self.assertEqual(args.timeout, 120)
        self.assertIsNone(args.api_key)

    def test_model_from_env(self):
        with mock.patch.dict(os.environ, {"CLEAN_VLM_MODEL": "env-model"}):
            args = parse_args(["--coco-json", "in.json", "--image-dir", "imgs"])
        self.assertEqual(args.model, "env-model")


class MainValidationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _write_input(self) -> Path:
        p = self.tmp / "in.json"
        p.write_text(
            json.dumps({"images": [], "annotations": [], "categories": []}),
            encoding="utf-8",
        )
        return p

    def test_missing_input_exits(self):
        with self.assertRaises(SystemExit):
            main(["--coco-json", str(self.tmp / "nope.json"), "--image-dir", ".", "--model", "m"])

    def test_output_same_as_input_exits(self):
        p = self._write_input()
        with self.assertRaises(SystemExit):
            main(["--coco-json", str(p), "--image-dir", ".", "--model", "m",
                  "--output", str(p)])

    def test_missing_model_exits(self):
        p = self._write_input()
        with mock.patch.dict(os.environ, {"CLEAN_VLM_MODEL": ""}):
            with self.assertRaises(SystemExit):
                main(["--coco-json", str(p), "--image-dir", "."])

    def test_valid_input_prints_stats(self):
        p = self.tmp / "in.json"
        p.write_text(json.dumps({
            "images": [{"id": 0, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [1, 1, 2, 2]},
            ],
            "categories": [{"id": 0, "name": "x"}],
        }), encoding="utf-8")
        with mock.patch.dict(os.environ, {"CLEAN_VLM_MODEL": "env-model"}):
            main(["--coco-json", str(p), "--image-dir", ".",
                  "--output", str(self.tmp / "out.json")])


class ValidateRefsTest(unittest.TestCase):
    def test_bad_reference_exits(self):
        coco = {
            "images": [{"id": 0, "file_name": "a.jpg", "width": 10, "height": 10}],
            "categories": [{"id": 0, "name": "x"}],
            "annotations": [
                {"id": 0, "image_id": 9, "category_id": 0, "bbox": [0, 0, 1, 1]},
            ],
        }
        with self.assertRaises(SystemExit):
            validate_refs(coco)

    def test_good_reference_passes(self):
        coco = {
            "images": [{"id": 0, "file_name": "a.jpg", "width": 10, "height": 10}],
            "categories": [{"id": 0, "name": "x"}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1]},
            ],
        }
        validate_refs(coco)  # 不抛异常即通过


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clean_pseudo_labels'`

- [ ] **Step 3: 写最小实现**

创建 `distill/clean_pseudo_labels.py`：

```python
"""步骤4.5（可选）：VLM 清洗伪标签——本地 IoU-NMS 去重 + 多模态模型逐框验证。

用法示例:
    python distill/clean_pseudo_labels.py \
        --coco-json distill/data/pseudo_labels.json \
        --image-dir distill/data/images \
        --base-url http://localhost:8000/v1 \
        --model qwen3-vl-8b

流程: 阶段1 同图同类 IoU-NMS 去重（零 API 成本）→ 阶段2 每个标注框裁成局部小图
发给 OpenAI 兼容多模态服务判断「主要拍摄对象是否属于该类别」（「否」删除，
失败保守保留）→ 阶段3 写出清洗后 COCO 与统计报告。
决策日志 JSONL 即时落盘，支持中断后原命令重跑（已判定框从日志回放，不发请求）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill.coco_utils import load_coco  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM 清洗伪标签：本地 IoU-NMS 去重 + 多模态模型逐框裁剪验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coco-json", required=True, help="输入伪标签 COCO JSON")
    p.add_argument("--image-dir", required=True, help="图像目录（COCO file_name 相对此目录）")
    p.add_argument("--output", default=None,
                   help="输出 COCO 路径，默认 <input>.cleaned.json，永不覆盖输入")
    p.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI 兼容服务地址")
    p.add_argument("--model", default=os.environ.get("CLEAN_VLM_MODEL"),
                   help="模型名（vLLM --served-model-name）；默认取环境变量 CLEAN_VLM_MODEL")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="API key（本地 vLLM 通常不需要）")
    p.add_argument("--concurrency", type=int, default=16, help="线程池大小")
    p.add_argument("--iou-threshold", type=float, default=0.55, help="同图同类 NMS 的 IoU 阈值")
    p.add_argument("--no-dedup", action="store_true", help="跳过本地去重阶段")
    p.add_argument("--max-side", type=int, default=512, help="裁剪图最长边超过则等比缩小")
    p.add_argument("--min-crop-pad", type=float, default=0.12,
                   help="裁剪框外扩比例（相对框长边），最小边不足 32px 时中心扩展至 32px")
    p.add_argument("--decision-log", default=None,
                   help="决策日志 JSONL 路径，默认 <output>.decisions.jsonl（断点续跑依据）")
    p.add_argument("--report", default=None, help="统计报告 JSON 输出路径（默认仅终端打印）")
    p.add_argument("--max-retries", type=int, default=3, help="单框请求最大重试次数")
    p.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数")
    return p.parse_args(argv)


def validate_refs(coco: dict) -> None:
    """校验每条标注引用的 image_id/category_id 均存在，缺失则报错退出。"""
    image_ids = {img["id"] for img in coco["images"]}
    category_ids = {c["id"] for c in coco["categories"]}
    bad = [
        a for a in coco["annotations"]
        if a["image_id"] not in image_ids or a["category_id"] not in category_ids
    ]
    if bad:
        sample = ", ".join(str(a["id"]) for a in bad[:10])
        sys.exit(f"错误：{len(bad)} 条标注引用了不存在的 image_id/category_id（示例 ann id: {sample}）")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.coco_json)
    if not input_path.is_file():
        sys.exit(f"错误：输入文件不存在: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".cleaned.json")
    if output_path.resolve() == input_path.resolve():
        sys.exit(f"错误：--output 不能与输入文件相同: {output_path}")
    decision_log_path = (
        Path(args.decision_log) if args.decision_log else Path(f"{output_path}.decisions.jsonl")
    )
    if not args.model:
        sys.exit("错误：未指定 --model，且环境变量 CLEAN_VLM_MODEL 未设置")

    coco = load_coco(input_path)
    validate_refs(coco)
    # Task 6 将在此接入 run_pipeline(args, coco) 与报告写出
    print(f"加载完成：{len(coco['images'])} 图 / {len(coco['annotations'])} 框 / "
          f"{len(coco['categories'])} 类")
    print(f"输出将写入: {output_path}")
    print(f"决策日志: {decision_log_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: PASS — ParseArgsTest 2 项、MainValidationTest 4 项、ValidateRefsTest 2 项全部 OK；另跑 `python -m py_compile distill/clean_pseudo_labels.py` 无错误。

---

### Task 2: 决策日志模块（断点续跑基础）

**Files:**
- Modify: `distill/clean_pseudo_labels.py`（追加两个顶层符号）
- Modify: `distill/tests/test_clean_pseudo_labels.py`（追加测试类与 import）

**Interfaces:**
- Produces:
  - `DecisionLog(path: Path, meta: dict)`：构造时父目录自动创建；空文件写入首行 `{"_meta": meta}`；已有一致旧日志则以追加模式继续（尾部若有上次中断残留的半行，先补 `\n` 隔离）；`.append(record: dict)` 逐条即时落盘（flush）；`.close()` 关闭句柄。
  - `load_previous_decisions(path: Path, meta: dict) -> tuple[dict[tuple[str, int], dict], bool]`：返回 `(索引, 是否可续跑)`。文件不存在返回 `({}, True)`；首行 `_meta` 与当前参数不一致打印告警并返回 `({}, False)`；损坏行（半行 JSON）静默跳过；索引键为 `(file_name, ann_id)`。

- [ ] **Step 1: 写失败测试**

把测试文件顶部 import 区改为：

```python
from clean_pseudo_labels import (  # noqa: E402
    DecisionLog,
    load_previous_decisions,
    main,
    parse_args,
    validate_refs,
)
```

并在 `if __name__ == "__main__":` 之前追加：

```python
class DecisionLogTest(unittest.TestCase):
    META = {"model": "m", "coco_json": "a.json", "iou_threshold": 0.55}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "sub" / "decisions.jsonl"

    def _rec(self, ann_id: int = 3, verdict: str = "keep") -> dict:
        return {
            "file_name": "a.jpg", "ann_id": ann_id, "category_id": 0,
            "category_name": "x", "verdict": verdict, "raw_reply": "是",
            "elapsed_ms": 10,
        }

    def test_roundtrip_and_index(self):
        log = DecisionLog(self.path, self.META)
        rec = self._rec()
        log.append(rec)
        log.close()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0]), {"_meta": self.META})  # 首行元信息
        index, ok = load_previous_decisions(self.path, self.META)
        self.assertTrue(ok)
        self.assertEqual(index[("a.jpg", 3)], rec)

    def test_append_multiple_creates_parent_dir(self):
        log = DecisionLog(self.path, self.META)
        log.append(self._rec(1))
        log.append(self._rec(2, "delete"))
        log.close()
        _, ok = load_previous_decisions(self.path, self.META)
        self.assertTrue(ok)

    def test_meta_mismatch_reports_not_reusable(self):
        DecisionLog(self.path, self.META).close()
        other = {"model": "other", "coco_json": "a.json", "iou_threshold": 0.55}
        index, ok = load_previous_decisions(self.path, other)
        self.assertEqual(index, {})
        self.assertFalse(ok)

    def test_missing_file_ok(self):
        index, ok = load_previous_decisions(Path("Z:/definitely/not/here.jsonl"), self.META)
        self.assertEqual(index, {})
        self.assertTrue(ok)

    def test_corrupt_tail_line_skipped(self):
        # 模拟上次运行中断留下的半行
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"_meta": self.META}, ensure_ascii=False) + "\n"
            + '{"file_name": "a.jpg", "ann_id": 9',  # 故意截断
            encoding="utf-8",
        )
        index, ok = load_previous_decisions(self.path, self.META)
        self.assertTrue(ok)
        self.assertEqual(index, {})  # 只有 meta 行有效
        # 续写后新记录正常追加，且半行被换行隔离
        log = DecisionLog(self.path, self.META)
        log.append(self._rec(3))
        log.close()
        index2, ok2 = load_previous_decisions(self.path, self.META)
        self.assertTrue(ok2)
        self.assertIn(("a.jpg", 3), index2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: FAIL — `ImportError: cannot import name 'DecisionLog' from 'clean_pseudo_labels'`

- [ ] **Step 3: 写最小实现**

在脚本顶部 import 区补充 `import json`（置于 `argparse` 与 `os` 之间保持字母序）。在 `validate_refs` 之后追加：

```python
class DecisionLog:
    """JSONL 决策日志：逐条判定即时落盘，支持断点续跑回放。"""

    def __init__(self, path: Path, meta: dict) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_meta = True
        if self.path.exists() and self.path.stat().st_size > 0:
            needs_meta = False
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":  # 上次中断留下的半行，补换行隔离
                    with open(self.path, "ab") as fb:
                        fb.write(b"\n")
        self._fh = open(self.path, "a", encoding="utf-8")
        if needs_meta:
            self.append({"_meta": meta})

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def load_previous_decisions(path: Path, meta: dict) -> tuple[dict[tuple[str, int], dict], bool]:
    """读取旧决策日志并以 (file_name, ann_id) 建索引。

    返回 (索引, 日志是否可续跑)：文件不存在视为可续跑（空索引）；
    首行 _meta 与当前参数不一致则告警并返回不可续跑（调用方应删除旧日志重来）。
    """
    path = Path(path)
    if not path.is_file():
        return {}, True
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 中断产生的半行，跳过
    if not records or records[0].get("_meta") != meta:
        print(f"[warn] 旧决策日志 {path} 参数不一致或为空，将忽略并重新开始", file=sys.stderr)
        return {}, False
    index: dict[tuple[str, int], dict] = {}
    for rec in records[1:]:
        index[(rec["file_name"], rec["ann_id"])] = rec
    return index, True
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: PASS — 原有 8 项 + DecisionLogTest 5 项全部 OK。

### Task 3: IoU-NMS 本地去重

**Files:**
- Modify: `distill/clean_pseudo_labels.py`
- Modify: `distill/tests/test_clean_pseudo_labels.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `iou_xywh(a: list[float], b: list[float]) -> float`：xywh 格式交并比，不相交返回 0.0。
  - `dedup_annotations(coco: dict, threshold: float) -> tuple[list[dict], list[dict]]`：每张图内按类别分组，组内按面积降序（`area` 缺失时退回 `w*h`），逐一与已保留框算 IoU，`> threshold` 判重删除。返回 `(保留标注列表, dedup 决策记录列表)`；保留列表保持原 COCO 顺序（保证二次运行输出一致）。决策记录结构：`{"file_name", "ann_id", "category_id", "category_name", "verdict": "dedup", "raw_reply": "", "elapsed_ms": 0}`。

- [ ] **Step 1: 写失败测试**

更新测试文件 import 区加入 `dedup_annotations, iou_xywh`（字母序插入），并在 `if __name__ == "__main__":` 之前追加：

```python
class IouTest(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertAlmostEqual(iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_is_zero(self):
        self.assertEqual(iou_xywh([0, 0, 10, 10], [20, 20, 10, 10]), 0.0)

    def test_partial_overlap_value(self):
        # inter = 5*5 = 25, union = 100+100-25 = 175
        self.assertAlmostEqual(iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]), 25 / 175)


class DedupAnnotationsTest(unittest.TestCase):
    @staticmethod
    def _coco() -> dict:
        return {
            "images": [
                {"id": 0, "file_name": "a.jpg", "width": 500, "height": 400},
                {"id": 1, "file_name": "b.jpg", "width": 500, "height": 400},
            ],
            "categories": [{"id": 0, "name": "orange"}, {"id": 1, "name": "apple"}],
            "annotations": [
                # 同图同类高 IoU 对：IoU(框0, 框1) = 80*80/(10000+4900-6400) ≈ 0.68
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 100, 100], "area": 10000},
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [20, 20, 70, 70], "area": 4900},
                # 同图不同类、位置相同 → 不受影响
                {"id": 2, "image_id": 0, "category_id": 1, "bbox": [0, 0, 100, 100], "area": 10000},
                # 不同图同类、位置相同 → 不受影响
                {"id": 3, "image_id": 1, "category_id": 0, "bbox": [0, 0, 100, 100], "area": 10000},
            ],
        }

    def test_keeps_larger_box_drops_duplicate(self):
        coco = self._coco()
        kept, records = dedup_annotations(coco, 0.55)
        self.assertEqual([a["id"] for a in kept], [0, 2, 3])  # 保持原顺序
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["ann_id"], 1)
        self.assertEqual(rec["file_name"], "a.jpg")
        self.assertEqual(rec["verdict"], "dedup")
        self.assertEqual(rec["category_name"], "orange")

    def test_below_threshold_kept(self):
        coco = self._coco()
        kept, records = dedup_annotations(coco, 0.95)  # 阈值高于实际 IoU
        self.assertEqual(len(records), 0)
        self.assertEqual(len(kept), 4)

    def test_area_fallback_when_missing(self):
        coco = self._coco()
        for a in coco["annotations"]:
            del a["area"]  # 触发 w*h 退化路径
        kept, records = dedup_annotations(coco, 0.55)
        self.assertEqual([a["id"] for a in kept], [0, 2, 3])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: FAIL — `ImportError: cannot import name 'dedup_annotations'`

- [ ] **Step 3: 写最小实现**

在 `distill/clean_pseudo_labels.py` 的 `load_previous_decisions` 之后追加：

```python
def iou_xywh(a: list[float], b: list[float]) -> float:
    """计算两个 xywh 框的交并比（IoU），不相交返回 0.0。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedup_annotations(coco: dict, threshold: float) -> tuple[list[dict], list[dict]]:
    """同图同类 IoU-NMS：面积降序保留更大框（滑窗重复对中较大者边界通常更完整）。

    返回 (保留标注列表, dedup 决策记录列表)。保留列表保持原 COCO 顺序，
    保证同一输入多次运行产出一致。
    """
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    file_by_id = {img["id"]: img["file_name"] for img in coco["images"]}

    groups: dict[tuple[int, int], list[dict]] = {}  # (image_id, category_id) -> anns
    for ann in coco["annotations"]:
        groups.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)

    keep_ids: set[int] = set()
    dedup_records: list[dict] = []
    for (image_id, category_id), anns in groups.items():
        ordered = sorted(
            anns,
            key=lambda a: a.get("area") or a["bbox"][2] * a["bbox"][3],
            reverse=True,
        )
        kept_boxes: list[list[float]] = []
        for ann in ordered:
            if any(iou_xywh(ann["bbox"], kb) > threshold for kb in kept_boxes):
                dedup_records.append({
                    "file_name": file_by_id[image_id],
                    "ann_id": ann["id"],
                    "category_id": category_id,
                    "category_name": cat_names[category_id],
                    "verdict": "dedup",
                    "raw_reply": "",
                    "elapsed_ms": 0,
                })
            else:
                kept_boxes.append(list(ann["bbox"]))
                keep_ids.add(ann["id"])

    kept = [a for a in coco["annotations"] if a["id"] in keep_ids]
    return kept, dedup_records
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: PASS — 原有 13 项 + IouTest 3 项 + DedupAnnotationsTest 3 项全部 OK。

---

### Task 4: 裁剪与编码工具

**Files:**
- Modify: `distill/clean_pseudo_labels.py`
- Modify: `distill/tests/test_clean_pseudo_labels.py`

**Interfaces:**
- Produces: `crop_encode(image: Image.Image, bbox_xywh: list[float], min_crop_pad: float = 0.12, max_side: int = 512) -> bytes`
  - 外扩 `min_crop_pad × max(w, h)`；外扩后最小边不足 32px 时以中心扩展至 32px；越界部分钳制到图像边界（钳制后允许小于 32px——图本身更小时无解）。
  - 最长边超过 `max_side` 等比缩小。
  - 返回 JPEG quality=85 字节流。

- [ ] **Step 1: 写失败测试**

更新测试文件 import 区加入 `crop_encode`，并追加：

```python
class CropEncodeTest(unittest.TestCase):
    def setUp(self):
        self.img500 = Image.new("RGB", (500, 399), (120, 40, 40))

    @staticmethod
    def _decode(data: bytes) -> Image.Image:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im

    def test_pad_and_jpeg_output(self):
        # pad = 0.12*90 = 10.8 → 外扩后裁剪边长 ≈112
        data = crop_encode(self.img500, [120, 90, 90, 90])
        self.assertEqual(data[:2], b"\xff\xd8")  # JPEG 魔数
        self.assertEqual(self._decode(data).size, (112, 112))
        self.assertEqual(self._decode(data).format, "JPEG")

    def test_clamp_to_image_bounds(self):
        # 右下越界 → 钳制后尺寸明显小于完整外扩尺寸且为正
        im = self._decode(crop_encode(self.img500, [450, 350, 90, 90]))
        self.assertGreater(min(im.size), 0)
        self.assertLess(max(im.size), 112)

    def test_min_side_32(self):
        # 小框中心扩展至 ≥32px（位置不贴边，钳制不影响）
        im = self._decode(crop_encode(self.img500, [100, 100, 5, 5]))
        self.assertGreaterEqual(min(im.size), 32)

    def test_max_side_downscale(self):
        big = Image.new("RGB", (1024, 731), (60, 60, 200))
        im = self._decode(crop_encode(big, [0, 0, 600, 600], min_crop_pad=0, max_side=512))
        self.assertEqual(max(im.size), 512)

    def test_negative_xy_clamped_to_zero(self):
        # 左上越界（负坐标）同样钳制：cx=cy=20，cw=ch=100 → [max(0,-30), min(500,70)) = 70px
        im = self._decode(crop_encode(self.img500, [-30, -30, 100, 100], min_crop_pad=0))
        self.assertEqual(im.size, (70, 70))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: FAIL — `ImportError: cannot import name 'crop_encode'`

- [ ] **Step 3: 写最小实现**

在脚本顶部 import 区补充 `import base64`、`import io`、`import time`、`from collections import deque`、`from concurrent.futures import ThreadPoolExecutor`、`from PIL import Image`（base64/time/deque/ThreadPoolExecutor 在 Task 5/6 使用，一次补齐；按字母序排列）。在 `dedup_annotations` 之后追加：

```python
def crop_encode(
    image: Image.Image,
    bbox_xywh: list[float],
    min_crop_pad: float = 0.12,
    max_side: int = 512,
) -> bytes:
    """按框裁剪局部小图并编码为 JPEG 字节流。

    - 外扩 ``min_crop_pad × max(w, h)``；最小边不足 32px 时以中心扩展至 32px；
      越界部分钳制到图像边界。
    - 最长边超过 ``max_side`` 时等比缩小。
    """
    x, y, w, h = bbox_xywh
    pad = min_crop_pad * max(w, h)
    cw = max(w + 2 * pad, 32.0)
    ch = max(h + 2 * pad, 32.0)
    cx, cy = x + w / 2, y + h / 2
    img_w, img_h = image.size
    left = max(0, int(round(cx - cw / 2)))
    top = max(0, int(round(cy - ch / 2)))
    right = min(img_w, int(round(cx + cw / 2)))
    bottom = min(img_h, int(round(cy + ch / 2)))
    crop = image.crop((left, top, right, bottom))
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        )
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: PASS — 原有 19 项 + CropEncodeTest 5 项全部 OK。

### Task 5: VLM 验证客户端

**Files:**
- Modify: `distill/clean_pseudo_labels.py`
- Modify: `distill/tests/test_clean_pseudo_labels.py`

**Interfaces:**
- Consumes: Task 4 无直接依赖（`verify` 只吃 JPEG 字节）
- Produces:
  - `class ServiceUnreachable(Exception)`：首个请求因连接拒绝/域名解析失败抛出，由主流程转为快速退出。
  - `VLMVerifier(base_url, model, api_key=None, max_retries=3, timeout=120, backoff_base=2.0)`：
    - `.verify(image_bytes: bytes, category_name: str) -> tuple[str, str, int]`，返回 `(verdict, raw_reply, elapsed_ms)`；`verdict ∈ {"keep", "delete", "error_keep"}`。
    - 请求：`POST {base_url}/chat/completions`，`temperature=0`、`max_tokens=8`，图片走 base64 data URI；回复 strip 后以「是」开头 → keep、以「否」开头 → delete，其余视为失败进入重试。
    - 重试：总尝试次数 = `max_retries + 1`，指数退避 `sleep(backoff_base ** attempt * 0.5)`。
    - 快速退出规则：`requests.exceptions.ConnectionError` 且此前从未收到过任何 HTTP 响应时抛 `ServiceUnreachable`；一旦收到过响应（无论成败），后续连接错误一律走重试 + fail-open。
    - 计数属性：`.calls`（HTTP 尝试总数）、`.failures`(失败尝试数)。

- [ ] **Step 1: 写失败测试**

更新测试文件 import 区加入 `ServiceUnreachable, VLMVerifier`，并在 `if __name__ == "__main__":` 之前追加 mock 服务与测试：

```python
class _Handler(BaseHTTPRequestHandler):
    """按 scenario 序列依次应答的 mock vLLM；元素为 str（回复内容）或 int（HTTP 状态码）。"""

    scenario: list = ["是"]
    calls: int = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        _Handler.calls += 1
        seq = _Handler.scenario
        content = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(content, int):  # 模拟 HTTP 错误
            body = json.dumps({"error": {"message": "mock error"}}).encode("utf-8")
            self.send_response(content)
        else:
            body = json.dumps(
                {"choices": [{"message": {"content": content}}]}
            ).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # 静默访问日志
        pass


class MockVLMBase(unittest.TestCase):
    """启动一次性 mock 服务，供 VLMVerifier 与端到端测试共用。"""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.scenario = ["是"]
        _Handler.calls = 0


class VLMVerifierTest(MockVLMBase):
    def _verifier(self, max_retries: int = 0) -> VLMVerifier:
        return VLMVerifier(self.base_url, "mock-model", max_retries=max_retries,
                           timeout=10, backoff_base=0.001)

    def test_yes_returns_keep(self):
        _Handler.scenario = ["是"]
        verdict, raw, _ = self._verifier().verify(b"fake-image-bytes", "orange")
        self.assertEqual((verdict, raw), ("keep", "是"))

    def test_no_returns_delete(self):
        _Handler.scenario = ["否"]
        verdict, _, _ = self._verifier().verify(b"fake-image-bytes", "orange")
        self.assertEqual(verdict, "delete")

    def test_garbage_exhausts_retries_then_error_keep(self):
        _Handler.scenario = ["也许吧"]  # 不以是/否开头 → 每次都判失败
        v = self._verifier(max_retries=2)
        verdict, _, _ = v.verify(b"fake-image-bytes", "orange")
        self.assertEqual(verdict, "error_keep")
        self.assertEqual(v.calls, 3)  # 首次 + 2 次重试
        self.assertEqual(_Handler.calls, 3)

    def test_flaky_retry_succeeds(self):
        _Handler.scenario = ["嗯", "否"]  # 第一次乱码，第二次正常
        verdict, _, _ = self._verifier(max_retries=2).verify(b"fake-image-bytes", "orange")
        self.assertEqual(verdict, "delete")

    def test_http_500_fails_open(self):
        _Handler.scenario = [500, 500, 500]
        verdict, _, _ = self._verifier(max_retries=2).verify(b"fake-image-bytes", "orange")
        self.assertEqual(verdict, "error_keep")

    def test_first_request_unreachable_fast_exit(self):
        # 端口 9（discard）通常无监听 → 连接拒绝
        v = VLMVerifier("http://127.0.0.1:9/v1", "m", max_retries=0, timeout=3)
        with self.assertRaises(ServiceUnreachable):
            v.verify(b"fake-image-bytes", "orange")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: FAIL — `ImportError: cannot import name 'ServiceUnreachable'`

- [ ] **Step 3: 写最小实现**

在 `crop_encode` 之后追加：

```python
SYSTEM_PROMPT = '你是严格的图像内容审核助手，只回答"是"或"否"。'
USER_PROMPT = '这张从大图裁出的局部区域中，主要拍摄对象是否属于类别「{name}」？只回答"是"或"否"。'


class ServiceUnreachable(Exception):
    """服务不可达（连接拒绝/域名解析失败），仅限首个请求触发快速退出。"""


class VLMVerifier:
    """OpenAI 兼容 chat/completions 逐框验证客户端（temperature=0 强制短回答）。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 120,
        backoff_base: float = 2.0,
    ) -> None:
        import requests  # 延迟导入，与仓库其他脚本一致

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.calls = 0      # HTTP 尝试总数
        self.failures = 0   # 失败尝试数
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.headers = headers
        self.got_any_response = False  # 是否收到过任何 HTTP 响应
        self._post = requests.post
        self._conn_error_cls = requests.exceptions.ConnectionError

    def verify(self, image_bytes: bytes, category_name: str) -> tuple[str, str, int]:
        """验证单框。返回 (verdict, raw_reply, elapsed_ms)，失败耗尽重试后 fail-open。"""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": USER_PROMPT.format(name=category_name)},
                    ],
                },
            ],
        }
        t0 = time.perf_counter()
        last_err = ""
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            try:
                resp = self._post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                self.got_any_response = True  # 收到过响应后不再触发快速退出
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                reply = resp.json()["choices"][0]["message"]["content"].strip()
                if reply.startswith("是"):
                    return "keep", reply, int((time.perf_counter() - t0) * 1000)
                if reply.startswith("否"):
                    return "delete", reply, int((time.perf_counter() - t0) * 1000)
                raise ValueError(f"无法解析回复: {reply!r}")
            except Exception as exc:  # noqa: BLE001 网络/HTTP/解析错误统一重试
                last_err = f"{type(exc).__name__}: {exc}"
                self.failures += 1
                if isinstance(exc, self._conn_error_cls) and not self.got_any_response:
                    raise ServiceUnreachable(
                        f"服务不可达（{last_err}）。请检查 --base-url 是否正确、"
                        f"vLLM 服务是否已启动。"
                    ) from exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base ** attempt * 0.5)
        return "error_keep", last_err, int((time.perf_counter() - t0) * 1000)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: PASS — 原有 24 项 + VLMVerifierTest 6 项全部 OK。

### Task 6: 主流程编排、进度条与端到端验证

**Files:**
- Modify: `distill/clean_pseudo_labels.py`
- Modify: `distill/tests/test_clean_pseudo_labels.py`

**Interfaces:**
- Consumes: Task 1-5 全部符号（`parse_args`/`validate_refs`/`DecisionLog`/`load_previous_decisions`/`dedup_annotations`/`crop_encode`/`ServiceUnreachable`/`VLMVerifier`）
- Produces:
  - `write_output(coco: dict, kept_anns: list[dict], output_path: Path) -> None`：复制 `images`（全量保留，0 标注图作负样本）/`categories`，过滤后 annotations 的 id 从 0 重新连续编号，其余字段原样保留；经 `save_coco` 写盘。
  - `run_pipeline(args: argparse.Namespace, coco: dict) -> dict`：执行三阶段流水线并返回统计报告 dict。
  - `main()` 最终形态：校验 → `run_pipeline` → 打印报告 → `--report` 时写 JSON。

- [ ] **Step 1: 写失败测试**

更新测试文件 import 区加入 `write_output, run_pipeline`，并在 `if __name__ == "__main__":` 之前追加：

```python
def _write_fixture(img_dir: Path) -> Path:
    """生成两张纯色图 + 一份含高 IoU 同类框对的 COCO，返回 json 路径。

    IoU(ann0, ann1) = 90*90/(10000+10000-8100) ≈ 0.68 > 0.55 → ann1 判 dedup。
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (500, 399), (200, 60, 60)).save(img_dir / "img_a.jpg", quality=90)
    Image.new("RGB", (1024, 731), (60, 120, 200)).save(img_dir / "img_b.jpg", quality=90)
    coco = {
        "images": [
            {"id": 0, "file_name": "img_a.jpg", "width": 500, "height": 399},
            {"id": 1, "file_name": "img_b.jpg", "width": 1024, "height": 731},
        ],
        "categories": [{"id": 0, "name": "orange"}, {"id": 1, "name": "apple"}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 0, "bbox": [50, 50, 100, 100], "area": 10000},
            {"id": 1, "image_id": 0, "category_id": 0, "bbox": [60, 60, 100, 100], "area": 10000},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [300, 220, 100, 100], "area": 10000},
            {"id": 3, "image_id": 1, "category_id": 0, "bbox": [400, 300, 120, 120], "area": 14400},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [700, 400, 110, 110], "area": 12100},
        ],
    }
    path = img_dir.parent / "pseudo_labels.json"
    path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return path


class PipelineTest(MockVLMBase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.img_dir = self.tmp / "images"
        self.in_json = _write_fixture(self.img_dir)
        self.out_json = self.tmp / "cleaned.json"
        self.log_json = self.tmp / "decisions.jsonl"

    def _args(self, base_url=None, image_dir=None):
        return parse_args([
            "--coco-json", str(self.in_json),
            "--image-dir", str(image_dir or self.img_dir),
            "--output", str(self.out_json),
            "--decision-log", str(self.log_json),
            "--base-url", base_url or self.base_url,
            "--model", "mock-model",
            "--concurrency", "4",
        ])

    def test_yes_keeps_and_replay_zero_calls(self):
        _Handler.scenario = ["是"]
        report = run_pipeline(self._args(), load_coco(self.in_json))
        self.assertEqual(report["dedup_removed"], 1)  # ann1 与 ann0 重叠
        out = load_coco(self.out_json)
        self.assertEqual([a["id"] for a in out["annotations"]], [0, 1, 2, 3])  # id 连续重排
        self.assertEqual(len(out["images"]), 2)
        calls_after_run1 = _Handler.calls
        self.assertGreater(calls_after_run1, 0)
        # 断点续跑：第二次运行零新增请求，判定全部来自日志回放，输出一致
        report2 = run_pipeline(self._args(), load_coco(self.in_json))
        self.assertEqual(report2["replayed_from_log"], 4)
        self.assertEqual(_Handler.calls, calls_after_run1)
        self.assertEqual(load_coco(self.out_json), out)

    def test_all_delete_keeps_zero_annotation_images(self):
        _Handler.scenario = ["否"]
        report = run_pipeline(self._args(), load_coco(self.in_json))
        self.assertEqual(report["vlm_removed"], 4)
        out = load_coco(self.out_json)
        self.assertEqual(out["annotations"], [])
        self.assertEqual(len(out["images"]), 2)  # 清零图保留为负样本
        self.assertEqual(report["images_emptied"], 2)

    def test_missing_images_fail_open(self):
        empty_dir = self.tmp / "no_images"
        empty_dir.mkdir()
        _Handler.scenario = ["是"]
        report = run_pipeline(
            self._args(image_dir=empty_dir), load_coco(self.in_json)
        )
        self.assertEqual(report["dedup_removed"], 1)   # 去重仍本地生效
        self.assertEqual(report["error_keep"], 4)      # 其余框 fail-open
        self.assertEqual(report["final_annotations"], 4)
        self.assertTrue(report["missing_images"])

    def test_unreachable_base_url_fast_exit(self):
        with self.assertRaises(SystemExit):
            run_pipeline(
                self._args(base_url="http://127.0.0.1:9/v1"), load_coco(self.in_json)
            )


class WriteOutputTest(unittest.TestCase):
    def test_renumber_and_keep_empty_images(self):
        coco = {
            "info": {"description": "demo"},
            "images": [
                {"id": 0, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 1, "file_name": "b.jpg", "width": 10, "height": 10},
            ],
            "categories": [{"id": 0, "name": "x"}],
            "annotations": [
                {"id": 7, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1]},
                {"id": 9, "image_id": 0, "category_id": 0, "bbox": [2, 2, 1, 1]},
            ],
        }
        kept = [coco["annotations"][1]]  # 只留一条
        out_path = Path(tempfile.mkdtemp()) / "o.json"
        write_output(coco, kept, out_path)
        out = load_coco(out_path)
        self.assertEqual(out["annotations"][0]["id"], 0)          # id 从 0 连续编号
        self.assertEqual(out["annotations"][0]["image_id"], 0)
        self.assertEqual(out["annotations"][0]["bbox"], [2, 2, 1, 1])
        self.assertEqual(len(out["images"]), 2)                   # 图像全量保留
        self.assertEqual(out["info"]["description"], "demo")      # 可选字段透传
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: FAIL — `ImportError: cannot import name 'write_output'`

- [ ] **Step 3: 写最小实现**

在 `VLMVerifier` 类之后追加 `write_output` 与 `run_pipeline`：

```python
def write_output(coco: dict, kept_anns: list[dict], output_path: Path) -> None:
    """写出清洗后的 COCO：图像全量保留（0 标注图作负样本），ann id 连续重编号。"""
    out: dict = {
        "images": coco["images"],
        "categories": coco["categories"],
        "annotations": [],
    }
    if "info" in coco:
        out["info"] = coco["info"]
    for new_id, ann in enumerate(kept_anns):
        item = dict(ann)
        item["id"] = new_id
        out["annotations"].append(item)
    save_coco(out, output_path)


def run_pipeline(args: argparse.Namespace, coco: dict) -> dict:
    """三阶段流水线：本地去重 → 并发逐框验证 → 写出与统计报告。"""
    t0 = time.time()
    image_dir = Path(args.image_dir)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    file_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    total_orig = len(coco["annotations"])

    meta = {
        "model": args.model,
        "coco_json": args.coco_json,
        "iou_threshold": args.iou_threshold,
    }
    log_path = Path(args.decision_log)
    prev, reusable = load_previous_decisions(log_path, meta)
    if not reusable and log_path.exists():
        log_path.unlink()
        print("[warn] 已删除参数不一致的旧决策日志，重新开始", file=sys.stderr)
    log = DecisionLog(log_path, meta)

    # ---- 阶段1：本地 IoU-NMS 去重（零 API 成本）----
    if args.no_dedup:
        working = list(coco["annotations"])
        n_dedup = 0
    else:
        working, dedup_records = dedup_annotations(coco, args.iou_threshold)
        n_dedup = len(dedup_records)
        known_keys = set(prev)
        for rec in dedup_records:
            # 续跑时同一重复对可能已记录过，避免日志冗余
            if (rec["file_name"], rec["ann_id"]) not in known_keys:
                log.append(rec)
    print(f"[阶段1] 去重删除 {n_dedup} 框，待验证 {len(working)} 框", file=sys.stderr)

    by_image: dict[int, list[dict]] = {}
    for ann in working:
        by_image.setdefault(ann["image_id"], []).append(ann)

    # 预扫描：需真实调用 VLM 的框数（进度条 total / ETA 依据），并标记缺图
    missing_images: set[int] = set()
    pending_total = 0
    for image_id, anns in by_image.items():
        file_name = file_by_id[image_id]
        if not (image_dir / file_name).is_file():
            missing_images.add(image_id)
            continue
        pending_total += sum(1 for a in anns if (file_name, a["id"]) not in prev)

    verifier = VLMVerifier(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    from tqdm import tqdm

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    inflight: deque = deque()  # 有界 future 队列，控制内存峰值
    max_inflight = max(args.concurrency * 4, 8)
    counters = {"keep": 0, "delete": 0, "error_keep": 0}
    verdicts: dict[int, str] = {}
    n_replay = 0

    outer = tqdm(total=len(by_image), desc="读取图像", unit="图",
                 file=sys.stderr, disable=len(by_image) <= 1)
    inner = tqdm(total=pending_total, desc="VLM 验证", unit="框",
                 file=sys.stderr, disable=pending_total <= 1)

    def drain(item) -> None:
        """收割单个 future：落盘决策、更新计数与内层进度。"""
        future, ann, fname = item
        try:
            verdict, raw, elapsed_ms = future.result()
        except ServiceUnreachable as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(f"错误：{exc}")
        except Exception as exc:  # noqa: BLE001 线程内意外异常同样 fail-open
            verdict, raw, elapsed_ms = "error_keep", f"{type(exc).__name__}: {exc}", 0
        verdicts[ann["id"]] = verdict
        log.append({
            "file_name": fname,
            "ann_id": ann["id"],
            "category_id": ann["category_id"],
            "category_name": cat_names[ann["category_id"]],
            "verdict": verdict,
            "raw_reply": raw,
            "elapsed_ms": elapsed_ms,
        })
        counters[verdict] += 1
        inner.update(1)
        inner.set_postfix_str(f"留{counters['keep']} 删{counters['delete']} 错{counters['error_keep']}")

    # ---- 阶段2：逐图读图裁剪 + 并发验证（流水线）----
    for image_id, anns in by_image.items():
        fname = file_by_id[image_id]
        outer.update(1)

        todo: list[dict] = []
        for ann in anns:
            rec = prev.get((fname, ann["id"]))
            if rec is None:
                todo.append(ann)
            else:  # 断点回放：直接复用判定，不发请求
                verdicts[ann["id"]] = rec["verdict"]
                counters[rec["verdict"]] += 1
                n_replay += 1

        if image_id in missing_images:
            print(f"[warn] 图片缺失，{len(todo)} 框记 error_keep: {fname}", file=sys.stderr)
            for ann in todo:
                verdicts[ann["id"]] = "error_keep"
                log.append({
                    "file_name": fname,
                    "ann_id": ann["id"],
                    "category_id": ann["category_id"],
                    "category_name": cat_names[ann["category_id"]],
                    "verdict": "error_keep",
                    "raw_reply": "image file missing",
                    "elapsed_ms": 0,
                })
                counters["error_keep"] += 1
            continue

        if not todo:
            continue
        image = Image.open(image_dir / fname).convert("RGB")
        for ann in todo:
            data = crop_encode(image, ann["bbox"], args.min_crop_pad, args.max_side)
            future = executor.submit(verifier.verify, data, cat_names[ann["category_id"]])
            inflight.append((future, ann, fname))
            while len(inflight) >= max_inflight:
                drain(inflight.popleft())

    while inflight:
        drain(inflight.popleft())
    executor.shutdown()
    outer.close()
    inner.close()
    log.close()

    # ---- 阶段3：写出与统计报告 ----
    kept_anns = [a for a in working if verdicts.get(a["id"]) != "delete"]
    kept_imgs = {a["image_id"] for a in kept_anns}
    emptied = sum(1 for iid in by_image if iid not in kept_imgs)

    del_per_cat: dict[str, int] = {}
    for a in working:
        if verdicts.get(a["id"]) == "delete":
            name = cat_names[a["category_id"]]
            del_per_cat[name] = del_per_cat.get(name, 0) + 1
    top = sorted(del_per_cat.items(), key=lambda kv: kv[1], reverse=True)[:10]

    elapsed = max(time.time() - t0, 1e-6)
    judged = len(working)
    report = {
        "total_annotations": total_orig,
        "dedup_removed": n_dedup,
        "vlm_removed": counters["delete"],
        "kept": counters["keep"],
        "error_keep": counters["error_keep"],
        "final_annotations": len(kept_anns),
        "replayed_from_log": n_replay,
        "per_category_deleted_top": [
            {"category_name": k, "deleted": v} for k, v in top
        ],
        "elapsed_sec": round(elapsed, 1),
        "throughput_boxes_per_sec": round(judged / elapsed, 2),
        "vlm_calls": verifier.calls,
        "vlm_failures": verifier.failures,
        "vlm_failure_rate": round(verifier.failures / verifier.calls, 4) if verifier.calls else 0.0,
        "images_total": len(coco["images"]),
        "images_emptied": emptied,
        "missing_images": sorted(file_by_id[i] for i in missing_images),
    }

    write_output(coco, kept_anns, Path(args.output))
    return report


def print_report(report: dict) -> None:
    """把统计报告打印为终端友好的中文摘要。"""
    print("\n===== 清洗统计 =====")
    print(f"输入总框数: {report['total_annotations']}")
    print(
        f"去重删除 {report['dedup_removed']} | VLM 删除 {report['vlm_removed']} | "
        f"保留 {report['kept']} | 失败保留 {report['error_keep']} | "
        f"日志回放 {report['replayed_from_log']}"
    )
    print(
        f"输出标注 {report['final_annotations']} 条；图像 {report['images_total']} 张"
        f"（清零保留 {report['images_emptied']} 张作负样本）"
    )
    if report["missing_images"]:
        print(f"缺失图片 {len(report['missing_images'])} 张（已 fail-open）")
    if report["per_category_deleted_top"]:
        print("各类别删除 Top:")
        for item in report["per_category_deleted_top"]:
            print(f"  {item['category_name']}: {item['deleted']}")
    print(
        f"耗时 {report['elapsed_sec']}s | 吞吐 {report['throughput_boxes_per_sec']} 框/s | "
        f"VLM 调用 {report['vlm_calls']} 次（失败率 {report['vlm_failure_rate']}）"
    )
```

最后把 `main()` 中「Task 6 将在此接入」注释及之后的三行 print 替换为：

```python
    coco = load_coco(input_path)
    validate_refs(coco)
    report = run_pipeline(args, coco)
    print_report(report)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"统计报告已写入 {report_path}")
```

同时在顶部 import 区把 `from distill.coco_utils import load_coco` 改为 `from distill.coco_utils import load_coco, save_coco`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: PASS — 原有 30 项 + PipelineTest 4 项 + WriteOutputTest 1 项全部 OK。

### Task 7: 更新 distill/README.md

**目标**：把新脚本接入 README 文档链路（目录树、环境要求、完整使用节、下游衔接、注意事项），全部锚点已在编写本计划时逐行核实。

**Files**:
- Modify: `distill/README.md`（5 处修改）

**Interfaces**: 无代码接口，纯文档。

- [ ] **Step 1: 目录树插入脚本行**

old_str:
```
├── generate_pseudo_labels.py   # 步骤4：VLX-Seek → COCO 伪标签
├── convert_annotations.py      # 标注格式互转：COCO / YOLO / LabelMe（6 个方向）
```

new_str:
```
├── generate_pseudo_labels.py   # 步骤4：VLX-Seek → COCO 伪标签
├── clean_pseudo_labels.py      # 步骤4.5（可选）：IoU-NMS 去重 + VLM 逐框清洗伪标签
├── convert_annotations.py      # 标注格式互转：COCO / YOLO / LabelMe（6 个方向）
```

- [ ] **Step 2: 环境要求插入步骤4.5 条目**

old_str:
```
- **步骤4（伪标签生成）**：需要 GPU + VLX-Seek 权重（`resources/VLX-Seek-1.5-10B`）与 WeDetect 权重（`resources/wedetect_base_uni.pth`，缺失时自动下载）。依赖见项目根 `requirements.txt`。
```

new_str:
```
- **步骤4（伪标签生成）**：需要 GPU + VLX-Seek 权重（`resources/VLX-Seek-1.5-10B`）与 WeDetect 权重（`resources/wedetect_base_uni.pth`，缺失时自动下载）。依赖见项目根 `requirements.txt`。
- **步骤4.5（VLM 清洗伪标签，可选）**：需 `requests` 与 OpenAI 兼容多模态服务（如 vLLM 部署的 Qwen3-VL）；本脚本本身纯 CPU 运行。
```

- [ ] **Step 3: 「## 标注格式转换」前插入完整使用节**

old_str:
```
## 标注格式转换（convert_annotations.py）
```

new_str:
````
## 步骤4.5（可选）：VLM 清洗伪标签

步骤4 产出的伪标签存在两类噪声：滑窗重叠导致的跨块重复框、WeDetect 召回局限导致的误检。`clean_pseudo_labels.py` 借助 OpenAI 兼容的多模态服务（如 vLLM 部署的 Qwen3-VL）对每条标注裁剪出小图并让 VLM 判断"图中是否存在该类目标"，只回"是/否"，据此过滤伪标签。

先用 vLLM 部署服务端（示例）：

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

再运行清洗（本地去重 + VLM 验证一步完成）：

```bash
python distill/clean_pseudo_labels.py \
  --coco-json data/pseudo_labels.json \
  --image-dir data/images \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --decision-log data/pseudo_labels.decisions.jsonl \
  --report data/clean_report.json
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--coco-json` | （必填） | 输入伪标签 COCO JSON |
| `--image-dir` | （必填） | 图像目录（COCO `file_name` 相对此目录） |
| `--output` | `<输入名>.cleaned.json` | 清洗后 COCO 输出路径（永不覆盖输入） |
| `--base-url` | `http://localhost:8000/v1` | OpenAI 兼容服务地址 |
| `--model` | 环境变量 `CLEAN_VLM_MODEL` | 多模态模型名（两者均缺则报错退出） |
| `--api-key` | 环境变量 `OPENAI_API_KEY` | API Key（本地 vLLM 可不设） |
| `--concurrency` | `16` | 并发请求线程数（按服务吞吐调整） |
| `--iou-threshold` | `0.55` | 同图同类别 NMS 的 IoU 阈值 |
| `--no-dedup` | 关 | 跳过本地去重阶段，仅做 VLM 验证 |
| `--max-side` | `512` | 裁剪图最长边超过则等比缩小后再上传 |
| `--min-crop-pad` | `0.12` | 裁剪框四周外扩比例（保留上下文） |
| `--decision-log` | 无 | 决策日志 JSONL 路径（断点续跑必需） |
| `--max-retries` | `3` | 单框请求最大重试次数 |
| `--timeout` | `120` | 单次请求超时秒数 |
| `--report` | 无 | 统计报告 JSON 输出路径（默认仅终端打印） |

行为要点：

1. **三阶段流水线**：阶段1 本地按 (图, 类别) 分组做 IoU-NMS 去重（面积大的框优先保留）；阶段2 逐框裁剪（外扩 12%、最小边 32px、JPEG 质量 85）并发送 VLM 判定；阶段3 写出——标注 id 重排连续、info/categories 原样透传、清零图像保留作负样本。
2. **断点续跑**：每个框的判定实时追加写入决策日志；中断后原命令重跑即自动跳过已完成框（进度条 total 会扣除已判定数量）。若更改了模型/IoU 阈值等关键参数，需先删除旧日志再重跑（脚本检测到参数不一致会告警并不沿用）。
3. **fail-open 原则**：VLM 无法判断/回复乱码/重试耗尽时保守保留该框，只有明确的"否"才删除；仅当首个请求出现连接拒绝或域名解析失败时快速退出，提示检查服务地址。
4. **两级进度条**：外层按图像推进显示整体 ETA，内层按真实 VLM 调用框数推进显示剩余时间。

## 标注格式转换（convert_annotations.py）
````

- [ ] **Step 4: 步骤5 bullet 区补下游衔接提示**

old_str:
```
- 首次运行会下载 YOLO-World 预训练权重与 CLIP 文本编码器权重。

### 多 GPU 训练
```

new_str:
```
- 首次运行会下载 YOLO-World 预训练权重与 CLIP 文本编码器权重。
- 若运行过步骤4.5 清洗，把上面命令中的 `--coco-json` 换成清洗输出（如 `data/pseudo_labels.cleaned.json`）。

### 多 GPU 训练
```

- [ ] **Step 5: 注意事项第5条改为指向新脚本**

old_str:
```
5. **滑窗重叠会产生重复框**：裁剪推理默认 10% 重叠，同一目标在相邻切块各检一次会写入两条近似重复标注，合并时没有跨块去重；如需更干净的伪标签，可在 COCO 上按类别做一次 IoU-NMS 过滤。
```

new_str:
```
5. **滑窗重叠会产生重复框**：裁剪推理默认 10% 重叠，同一目标在相邻切块各检一次会写入两条近似重复标注，合并时没有跨块去重；可运行步骤4.5 的 `clean_pseudo_labels.py`，其第一阶段即按类别做 IoU-NMS 去重，并可用 VLM 顺带过滤误检。
```

- [ ] **Step 6: 人工核对**

通读修改后的 README 步骤4.5 节：表格渲染正常、命令可直接复制执行、五处修改无残留旧文本。

---

### Task 8: 最终验收

**目标**：确认交付物编译通过、测试全绿、可选真实服务冒烟。

**Files**: 无新增修改（只读验证）。

- [ ] **Step 1: 语法编译检查**

Run: `python -m py_compile distill/clean_pseudo_labels.py distill/tests/test_clean_pseudo_labels.py`
Expected: 无任何输出、退出码 0。

- [ ] **Step 2: 全量单元测试**

Run: `python distill/tests/test_clean_pseudo_labels.py -v`
Expected: 全部 PASS（预期 35 项：CLI/校验/日志/去重/裁剪/VLM 客户端 30 项 + PipelineTest 4 项 + WriteOutputTest 1 项）。测试过程中 stderr 出现进度条片段属正常现象（PipelineTest 会触发真实的 tqdm 渲染），不影响结果判定。

- [ ] **Step 3: 可选真实服务冒烟（用户侧 vLLM 可用时）**

用仓库自带 examples 数据试跑（该数据同图同类框不重叠，去重阶段无事可做，重点验证 VLM 链路与进度条）：

```powershell
python distill/clean_pseudo_labels.py `
  --coco-json distill/examples/pseudo_labels.json `
  --image-dir distill/examples/images `
  --base-url http://localhost:8000/v1 `
  --model Qwen/Qwen3-VL-8B-Instruct
```

Expected: 两级进度条走完；终端打印中文统计摘要；生成 `distill/examples/pseudo_labels.cleaned.json` 且标注数 ≤ 输入的 5 条；验证完成后删除该 cleaned 文件以免污染示例目录。

---

## Self-Review 清单

- [x] 每个 Task 均有：目标 / Files / Interfaces / 编号 Steps / 每步期望输出
- [x] 所有代码块完整可落地，无 TODO / 省略号 / "此处略" 占位
- [x] 跨任务接口签名与「接口契约」表逐一核对一致（parse_args、validate_refs、DecisionLog、load_previous_decisions、iou_xywh、dedup_annotations、crop_encode、ServiceUnreachable、VLMVerifier.verify、write_output、run_pipeline、print_report）
- [x] 测试数量口径统一：T1–T5 合计 30 项 + Task 6 新增 5 项 = 35 项，Task 6 Step 4 与 Task 8 Step 2 表述一致
- [x] 关键数值与规格文档一致：IoU 阈值 0.55、外扩 12%、最小边 32px、JPEG q85、最长边 512、concurrency 16、max-retries 3、timeout 120
- [x] Global Constraints 全部满足：零新依赖（requests/Pillow/tqdm 均已有）、tqdm 走 stderr、import 惯例照抄 generate_pseudo_labels.py、全中文注释、输出永不覆盖输入、仓库无 .git 故不含 commit 步骤
- [x] 测试自足性：mock 服务内嵌于测试文件（http.server），端口 0 随机分配，不依赖外部网络与真实 vLLM
- [x] 幂等性：二次运行经决策日志回放后输出与首次完全一致（PipelineTest 已覆盖）

---

## 执行方式

Plan complete and saved to `docs/superpowers/plans/2026-08-26-vlm-clean-pseudo-labels.md`，共 8 个 Task。两种执行方式：

1. **Subagent-Driven（推荐）**：主会话逐 Task 派发子代理实现与验证，上下文隔离、每步可审查。
2. **Inline 执行**：主会话直接按 Task 顺序实现，速度更快但占用主上下文。

请选择执行方式。
