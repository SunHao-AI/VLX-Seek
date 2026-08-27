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

from clean_pseudo_labels import (  # noqa: E402
    DecisionLog,
    ServiceUnreachable,
    VLMVerifier,
    crop_decode,
    dedup_annotations,
    iou_xywh,
    load_previous_decisions,
    main,
    parse_args,
    run_pipeline,
    validate_refs,
    write_output,
)
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
        self.assertEqual(args.max_side, 960)
        self.assertEqual(args.min_crop_size, 640)
        self.assertEqual(args.box_color, "red")
        self.assertFalse(args.no_draw_box)
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

    def test_torn_utf8_tail_skipped(self):
        # 中断留下的撕裂多字节 UTF-8 尾行不应让续跑崩溃
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = self._rec(5)
        with open(self.path, "wb") as f:
            f.write((json.dumps({"_meta": self.META}, ensure_ascii=False) + "\n")
                    .encode("utf-8"))
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            f.write(b"\xe4\xb8")  # 「中」字的前两个字节，故意撕裂
        index, ok = load_previous_decisions(self.path, self.META)
        self.assertTrue(ok)
        self.assertEqual(index, {("a.jpg", 5): rec})  # 前两行正常解析
        # 可续跑：续写后新记录正常追加且坏尾行被换行隔离
        log = DecisionLog(self.path, self.META)
        log.append(self._rec(6))
        log.close()
        index2, ok2 = load_previous_decisions(self.path, self.META)
        self.assertTrue(ok2)
        self.assertIn(("a.jpg", 6), index2)


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
                # 同图同类高 IoU 对：交集 80*80，IoU(框0, 框1) = 6400/(10000+6400-6400) = 0.64 > 0.55
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 100, 100], "area": 10000},
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [20, 20, 80, 80], "area": 6400},
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


class _Handler(BaseHTTPRequestHandler):
    """按 scenario 序列依次应答的 mock vLLM；元素为 str（回复内容）或 int（HTTP 状态码）。"""

    scenario: list = ["是"]
    calls: int = 0
    last_request_body: str = ""  # 最近一次请求体（用于断言提示词内容）

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        _Handler.last_request_body = self.rfile.read(length).decode("utf-8")
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
        _Handler.last_request_body = ""


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
        # "abc" 不在 classify_reply 任一分支, 触发 ValueError → 退避重试直到耗尽
        _Handler.scenario = ["abc"]
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

    def test_prompt_red_box_word(self):
        _Handler.scenario = ["是"]
        self._verifier().verify(b"fake-image-bytes", "orange", (11, 11, 90, 90), box_color="red")
        payload = json.loads(_Handler.last_request_body)
        texts = [c["text"] for c in payload["messages"][1]["content"]
                 if c.get("type") == "text"]
        self.assertEqual(len(texts), 1)
        self.assertIn("红色", texts[0])
        self.assertIn("矩形框已标注了待审核目标", texts[0])
        self.assertNotIn("x=11, y=11, w=90, h=90", texts[0])
        self.assertIn("orange", texts[0])
        self.assertIn('只回答"是"或"否"', texts[0])

    def test_prompt_yellow_box_word(self):
        _Handler.scenario = ["是"]
        self._verifier().verify(b"fake-image-bytes", "orange", (11, 11, 90, 90), box_color="yellow")
        payload = json.loads(_Handler.last_request_body)
        texts = [c["text"] for c in payload["messages"][1]["content"]
                 if c.get("type") == "text"]
        self.assertEqual(len(texts), 1)
        self.assertIn("黄色", texts[0])

    def test_prompt_without_box_uses_legacy(self):
        _Handler.scenario = ["是"]
        self._verifier().verify(b"fake-image-bytes", "orange")
        payload = json.loads(_Handler.last_request_body)
        texts = [c["text"] for c in payload["messages"][1]["content"]
                 if c.get("type") == "text"]
        self.assertEqual(len(texts), 1)
        self.assertNotIn("矩形框", texts[0])
        self.assertIn("orange", texts[0])

    def test_first_request_unreachable_fast_exit(self):
        # 端口 9（discard）通常无监听 → 连接拒绝
        v = VLMVerifier("http://127.0.0.1:9/v1", "m", max_retries=0, timeout=3)
        with self.assertRaises(ServiceUnreachable):
            v.verify(b"fake-image-bytes", "orange")


def _write_fixture(img_dir: Path) -> Path:
    """生成两张纯色图 + 一份含高 IoU 同类框对的 COCO，返回 json 路径。

    IoU(ann0, ann1) = 90*90/(10000+10000-8100) ≈ 0.68 > 0.55 → ann1 判 dedup。
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (700, 640), (200, 60, 60)).save(img_dir / "img_a.jpg", quality=90)
    Image.new("RGB", (1024, 731), (60, 120, 200)).save(img_dir / "img_b.jpg", quality=90)
    coco = {
        "images": [
            {"id": 0, "file_name": "img_a.jpg", "width": 700, "height": 640},
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

    def test_no_dedup_rerun_reverifies_dedup_boxes(self):
        # 第一次带去重运行：ann1 判 dedup 并写入日志
        _Handler.scenario = ["是"]
        report1 = run_pipeline(self._args(), load_coco(self.in_json))
        self.assertEqual(report1["dedup_removed"], 1)
        calls_after_run1 = _Handler.calls
        lines = self.log_json.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(json.loads(ln).get("verdict") == "dedup" for ln in lines))
        # 第二次以 --no-dedup 等效参数重跑同一日志与输出：
        # 曾被判 dedup 的 ann1 不算已完成决策，应重新送验而非裸 KeyError 崩溃
        args2 = parse_args([
            "--coco-json", str(self.in_json),
            "--image-dir", str(self.img_dir),
            "--output", str(self.out_json),
            "--decision-log", str(self.log_json),
            "--base-url", self.base_url,
            "--model", "mock-model",
            "--concurrency", "4",
            "--no-dedup",
        ])
        report2 = run_pipeline(args2, load_coco(self.in_json))
        self.assertEqual(_Handler.calls, calls_after_run1 + 1)  # 只有 dedup 框被送验
        self.assertEqual(report2["replayed_from_log"], 4)       # 其余框正常回放
        self.assertEqual(report2["dedup_removed"], 0)           # no-dedup 无本地去重
        self.assertEqual(report2["kept"], 5)                    # ann1 重验回"是"保留
        out = load_coco(self.out_json)
        self.assertEqual([a["id"] for a in out["annotations"]], [0, 1, 2, 3, 4])


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


if __name__ == "__main__":
    unittest.main()
