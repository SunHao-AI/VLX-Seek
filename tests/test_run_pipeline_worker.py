"""测试 run_pipeline：注入 worker 复用、_create_worker 选择后端、返回成功列表、resume 跳过。"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "distill"))

import generate_pseudo_labels as gpl

# 关闭裁剪推理，避免加载 WeDetect；关闭类别分批，走 worker.detect 分支
def make_args(output: str, images: list[Path]) -> argparse.Namespace:
    return argparse.Namespace(
        categories="person; car",
        output=output,
        prompt_map=str(ROOT / "distill" / "data" / "category_prompts.json"),
        model_path="fake-model",
        detector_checkpoint="fake-ckpt.pth",
        device="cpu",
        backend="hf",
        lang="en",
        max_new_tokens=128,
        temperature=0.0,
        min_area=0.0,
        resume=True,
        start_index=0,
        end_index=None,
        crop_inference=False,
        slice_width=1000,
        slice_height=1000,
        overlap_width_ratio=0.1,
        overlap_height_ratio=0.1,
        prompt_batch_size=0,
        max_proposals=100,
        letterbox_size=0,
        log_timing=False,
    )


class FakeWorker:
    def __init__(self):
        self.log_timing = False
        self.calls = 0

    def detect(self, image, boxes, categories, **kwargs):
        self.calls += 1
        return {"result_bbox_list": [{"label": "person", "xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40}]}


def _make_image(path: Path) -> None:
    Image.new("RGB", (64, 64)).save(path)


def test_run_pipeline_reuses_injected_worker_and_returns_success():
    real_detect_with_crop = gpl.detect_with_crop
    real_create_worker = gpl._create_worker
    real_load_proposals = gpl.load_proposals
    try:
        gpl.detect_with_crop = lambda image, worker, categories, args, cat_id_map: worker.detect(image, [], categories)
        gpl._create_worker = lambda args: (_ for _ in ()).throw(AssertionError("不应自建 worker"))
        # crop_inference=False 时 run_pipeline 会调 load_proposals 构建真实 WeDetect，必须 stub 掉
        gpl.load_proposals = lambda image, detector_checkpoint, max_proposals: []

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imgs = [tmp / "x1.jpg", tmp / "x2.jpg"]
            for p in imgs:
                _make_image(p)
            out = tmp / "out.json"
            worker = FakeWorker()

            done = gpl.run_pipeline(make_args(str(out), imgs), image_paths=imgs, worker=worker)

            assert done == ["x1.jpg", "x2.jpg"], done
            assert worker.calls == 2, "注入的 worker 应被复用"
            with open(out, encoding="utf-8") as f:
                coco = json.load(f)
            assert len(coco["images"]) == 2 and len(coco["annotations"]) == 2
    finally:
        gpl.detect_with_crop = real_detect_with_crop
        gpl._create_worker = real_create_worker
        gpl.load_proposals = real_load_proposals


def test_run_pipeline_creates_worker_when_none_and_resume_skips():
    real_detect_with_crop = gpl.detect_with_crop
    real_create_worker = gpl._create_worker
    real_load_proposals = gpl.load_proposals
    try:
        gpl.detect_with_crop = lambda image, worker, categories, args, cat_id_map: worker.detect(image, [], categories)
        gpl._create_worker = lambda args: FakeWorker()
        gpl.load_proposals = lambda image, detector_checkpoint, max_proposals: []

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imgs = [tmp / "a.jpg", tmp / "b.jpg"]
            for p in imgs:
                _make_image(p)
            out = tmp / "out.json"
            # 预置已有输出，模拟断点续跑：a.jpg 已处理
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"images": [{"id": 0, "file_name": "a.jpg", "width": 64, "height": 64}], "annotations": [], "categories": []}, f)

            done = gpl.run_pipeline(make_args(str(out), imgs), image_paths=imgs)

            assert done == ["b.jpg"], "resume 应跳过 a.jpg"
            with open(out, encoding="utf-8") as f:
                coco = json.load(f)
            assert {img["file_name"] for img in coco["images"]} == {"a.jpg", "b.jpg"}
    finally:
        gpl.detect_with_crop = real_detect_with_crop
        gpl._create_worker = real_create_worker
        gpl.load_proposals = real_load_proposals


if __name__ == "__main__":
    test_run_pipeline_reuses_injected_worker_and_returns_success()
    test_run_pipeline_creates_worker_when_none_and_resume_skips()
    print("test_run_pipeline_worker OK")
