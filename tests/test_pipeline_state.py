"""测试 _PipelineState：跨 run_pipeline 调用累积进度、resume 只恢复一次、定期/收尾落盘。

多卡 worker 每进程一个 state：首次创建时从输出文件恢复（--resume），之后跨批次
在内存累积，由 maybe_save()（每 N 张）与 save()（收尾）控制落盘频率，避免
batch=1 逐张拉取时每次全量重读/重写 JSON 的 I/O 放大。
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "distill"))

import generate_pseudo_labels as gpl


def make_args(output: str) -> argparse.Namespace:
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


def _patch(worker: FakeWorker):
    """monkeypatch 外部依赖，返回恢复函数。"""
    real_detect_with_crop = gpl.detect_with_crop
    real_load_proposals = gpl.load_proposals
    gpl.detect_with_crop = lambda image, w, categories, args, cat_id_map: w.detect(image, [], categories)
    gpl.load_proposals = lambda image, detector_checkpoint, max_proposals: []
    gpl._create_worker = lambda args: worker

    def restore():
        gpl.detect_with_crop = real_detect_with_crop
        gpl.load_proposals = real_load_proposals
        gpl._create_worker = lambda args: worker

    return restore


def test_state_accumulates_with_defer_save():
    worker = FakeWorker()
    restore = _patch(worker)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imgs = [tmp / "x1.jpg", tmp / "x2.jpg"]
            for p in imgs:
                _make_image(p)
            out = tmp / "out.json"
            state = gpl._PipelineState(make_args(str(out)))

            # 两次 defer_save 调用，各处理 1 张，跨调用累积
            done1 = gpl.run_pipeline(make_args(str(out)), image_paths=imgs[:1], worker=worker, state=state, defer_save=True)
            done2 = gpl.run_pipeline(make_args(str(out)), image_paths=imgs[1:], worker=worker, state=state, defer_save=True)
            assert done1 == ["x1.jpg"] and done2 == ["x2.jpg"]
            # defer_save：2 张 < interval(10)，不应落盘
            assert not out.is_file(), "defer_save 且不足 interval 时不应落盘"
            assert len(state.coco["images"]) == 2
            assert {img["file_name"] for img in state.coco["images"]} == {"x1.jpg", "x2.jpg"}
            assert state.next_image_id == 2, "id 游标应跨调用衔接"

            # 收尾 save 落盘
            state.save()
            assert out.is_file()
            with open(out, encoding="utf-8") as f:
                coco = json.load(f)
            assert len(coco["images"]) == 2
    finally:
        restore()


def test_state_resume_restores_once():
    worker = FakeWorker()
    restore = _patch(worker)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imgs = [tmp / "a.jpg", tmp / "b.jpg"]
            for p in imgs:
                _make_image(p)
            out = tmp / "out.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"images": [{"id": 0, "file_name": "a.jpg", "width": 64, "height": 64}], "annotations": [], "categories": []}, f)

            # 创建 state 时从输出文件恢复一次
            state = gpl._PipelineState(make_args(str(out)))
            assert state.next_image_id == 1, "恢复后 id 游标应为 1"
            assert state.done_names == {"a.jpg"}

            # 后续调用复用 state，不再重读磁盘；处理新图 b.jpg
            done = gpl.run_pipeline(make_args(str(out)), image_paths=[imgs[1]], worker=worker, state=state, defer_save=True)
            assert done == ["b.jpg"]
            state.save()
            with open(out, encoding="utf-8") as f:
                coco = json.load(f)
            assert {img["file_name"] for img in coco["images"]} == {"a.jpg", "b.jpg"}
            ids = sorted(img["id"] for img in coco["images"])
            assert ids == [0, 1], "新图 id 应从恢复后的游标衔接，不冲突"
    finally:
        restore()


def test_state_maybe_save_interval():
    worker = FakeWorker()
    restore = _patch(worker)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            state = gpl._PipelineState(make_args(str(out)))
            # 累积 9 张：不足 interval(10)，不落盘
            for i in range(9):
                state.coco["images"].append({"id": i, "file_name": f"f{i}.jpg", "width": 64, "height": 64})
            state.maybe_save(10)
            assert not out.is_file(), "9 < 10 不应落盘"
            # 第 10 张触发落盘
            state.coco["images"].append({"id": 9, "file_name": "f9.jpg", "width": 64, "height": 64})
            state.maybe_save(10)
            assert out.is_file(), "累计满 10 张应落盘"
            with open(out, encoding="utf-8") as f:
                assert len(json.load(f)["images"]) == 10
            # 落盘后 saved_count 更新：再累积 1 张不落盘
            state.coco["images"].append({"id": 10, "file_name": "f10.jpg", "width": 64, "height": 64})
            state.maybe_save(10)
            with open(out, encoding="utf-8") as f:
                assert len(json.load(f)["images"]) == 10, "saved_count 更新后不应重复落盘"
    finally:
        restore()


if __name__ == "__main__":
    test_state_accumulates_with_defer_save()
    test_state_resume_restores_once()
    test_state_maybe_save_interval()
    print("test_pipeline_state OK")
