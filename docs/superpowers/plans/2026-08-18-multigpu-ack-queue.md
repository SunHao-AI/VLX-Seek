# 多卡伪标签生成：共享队列 + ack 调度实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写 `distill/generate_pseudo_labels.py` 的多卡调度：共享任务队列 + 消费确认（ack）+ 主进程哨兵退出，消除部分 GPU 提前退出的竞态，并让每个 worker 进程只创建一次模型。

**Architecture:** 主进程全局去重后把所有未处理图像路径放入共享任务队列；每个 spawn worker 进程内只创建一次 VLXSeek worker，阻塞地从任务队列取路径、凑成 32 张一批调用 `run_pipeline`，处理完把该批路径放入完成队列（ack）；主进程轮询完成队列统计消费数，等于分发数后向每个 worker 发哨兵，join 后合并 shard。

**Tech Stack:** Python 3.12，multiprocessing（spawn），现有 vllm/hf 后端 worker。

**设计文档:** `docs/superpowers/specs/2026-08-18-multigpu-ack-queue-design.md`

## Global Constraints

- 只修改 `distill/generate_pseudo_labels.py` 与 `tests/` 目录；不引入新依赖。
- 单卡模式（无 `--gpu-ids`）行为必须零变化。
- 全部队列/进程使用 `ctx = mp.get_context("spawn")` 统一创建。
- ack 语义是"已消费"而非"已成功"：批内失败由 `run_pipeline` 内部 try/except 容错，不重试。
- 中文注释与日志，与现有代码风格一致。

---

### Task 1: 全局去重工具函数 `_collect_done_names`

**Files:**
- Modify: `distill/generate_pseudo_labels.py`（在 `merge_shards` 附近新增函数）
- Test: `tests/test_done_names.py`（新建）

**Interfaces:**
- Consumes: 无（纯函数）。
- Produces: `_collect_done_names(shard_paths: list[str]) -> set[str]` —— 扫描各路径对应 JSON 的 `images[].file_name`，文件缺失/损坏时跳过，返回并集。

- [ ] **Step 1: 写失败测试**

`tests/test_done_names.py`（风格与 `tests/test_worker_cache.py` 一致：自包含 + `__main__`）：

```python
"""测试 _collect_done_names：全局去重扫描已有输出与 shard 文件。"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "distill"))

import generate_pseudo_labels as gpl


def _write_shard(path: Path, names: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": [{"id": i, "file_name": n} for i, n in enumerate(names)], "annotations": []}, f)


def test_collect_done_names_merges_and_tolerates_bad_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good1 = tmp / "out.shard0.json"
        good2 = tmp / "out.shard1.json"
        broken = tmp / "out.shard2.json"
        missing = tmp / "out.shard3.json"
        _write_shard(good1, ["a.jpg", "b.jpg"])
        _write_shard(good2, ["b.jpg", "c.jpg"])
        broken.write_text("{ 不是合法 json", encoding="utf-8")

        done = gpl._collect_done_names([str(good1), str(good2), str(broken), str(missing)])
        assert done == {"a.jpg", "b.jpg", "c.jpg"}, done


if __name__ == "__main__":
    test_collect_done_names_merges_and_tolerates_bad_files()
    print("test_done_names OK")
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests/test_done_names.py`
Expected: FAIL（`AttributeError: module ... has no attribute '_collect_done_names'`）

- [ ] **Step 3: 实现**

在 `distill/generate_pseudo_labels.py` 的 `merge_shards`（约 L518）之前新增：

```python
def _collect_done_names(shard_paths: list[str]) -> set[str]:
    """扫描输出文件与各 shard 文件，收集已处理图像的 file_name（全局去重）。

    文件缺失或损坏时跳过，保证 --resume 在共享队列分发下不重复处理已完成图像。
    """
    done: set[str] = set()
    for path in shard_paths:
        if not Path(path).is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        done.update(img["file_name"] for img in data.get("images", []))
    return done
```

- [ ] **Step 4: 运行确认通过**

Run: `python tests/test_done_names.py`
Expected: PASS（`test_done_names OK`）

- [ ] **Step 5: 提交**

```bash
git add tests/test_done_names.py distill/generate_pseudo_labels.py
git commit -m "feat: 新增多卡全局去重工具 _collect_done_names"
```

---

### Task 2: worker 创建下沉 `_create_worker`，`run_pipeline` 支持注入 worker 并返回成功列表

**Files:**
- Modify: `distill/generate_pseudo_labels.py`（`run_pipeline` 约 L339-464）
- Test: `tests/test_run_pipeline_worker.py`（新建）

**Interfaces:**
- Consumes: `run_pipeline` 现有实现；`args.backend`（"vllm" / "hf"）。
- Produces:
  - `_create_worker(args: argparse.Namespace) -> object` —— 按 `args.backend` 创建 vllm 或 hf 后端 worker，返回 worker 实例。
  - `run_pipeline(args, image_paths=None, worker=None) -> list[str]` —— 返回本次成功处理的 `file_name` 列表；`worker` 传入时复用（多卡），否则自建（单卡，行为不变）。

- [ ] **Step 1: 写失败测试**

`tests/test_run_pipeline_worker.py`：

```python
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
    try:
        gpl.detect_with_crop = lambda image, worker, categories, args, cat_id_map: worker.detect(image, [], categories)
        gpl._create_worker = lambda args: (_ for _ in ()).throw(AssertionError("不应自建 worker"))

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


def test_run_pipeline_creates_worker_when_none_and_resume_skips():
    real_detect_with_crop = gpl.detect_with_crop
    real_create_worker = gpl._create_worker
    try:
        gpl.detect_with_crop = lambda image, worker, categories, args, cat_id_map: worker.detect(image, [], categories)
        gpl._create_worker = lambda args: FakeWorker()

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


if __name__ == "__main__":
    test_run_pipeline_reuses_injected_worker_and_returns_success()
    test_run_pipeline_creates_worker_when_none_and_resume_skips()
    print("test_run_pipeline_worker OK")
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests/test_run_pipeline_worker.py`
Expected: FAIL（`_create_worker` 不存在；且 `run_pipeline` 签名尚不支持 `worker` 参数、无返回值）

- [ ] **Step 3: 实现**

在 `distill/generate_pseudo_labels.py` 中：

1. 新增 `_create_worker`（放在 `run_pipeline` 之前，约 L339）：

```python
def _create_worker(args: argparse.Namespace):
    """按后端创建 VLX-Seek worker。多卡模式下每个进程只调用一次。"""
    if args.backend == "vllm":
        from vllm_serve.vlx_seek_vllm_worker import VLXSeekVLLMWorker

        return VLXSeekVLLMWorker(
            args.model_path,
            device=args.device,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            max_model_len=8192,
            letterbox_size=args.letterbox_size,
        )
    from vlx_seek_worker import VLXSeekWorker

    return VLXSeekWorker(args.model_path, device=args.device, letterbox_size=args.letterbox_size)
```

2. 改 `run_pipeline` 签名与 worker 创建段（原 L375-391），并收集成功列表：

```python
def run_pipeline(
    args: argparse.Namespace,
    image_paths: list[Path] | None = None,
    worker=None,
) -> list[str]:
    """单进程处理一份图像列表，写入 args.output，返回成功处理的 file_name 列表。

    多卡模式下由子进程调用：image_paths 为该进程本批图像，worker 已由调用方创建并传入。
    单卡模式（worker=None）时按 args.backend 自建 worker，行为与之前完全一致。
    """
```

worker 创建段替换为：

```python
    if worker is None:
        worker = _create_worker(args)
    worker.log_timing = args.log_timing
```

处理循环（原 L411-460）中，成功路径收集 + 返回：

```python
    succeeded: list[str] = []
    pbar = tqdm(image_paths, desc="生成伪标签", unit="图", file=sys.stderr)
    for i, img_path in enumerate(pbar):
        if img_path.name in done_names:
            continue
        ...
        except Exception as exc:  # 单张失败不中断整体
            pbar.write(f"[{i + 1}/{total}] 失败 {img_path.name}: {exc}")
            if use_batch:
                worker.clear_image_cache()
            continue
        succeeded.append(img_path.name)
        ...
    save_coco(coco, args.output)
    print(f"伪标签已保存到 {args.output}")
    print(f"图像数: {len(coco['images'])}，标注数: {len(coco['annotations'])}")
    return succeeded
```

（保留原有 docstring 中关于裁剪推理/类别还原的说明段落，仅同步修改函数签名注释。）

- [ ] **Step 4: 运行确认通过**

Run: `python tests/test_run_pipeline_worker.py`
Expected: PASS（`test_run_pipeline_worker OK`）

- [ ] **Step 5: 提交**

```bash
git add tests/test_run_pipeline_worker.py distill/generate_pseudo_labels.py
git commit -m "feat: worker 创建下沉 _create_worker，run_pipeline 支持注入并返回成功列表"
```

---

### Task 3: 重写 `_worker_shard`：阻塞取任务 + 哨兵 + 按批 ack

**Files:**
- Modify: `distill/generate_pseudo_labels.py`（`_worker_shard` 约 L494-516，`_split_shards` 约 L467-472）
- Test: 无独立单测（依赖 GPU 模型），验证靠 Task 4 冒烟 + `py_compile`。

**Interfaces:**
- Consumes: `_create_worker(args)`（Task 2）、`run_pipeline(args, image_paths, worker)`（Task 2）、`run_multigpu` 传入的 `task_queue`/`done_queue`。
- Produces: `_worker_shard(args, gpu_id, task_queue, done_queue, output_path) -> None` —— 新签名；模块级哨兵 `_SENTINEL = None`。

- [ ] **Step 1: 实现**

新增模块级哨兵常量（放在 `IMAGE_EXTS` 附近）：

```python
# 任务队列哨兵：worker 取到后处理完当前批即退出（主进程确认全部消费后统一发送）
_SENTINEL = None
```

删除 `_split_shards`（L467-472，不再使用）。

重写 `_worker_shard`：

```python
def _worker_shard(
    args: argparse.Namespace,
    gpu_id: int,
    task_queue: mp.Queue,
    done_queue: mp.Queue,
    output_path: str,
) -> None:
    """子进程入口：CUDA_VISIBLE_DEVICES 隔离到指定 GPU。

    进程内只创建一次模型；从共享任务队列阻塞拉取图像，凑满一批处理完后向
    完成队列发 ack（该批全部路径，表示"已消费"）。取到哨兵即退出。
    """
    _setup_logging(output_path + ".log")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shard_args = argparse.Namespace(**vars(args))
    shard_args.device = "cuda:0"  # 隔离后 cuda:0 即物理 GPU gpu_id
    shard_args.output = output_path

    worker = _create_worker(shard_args)
    worker.log_timing = args.log_timing

    batch_size = 32
    batch: list[Path] = []
    while True:
        item = task_queue.get()  # 阻塞：不会因"队列暂时为空"提前退出
        if item is _SENTINEL:
            break
        batch.append(item)
        if len(batch) >= batch_size:
            run_pipeline(shard_args, image_paths=batch, worker=worker)
            done_queue.put([str(p) for p in batch])  # ack：本批已消费
            batch = []
    if batch:  # 收到哨兵时兜底处理剩余不足一批的图像
        run_pipeline(shard_args, image_paths=batch, worker=worker)
        done_queue.put([str(p) for p in batch])
```

- [ ] **Step 2: 语法检查**

Run: `python -m py_compile distill/generate_pseudo_labels.py`
Expected: 退出码 0，无输出。

- [ ] **Step 3: 提交**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "feat: _worker_shard 改共享队列阻塞消费 + 哨兵退出 + 按批 ack"
```

---

### Task 4: 重写 `run_multigpu`：全局去重、分发、ack 统计、哨兵、合并

**Files:**
- Modify: `distill/generate_pseudo_labels.py`（`run_multigpu` 约 L550-587）
- Test: 无独立单测（依赖 spawn + GPU），验证靠 Task 5 服务器冒烟 + `py_compile`。

**Interfaces:**
- Consumes: `_collect_done_names(shard_paths)`（Task 1）、`_worker_shard(args, gpu_id, task_queue, done_queue, output_path)`（Task 3）、`_SENTINEL`（Task 3）、`merge_shards`（现有）。
- Produces: `run_multigpu(args) -> None` —— 新调度主循环。

- [ ] **Step 1: 实现**

重写 `run_multigpu`：

```python
def run_multigpu(args: argparse.Namespace) -> None:
    """多卡入口：共享任务队列 + 消费确认 + 哨兵退出。

    1. 主进程扫描已有输出与 shard 文件做全局去重，只分发未处理图像。
    2. 各 GPU 子进程从任务队列阻塞拉取、处理后向完成队列发 ack。
    3. 主进程统计消费数，全部确认后发哨兵，join 后合并 shard。
    """
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids 不能为空")

    image_paths = collect_image_paths(args.image_dir)
    output = Path(args.output)
    shard_outputs = [str(output.with_name(f"{output.stem}.shard{i}.json")) for i in range(len(gpu_ids))]

    # 断点续跑：全局去重（扫描最终输出 + 各 shard），只分发未处理图像
    done_names = _collect_done_names([args.output, *shard_outputs])
    pending = [p for p in image_paths if p.name not in done_names]
    if done_names:
        print(f"断点续跑：跳过 {len(done_names)} 张已处理图像", file=sys.stderr)
    print(f"待处理图像: {len(pending)} 张，共 {len(gpu_ids)} 卡", file=sys.stderr)

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    done_queue: mp.Queue = ctx.Queue()
    for p in pending:
        task_queue.put(p)

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_shard,
            args=(args, gpu_id, task_queue, done_queue, shard_outputs[i]),
        )
        p.start()
        processes.append(p)

    # 主进程统计消费数：worker 全部退出但未消费完 -> 有任务丢失，中止
    consumed = 0
    while consumed < len(pending):
        try:
            ack_batch = done_queue.get(timeout=5)
        except mp.queues.Empty:
            if not any(p.is_alive() for p in processes):
                raise RuntimeError(
                    f"所有 worker 已退出但仅消费 {consumed}/{len(pending)} 张，"
                    f"任务可能丢失，请用 --resume 重跑"
                )
            continue
        consumed += len(ack_batch)

    # 全部确认消费：向每个 worker 发哨兵，优雅退出
    for _ in gpu_ids:
        task_queue.put(_SENTINEL)
    for p in processes:
        p.join()

    failed = [p.exitcode for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"部分 GPU 分片失败，exit codes: {failed}")

    merge_shards(shard_outputs, args.output)
    with open(args.output, "r", encoding="utf-8") as f:
        merged = json.load(f)
    print(f"多卡伪标签已合并保存到 {args.output}")
    print(f"图像数: {len(merged['images'])}，标注数: {len(merged['annotations'])}")
```

同步删除旧 `run_multigpu` 中的 `queue`/`_split_shards` 引用（`_split_shards` 已在 Task 3 删除）。

- [ ] **Step 2: 语法检查**

Run: `python -m py_compile distill/generate_pseudo_labels.py`
Expected: 退出码 0，无输出。

- [ ] **Step 3: 提交**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "feat: run_multigpu 改全局去重+共享队列+ack 统计+哨兵退出"
```

---

### Task 5: 文件头注释更新 + 服务器冒烟验证

**Files:**
- Modify: `distill/generate_pseudo_labels.py`（文件 docstring 多卡说明，约 L24-28）

- [ ] **Step 1: 更新 docstring**

把多卡说明段（L24-28）替换为：

```
多卡说明：
    --gpu-ids 指定参与推理的 GPU 索引（逗号分隔）。脚本启动一个主进程和每卡一个
    spawn 子进程：主进程对输出与各 shard 文件做全局去重后，把待处理图像放入共享
    任务队列；各子进程用 CUDA_VISIBLE_DEVICES 隔离到对应 GPU，进程内只创建一次
    模型，从队列动态拉取图像（每 32 张一批），处理完向完成队列发 ack；主进程统计
    全部确认后发哨兵，子进程优雅退出。各子进程分别写入 <output>.shard<i>.json，
    全部完成后合并为最终输出。分片文件保留，配合 --resume 可断点续跑。
```

- [ ] **Step 2: 本机全量检查**

Run: `python -m py_compile distill/generate_pseudo_labels.py; python tests/test_done_names.py; python tests/test_run_pipeline_worker.py`
Expected: 全部退出码 0；两个测试打印 `OK`。

- [ ] **Step 3: 提交**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "docs: 更新多卡调度 docstring，补充服务器冒烟命令"
```

- [ ] **Step 4: 服务器冒烟验证（需用户执行，Linux + GPU）**

在服务器 `.venv-vllm` 环境运行（小规模：先验证调度正确性）：

```bash
# 准备 8 张测试图（无 GPU 依赖也可用已有 distill/data/images 前若干张）
mkdir -p /tmp/mgpu_smoke && cp /raid5/sh/code/VLX-Seek/distill/data/images/*.jpg /tmp/mgpu_smoke/ | head -n 8
python distill/generate_pseudo_labels.py \
  --image-dir /tmp/mgpu_smoke \
  --categories "person; car" \
  --output /tmp/mgpu_smoke/out.json \
  --model-path resources/VLX-Seek-1.5-10B \
  --backend vllm \
  --prompt-batch-size 50 \
  --gpu-ids "0,1,2,3,4,5,6,7" \
  --slice-width 2500 --slice-height 2500
```

预期：
- 8 个进程都出现 `Loading weights`（或 vLLM 初始化日志），**无进程提前退出**；
- 结束时打印 `图像数: 8`（8 张全部处理）；
- 再跑一次加 `--resume`，打印 `断点续跑：跳过 8 张` 后立即完成。

---

## Self-Review

- **Spec 覆盖**：全局去重（Task 1/4）✓；共享队列 + 按批 ack（Task 3）✓；主进程统计 + 哨兵退出（Task 4）✓；worker 只创建一次模型（Task 2/3）✓；崩溃处理（Task 4 中 `consumed < len(pending)` 且进程全退时 `RuntimeError`）✓；合并 shard（Task 4 复用 `merge_shards`）✓；单卡行为零变化（Task 2 中 `worker=None` 时自建）✓。
- **占位符**：无 TBD/TODO；每个代码步骤给出完整代码。
- **类型一致性**：`_create_worker`、`run_pipeline(worker=)`、`_worker_shard(task_queue, done_queue, output_path)`、`_collect_done_names(shard_paths)`、`_SENTINEL` 在 Task 2/3/4 中签名一致。
