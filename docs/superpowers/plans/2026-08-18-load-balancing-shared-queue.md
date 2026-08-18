# Load Balancing via Shared Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the multi‑GPU pseudo‑label generation script to use a shared work queue so that GPU workers dynamically pull images as they become available, eliminating static load imbalance and reducing overall makespan.

**Architecture:** The main process creates a `multiprocessing.Queue` filled with all image paths. Each worker process repeatedly `get_nowait()` an image from the queue, processes it with the existing inference pipeline, and appends results to its own shard file (preserving the current per‑GPU output format). When the queue is empty, the worker exits. After all workers finish, the main process merges the shard files as before.

**Tech Stack:** Python 3.12+, multiprocessing, existing VLX‑Seek inference pipeline, tqdm for progress reporting.

## Global Constraints

- Preserve existing command‑line interface (`--gpu-ids`, `--resume`, `--output`, etc.).
- Keep per‑GPU shard files (`<output>.shard<i>.json`) for fault tolerance and resume capability.
- Do not modify the core inference logic (`detect_with_crop`, `run_pipeline`, etc.).
- Maintain the same logging setup (per‑shard log files).
- Ensure the script works on Windows (using `spawn` context) and Linux.

---
### Task 1: Understand Current Multiprocessing Flow

**Files:**
- Modify: `distill/generate_pseudo_labels.py:534-568` (run_multigpu and _worker_shard)

**Interfaces:**
- Consumes: `args` (parsed arguments), `image_paths` (list of Path)
- Produces: spawns worker processes, waits for them, calls `merge_shards`

**Steps:**
- [ ] **Step 1: Write a comment summarizing the current flow**

```python
# Current flow: round-robin split of image_paths into shards, one shard per GPU.
# Each worker processes its shard independently and writes <output>.shard<i>.json.
```

- [ ] **Step 2: Run the script in single‑GPU mode to verify baseline**

Run: `python distill/generate_pseudo_labels.py --image-dir distill/data/images --categories "person;car" --output test.json --device cuda`
Expected: Should run without error and produce test.json.

- [ ] **Step 3: Commit**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "docs: comment current multiprocessing flow"
```

### Task 2: Introduce Shared Queue in run_multigpu

**Files:**
- Modify: `distill/generate_pseudo_labels.py:534-568`

**Interfaces:**
- Consumes: `args`, `image_paths`
- Produces: `mp.Queue` filled with image paths, passed to each worker

**Steps:**
- [ ] **Step 1: Import multiprocessing.Queue at top if not already**

```python
import multiprocessing as mp
```

- [ ] **Step 2: Replace static shard splitting with queue creation**

```python
def run_multigpu(args: argparse.Namespace) -> None:
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids 不能为空")

    image_paths = collect_image_paths(args.image_dir)
    # Create a queue and put all image paths
    queue = mp.Queue()
    for p in image_paths:
        queue.put(p)

    output = Path(args.output)
    shard_outputs = [str(output.with_name(f"{output.stem}.shard{i}.json")) for i in range(len(gpu_ids))]

    ctx = mp.get_context("spawn")
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_shard,
            args=(args, gpu_id, queue, shard_outputs[i]),
        )
        p.start()
        processes.append(p)

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

- [ ] **Step 3: Run a quick sanity check (no actual inference) to ensure queue logic does not break**

Run: `python -c "import multiprocessing as mp; q=mp.Queue(); [q.put(i) for i in range(5)]; assert not q.empty()"`
Expected: No assertion error.

- [ ] **Step 4: Commit**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "feat: replace static sharding with shared queue in run_multigpu"
```

### Task 3: Modify _worker_shard to Consume from Queue

**Files:**
- Modify: `distill/generate_pseudo_labels.py:494-502` (_worker_shard)

**Interfaces:**
- Consumes: `args`, `gpu_id`, `queue` (mp.Queue), `output_path` (shard file)
- Produces: processes images from queue until empty, writes to output_path

**Steps:**
- [ ] **Step 1: Update _worker_shard signature and docstring**

```python
def _worker_shard(args: argparse.Namespace, gpu_id: int, queue: mp.Queue, output_path: str) -> None:
    """子进程入口：用 CUDA_VISIBLE_DEVICES 隔离到指定 GPU 后，从共享队列拉取图像进行推理。"""
```

- [ ] **Step 2: Replace shard iteration with queue‑get loop**

```python
def _worker_shard(args: argparse.Namespace, gpu_id: int, queue: mp.Queue, output_path: str) -> None:
    _setup_logging(output_path + ".log")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shard_args = argparse.Namespace(**vars(args))
    shard_args.device = "cuda:0"  # 隔离后 cuda:0 即物理 GPU gpu_id
    shard_args.output = output_path
    # Process images from the shared queue until it is empty
    run_pipeline(shard_args, image_paths=None)  # we will ignore image_paths and pull from queue inside run_pipeline? 
```

But we need to adjust run_pipeline to accept a queue or we need to pull inside _worker_shard and pass a list. Simpler: keep run_pipeline expecting a list of image paths; we will collect from queue into a local list? That defeats purpose. Instead we modify run_pipeline to accept an optional iterator or we change _worker_shard to loop and call a helper that processes a single image.

Let's refactor: create a function `process_single_image` that does the work for one image path, then _worker_shard loops over queue.

We'll need to extract the per‑image processing logic from run_pipeline.

Given time, we can adjust run_pipeline to accept an optional iterable of image paths; if None, it reads from a queue attribute? Better to keep changes localized.

Alternative: In _worker_shard, we continuously `queue.get_nowait()` and call a new helper `process_image_and_append_to_coco` that mimics the per‑image logic but writes directly to the shard file.

We'll need to replicate the COCO building logic per worker.

Simpler: we can keep run_pipeline but change its signature to accept an optional queue; if queue provided, it pulls until empty.

Let's do that: modify run_pipeline to accept `queue: mp.Queue | None = None` and if provided, ignore `image_paths` and loop over queue.

We'll need to adjust the function accordingly.

Let's outline steps:

- Modify run_pipeline signature: add `queue: mp.Queue | None = None`
- At start, if queue is not None, we will iterate: `while True: try: img_path = queue.get_nowait() except Empty: break`
- For each img_path, run the same processing as before (the for loop inside run_pipeline).
- Remove the existing for loop over image_paths and replace with while loop that gets from queue or iterates over image_paths.

We'll need to move the per‑image processing block into a helper `_process_one_image` to avoid duplication.

Given the scope, we can implement directly in run_pipeline.

Let's write the changes.

**Files:**
- Modify: `distill/generate_pseudo_labels.py:339-465` (run_pipeline)

**Steps:**
- [ ] **Step 1: Add queue parameter to run_pipeline**

```python
def run_pipeline(args: argparse.Namespace, image_paths: list[Path] | None = None, queue: mp.Queue | None = None) -> None:
```

- [ ] **Step 2: At the top, if queue is not None, we will use it; else use image_paths**

```python
    if queue is not None:
        # We'll pull from queue until empty
        use_queue = True
    else:
        use_queue = False
        if image_paths is None:
            image_paths = collect_image_paths(args.image_dir)
            image_paths = image_paths[args.start_index : args.end_index]
```

- [ ] **Step 3: Replace the for loop with a while loop that gets from queue when use_queue, else iterates over image_paths**

We'll need to restructure the loop:

```python
    pbar = None
    if not use_queue:
        total = len(image_paths)
        pbar = tqdm(image_paths, desc="生成伪标签", unit="图", file=sys.stderr)
    else:
        # For queue we don't know total; we can still show progress via a counter
        pbar = tqdm(desc="生成伪标签 (队列)", unit="图", file=sys.stderr)

    processed_count = 0
    while True:
        if use_queue:
            try:
                img_path = queue.get_nowait()
            except mp.queues.Empty:
                break
        else:
            # image_paths iteration
            if processed_count >= len(image_paths):
                break
            img_path = image_paths[processed_count]
            processed_count += 1

        if img_path.name in done_names:
            if not use_queue:
                pbar.update(1)
            continue

        # ... existing per‑image processing (lines 415-460) ...

        if (i + 1) % 10 == 0 or (i + 1) == total:  # need to adjust logic for queue
```

This is getting complex. Given the time, perhaps a simpler approach: keep run_pipeline as is (expects a list) and in _worker_shard we accumulate images from queue into a local list until a batch size (e.g., 64) then call run_pipeline on that batch, repeating until queue empty. This reduces overhead of calling run_pipeline many times.

We can set a reasonable batch size (e.g., 32) to amortize the setup cost.

Let's adopt batch approach.

**Steps for _worker_shard:**
- [ ] **Step 1: In _worker_shard, pull images from queue in batches**

```python
def _worker_shard(args: argparse.Namespace, gpu_id: int, queue: mp.Queue, output_path: str) -> None:
    _setup_logging(output_path + ".log")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shard_args = argparse.Namespace(**vars(args))
    shard_args.device = "cuda:0"
    shard_args.output = output_path

    batch_size = 32
    batch: list[Path] = []
    while True:
        try:
            # get with timeout to allow breaking when empty? We'll use get_nowait and break on Empty
            img_path = queue.get_nowait()
            batch.append(img_path)
            if len(batch) >= batch_size:
                run_pipeline(shard_args, image_paths=batch)
                batch.clear()
        except mp.queues.Empty:
            break
    # process remaining batch
    if batch:
        run_pipeline(shard_args, image_paths=batch)
```

- [ ] **Step 2: Ensure run_pipeline still works with a list (no queue param needed)**

Thus we keep run_pipeline unchanged.

- [ ] **Step 3: Test that the logic works with a dummy queue**

Run: a small script that creates a queue, puts a few paths, spawns a worker that uses the above logic and verifies it processes all.

- [ ] **Step 4: Commit**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "feat: modify _worker_shard to consume images from shared queue in batches"
```

### Task 4: Preserve Resume Functionality

**Files:**
- Modify: `distill/generate_pseudo_labels.py:362-369` (resume logic inside run_pipeline)

**Interfaces:**
- Consumes: `args.resume`, `args.output`
- Produces: `done_names` set of already processed file names

**Steps:**
- [ ] **Step 1: Verify that the existing resume logic works unchanged when each worker writes to its own shard file**

Because each worker gets its own `output_path` (shard file), the resume logic inside run_pipeline will read that shard file and skip already processed images. This should work as before.

- [ ] **Step 2: Add a comment to clarify**

```python
# 断点续跑：读取已有输出中的 file_name（每个卡的分片文件独立）
```

- [ ] **Step 3: Run a quick test with --resume to ensure no regression**

Run: 
1. First run with 2 GPUs on a tiny subset (--end-index 10) to generate shards.
2. Second run with same args plus --resume; ensure it skips already processed images.

Expected: No duplicate annotations, second run finishes quickly.

- [ ] **Step 4: Commit**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "docs: confirm resume logic remains valid with per‑worker shards"
```

### Task 5: Test End‑to‑End Multi‑GPU Load Balancing

**Files:**
- Modify: None (just run verification)

**Steps:**
- [ ] **Step 1: Create a small test image set (e.g., copy 20 images to a temp folder)**

Run: 
```bash
mkdir -p test_images
copy /distill/data/images/*.jpg test_images\ (first 20 files)
```

- [ ] **Step 2: Run the script with 2 GPU IDs (0,1) simulating multi‑GPU (even if only one GPU present, the script will still spawn processes; CUDA_VISIBLE_DEVICES will isolate them; if only one GPU, both processes will see GPU 0, but that's okay for testing logic).**

Run: 
```bash
python distill/generate_pseudo_labels.py --image-dir test_images --categories "person;car" --output test_multi.json --gpu-ids 0,1
```

- [ ] **Step 3: Verify output JSON is valid COCO and contains annotations for all images**

Run: 
```bash
python -c "import json; data=json.load(open('test_multi.json')); print('images:', len(data['images']), 'anns:', len(data['annotations']))"
```
Expected: images count equals number of input images (20), annotations >=0.

- [ ] **Step 4: Check that both shard files were created and merged**

Run: 
```bash
dir test_multi.shard*.json
```
Expected: two shard files present.

- [ ] **Step 5: Commit**

```bash
git add test_multi.json test_multi.shard0.json test_multi.shard1.json
git commit -m "test: verify shared queue load balancing produces correct COCO output"
```

### Task 6: Performance Validation (Optional Benchmark)

**Files:**
- None

**Steps:**
- [ ] **Step 1: Run a timing comparison between static sharding (old version) and shared queue (new version) on a moderate dataset (e.g., 200 images).** Use two separate branches or stash.

We can approximate by checking total runtime printed at end.

- [ ] **Step 2: Record makespan (total time) and note improvement.**

- [ ] **Step 3: Commit a benchmark note**

```bash
git commit -m "docs: record benchmark showing X% reduction in makespan with shared queue"
```

### Task 7: Clean Up Temporary Test Files

**Files:**
- Modify: None

**Steps:**
- [ ] **Step 1: Remove test images and output files**

Run: 
```bash
rmdir /s /q test_images
del test_multi.json test_multi.shard*.json
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove temporary benchmark files"
```

## Summary

By following these tasks we will have modified `distill/generate_pseudo_labels.py` to:
- Replace static round‑robin sharding with a dynamic shared queue.
- Keep the existing per‑GPU shard file output for fault tolerance and resume.
- Ensure minimal changes to core inference logic.
- Validate correctness with end‑to‑end tests.

Once the plan is approved, we can proceed with implementation using the `subagent-driven-development` skill.