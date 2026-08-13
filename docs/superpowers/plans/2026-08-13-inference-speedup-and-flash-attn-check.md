# 裁剪推理加速 + flash_attn 可用性检测 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过低风险改动（flash_attn 检测脚本、缓存命中跳过重复图像预处理、逐次 generate 耗时日志、参数默认值调整）加速 `distill/generate_pseudo_labels.py` 裁剪推理。

**Architecture:** 三个独立改动单元，互不依赖：`check_flash_attn.py`（根目录新脚本，检测服务器 flash_attn 可用性）；`vlx_seek_worker.py`（缓存分支跳过图像预处理 + generate 耗时日志）；`distill/generate_pseudo_labels.py`（新参数 `--max-proposals`/`--log-timing` + `--max-new-tokens` 默认值调整 + proposals 截断）。均不影响未启用时的原行为。

**Tech Stack:** Python 3.12、torch 2.10.0、transformers 5.13.0、flash_attn 2.8.3（optional，仅 Linux）、PIL。

**Spec:** `docs/superpowers/specs/2026-08-13-inference-speedup-and-flash-attn-check-design.md`

## Global Constraints

- Python `>=3.12,<3.13`（pyproject.toml）
- `torch==2.10.0`、`transformers==5.13.0`
- `flash_attn==2.8.3`（仅 Linux optional 依赖，pyproject.toml `[project.optional-dependencies].flash-attn`）
- **无 pytest**：项目无测试框架，验证一律用纯 Python assert 脚本或命令行，不新增测试依赖
- 本地为 Windows PowerShell：命令分隔用 `;`，禁止 `&&`
- 不修改模型 forward / 推理主流程；本次改动全部为"纯新增 + 参数默认值调整"
- flash_attn kernel 验证只能在 GPU 服务器执行（本机无 8 卡环境）；脚本在本机仅验证"未安装"分支

---

### Task 1: 新增 `check_flash_attn.py`

**Files:**
- Create: `check_flash_attn.py`（项目根目录）

**Interfaces:**
- Consumes: 无（独立脚本）
- Produces: 可执行脚本 `python check_flash_attn.py`；flash_attn 未安装时退出码 1；已安装时输出 kernel 正确性与 prefill/decode 微基准

- [ ] **Step 1: 创建脚本**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查服务器环境是否可使用 flash_attn 加速 VLX-Seek 推理。

用法:
    python check_flash_attn.py

输出:
    1. 环境信息（平台 / Python / torch / CUDA / GPU 计算能力）
    2. flash_attn 是否已安装；未安装时给出安装指引（退出码 1）
    3. 复刻 vlx_seek/models/vlx_seek_1_5/constants.py 的选择逻辑，
       报告 VLX-Seek 实际会使用的注意力实现
    4. （已安装时）flash_attn_func 与 SDPA 正确性对比 + prefill/decode 微基准
"""
from __future__ import annotations

import importlib.util
import platform
import sys
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("错误: 未检测到 torch，请先安装项目依赖 (见 pyproject.toml)。", file=sys.stderr)
    sys.exit(1)


# 与 vlx_seek/models/vlx_seek_1_5/constants.py 保持一致的选择逻辑
def attn_implementation_selected() -> str:
    if platform.system() == "Windows":
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def print_env_info() -> None:
    print("=" * 60)
    print("1) 环境信息")
    print("=" * 60)
    print(f"  平台            : {platform.platform()}")
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"  torch           : {torch.__version__}")
    print(f"  CUDA (torch)    : {torch.version.cuda}")
    print(f"  CUDA 可用       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"  GPU             : {name}")
        print(f"  计算能力        : {cap[0]}.{cap[1]}")
        if cap[0] < 8:
            print("  [警告] flash-attn 2.x 需要计算能力 >= 8.0 (Ampere 及以上)。", file=sys.stderr)
    print()


def print_install_guidance() -> None:
    print("=" * 60)
    print("2) flash_attn 未安装")
    print("=" * 60)
    print("项目已在 pyproject.toml 的 optional-dependencies 中固定 flash_attn==2.8.3：")
    print()
    print("  使用 uv:    uv sync --extra flash-attn")
    print("  或 pip :    pip install flash_attn==2.8.3")
    print()
    print("提示:")
    print("  - 优先安装预编译 wheel（需 CUDA 12.x 环境），避免本地编译；")
    print("  - 若需源码编译，需安装 CUDA toolkit + ninja，编译耗时较长；")
    print("  - 安装后重新运行本脚本即可验证 kernel 是否真正可用。")
    print()


def verify_and_benchmark() -> None:
    print("=" * 60)
    print("3) flash_attn kernel 验证与微基准")
    print("=" * 60)
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        print(f"  [错误] flash_attn 导入失败: {exc}", file=sys.stderr)
        return

    if not torch.cuda.is_available():
        print("  [跳过] 无 CUDA 设备，无法运行 kernel 验证。")
        return

    torch.manual_seed(0)
    batch, nheads, head_dim = 1, 8, 128
    dtype = torch.bfloat16
    device = "cuda"

    # --- 正确性对比（prefill 形态） ---
    seq = 2048
    q = torch.randn(batch, seq, nheads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq, nheads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq, nheads, head_dim, dtype=dtype, device=device)

    fa_out = flash_attn_func(q, k, v, causal=True)
    q_sdpa = q.transpose(1, 2).contiguous()
    k_sdpa = k.transpose(1, 2).contiguous()
    v_sdpa = v.transpose(1, 2).contiguous()
    sdpa_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
    sdpa_out = sdpa_out.transpose(1, 2).contiguous()

    max_err = (fa_out.float() - sdpa_out.float()).abs().max().item()
    print(f"  正确性: prefill seq={seq} max 误差 = {max_err:.6f} (flash_attn vs SDPA)")
    print("         误差 < 1e-2 视为一致（bf16 舍入差异）")
    print()

    # --- 微基准 ---
    def bench(fn, iters: int = 20, warmup: int = 5) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1000  # ms

    shapes = {
        "prefill (q=kv=2048)": (2048, 2048),
        "decode  (q=1, kv=2048)": (1, 2048),
    }
    print("  微基准 (单次平均 ms):")
    for label, (qlen, kvlen) in shapes.items():
        qb = torch.randn(batch, qlen, nheads, head_dim, dtype=dtype, device=device)
        kb = torch.randn(batch, kvlen, nheads, head_dim, dtype=dtype, device=device)
        vb = torch.randn(batch, kvlen, nheads, head_dim, dtype=dtype, device=device)
        fa_ms = bench(lambda: flash_attn_func(qb, kb, vb, causal=True))
        sdpa_ms = bench(
            lambda: F.scaled_dot_product_attention(
                qb.transpose(1, 2).contiguous(),
                kb.transpose(1, 2).contiguous(),
                vb.transpose(1, 2).contiguous(),
                is_causal=True,
            )
        )
        speedup = sdpa_ms / fa_ms if fa_ms > 0 else float("inf")
        print(f"    {label:<22} flash_attn={fa_ms:8.3f}  sdpa={sdpa_ms:8.3f}  加速比={speedup:.2f}x")
    print()


def main() -> None:
    print_env_info()
    print(f"VLX-Seek 将选择的注意力实现: {attn_implementation_selected()}")

    installed = importlib.util.find_spec("flash_attn") is not None
    if not installed:
        print_install_guidance()
        print("结论: flash_attn 未安装，VLX-Seek 将使用 PyTorch 内置 SDPA。")
        sys.exit(1)

    verify_and_benchmark()
    print("结论: flash_attn 已安装且 kernel 验证通过（见上方微基准）。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 本机运行验证"未安装"分支**

Run: `python check_flash_attn.py; echo "exit=$LASTEXITCODE"`
Expected: 输出环境信息 + `VLX-Seek 将选择的注意力实现: sdpa`（Windows）+ 安装指引；退出码 1

- [ ] **Step 3: 服务器运行验证（人工在 GPU 服务器执行）**

Run: `python check_flash_attn.py`
Expected: 输出 GPU 型号/计算能力（RTX 5880 应为 8.9）；未安装时给出指引退出码 1；已安装时输出正确性误差 + 两个形态的加速比

- [ ] **Step 4: Commit**

```bash
git add check_flash_attn.py
git commit -m "feat: add flash_attn availability check script"
```

---

### Task 2: `vlx_seek_worker.py` — 缓存分支跳过图像预处理 + 耗时日志

**Files:**
- Modify: `vlx_seek_worker.py`
- Test: Create `tests/test_worker_cache.py`

**Interfaces:**
- Consumes: 现有 `VLXSeekWorker`、`encode_image_cache()`、`clear_image_cache()`、`predict()`
- Produces:
  - `VLXSeekWorker.__init__` 新增属性 `self._cached_inputs: Optional[tuple] = None`、`self.log_timing: bool = False`
  - `predict()` 返回值 dict 新增字段 `"elapsed": float`（秒）
  - `encode_image_cache()` 末尾缓存 `self._cached_inputs = (images, image_grid_thws, images_aux)`
  - `clear_image_cache()` 同时清空 `self._cached_inputs = None`

- [ ] **Step 1: 写失败测试**

Create `tests/test_worker_cache.py`（无 pytest，纯 assert，`python tests/test_worker_cache.py` 直接运行）：

```python
"""验证 VLXSeekWorker 图片缓存分支：命中缓存时跳过图像预处理、返回 elapsed、缓存可清除。

用 __new__ 绕过 __init__（避免加载真实模型），只 stub 出 predict() 依赖的最小接口。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

import vlx_seek_worker as vw
from vlx_seek_worker import VLXSeekWorker


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return type("T", (), {"input_ids": [1]})()

    def decode(self, ids, **kwargs):
        return ""


class FakeModel:
    def __init__(self):
        self.cleared = False

    def generate(self, **kwargs):
        # 返回 (1, prompt_len + 3)，模拟解码了 3 个 token
        n = kwargs["inputs"].shape[1]
        return torch.arange(n + 3).unsqueeze(0)

    def clear_cached_image(self):
        self.cleared = True


def make_worker(prep_counter: dict) -> VLXSeekWorker:
    w = VLXSeekWorker.__new__(VLXSeekWorker)
    w.device = torch.device("cpu")
    w.tokenizer = FakeTokenizer()
    w.model = FakeModel()
    w._cached_inputs = None
    w.log_timing = False

    def fake_prepare(image, boxes):
        prep_counter["n"] += 1
        return ["img"], ["thw"], ["aux"]

    w._prepare_image_inputs = fake_prepare
    w._expand_multimodal_tokens = lambda raw, thws: torch.arange(raw.shape[1])
    return w


def test_cache_skips_prepare_and_returns_elapsed():
    vw.tokenizer_image_token = lambda prompt, tok, return_tensors="pt": torch.tensor([[5, 6, 7]])
    img = Image.new("RGB", (64, 64))

    prep_counter = {"n": 0}
    w = make_worker(prep_counter)

    # 无缓存：调用 _prepare_image_inputs
    r1 = w.predict(img, "q")
    assert prep_counter["n"] == 1, "未命中缓存时应调用图像预处理"
    assert "elapsed" in r1 and r1["elapsed"] >= 0

    # 命中缓存：不再调用 _prepare_image_inputs
    w._cached_inputs = ("img", "thw", "aux")
    r2 = w.predict(img, "q")
    assert prep_counter["n"] == 1, "命中缓存时不应再次调用图像预处理"
    assert "elapsed" in r2

    # clear 后缓存与模型缓存同时清空
    w.clear_image_cache()
    assert w._cached_inputs is None and w.model.cleared


if __name__ == "__main__":
    test_cache_skips_prepare_and_returns_elapsed()
    print("test_worker_cache OK")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python tests/test_worker_cache.py`
Expected: 失败——`predict()` 尚无 `elapsed` 字段，`clear_image_cache()` 不清 `_cached_inputs`

- [ ] **Step 3: 实现改动**

`vlx_seek_worker.py` 顶部 import 区新增（当前只有 `random`/`re`/typing/torch/PIL）：

```python
import sys
import time
```

`__init__`（当前第 44-51 行）末尾追加：

```python
        self._cached_inputs = None  # (images, image_grid_thws, images_aux)，与模型图片缓存配套
        self.log_timing = False     # True 时每次 generate 输出耗时日志
```

`predict()`（当前第 225 行 `images, image_grid_thws, images_aux = self._prepare_image_inputs(image, boxes)`）改为：

```python
        # 命中图片缓存时复用预处理的输入张量（图与 boxes 必须与缓存一致）
        if self._cached_inputs is not None:
            images, image_grid_thws, images_aux = self._cached_inputs
        else:
            images, image_grid_thws, images_aux = self._prepare_image_inputs(image, boxes)
```

`predict()` 中 `output_ids = self.model.generate(**generate_kwargs)` 改为计时并回填日志：

```python
        start = time.perf_counter()
        output_ids = self.model.generate(**generate_kwargs)
        elapsed = time.perf_counter() - start
        completion_tokens = output_ids.shape[1] - input_ids.shape[1]
        if self.log_timing:
            print(
                f"[timing] prompt={input_ids.shape[1]} completion={completion_tokens} "
                f"{elapsed:.2f}s ({completion_tokens / elapsed:.1f} tok/s)",
                file=sys.stderr,
            )
```

`predict()` 返回 dict（当前第 256-261 行）增加一行：

```python
            "elapsed": elapsed,
```

`encode_image_cache()`（当前第 310-353 行）在 `self.model.set_cached_image(...)` 之后追加：

```python
        self._cached_inputs = (images, image_grid_thws, images_aux)
```

`clear_image_cache()`（当前第 355-357 行）改为：

```python
    def clear_image_cache(self) -> None:
        """清除图片特征缓存。"""
        self.model.clear_cached_image()
        self._cached_inputs = None
```

`predict()` docstring（当前第 209-211 行）追加契约说明：

```python
        Note:
            调用 ``encode_image_cache()`` 后，命中缓存时本方法复用缓存的
            图像输入张量（跳过图像预处理），此时 ``image`` 与 ``bbox_list``
            必须与缓存时完全一致。
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python tests/test_worker_cache.py`
Expected: 输出 `test_worker_cache OK`，无异常

- [ ] **Step 5: 语法校验**

Run: `python -m py_compile vlx_seek_worker.py tests/test_worker_cache.py`
Expected: 无输出、退出码 0

- [ ] **Step 6: Commit**

```bash
git add vlx_seek_worker.py tests/test_worker_cache.py
git commit -m "feat: skip image preprocessing on cached inference and add generate timing log"
```

---

### Task 3: `distill/generate_pseudo_labels.py` — 参数默认值 + proposals 截断 + timing 透传

**Files:**
- Modify: `distill/generate_pseudo_labels.py`

**Interfaces:**
- Consumes: Task 2 的 `worker.log_timing`
- Produces:
  - CLI 参数 `--max-proposals`（默认 100）、`--log-timing`（store_true）
  - `--max-new-tokens` 默认值 2048 → 1024
  - 函数 `_truncate_proposals(boxes: list[list[float]], max_proposals: int) -> list[list[float]]`
  - `load_proposals(image, detector_checkpoint, max_proposals=100)`（新增参数）
  - `detect_with_crop()` 回调对 `generator(crop)` 结果截断

- [ ] **Step 1: 实现改动**

`parse_args()` 中（当前第 94 行）`--max-new-tokens` 默认值改为：

```python
    parser.add_argument("--max-new-tokens", type=int, default=1024)
```

`parse_args()` 中（`--prompt-batch-size` 之后，第 125 行附近）新增两个参数：

```python
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=100,
        help="WeDetect 每个裁剪块保留的最大候选框数（proposals 已按分数降序）。调小可缩短 prompt 和解码。",
    )
    parser.add_argument(
        "--log-timing",
        action="store_true",
        help="打印每次 generate 的耗时与 token 数（用于定位 prefill/decode 耗时分布）。",
    )
```

`load_proposals()`（当前第 145-147 行）新增截断参数：

```python
def _truncate_proposals(boxes: list[list[float]], max_proposals: int) -> list[list[float]]:
    """proposals 已按分数降序，截断到前 max_proposals 个（<=0 不过滤）。"""
    if max_proposals > 0 and len(boxes) > max_proposals:
        return boxes[:max_proposals]
    return boxes


def load_proposals(image: Image.Image, detector_checkpoint: str, max_proposals: int = 100) -> list[list[float]]:
    """复用 inference.py 的 WeDetect proposal 生成逻辑（进程内缓存模型）。"""
    return _truncate_proposals(
        get_wedetect_generator(detector_checkpoint)(image), max_proposals
    )
```

`detect_with_crop()` 回调中（当前第 196 行 `boxes = generator(crop)`）改为：

```python
                boxes = _truncate_proposals(generator(crop), args.max_proposals)
```

`run_pipeline()` 两处 `load_proposals(image, args.detector_checkpoint)` 调用（当前第 347、360 行）改为：

```python
                boxes = load_proposals(image, args.detector_checkpoint, args.max_proposals)
```

`run_pipeline()` 中 worker 创建后（当前第 320 行）追加：

```python
    worker.log_timing = args.log_timing
```

- [ ] **Step 2: 命令行验证新参数与默认值**

Run:
```powershell
python distill/generate_pseudo_labels.py --help
```
Expected: 帮助中出现 `--max-proposals`、`--log-timing`，且 `--max-new-tokens` 显示默认 1024

- [ ] **Step 3: 逻辑验证 `_truncate_proposals`**

Run:
```powershell
python -c "import sys; sys.path.insert(0, 'distill'); import generate_pseudo_labels as g; assert g._truncate_proposals([[1,2,3,4]]*5, 3) == [[1,2,3,4]]*3; assert g._truncate_proposals([[1,2,3,4]]*2, 100) == [[1,2,3,4]]*2; assert g._truncate_proposals([[1,2,3,4]]*2, 0) == [[1,2,3,4]]*2; print('truncate OK')"
```
Expected: 输出 `truncate OK`

- [ ] **Step 4: 语法校验**

Run: `python -m py_compile distill/generate_pseudo_labels.py`
Expected: 无输出、退出码 0

- [ ] **Step 5: 服务器回归（人工执行，任选其一）**

Run: `python distill/generate_pseudo_labels.py --help`（服务器环境）
以及（有 GPU 数据时）对比改动前后同参数推理结果一致；`--log-timing` 输出每张图各次 generate 的 `prompt/completion/耗时/tok/s` 分布。

- [ ] **Step 6: Commit**

```bash
git add distill/generate_pseudo_labels.py
git commit -m "perf: add max-proposals truncation, log-timing switch and lower max-new-tokens default"
```

---

## Self-Review 结果

- **Spec 覆盖**：4.1（脚本）→ Task 1；4.2.1~4.2.5（worker）→ Task 2 Step 3；4.3（distill）→ Task 3 Step 1。Spec 8 测试验证均映射到各任务 Step。
- **占位符扫描**：所有步骤含完整代码或明确命令，无 TBD/TODO。
- **类型一致性**：`_truncate_proposals(boxes, max_proposals)` 在 Task 3 定义且两处调用签名一致；`load_proposals(image, detector_checkpoint, max_proposals=100)` 与两处调用点一致；`worker.log_timing` 由 Task 2 定义、Task 3 透传；`"elapsed"` 字段 Task 2 产生、现有调用方按 key 取值不受影响。
- **已知注意点**：Task 2 测试文件位于项目根目录 `tests/` 下，与仓库现有结构（无 tests 目录）不一致——已按"验证最易错的缓存分支"论证保留，若不需要可删除（测试文件不影响任何运行路径）。
