# 设计文档：裁剪推理加速 + flash_attn 可用性检测

**日期**: 2026-08-13
**状态**: 已确认，待实现

## 1. 背景与动机

`distill/generate_pseudo_labels.py` 裁剪推理耗时严重：14 组 prompt × 20 切片（一张大图）约 50 分钟，
即约 280 次串行 `model.generate` 调用，单次约 10.7 秒。此前分析确认：

1. **解码是单次耗时主体**：10B bf16 模型在 RTX 5880（48G，约 960 GB/s 显存带宽）上 batch=1 解码约 25~35 tok/s，
   检测输出需枚举 `<objN>` 索引，`--max-new-tokens 2048` 上限又放大了异常长解码的风险。
2. **零并行 + 重复 prefill**：280 次调用严格串行；`encode_image_cache` 只缓存了视觉特征（跳过 vision tower），
   LLM 层的 KV 未复用，14 组共享前缀的 prompt 每次都全量重算 prefill。
3. **无效重复开销**：缓存分支下每次 `predict()` 仍执行主+辅两套 image processor 的 CPU 图像预处理（280 次/图）。
4. **服务器可能未启用 flash_attn**：`constants.py` 在 Linux 且安装了 flash-attn 时才会选
   `flash_attention_2`，否则回退 `sdpa`。需要脚本确认服务器环境。

本设计实施低风险优化：flash_attn 可用性检测脚本、缓存命中时跳过重复图像预处理、
逐次 generate 耗时日志、以及参数默认值调整。不做深度改造（vLLM、prefix-KV 复用等）。

## 2. 方案选择

### 方案 A：仅调参（零代码改动）
只改启动命令参数（`--prompt-batch-size`、`--max-new-tokens`、`--slice-width/height`）。
- 优点：零风险，立即可用
- 缺点：无法验证 flash_attn 是否可用；重复预处理等代码层浪费依旧存在；没有耗时数据支撑后续决策

### 方案 B：调参 + 低风险代码优化（已选定）
在方案 A 基础上，新增 flash_attn 检测脚本、缓存分支跳过重复预处理、逐次 generate 耗时日志。
- 优点：收益覆盖参数和代码两层；耗时日志可为后续深度改造（vLLM 等）提供量化依据；改动均为纯新增/小改，风险低
- 缺点：不触及 LLM prefill 复用这一最大优化点（属深度改造，本次不做）

**决策**：采用方案 B。

## 3. 架构设计

### 3.1 分层改动

```
┌─ check_flash_attn.py（新增，项目根目录）─────────────────┐
│ 1. 环境信息（平台/torch/CUDA/GPU 计算能力）                │
│ 2. find_spec("flash_attn") 安装检测 + 未安装时打印指引     │
│ 3. 复刻 constants.py 分支逻辑，报告 VLX-Seek 实际选择      │
│ 4. flash_attn_func vs SDPA 正确性对比 + prefill/decode 基准│
└──────────────────────────────────────────────────────────┘

┌─ vlx_seek_worker.py（改动）──────────────────────────────┐
│ predict(): 命中图片缓存 → 跳过 _prepare_image_inputs       │
│ encode_image_cache(): 额外缓存 (images, thws, images_aux)  │
│ clear_image_cache(): 同步清空缓存输入                      │
│ 新增 log_timing 属性：每次 generate 输出 耗时/token/tok/s  │
│ predict() 结果增加 elapsed 字段                            │
└──────────────────────┬───────────────────────────────────┘
                       │
┌─ distill/generate_pseudo_labels.py（改动）────────────────┐
│ 新增 --max-proposals（默认 100=现状）                      │
│ 新增 --log-timing 开关                                     │
│ --max-new-tokens 默认 2048 → 1024                          │
└───────────────────────────────────────────────────────────┘
```

### 3.2 执行流程（缓存分支）

1. 图片加载 → WeDetect 生成 proposals
2. `worker.encode_image_cache(image, boxes)` → 编码并缓存视觉特征 + 缓存输入张量
3. 循环 N 次 `worker.detect()` → `predict()` 命中缓存 → 跳过图像预处理，直接走模型缓存分支
4. `worker.clear_image_cache()` 释放

### 3.3 性能收益估算

| 措施 | 预期收益 |
|---|---|
| flash_attn（若可安装） | prefill 提升明显；decode 在 batch=1 下收益有限（仍是带宽瓶颈） |
| 跳过重复图像预处理 | 每图省 14×20 次 CPU 预处理 |
| `--max-new-tokens` 1024 | 防止偶发长解码，单次耗时上界降低一半 |
| `--max-proposals` 50（可选） | prompt 少 100 token、输出枚举更短 |

## 4. 详细设计

### 4.1 新增 `check_flash_attn.py`（项目根目录，约 150 行）

命令行工具，无第三方新增依赖（flash_attn 本身除外，检测到即验证）：

1. **环境信息**：
   - `platform.platform()` / `sys.version` / `torch.__version__` / `torch.version.cuda`
   - `torch.cuda.get_device_name(0)` 与 `torch.cuda.get_device_capability(0)`（RTX 5880 Ada = (8, 9)，满足 flash-attn 2.x 的 sm80+）
2. **安装检测**：`importlib.util.find_spec("flash_attn")`
   - 未安装：打印安装指引（按 torch/CUDA/python 版本推荐官方 wheel：`pip install flash-attn --no-build-isolation` 或预编译 wheel；若需源码编译需 CUDA toolkit + ninja，耗时较长），退出码 1
   - 已安装：继续
3. **选择逻辑镜像**：复刻 `constants.py` 的 `ATTN_IMPLEMENTATION` 判定（Linux + flash_attn 存在 → flash_attention_2），报告 VLX-Seek 实际生效的注意力实现
4. **真实 kernel 验证**（仅已安装时）：
   - 正确性：构造随机 bf16 张量，`flash_attn_func(..., causal=True)` 与 `F.scaled_dot_product_attention(..., is_causal=True)` 输出对比 max 误差
   - 微基准：prefill 形态（seq=2048, head_dim=128）与 decode 形态（q_len=1, kv_len=2048）各跑若干次，输出两者耗时与加速比

> 注意：脚本需在 GPU 服务器上运行（本机无 8 卡环境），可放在任意能访问代码的位置，无路径依赖。

### 4.2 `vlx_seek_worker.py` 改动（约 25 行）

#### 4.2.1 `__init__` 新增

```python
self._cached_inputs = None  # (images, image_grid_thws, images_aux)，与模型缓存配套
self.log_timing = False     # True 时每次 generate 输出耗时日志
```

#### 4.2.2 `encode_image_cache()`

现有逻辑末尾追加：

```python
self._cached_inputs = (images, image_grid_thws, images_aux)
```

#### 4.2.3 `predict()` 缓存分支

在 `_prepare_image_inputs` 调用处改为：

```python
if self._cached_inputs is not None:
    images, image_grid_thws, images_aux = self._cached_inputs
else:
    images, image_grid_thws, images_aux = self._prepare_image_inputs(image, boxes)
```

契约（docstring 注明）：命中缓存时要求用与缓存相同的图和 boxes 调用（与现有图片缓存语义一致）。

#### 4.2.4 `clear_image_cache()`

```python
def clear_image_cache(self) -> None:
    self.model.clear_cached_image()
    self._cached_inputs = None
```

#### 4.2.5 耗时日志与 elapsed

`predict()` 中：

```python
import sys
import time  # 顶部新增（worker 当前未导入这两个模块）

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

结果 dict 增加 `"elapsed": elapsed`。

### 4.3 `distill/generate_pseudo_labels.py` 改动（约 15 行）

1. `--max-new-tokens` 默认值 `2048 → 1024`
2. 新增参数：

```python
parser.add_argument(
    "--max-proposals",
    type=int,
    default=100,
    help="WeDetect 每个裁剪块保留的最大候选框数（proposals 已按分数降序）。调小可缩短 prompt 和解码。",
)
```

3. 新增参数 `--log-timing`（`action="store_true"`，透传给 worker：`worker.log_timing = args.log_timing`）
4. WeDetect 调用处截断。proposals 产生于两处，需分别处理：

   - `load_proposals()` 函数（非裁剪路径，[generate_pseudo_labels.py](file:///d:/WorkPlace/Pycharm/VLX-Seek/distill/generate_pseudo_labels.py#L145-L147)）：
     对 `get_wedetect_generator(detector_checkpoint)(image)` 的返回值截断；
   - `detect_with_crop()` 回调（裁剪路径，[generate_pseudo_labels.py](file:///d:/WorkPlace/Pycharm/VLX-Seek/distill/generate_pseudo_labels.py#L196)）：
     对 `generator(crop)` 的返回值截断。

   抽一个小工具函数 `_truncate_proposals(boxes, max_proposals)`，两处调用，避免重复逻辑：

```python
def _truncate_proposals(boxes: list[list[float]], max_proposals: int) -> list[list[float]]:
    """proposals 已按分数降序，截断到前 max_proposals 个（<=0 不过滤）。"""
    if max_proposals > 0 and len(boxes) > max_proposals:
        return boxes[:max_proposals]
    return boxes
```

## 5. 错误处理

| 场景 | 处理方式 |
|---|---|
| flash_attn 未安装 | 脚本打印安装指引后退出码 1，不抛异常 |
| flash_attn 已装但 kernel 验证失败（如计算能力不匹配） | 脚本输出失败详情并建议回退 sdpa，VLX-Seek 仍可正常使用 |
| 缓存命中但图/boxes 与缓存不一致 | 属调用方契约违反；与现有图片缓存语义一致，不额外加校验 |
| `--max-proposals <= 0` | 视为不过滤，保持全部 proposals |

## 6. 向后兼容性

- `--max-new-tokens` 默认值变更：显式传参的调用不受影响；未传参的默认行为从"长解码上限 2048"变为 1024（检测场景输出远短于此，无实际影响）
- `--max-proposals` 默认 100 = 现状，零行为变化
- `--log-timing` 默认关闭，零行为变化
- Worker 缓存分支为纯新增 `if`，未命中缓存时与原路径完全一致
- `predict()` 返回值新增 `elapsed` 字段，现有调用方按 key 取值，不受影响

## 7. 改动文件清单

| 文件 | 改动类型 | 改动量 |
|---|---|---|
| `check_flash_attn.py` | 新增 | ~150 行 |
| `vlx_seek_worker.py` | 修改 | ~25 行 |
| `distill/generate_pseudo_labels.py` | 修改 | ~15 行 |

## 8. 测试验证

1. **脚本**：在 GPU 服务器 `python check_flash_attn.py`，确认输出环境信息、安装状态、VLX-Seek 实际选择，已安装时验证 kernel 正确性与加速比
2. **默认路径回归**：`--prompt-batch-size 0`（缓存不启用）输出与改动前一致
3. **分批路径**：分批推理结果与改动前一致，且每图耗时下降（图像预处理被跳过）
4. **参数**：`--max-proposals 50` 与默认 100 结果对比（应一致或仅遗漏低置信小目标）
5. **日志**：`--log-timing` 输出每张图各次 generate 的耗时分布，用于确认 prefill/decode 占比
