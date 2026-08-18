# 多卡伪标签生成：共享队列 + 消费确认（ack）调度设计

日期：2026-08-18

## 背景与问题

`distill/generate_pseudo_labels.py --gpu-ids` 当前的多卡调度（改动前）存在两类问题：

1. **部分 GPU 提前退出**：`_worker_shard` 用 `queue.get_nowait()` + `except mp.queues.Empty: break` 判断结束。
   8 个 spawn 子进程加载 10B 模型耗时不同，先加载完成的 worker 会把队列剩余任务批量拉光，
   后加载完成的 worker 拿到 `Empty` 直接退出，导致实际只有部分 GPU 在推理。
   根因：**队列为空 ≠ 任务完成**（分发与 worker 就绪存在竞态）。
2. **每批重建模型**：`run_pipeline` 每次调用都 `new` 一个 VLXSeek worker（vLLM 后端会
   创建完整 `vLLM.LLM` 引擎，加载全部权重）。`_worker_shard` 每 32 张图调用一次
   `run_pipeline`，存在反复加载模型的开销隐患。

目标：采用**共享任务队列 + 消费确认（ack）+ 主进程哨兵退出**的调度，彻底消除竞态，
并保证 worker 进程内只创建一次模型。

## 方案对比（已收敛）

| 方案 | 机制 | 结论 |
|---|---|---|
| A. 任务队列 + 完成确认 + 主进程哨兵退出 | 主进程全局去重后分发；worker 阻塞取任务、处理后 ack；主进程统计完成数，全完成后发哨兵 | ✅ 采用 |
| B. 任务队列尾部放哨兵 | 主进程 put 任务后追加 N 个哨兵 | 无完成统计，崩溃静默丢失任务，不采用 |
| C. 中心状态表（manager dict） | 主进程维护 path→status | 过重，多进程锁开销，不采用 |

## 已确认语义

1. **确认消费成功 = 仅完成确认（ack）**：worker 处理完一批图像后向完成队列发 ack，
   主进程据此判断任务是否全部完成；不做失败重试。
2. **断点续跑去重 = 主进程全局去重**：分发前扫描所有已有 shard 文件，构建全局
   `done set`，只分发未处理图像。
3. **ack 粒度 = 按批（32 张）**：每批一条 ack 消息，消息开销小。
4. **ack 语义为"已消费"而非"已成功"**：批内个别图像失败由 `run_pipeline` 内部
   try/except 容错（打印日志后 continue），失败数在最终汇总报告，不重试、不阻塞完成判断。
5. **崩溃处理 = 中止并靠 --resume 重跑**：主进程检测到所有 worker 已退出但消费数
   < 分发数时，中止并报错；全局去重保证 `--resume` 重跑不重复处理已完成图像。

## 架构

```
主进程                                  每个 worker（spawn 子进程）
─────────                              ─────────────────────────
collect_image_paths                    创建 VLXSeek worker 一次
  ↓                                      ↓
全局去重（扫描已有 shard）               循环:
  ↓                                      task_queue.get() 阻塞
task_queue.put(未处理路径)  ←─────       ├─ 哨兵 → 处理完当前批后退出
  ↓                                      ├─ 路径 → 凑成 batch(32)
启动 N 个 worker                          ↓  batch 满
  ↓                                      run_pipeline(worker=..., batch)
循环:                                    done_queue.put(batch路径)   ← ack
  done_queue.get(timeout)                 ↑
  累计消费数 + 监控进程存活               │
  └─ 消费数 == 分发数 → put N 个哨兵      │
      ↓                                  │
  join 全部进程 → 检查 exitcode           │
      ↓                                  │
  merge_shards → 最终 COCO               │
```

## 关键决策

- **worker 退出不依赖队列空**：阻塞 `get()` + 收到哨兵才退出。先加载完的 worker
  拉光任务不影响其余 worker——它们会阻塞等待，不会提前退出。任务全部分发且全部
  ack 后，主进程统一发 N 个哨兵（每个 worker 一个）。
- **worker 进程内只创建一次模型**：新增 `_create_worker(args)`，在 `_worker_shard`
  中调用一次；`run_pipeline` 增加可选参数 `worker=None`——传入则复用，单卡模式
  （无 `--gpu-ids`）仍自建。同时修复"每批重载模型"隐患。
- **ack 计数**：主进程按 `done_queue` 收到的路径总数累加，`消费数 == 分发数` 时
  判定完成。分发数 = 全局去重后 put 进任务队列的路径数。
- **主进程 poll 超时**：`done_queue.get(timeout=5)`，超时则检查子进程存活状态，
  避免死等，也能及时发现 worker 崩溃。

## 改动范围（全部在 distill/generate_pseudo_labels.py）

| 函数 | 改动 |
|---|---|
| `run_multigpu` | 重写：全局去重 → 分发 → ack 统计循环 → 哨兵 → join → 合并 |
| `_worker_shard` | 重写：创建 worker 一次 → 阻塞取任务循环 → ack |
| `run_pipeline` | 加 `worker=None` 参数；返回成功处理的 file_name 列表（供汇总报告） |
| 新增 `_create_worker(args)` | 封装 vllm/hf 两种后端的 worker 创建 |
| 新增 `_collect_done_names(shard_paths)` | 扫描输出文件 + 各 shard 文件，返回全局已处理 file_name 集合 |

## 错误处理

- worker 单张失败：`run_pipeline` 内部已 try/except，打印日志后 continue，不中断批次。
- worker 崩溃（exitcode != 0）：主进程在 poll 循环中发现所有进程已退出但消费数
  < 分发数 → `RuntimeError`，报告未消费任务数，提示用 `--resume` 重跑。
- 任务队列/完成队列类型：均为 `ctx.Queue()`（spawn 上下文），与 `ctx.Process` 一致。

## 验证方式

1. 构造小规模图像目录（如 8 张图 + 少量类别），`--gpu-ids "0,1"` 运行，确认两张卡
   都有输出、合并结果图像数 = 8。
2. 用远超 GPU 数的任务量（如 100 张图 / 2 卡）运行，确认不存在提前退出、消费数
   最终等于分发数。
3. 运行中途 kill 一个 worker 进程，确认主进程报错中止，`--resume` 后可继续。
