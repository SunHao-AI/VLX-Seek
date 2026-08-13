# vLLM 迁移（prefix caching + continuous batching）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 VLX-Seek 推理接入 vLLM 引擎，利用 automatic prefix caching 复用图像/object 共享前缀 KV，并用 continuous batching 把 `distill/generate_pseudo_labels.py` 的 280 次串行 generate 变为并发调度，使单张大图 50 分钟的推理降到分钟级。

**Architecture:** 新增 vLLM 推理路径（`VLXSeekVLLMWorker`），自定义多模态模型注册（vision tower + image grid + object features + aux tower 移植进 vLLM 图），保留 WeDetect proposals 生成与结果解析在 Python 侧；`generate_pseudo_labels.py` 通过统一接口切换到新 worker，改动面收敛在 worker 层。**M0 可行性尖峰（输出一致性验证）是本计划的 gate**，不通过则回退评估 SGLang 或 HF 自定义 prefix-KV。

**Tech Stack:** vLLM（版本需在服务器验证与 torch 2.10.0 / transformers 5.13.0 兼容）、torch 2.10.0、transformers 5.13.0、Python 3.12、flash_attn 2.8.3（vLLM 内部使用）、8×RTX 5880 48G。

**前置已完成:** `check_flash_attn.py`（flash_attn 可用性检测）、`vlx_seek_worker.py` 缓存分支与 `--log-timing`（耗时数据采集）、`generate_pseudo_labels.py` 的 `--max-proposals`/`--max-new-tokens`/`--log-timing`。设计见 `docs/superpowers/specs/2026-08-13-inference-speedup-and-flash-attn-check-design.md`。

## 现状与瓶颈（已实测/估算）

- 14 组 prompt × 20 切片 = 280 次串行 `model.generate`，单次 ≈ 10.7s，全图 ≈ 50 分钟
- 每次调用 prompt 约 1500~2000 token（1000×1000 切片 → ~1200 视觉 token + 100 proposals → 200 object token + 类别文本）
- 单次耗时构成：decode 是主体（batch=1，10B bf16 在 5880 ≈ 960 GB/s 带宽 → 25~35 tok/s）+ 每次全量重算共享前缀 prefill
- `encode_image_cache` 只跳过 vision tower，LLM 层 KV 不复用；14 组 prompt 的图像+object 前缀（占 prompt ~90%）完全相同

## 版本兼容性调研结论（2026-08-13 Web 检索，M0 需在服务器实测确认）

| 候选 | torch | transformers | Qwen3.5 支持 | 备注 |
|---|---|---|---|---|
| **vLLM 0.17.0（首选）** | 2.10.0（**与项目一致**） | 社区记录 4.56.2 规避 AuxRequest/flex_attention 问题；**对 transformers 5.x 兼容性未知** | ✅ 原生支持 | 与项目 torch 完全匹配；需在 M0 实测 transformers 5.13.0 能否加载，若报错则需在 a) 项目 transformers 降级（影响 HF 路径，风险大）与 b) 换 vLLM 0.20+ 之间抉择 |
| vLLM 0.20.0（2026-04） | 2.11（默认 CUDA 13） | v5 兼容性修复（>=5） | 部分（Qwen3-Next/3.5 相关支持） | torch 2.11 与项目 2.10 不匹配，需项目级 torch 升级 |
| vLLM 0.22.0 | 要求 torch==2.11.0 | >=4.56.0 | — | 同上，torch 升级成本 |
| NVIDIA NGC 26.02 容器 | 2.11.0a0 | 4.57.5 | — | 容器化备选（含 flash-attn 2.7.4.post1） |

**M0 验证顺序**：vLLM 0.17.0 + 项目 torch 2.10.0 + transformers 5.13.0 直接试装 → 不兼容则按上表决策。

**vLLM 自定义多模态 API 现状（Next/latest 文档）**：`ModelRegistry.register_model("VLXSeek1_5ForCausalLM", "vllm_serve.vlx_seek_vlm:VLXSeek1_5ForCausalLM")`（插件式注册，建议放 vLLM 插件避免 fork 子进程 CUDA 重初始化问题）；模型类实现 `SupportsMultiModal`，提供 `get_multimodal_embeddings()`（返回 `(num_items, feature_size, hidden_size)` 3D 张量）与 `get_input_embeddings()`（用 `merge_multimodal_embeddings` 合并占位 token）；`MULTIMODAL_REGISTRY.register_processor(processor, info=..., dummy_inputs=...)` 注册处理器。注意旧版（v0.7 系）API 是 `register_image_input_mapper`，以实际安装版本为准（M0 时在服务器 `pip show vllm` + 查对应版本文档确认）。

## vLLM 收益点

| 收益 | 机制 | 预期 |
|---|---|---|
| 消除重复 prefill | automatic prefix caching（`enable_prefix_caching`），同 crop 的 14 个请求共享图像+object 前缀 KV | 每请求省 ~90% prefill |
| 消除串行 | continuous batching：280 请求一次提交，引擎自行调度 | 多请求并发 decode |
| 提高 decode 利用率 | 多请求共享一个 decode step 的带宽/算力 | 吞吐 5-20× |
| 多卡利用 | tensor parallel / 引擎层多卡调度 | 与切片级分片正交 |

## M0 实测结果（2026-08-13，服务器）

**环境（.venv-vllm 独立环境）**：8×RTX 5880 Ada（sm89）、torch 2.10.0+cu130（`uv pip install --torch-backend=auto` 自动选择）、vllm 0.17.0、transformers 4.57.6、flash_attn 未装（警告，非必需）。

**直接加载结果**：`LLM(model=..., trust_remote_code=True)` **失败**（预期内）——
```
Value error, The checkpoint you are trying to load has model type `vlx_seek_1_5`
but Transformers does not recognize this architecture.
```
vLLM 的 ModelConfig 校验走 transformers 架构注册表，`model_type: vlx_seek_1_5` 不被识别；`trust_remote_code` 参数被忽略（"It has no effect here"）。vLLM 0.17 虽原生支持 Qwen3.5，但只认标准架构名，不会映射自定义名。

**Gate 结论：通过（按预期失败路径）→ Task 1 自定义注册**。两条修正：
1. `ModelRegistry.register_model` 的注册键应使用 config 的 architecture 名 `VLXSeek1_5ForCausalLM`（vLLM 用架构名解析，不是 model_type）
2. HF 基线对比在 vllm 环境不可行（transformers 4.57.6 vs 项目 5.13 + 缺项目依赖），一致性回归统一放到项目环境做（Task 3）

## vLLM 0.17 自定义注册 API 调研结论（2026-08-13）

- **语言主干直接复用**：vLLM 0.17 内置 `Qwen3_5ForConditionalGeneration`（`vllm/model_executor/models/qwen3_5.py:630`），VLX-Seek 继承它即可
- **新协议**（旧版 `get_multimodal_embeddings/get_input_embeddings/merge_multimodal_embeddings` 已不存在）：`embed_multimodal(**kwargs) -> MultiModalEmbeddings`（返回待合并嵌入）+ `embed_input_ids(input_ids, multimodal_embeddings, *, is_multimodal)`（默认实现内含合并）
- **注册**：`ModelRegistry.register_model(架构名, "模块:类"懒加载串)`，经 `vllm.general_plugins` entry point 在引擎前加载；注册后 ModelConfig 校验直接命中注册表，**绕过 transformers 架构校验**（M0 那条 `vlx_seek_1_5 not recognized` 报错消失）
- **config 加载**：vLLM 用 transformers AutoConfig 读 config.json，`model_type=vlx_seek_1_5` 需自行 `AutoConfig.register`（vllm_serve 里定义了最小 `VLXSeek1_5Config(Qwen3_5Config)`）
- **权重**：`load_weights` 钩子 + `WeightsMapper(orig_to_new_prefix=...)` + `AutoWeightsLoader`（照抄 Qwen3VL 的 `model.language_model.→language_model.model.`、`lm_head.→language_model.lm_head.` 模式）
- **bbox/object feature 通道 vLLM 无原生支持**：需自定义 MultiModalProcessor / data parser（最大工作量，下个里程碑）

## Task 1 实施状态（2026-08-13）

已提交 `vllm_serve/` 三件套（v0 实现，待服务器验证迭代）：
- `vlx_seek_vlm.py`：`VLXSeek1_5Config` + `VLXSeek1_5ForCausalLM`（继承 vLLM Qwen3.5 VLM，替换视觉栈为项目模块：vision_tower / mm_projector / vision_tower_aux / HFRE / mm_projector_aux；`encode_images/encode_objects` 逐字移植；`embed_multimodal` 图像+object 双通道；`load_weights` + mapper）
- `plugin.py`：注册 config + 模型（懒加载串）
- `test_vllm.py`：text-only + 图像冒烟测试

**里程碑 1a 验证**（服务器）：text-only 生成通过 = 注册+config+权重+引擎全链路打通
**里程碑 1b（未做）**：自定义 MultiModalProcessor——处理 `images_aux/image_grid_thws/bbox_list` 输入、让 `<objfeat>` 占位符进入 `is_multimodal` 掩码、图像 token 数按 grid thw 展开

**1a 实测发现（2026-08-13）**：vllm 环境 transformers 4.57.6 **无 `transformers.models.qwen3_5`**（qwen3_5 modeling 属 transformers 5.x）→ 项目视觉塔（依赖 `Qwen3_5VisionModel`）无法导入。修复：vllm 环境升级 transformers==5.13.0（与项目一致），vLLM 0.17 对 transformers 5.x 的兼容性待实测（若 `import vllm` 崩则回退 4.57.6 并在 vllm_serve 内自实现视觉塔）。

**1a 调试记录（服务器 + 代码修复）**：
1. `VLXSeek1_5Config` 必须继承 **vLLM 自己的** `Qwen3_5Config`（`vllm.transformers_utils.configs.qwen3_5`），否则处理器 isinstance 校验失败；
2. transformers 5.13 的 `PretrainedConfig` 改为 dataclass 机制，`from_dict` 会先把嵌套 `text_config`/`vision_config` 转成对象传入 `__init__`，而 vLLM 的 `Qwen3_5Config.__init__` 只处理 dict/None → 对象被静默丢弃 → 需在 `VLXSeek1_5Config.__init__` 兜底解析并强制写回属性（已修复）；
3. vLLM 为 Qwen3.5 VLM 初始化 mm budget 时会加载 **video 处理器**（VLX-Seek 纯图像模型用不到）：checkpoint 需补 `video_preprocessor_config.json`，且 **`video_processor_type` 的值必须是处理器类名 `"Qwen3VLVideoProcessor"`（不是 model_type "qwen3_5"）**，否则 `video_processor_class_from_name` 解析失败直接报 Unrecognized。

## Task 1 实际完成情况（2026-08-13，1a + 1b 全部打通）

**1a 调试链（worker 注册丢失 → 权重加载）**：
1. **worker 进程注册丢失**：CUDA 初始化后 vLLM 强制 spawn，主进程 `plugin.init()` 注册不传子进程（EngineCore_DP0 报 `Model architectures ['VLXSeek1_5ForCausalLM'] are not supported`）→ 方案：pyproject 声明 `[project.entry-points."vllm.general_plugins"] vllm-seek = "vllm_serve.plugin:init"` + `package=true` + 服务器 `uv pip install -e . --no-deps`，`plugin.py` 幂等守卫
2. **权重键映射**：checkpoint `model.vision_tower.visual.blocks.*` → 模型 `visual.visual.blocks.*`（只替换第一段 `vision_tower`→`visual`）；`model.language_model.*` → `language_model.model.*` + `lm_head` 模式照抄 Qwen3VL
3. **aux 塔 buffer**：C-RADIOv4 的 `input_conditioner.norm_mean/norm_std`、`radio_model.summary_idxs` 是 nn.Buffer（非 Parameter），AutoWeightsLoader 报缺失 → `c_radio_v4_aux_encoder.py` 自定义 `load_weights` 分离 buffer 键手动 copy_；外层用 `ignore_unexpected_prefixes` 防下钻
4. **内层 loader 前缀**：`CRadioV4AuxEncoder.load_weights` 收到的键已去掉 `vision_tower_aux.` 前缀但仍有 `image_tower.` → 传给内层 `AutoWeightsLoader(self.image_tower)` 前剥掉 `image_tower.`

**1a 性能结论（关键）**：CUDA graph 对 hybrid mamba 架构捕获不匹配，text-only 生成 61.5s（0.23 tok/s）；**`enforce_eager=True` 后 2.7s（5.63 tok/s），23× 提升** → 所有 vLLM 推理统一走 enforce_eager

**1b 调试链（多模态输入路径）**：
1. `_parse_and_validate_multimodal_inputs` 找 `pixel_values` 键 → 删除自定义 `embed_multimodal` 解析，改为覆盖 `_process_image_input`（基类解析 + 自定义视觉编码）
2. `_call_hf_processor` 签名是 `(prompt, mm_data, mm_kwargs, tok_kwargs)`（非旧版 2 参）；`bbox_list/images_aux` 从 mm_kwargs 提取、不传 HF processor，合并回 BatchFeature
3. `MULTIMODAL_REGISTRY` 导入路径是 `vllm.multimodal`（非 `vllm.model_executor.models.registry`）
4. **字段 batch 合并**：`images_aux` 须用 CLIPImageProcessor（C-RADIOv4 配置）输出 `[1,C,H,W]`；`bbox_list` 带 batch 维 `[1,N,4]`；`_get_mm_fields_config` 声明为 batched image 字段
5. **embeddings 数量校验**：`sanity_check_mm_encoder_outputs` 要求 `len(mm_embeddings) == num_items`（1 个 image item → 1 个元素）→ `embed_multimodal` 返回 `(torch.cat([image_embeds, object_embeds]),)` 单元素 tuple，图像+object 合并

**1b-2 实测（服务器）**：`test_object_features.py`（纯红图 + 2 bbox + `<objfeat>` 占位符）→ 输出 `'一个红色的背景'`，22.3s（含 11s rendering）。**1a/1b 里程碑全部达成**。commit 历史：`6584440`→`94aa62a`→`fcd325c`→`3637a83`→`f507c81`→`fb99313`→`e7978de`→`25c51b5`→`f8def74`→`39851ca`→`3b5c34f`→`4e9c2b9`

## Task 2 实测结论（2026-08-13，bench_prefix_cache.py）

14 个共享前缀请求（同图 + 同 bbox + 不同类别后缀）批量提交：

| 配置 | 总耗时 | 吞吐 |
|---|---|---|
| APC OFF | **24.3s** | 227 tok/s |
| APC ON | **24.6s** | 223 tok/s |

**核心结论：连续批处理（continuous batching）已覆盖共享前缀问题，无需 APC。** 同批提交时 scheduler 层直接共享 common prefix（`num_common_prefix_blocks`），第一个请求 prefill 后其余复用；APC 只对跨 batch 有价值。且 APC ON 触发 Mamba `align` 模式实验性警告（hybrid 架构支持不成熟）→ **统一保持 APC OFF，靠批量提交获得收益**。

**收益实测**：14 个共享前缀请求 24.3s vs 串行 ~280s+，**~11× 加速**；输出与单请求一致（`'一个红色的背景'` 等）。

## Task 3 实施状态（2026-08-13）

已提交 `vllm_serve/vlx_seek_vllm_worker.py`（与 VLXSeekWorker 同接口：`predict/predict_batch/run_task/detect/detect_multi_prompt/ground/count/encode_image_cache(no-op)/clear_image_cache(no-op)`）：
- prompt 构建：`<image>` → `<|image_pad|>`（vLLM processor 自动展开），`<objfeat>` 占位符由自定义 processor 替换
- 批量提交：detect_multi_prompt 一次 llm.generate 提交所有类别批次（同图共享前缀）
- 结果解析：复用 VLXSeekWorker 静态工具（_validate_boxes/_remap_*/_build_result_bbox_list）
- `distill/generate_pseudo_labels.py` 新增 `--backend hf|vllm`（默认 hf 零行为变化）

**冒烟测试通过（服务器）**：detect_multi_prompt（2 批 21.25s）、detect（5.72s）、predict_batch（5.64s，批量共享前缀生效）。接口格式与 HF worker 一致。

**待办（Task 3 Step 3）**：HF vs vLLM 输出一致性回归（真实图片，两环境各跑 1 张图对比检测结果）。

## Task 4 待办（2026-08-13）

## 关键架构决策

1. **自定义多模态模型注册（本计划最大工程点）**：vLLM 不识别 OmChat 式自定义 VLM。需要：
   - `@multimodal_model` 注册 `VLXSeek1_5ForCausalLM`（vLLM 的 `ModelRegistry`），实现 `MultiModalEmbeddings` / `MultiModalProcessor` 接口
   - 移植 `encode_images`（vision tower + image grid）与 `encode_objects`（object features + bbox_list + aux tower）为 vLLM 图内模块
   - 自定义 input mapper：HF 侧 `mm_utils.py` 的图像预处理（主+辅 processor、grid thw、bbox 裁剪）须在 `MultiModalProcessor` 中复现
2. **prefix caching 的共享前缀条件**：同一 crop 的 14 个 prompt，其 token 序列（图像 token 展开 + object token + 公共文本前缀）必须**字节级一致**。因此 prompt 拼接顺序固定为 `图像+objects+类别文本+输出格式`，类别文本作为可变后缀放在最后（与当前 `_build_prompt` 一致即可满足）。
3. **采样参数映射**：temperature/top_p 直接映射 vLLM `SamplingParams`；repetition_penalty 需确认 vLLM 行为与 HF 一致；`KeywordsStoppingCriteria`（mm_utils.py）用 vLLM 的 `stopping_criteria`（或 stop 字符串）替代。
4. **接口收敛**：`generate_pseudo_labels.py` 只依赖 worker 的 `detect/detect_multi_prompt/encode_image_cache/clear_image_cache/log_timing` 接口；新 worker 实现同接口，distill 脚本经 `--backend hf|vllm` 切换，默认 hf（零行为变化）。

## 全局约束

- Python `>=3.12,<3.13`；torch==2.10.0；transformers==5.13.0（vLLM 版本须与二者兼容，在 M0 验证，必要时升级锁定）
- flash_attn==2.8.3（vLLM 的注意力后端依赖）
- 无 pytest；验证用纯 Python assert / 命令行
- PowerShell 命令用 `;` 分隔；服务器为 Linux（WSL bash 路径用 /mnt/c/...）
- 默认路径零行为变化：`--backend` 默认 hf
- 不修改 HF 推理路径（作为一致性对照基准保留）

---

### Task 0: M0 可行性尖峰——vLLM 跑通最小推理并与 HF 输出一致（整个计划的 gate）

**Files:**
- Create: `vllm_serve/minimal_spike.py`（临时验证脚本，gate 通过后可删）
- Create: `vllm_serve/README.md`（可选，记录版本与踩坑）

**Goal:** 在 GPU 服务器上把模型加载进 vLLM，用 1 图 + 1 prompt 跑通生成，输出与 HF 基线一致。

- [ ] **Step 1: 服务器环境准备**
  - `python check_flash_attn.py` 确认 flash_attn 可用（无则按指引安装）
  - 确认 vLLM 版本：`pip index versions vllm`，选与 torch 2.10/transformers 5.13 兼容的最新稳定版；记录实际版本到 `pyproject.toml` 的 optional-dependencies（新增 `vllm` 组）
- [ ] **Step 2: 最小注册尝试**
  - 先试 vLLM 对 Qwen3_5 系模型的直接加载（`--trust-remote-code`）；失败则按 Task 1 的自定义注册走最小路径（vision tower 除外，仅验证 LLM 主干能跑：无图、纯文本生成）
- [ ] **Step 3: 一致性验证**
  - 同一 prompt（含图像，走自定义多模态路径），vLLM 输出与 `vlx_seek_worker.predict` 的 HF 输出对比（采样 seed 固定、temperature=0），要求检测结果逐字一致
  - 不一致则记录差异来源（数值误差 vs 逻辑差异），判断是否可接受
- [ ] **Step 4: Gate 判定**
  - 通过（输出一致且速度不劣化）→ 继续 Task 1；不通过 → 上报人工，评估回退 SGLang 或 HF 自定义 prefix-KV
  - 产出：报告（环境版本、输出对比、gate 结论），commit 到 `docs/superpowers/plans/` 下记录

---

### Task 1: 自定义多模态模型注册（vision tower + object features）

**Files:**
- Create: `vllm_serve/vlx_seek_vlm.py`（`@multimodal_model` 注册类 + `MultiModalProcessor`）
- Create: `vllm_serve/vlx_seek_input_mapper.py`（HF mm_utils 图像预处理的 vLLM 复刻）
- Modify: `vllm_serve/__init__.py`

**Goal:** 把 `modeling_vlx_seek_1_5.py` 的多模态路径移植进 vLLM，注册后可接收 `images/images_aux/image_grid_thws/bbox_list` 输入。

- [ ] **Step 1: 移植 encode_images / encode_objects**
  - 从 `VLXSeek1_5Model.forward`（modeling 文件 145-240 行）提取图像编码与 object 特征路径，封装为 vLLM 图内模块
- [ ] **Step 2: 复刻输入预处理**
  - 对照 `mm_utils.py` 与 `VLXSeekWorker._prepare_image_inputs`（worker 189 行）实现 `MultiModalProcessor`，输出与 HF 完全一致的 `images/images_aux/image_grid_thws`
- [ ] **Step 3: 一致性单测（服务器）**
  - 1 图 + 固定 seed，vLLM vs HF 逐层对比图像特征 embedding（`torch.max(|diff|)` < 1e-3）
- [ ] **Step 4: Commit**
  - `feat(vllm): register custom multimodal model for VLXSeek`

---

### Task 2: prefix caching 验证

**Files:**
- Modify: `vllm_serve/vlx_seek_vlm.py`（如需要为 APC 调整）
- Create: `vllm_serve/bench_prefix_cache.py`

**Goal:** 验证同 crop 的 14 个共享前缀请求确实命中 APC。

- [ ] **Step 1: 基准脚本**
  - 构造 N 个共享前缀请求（同图同 objects、不同类别后缀），开启 `enable_prefix_caching`，用 vLLM 日志/`--vllm_engine_args` 观察 prefix cache 命中
- [ ] **Step 2: 测量**
  - 记录：单请求 prefill token 数、cache 命中 block 数、TTFT（首 token 延迟）；对比关闭 APC 的基线
- [ ] **Step 3: 验收**
  - TTFT 显著下降（共享前缀越长越明显）；命中率 > 80%；commit `feat(vllm): prefix cache benchmark`

---

### Task 3: VLXSeekVLLMWorker——统一 worker 接口

**Files:**
- Create: `vllm_serve/vlx_seek_vllm_worker.py`
- Modify: `vlx_seek_worker.py`（抽出公共接口，可选）或直接由 distill 脚本 `--backend` 选择

**Goal:** 新 worker 实现与 `VLXSeekWorker` 相同接口（`detect / detect_multi_prompt / encode_image_cache / clear_image_cache / log_timing`），distill 脚本切到 vLLM 后端时行为一致。

- [ ] **Step 1: 引擎封装**
  - `LLM`/`AsyncLLM` 单例封装（模型加载一次）；SamplingParams 映射（temperature/top_p/repetition_penalty/max_tokens）
  - stopping 适配：`KeywordsStoppingCriteria` → vLLM `stopping_criteria`
- [ ] **Step 2: 结果解析**
  - 复用现有 `<ground><objects><objN>` 解析逻辑（从 worker 提取公共函数或直接复用），vLLM 输出转同一 `result_bbox_list` 格式
- [ ] **Step 3: 一致性回归（服务器）**
  - 同图同 prompt，`--backend vllm` 与 `--backend hf` 输出一致（固定 seed）；`--log-timing` 在 vLLM 后端输出每次请求耗时
- [ ] **Step 4: Commit**
  - `feat(vllm): vLLM worker with unified interface`

---

### Task 4: 批量调度与多卡

**Files:**
- Modify: `vllm_serve/vlx_seek_vllm_worker.py`（batch submit）
- Modify: `distill/generate_pseudo_labels.py`（`--backend`、batch 提交逻辑）

**Goal:** 单图 280 请求一次提交，引擎并发调度；tensor parallel 用满 8 卡。

- [ ] **Step 1: 批量提交**
  - 一个 crop 的 14 个请求（或全图 280 个）作为一批 submit；引擎侧 continuous batching 自动调度
- [ ] **Step 2: 多卡**
  - `tensor_parallel_size=8`（或 4×2 与切片分片正交组合）；验证显存（48G×8）足够
- [ ] **Step 3: 端到端回归**
  - 一张大图完整跑 `generate_pseudo_labels.py --backend vllm`，与 hf 基线对比：输出一致 + 总耗时
- [ ] **Step 4: 收益测量与文档**
  - 记录加速比（对照 ~50 分钟基线）；`--log-timing` 数据归档；commit

---

### Task 5: 收尾——回归、部署与文档

- [ ] **Step 1: 双后端回归**：`--backend hf` 默认路径零变化（改动前后输出一致）；`--backend vllm` 完整流程通过
- [ ] **Step 2: 部署文档**：`docker/` 或服务器一键安装脚本（vLLM 版本锁定、flash_attn 校验）
- [ ] **Step 3: 最终整体代码审查**（subagent-driven-development 流程）并合并

## 风险与回退

| 风险 | 影响 | 应对 |
|---|---|---|
| vLLM 与 torch 2.10/transformers 5.13 不兼容 | M0 无法进行 | 锁定可用版本组合；必要时与项目 pyproject 对齐升级 |
| 自定义多模态注册复杂度过高 | M0/Task1 延宕 | 先跑纯 LLM 主干验证价值；不行则回退 HF 自定义 prefix-KV（改造成本中等，只解决重复 prefill） |
| prefix caching 未命中 | 收益减半 | 检查共享前缀字节一致性；用 hash 前缀调试日志定位 |
| 输出一致性差异 | 伪标签质量风险 | 固定 seed + temperature=0 回归；差异超阈值则逐 token 对比定位 |

## 收益预期（供验收参考）

- M0-Task1 达成（一致性）+ Task2 达成（APC 命中）→ Task3 单 crop 14 请求批量：TTFT 复用后总时间预计降 5-10×
- Task4 全图批量 + 8 卡：预计 50 分钟 → 2-10 分钟（受 decode 总 token 数下界约束）
- 验收基线：`--backend vllm` 全图耗时 + `--log-timing` 分布 vs hf 基线

## 不做的事

- SGLang 迁移（仅当 vLLM gate 失败时评估）
- HF 自定义 prefix-KV（仅当 vLLM gate 失败时回退）
- WeDetect proposals 移入 GPU（保持 Python 侧）
