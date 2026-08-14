# VLX-Seek vLLM 后端：实现功能、与 HF 的区别、优缺点

> 本文档总结 `vllm_serve/` 模块已实现的功能，并与 HF 后端（`VLXSeekWorker`）
> 逐项对比，说明设计取舍与已知限制。数据流细节见 [data_flow.md](./data_flow.md)。

---

## 1. 已实现功能

### 1.1 推理接口（与 HF `VLXSeekWorker` 同签名）

| 接口 | 说明 | HF 对照 |
| --- | --- | --- |
| `predict(image, question, bbox_list, ...)` | 单图单问（可选 object 提示） | 同签名 |
| `predict_batch(requests, ...)` | 一次 `llm.generate` 批量提交 | HF 为串行 for 循环 |
| `run_task(image, task, text, lang, bbox_list, ...)` | 任务模板 prompt + 推理 | 同签名 |
| `detect / ground / count` | 检测 / 单目标 grounding / 计数 | 同签名 |
| `detect_multi_prompt(image, bbox_list, category_batches, ...)` | 多批类别一次提交，并集输出 | HF 逐批调用 |
| `encode_image_cache / clear_image_cache` | no-op（引擎常驻，无需显式缓存） | HF 缓存图像张量 |

返回 dict 结构与 HF 完全一致：`answer / result_bbox_list / prompt_tokens /
completion_tokens / elapsed`。

### 1.2 模型集成（`vllm_serve/vlx_seek_vlm.py`）

- **`VLXSeek1_5ForCausalLM`**：继承 vLLM 原生 `Qwen3_5ForConditionalGeneration`，
  语言主干 / 采样 / KV 缓存 / 引擎集成全部复用；
- **视觉栈替换**：`vision_tower`（Qwen3_5_VlVisionTower）+ `mm_projector` +
  `vision_tower_aux`（CRadioV4AuxEncoder）+ `object_vp_extractor`（HFRE）+
  `mm_projector_aux`，全部复用项目 nn.Module，属性名与 HF 完全一致；
- **`embed_multimodal`**：图像 + object 双通道编码，合并为一个 tensor
  返回（行序 = prompt 占位符顺序，由 is_multimodal 掩码 scatter）；
- **`VLXSeekMultiModalProcessor`**：透传 `images_aux` / `bbox_list`，并把
  `<objfeat>`（248181）并入 is_multimodal 掩码；
- **权重加载**：`AutoWeightsLoader` + `WeightsMapper`（键名前缀改写），
  C-RADIOv4 的 buffer（input_conditioner / summary_idxs）单独处理；
- **插件注册**（`plugin.py`）：config + 模型类幂等注册，覆盖所有 worker 子进程。

### 1.3 已修复的关键问题

1. **多 prompt update 只应用第一个**（vLLM 0.17 `_find_matches` 行为）：
   独立追加的 `<objfeat>` replacement 永远不生效，导致 248181 不进掩码、
   对象嵌入被 `masked_scatter_` 静默丢弃（输出发散为 `'小汽车None'`）。
   改为**单个合并 replacement**（target 覆盖 image_pad 段 + objfeat 行）。
2. **`outputs[0].text` 默认 `skip_special_tokens=True`**：`<ground>`/`<objN>`
   等特殊 token 在 decode 时被丢弃，只剩类别名，`_build_result_bbox_list`
   解析为空。改为 `_decode_answer()`（token_ids + `skip_special_tokens=False`）。
3. **`images_aux` / `bbox_list` 的 batch 维度规整**：兼容 vLLM batched
   字段切分后的多种形状（`[C,H,W]` / `[1,C,H,W]` / `[1,N,4]` / `[N,4]`）。

### 1.4 验证手段

- `test_consistency.py`：HF / vLLM 同图同输入（temperature=0 贪心解码）对比
  `detect` / `detect_multi_prompt` 的 answer 与 result_bbox_list；
- `test_object_features.py`：object features 生成冒烟；
- `test_vllm.py` / `test_vllm_worker.py` / `minimal_spike.py`：引擎加载冒烟；
- `bench_prefix_cache.py`：同图同 object、不同类别后缀的批量请求基准。

---

## 2. 与 HF 的区别

### 2.1 推理执行方式

| 维度 | HF（`VLXSeekWorker`） | vLLM（`VLXSeekVLLMWorker`） |
| --- | --- | --- |
| 模型加载 | 每次构造 worker 加载权重 | 引擎常驻，一次加载 |
| 批量推理 | `predict_batch` 串行 for 循环 | 一次 `llm.generate` 连续批处理 |
| 图像缓存 | `encode_image_cache` 显式缓存张量 | no-op（引擎按请求自动编码） |
| 停止条件 | `KeywordsStoppingCriteria("<\|im_end\|>")` | `SamplingParams.stop=["<\|im_end\|>"]` |
| 采样 | `do_sample + temperature/top_p/repetition_penalty` | `SamplingParams` 等价参数 |
| 输出 decode | `tokenizer.decode(ids, skip_special_tokens=False)` | 需手动 `_decode_answer()` 模拟 |

### 2.2 Prompt 与占位符

| 环节 | HF | vLLM |
| --- | --- | --- |
| 图像占位符 | `<image>` → 手工展开为 248056 × N（`_expand_multimodal_tokens`） | `<\|image_pad\|>` → processor 展开 |
| object 占位符 | `<objfeat>` → 248181（手工替换） | `<objfeat>` → 248181（PromptReplacement，进掩码） |
| 文本模板 | `<\|vision_start\|><image><\|vision_end\|>\n<obj0><objfeat>...` | 同构，`<image>` 换成 `<\|image_pad\|>` |

### 2.3 图像/object 预处理位置

- HF：主塔（Qwen3.5 processor）与 aux 塔（CLIPImageProcessor）预处理都在
  worker 内提前完成，张量随 `generate` 传入；
- vLLM：主塔预处理在 processor 层（`_call_hf_processor`），aux 预处理在
  worker 内完成（`_make_request`），经 `mm_processor_kwargs` 透传。

### 2.4 一致性结果（2026-08-14，temperature=0）

同一张图 / 同一 crop / 同一 5 个候选框 / `小汽车`：

| 方法 | HF answer | vLLM answer | result_bbox_list |
| --- | --- | --- | --- |
| detect | `<ground>小汽车</ground><objects><obj1><obj2><obj4><obj3></objects>` | 完全相同 | 4 items（一致） |
| detect_multi_prompt | 同上（顺序因采样略异） | `<obj0><obj1><obj2><obj4>` | 4 items（一致） |

> 采样细节：vLLM 与 HF 的 sampling RNG / 实现细节不同，`temperature>0` 时
> `<objN>` 顺序可能有差异；`temperature=0` 的贪心路径在回归测试中一致。

---

## 3. 优缺点

### 3.1 优点

- **吞吐与延迟**：引擎常驻 + continuous batching，多 prompt 同图请求共享
  图像/object 前缀 KV，批量检测（`detect_multi_prompt`）显著快于 HF 串行；
- **分布式扩展**：vLLM 原生支持 `tensor_parallel_size`、prefix caching、
  长文本 / chunked prefill，可平滑扩展多卡；
- **接口零成本切换**：`--backend hf|vllm` 一键切换，distill 流程无需改动；
- **内存复用**：权重 / KV cache 常驻，避免 HF 每 worker 重新加载。

### 3.2 缺点与限制

- **依赖版本锁定**：针对 vLLM **0.17.0** 的 `_find_matches` / processor 内部
  行为定制（合并 replacement 等），升级 vLLM 需要回归验证；
- **必须 `enforce_eager=True`**：hybrid mamba 架构下 CUDA graph 捕获不匹配，
  牺牲编译优化；
- **`max_model_len` 限制**：整张大图直出会因视觉 token 超长失败，需先
  crop/缩放（与 HF 侧相同的视觉规模约束）；
- **采样不完全等同**：`temperature>0` 时与 HF 的输出顺序可能有差异，
  依赖贪心解码的回归测试用 `temperature=0`；
- **`outputs[0].text` 陷阱**：必须手动 `_decode_answer()`，直接使用
  `.text` 会丢失 `<ground>`/`<objN>` 标签（已封装，勿绕过）；
- **首次启动慢**：引擎 profile + KV cache 初始化 + 权重加载需数秒～数十秒，
  单次调用场景（非批量）开销高于 HF 直出。

---

## 4. 参考

- 数据流 / 占位符 / 嵌入合并：[data_flow.md](./data_flow.md)
- 实现文件：[vlx_seek_vlm.py](./vlx_seek_vlm.py)、[vlx_seek_vllm_worker.py](./vlx_seek_vllm_worker.py)、[plugin.py](./plugin.py)
- 一致性回归：[test_consistency.py](./test_consistency.py)
- HF 对照实现：`vlx_seek_worker.py`
