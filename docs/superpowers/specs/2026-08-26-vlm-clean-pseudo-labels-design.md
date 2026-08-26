# VLM 伪标签清洗步骤设计（clean_pseudo_labels.py）

日期：2026-08-26
状态：已确认（用户批准）

## 1. 背景与目标

步骤4（`generate_pseudo_labels.py`）输出的伪标签存在两类噪声：

1. **滑窗重叠重复框**：裁剪推理默认 10% 重叠，同一目标在相邻切块各检一次，合并后写入两条近似重复标注，且合并没有跨块去重。
2. **误检框**：VLX-Seek 从 WeDetect 候选框中选择，存在幻觉/错标（false positive）。

本步骤插入在步骤4与步骤5之间，作为**可选的步骤4.5**：用外部多模态模型（用户通过 vLLM 部署的 Qwen3-VL 8B，OpenAI 兼容接口）对伪标签逐框验证过滤，输出更干净的 COCO 供微调使用。

## 2. 已确认的设计决策

| 决策点 | 结论 |
|---|---|
| 集成方式 | 独立脚本 `distill/clean_pseudo_labels.py`（一步一脚本模式，清洗一次可反复用于多次训练） |
| 验证粒度 | 逐框裁剪验证：每个标注框裁出局部小图单独判断 |
| 处置策略 | 默认仅删除不合规框；修正（relabel）不在本期范围，决策日志结构预留扩展 |
| 重复框 | 内置本地 IoU-NMS 去重（`--no-dedup` 可关），去重先于 VLM 验证以节省调用量 |
| 并发模型 | 线程池 + `requests`（与仓库现有依赖一致，不引入 httpx/aiohttp） |

## 3. CLI 接口

```bash
python distill/clean_pseudo_labels.py \
  --coco-json distill/data/pseudo_labels.json \
  --image-dir distill/data/images \
  --output distill/data/pseudo_labels.cleaned.json \
  --base-url http://localhost:8000/v1 \
  --model qwen3-vl-8b \
  --concurrency 16
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--coco-json` | 必填 | 输入伪标签 COCO |
| `--image-dir` | 必填 | 图像目录（COCO `file_name` 相对此目录） |
| `--output` | `<input>.cleaned.json`（同目录） | 清洗后 COCO；永不覆盖输入文件 |
| `--base-url` | `http://localhost:8000/v1` | OpenAI 兼容服务地址 |
| `--model` | 无默认；取环境变量 `CLEAN_VLM_MODEL`，仍缺失则启动报错 | 模型名（vLLM `--served-model-name`） |
| `--api-key` | 环境变量 `OPENAI_API_KEY` | 本地 vLLM 通常不需要 |
| `--concurrency` | 16 | 线程池大小 |
| `--iou-threshold` | 0.55 | 同图同类 NMS 的 IoU 阈值 |
| `--no-dedup` | 关 | 跳过去重阶段 |
| `--max-side` | 512 | 裁剪图最长边超过则等比缩小 |
| `--min-crop-pad` | 0.12 | 裁剪框外扩比例（相对框长边），并钳制最小边 ≥32px |
| `--decision-log` | `<output>.decisions.jsonl` | 决策日志路径（断点续跑依据） |
| `--report` | 无（仅终端打印） | 统计报告 JSON 输出路径 |
| `--max-retries` | 3 | 单框请求最大重试次数 |
| `--timeout` | 120 | 单次请求超时秒数 |

## 4. 处理流程

### 阶段0：加载与校验

- `coco_utils.load_coco` 读入；构建 `category_id → name` 映射。
- 校验每条标注引用的 `image_id`/`category_id` 存在，缺失则报错退出。

### 阶段1：本地去重（零 API 成本）

- 每张图内、按类别分组；组内按框面积**降序**排序（保留更大框：滑窗重复对中较大者通常边界更完整），逐一与已保留框算 IoU，> `--iou-threshold` 判为重复删除。
- 判定结果即时写入决策日志（`verdict: "dedup"`），不计入后续 VLM 调用总量。

### 阶段2：逐框 VLM 验证

- 按 `file_name` 分组待验证标注；同图的框共用一次图片读取/解码（CPU 同步完成），随后每框构造一个验证任务提交线程池。
- 裁剪：框外扩 `min_crop_pad × max(w, h)`，最小边不足 32px 时以中心扩展至 32px，越界部分钳制到图像边界。
- 编码：PIL crop → RGB → 最长边 > `max_side` 则等比缩放 → JPEG quality 85 → base64 data URI。
- 请求：`POST {base-url}/chat/completions`，`temperature=0`，`max_tokens=8`（强制短回答）。
- 判定解析：回复去空白后须以「是」或「否」开头；否则视为失败进入重试。

提示词模板：

```
system: 你是严格的图像内容审核助手，只回答"是"或"否"。
user:   （附裁剪图）这张从大图裁出的局部区域中，主要拍摄对象是否属于类别「{中文类名}」？只回答"是"或"否"。
```

`{中文类名}` 直接使用伪标签 COCO 的 `categories.name`（描述性中文文本，Qwen 中文理解无障碍）。

### 阶段3：写出与报告

- 「否」→ 删除；「是」→ 保留；重试耗尽仍失败 → **保守保留**（fail-open，`verdict: "error_keep"`）。
- 输出 COCO：复制 `images`/`categories`，过滤 `annotations`，annotation id 重新连续编号；**0 标注图像保留**（负样本对训练有价值）。
- 终端打印统计报告；`--report` 时另写 JSON。报告字段见 §8。

## 5. 进度展示

- 两级 `tqdm`（走 stderr，与现有脚本一致，不污染 stdout 报告）：
  - 外层按图像推进（覆盖阶段1+2 全过程）；
  - 内层框级进度，`total = 去重后待验证框数`，每判定一框 `update(1)`，后缀实时显示 `已删 X / 保留 Y / 失败 Z`。
- ETA 由 tqdm 按实际吞吐自动估算（框/秒），满足"预计完成时间"需求。
- 断点续跑兼容：重跑时已判定框从日志回放、瞬时计入进度，`total` 扣除已完成数，ETA 只反映剩余真实调用量。

## 6. 断点续跑与审计日志

- 决策日志 JSONL，逐条判定即时落盘：

```json
{"file_name": "a.jpg", "ann_id": 17, "category_id": 3, "category_name": "水面漂浮的垃圾或废弃物", "verdict": "keep|delete|dedup|error_keep", "raw_reply": "是", "elapsed_ms": 412}
```

- 首行为元信息行 `{"_meta": {"model": ..., "coco_json": ..., "iou_threshold": ...}}`；重跑时若 meta 与当前参数不一致则告警并忽略旧日志重新开始（避免陈旧判定污染新结果）。
- 重跑流程：读日志 → 以 `(file_name, ann_id)` 建索引 → 已记录的框直接复用判定 → 只对缺失框发请求 → 写出时合并全部判定。

## 7. 错误处理

| 场景 | 行为 |
|---|---|
| 服务不可达（连接拒绝/域名解析失败） | **仅限首个请求**因该类错误失败时快速退出，提示检查 `--base-url`；避免整批空转 |
| 运行中零星网络错误/超时/5xx | 指数退避重试至 `--max-retries`，仍失败 fail-open 保留并计数 |
| 回复无法解析为 是/否 | 同上（计入重试次数） |
| 图片文件缺失 | 该图所有框记 `error_keep`，警告列出 |
| 输出路径与输入相同 | 启动即报错退出 |

## 8. 输出与统计报告

报告（终端 + 可选 JSON）包含：

- 总框数 / dedup 删除数 / VLM 删除数 / 保留数 / error_keep 数
- 各类别删除量 Top-N（定位系统性误检类别）
- 总耗时、平均吞吐（框/秒）、VLM 调用次数与失败率
- 图像级统计：总图数、清零标注的图数（保留为负样本）

## 9. 文档更新（README.md）

- 目录结构补一行 `clean_pseudo_labels.py # 步骤4.5：VLM 清洗伪标签（去重+逐框验证）`
- 环境要求补：仅需 `requests` + 可达的 OpenAI 兼容 vLLM 服务
- 新增「步骤4.5（可选）：VLM 清洗伪标签」小节：部署前提、命令示例、参数表、决策日志/断点说明
- 步骤5 微调示例注释注明可用 `.cleaned.json` 作为输入

## 10. 测试与验收方式

1. `py_compile` 通过。
2. 离线管线测试（本机无 GPU/vLLM）：指向不可达 `--base-url` 运行 examples 数据——预期全部 fail-open 保留、去重正常生效、决策日志完整、输出 COCO 经 `load_coco` 可读且 id 连续。
3. 决策日志回放测试：同一命令重跑两次，第二次 VLM 请求数为 0，输出一致。
4. 真实验收由用户在其 vLLM 服务上小批量试跑（如 `--max-side` 不变、抽几百张），人工抽删正确率后再全量。

## 11. 非目标（YAGNI）

- 不实现自动改标（relabel）：决策日志 schema 已预留 `verdict` 扩展位，未来可加 `"relabel"`
- 不做跨图全局去重、不做置信度打分、不自动调 NMS 阈值
- 不支持非 OpenAI 兼容接口（vLLM/TGI 等均提供兼容层）
