# 伪标签质量改进 + 自改进迭代 设计

日期：2026-08-27
状态：用户已批准（待写入实现计划）

## 背景

当前 `distill/clean_pseudo_labels.py` 用 VLM（Qwen3.8-27B，vLLM 部署）逐框清洗伪标签，存在两类质量缺陷：

1. **小目标看不清**：裁剪策略「框外扩 12% + 最小 32px + 最长边 512px 等比缩小」，导致 30×30 的小目标最终只剩 32×32~40×40 的裁剪图，VLM 看不到目标周围环境，几乎只能"判否"。
2. **注意力不聚焦**：prompt 用纯文本注入目标框坐标 `(x, y, w, h)`，但 VLM 对"图上的坐标数字"的对应关系理解能力有限，实际不太能把注意力集中到指定区域。

此外用户的最终目标是把伪标签-训练流程从**单向蒸馏**升级为**自改进迭代**（SGDR 风格）：
```
d0(教师) → m0 → d1(m0 推理) → d2(清洗) → m1 → d3 → ... → mN
```
每一轮让 student 模型主导提案，再用清洗脚本剔除噪声，逐步摆脱对教师伪标签的依赖。

## 已确认的设计决策（用户拍板）

| 决策点 | 结论 |
|---|---|
| 裁剪图最小尺寸 | 640×640（目标居中，借鉴 cv_utils `crop_rect` 的越界反推；不用填充） |
| 目标框标注 | 图上画红框（4px），prompt 文本说明"红框内为待审核目标"，移除数值坐标注入 |
| 清洗时送入 VLM 的 max_side | 960（只缩不放；640² 裁剪保持原分辨率；2240×1680 裁剪缩到 960×720） |
| 自改进的 d_k 来源 | m_{k-1} 整图推理（**不再**调用 VLX-Seek 教师） |
| 训练分辨率 | 维持现状（默认 imgsz=640；用户根据效果调 960/1280；已启用随机裁剪增强兜底多尺度） |
| 迭代终止条件 | 达到 `--max-rounds` 或连续 N 轮 mAP50 无提升（`--early-stop-no-improve`，默认 2） |
| 实现形态 | 单体编排脚本 `distill/self_improve.py`，in-process 调用清洗/训练函数（非 subprocess） |
| 决策日志兼容 | 清洗脚本加上 `min_crop_size` 等新参数后，旧决策日志强制失效（参数一致性校验已有） |

## 第一部分：`clean_pseudo_labels.py` 改进

### 1.1 裁剪逻辑更换

新函数 `crop_decode`（取代之前的 `crop_encode`，名字含 decode 强调"原图解码头寸"）：

```python
def crop_decode(image, bbox_xywh, min_crop_size, max_side):
    """以目标为中心裁剪，宽高取 max(框, min_crop_size)，越界时反推到图内，
    再按需等比缩小到 max_side，画红框，编码 JPEG。

    返回 (jpeg 字节流, 目标框在裁剪图局部坐标系的 xywh)。"""
    x, y, w, h = bbox_xywh
    img_w, img_h = image.size
    # 1) 计算裁剪窗口（借鉴 cv_utils.crop_rect：先撑到 min，越界反推）
    cw = max(w, min_crop_size)
    ch = max(h, min_crop_size)
    cx, cy = x + w / 2, y + h / 2
    left = max(0, int(round(cx - cw / 2)))
    top = max(0, int(round(cy - ch / 2)))
    right = min(img_w, left + cw); bottom = min(img_h, top + ch)
    left = right - cw; top = bottom - ch  # 越界反推
    crop = image.crop((left, top, right, bottom))
    # 2) 等比缩放到 max_side（只缩不放）
    scale = 1.0
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    # 3) 目标框换算到局部坐标（含缩放），并在此处画红框
    bx = (x - left) * scale; by = (y - top) * scale
    bw = w * scale; bh = h * scale
    cw_px, ch_px = crop.size
    draw = ImageDraw.Draw(crop)
    px = max(0, min(cw_px, round(bx))); py = max(0, min(ch_px, round(by)))
    pw = max(1, min(cw_px - px, round(bw))); ph = max(1, min(ch_px - py, round(bh)))
    for i in range(4):  # 4px 线宽
        draw.rectangle([px + i, py + i, px + pw - i - 1, py + ph - i - 1], outline=(255, 0, 0))
    # 4) 编码
    buf = io.BytesIO(); crop.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), (px, py, pw, ph)
```

**边界条件**：当原图分辨率 < `min_crop_size` 时（如 500×500 图），启动时一次性报错并退出，避免静默产出不完整裁剪图。

### 1.2 Prompt 改造

新 prompt（无坐标数值）：

```python
SYSTEM_PROMPT = '你是严格的图像内容审核助手，只回答"是"或"否"。'
USER_PROMPT = ('这张图中红色矩形框已标注了待审核目标。'
               '请判断红框内的主要拍摄对象是否属于类别「{name}」。'
               '只回答"是"或"否"。')
LEGACY_USER_PROMPT = '这张从大图裁出的局部区域中，主要拍摄对象是否属于类别「{name}」？只回答"是"或"否"。'
```

`--no-draw-box` 模式下回退到 `LEGACY_USER_PROMPT`（保持向后兼容）。

### 1.3 新增 CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--min-crop-size` | 640 | 裁剪最小边（像素）；原图任一边小于该值时启动报错 |
| `--max-side` | **960**（原 512） | 裁剪图最长边超过则等比缩小（只缩不放） |
| `--box-color` | red | 目标框颜色，可选 `red` / `yellow` / `off` |
| `--no-draw-box` | 关 | 不画红框、回退到旧 prompt（兼容旧测试/旧数据） |

`min_crop_pad` 参数**保留但仅在新逻辑里失效**（被 640 触底压过）；为兼容性暂不删除，文档标注 deprecated。

### 1.4 决策日志 meta 字段扩展

`DecisionLog._meta` 增加 `min_crop_size / box_color / max_side / no_draw_box` 四个字段，参数变更时旧日志自动失效并删除（沿用现有"参数不一致则重新开始"机制）。

### 1.5 测试更新（`distill/tests/test_clean_pseudo_labels.py`）

`CropEncodeTest` 重命名为 `CropDecodeTest`（旧函数不再存在），用例改为：

- `crop_decode(img_1000, [0, 0, 100, 100], min_crop_size=640, max_side=960)` → 裁剪区域左上 `(0, 0)`，目标框局部坐标 `(0, 0, 100, 100)`
- `crop_decode(img_4000, [1000, 800, 2000, 1500], 640, 960)` → 裁剪宽 2240×1680，缩到 960×720
- `crop_decode(img_800, [0, 0, 30, 30], 640, 960)` → crop `[0, 0, 640, 640]`
- `crop_decode(img_500, [10, 10, 100, 100], 640, 960)` → 启动报错（原图 < 640）
- 关闭 `--no-draw-box` 时，prompt 是 LEGACY 文案；开启时新文案不含坐标数值
- 旧 XML mock vLLM 测试仍跑，回归响应分类路径

## 第二部分：自改进迭代 `distill/self_improve.py`

### 2.1 整体流程

```
输入: 
  --init-coco-json        教师伪标签 COCO（d0）
  --image-dir             图像目录
  --init-weights          m0 权重（可空：留空时直接用预训练 yolov8s-worldv2.pt）
  --max-rounds            最大轮数（默认 3）
  --val-ratio             验证集比例（默认 0.1；seed=42 固定划分一次）
  --category-map          category_prompts.json 路径（必需）
  --base-url / --model    VLM 服务（清洗步骤用）
  --train-device          训练用 GPU（如 "0,1"）
  --infer-device          推理用 GPU（如 "0"）

Round 0:
  A0. 训练 m0:
      若 --init-weights 缺失 → 用预训练 yolov8s-worldv2.pt 在 d0 上训出 m0
      若 --init-weights 给了 → 在 d0 上热启动
  E0. 在固定 val 上评估 m0 → baseline mAP50

Round 1..N:
  Bk. m_{k-1} 整图推理（YOLOWorld.predict, imgsz=--imgsz, conf=--conf-thresh, iou=--nms-iou）
       → 框坐标还原回原图像素 + 类名（train_name）反查中文 category_id
       → 写 round_k/raw_d_k.json (COCO)
  Ck. clean_pseudo_labels(raw_d_k, --base-url, --model) → round_k/clean_d_k.json
       （独立跑清洗脚本，断点续跑）
  Dk. finetune_yolo_world(clean_d_k, m_{k-1}.pt 热启动, --train-device) → m_k.pt
  Ek. val 集上评估 m_k → 输出 mAP50 / mAP / 每类 AP

终止:
  - k == max_rounds
  - 或 连续 early_stop_no_improve 轮 mAP50 不差于上一轮
  - 或早期 m_ap50 退化（每类 AP 任一连续 2 轮下降 > 20% 仅告警不终止）
```

### 2.2 关键设计点

**a. 验证集（一次划分，永远不变）**

从 d0 按 `--val-ratio`、`seed=42` 切一个 image-level val（基于 `file_name` 切，避免跨图同类框被拆开）。val 在**所有轮均不参与训练**：

- Round 0 训练 m0：用 d0 的 train 部分
- Round k 训练 m_k：用 clean_d_k 的 train 部分
- Ek 评估：固定 val 集

**实现细节**：`split_coco` 已存在但只按 annotation 比例切。这里改用 image-level 划分（`coco_utils.py` 可能新增 `split_coco_by_image`，或复用现有逻辑再筛 annotation）。划分结果（val 的 file_name 列表）写到 `run_dir/split.json`，所有轮次复用，保证 Ek 口径一致。

**b. m_{k-1} 推理（步骤 Bk）**

```python
from ultralytics import YOLOWorld
model = YOLOWorld(last_pt)   # m_{k-1}
results = model.predict(image_path, imgsz=640, conf=0.30, iou=0.50, device="0", verbose=False)
for box in results[0].boxes.xyxy.cpu().numpy():  # letterbox 像素
    # 反 letterbox 回原图坐标（除 padding、除 scale、除 翻转）
    # train_name 反查 category_id
    # 写 COCO
```

注：letterbox 回原图的坐标换算与 YOLO 官方推理模块相同，是一段固定公式（`scale = min(640/H, 640/W, 1.0)`，`pad_x, pad_y` 居中补边，原图 `x = (x_lb - pad_x) / scale`）。

**c. 类别名映射（train_name ↔ 中文 category_id）**

`category_prompts.json` 已有 `prompt → 中文名` 与 `中文名 → train_name`（参考 `generate_prompts.py` 与 `apply_category_map`）。本步骤需要反向索引：`train_name → ci_id`。该索引已在 `apply_category_map` 构造过，直接复用同一构造逻辑。

**d. 断点续跑（runs 目录约定）**

```
self_improve_runs/
└── run_<ts>/
    ├── config.json                     # 全部参数 + 已完成轮数
    ├── split.json                      # val 的 file_name 列表（固定）
    ├── summary.json                    # 所有轮 mAP 折线（每次 Ek 后增量写）
    ├── round_0/
    │   ├── m0.pt                       # round 0 训练产物
    │   └── eval.json                   # baseline mAP
    ├── round_1/
    │   ├── raw_d1.json                 # m0 推理产物
    │   ├── clean_d1.json               # 清洗产物
    │   ├── decisions_d1.jsonl          # 清洗决策日志（断点续跑）
    │   ├── m1.pt
    │   └── eval.json
    └── ...
```

- 启动时：若 `run_dir` 已存在，读 `config.json` + 各轮内容恢复状态，跳到下一未完成步骤。
- 清洗续跑：用 `--decision-log` 复用原清洗脚本逻辑。
- 训练续跑：检查 `m_{k-1}.pt` 是否存在且 last epoch 数 = 期望 → 跳过；否则重训。**不实现**按 epoch 级续训（ultralytics `resume=True` 依赖 last.pt 元数据，跨 venv 不稳），整轮重训即可。
- 推理续跑：若 `raw_d_k.json` 已存在且 image 数 == 期望 → 跳过。

**e. 编排脚本（单体）**

`self_improve.py` 在所有轮次循环里**直接 import 现有模块的函数**（不是 subprocess），原因：
- `clean_pseudo_labels.run_pipeline(args, coco)` 已是函数级入口
- `finetune_yolo_world.prepare_dataset` + `YOLOWorld.train` 也可拆开调用
- 这样异常处理、日志、进度条都能统一在一个进程里

注：训练部分的 `YOLOWorld` 在 ultralytics 内部会起 DDP 子进程，但没有"切断父进程"的副作用。

**f. GPU 调度**

- Bk：用 `--infer-device`（默认单卡）
- Ck：纯 CPU（清洗脚本就是 CPU 线程池）
- Dk：用 `--train-device`（可 DDP 多卡）
- 串行执行，不需要 GPU 共享调度

**g. summary.json 结构**

```json
{
  "run_id": "run_20260827_1430",
  "rounds": [
    {"k": 0, "m_map50": 0.329, "m_map": 0.201, "ap_per_class": {"人群密集": 0.45, ...}, "delta_map50": null},
    {"k": 1, "m_map50": 0.412, "m_map": 0.251, "ap_per_class": {...}, "delta_map50": 0.083},
    ...
  ],
  "early_stop_at_round": null,
  "final_model": "round_3/m3.pt"
}
```

### 2.3 CLI 参数

```bash
uv run python distill/self_improve.py \
  --init-coco-json distill/data/pseudo_labels.json \
  --image-dir distill/data/images \
  --category-map distill/data/category_prompts.json \
  --init-weights yolov8s-worldv2.pt \
  --max-rounds 3 --val-ratio 0.1 --imgsz 640 \
  --conf-thresh 0.30 --nms-iou 0.50 \
  --epochs 30 --batch 32 \
  --train-device 0,1,2 --infer-device 0 \
  --base-url http://127.0.0.1:8101/v1 --model qwen3.8-vllm \
  --api-key "$OPENAI_API_KEY" \
  --run-dir self_improve_runs/run_$(date +%Y%m%d_%H%M)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--init-coco-json` | 必填 | d0 |
| `--image-dir` | 必填 | 原图 |
| `--category-map` | 必填 | train_name 映射 |
| `--init-weights` | yolov8s-worldv2.pt | m0 权重；缺省即预训练 |
| `--max-rounds` | 3 | 最大迭代轮 |
| `--val-ratio` | 0.1 | 验证集比例（image-level） |
| `--imgsz` | 640 | 推理分辨率（Yolo-World 必须 640 倍数） |
| `--conf-thresh` | 0.30 | 推理置信度阈值 |
| `--nms-iou` | 0.50 | 推理 NMS 阈值 |
| `--epochs` / `--batch` / `--optimizer` / `--lr0` | 同 finetune 脚本 | 训练超参 |
| `--train-device` | 0 | 训练卡（可 DDP 多卡） |
| `--infer-device` | 0 | 推理卡 |
| `--base-url` / `--model` / `--api-key` | 清洗脚本默认 | VLM 清洗端点 |
| `--run-dir` | `self_improve_runs/run_<ts>` | 输出根目录 |
| `--early-stop-no-improve` | 2 | 连续 N 轮 mAP50 不上升即停 |
| `--ioup-per-class-alert` | 20 | 每类连续 2 轮 AP 跌 > 20% 告警（不终止） |

### 2.4 错误处理

| 场景 | 行为 |
|---|---|
| Bk 推理某图异常 | 该图 d_k 标注为空，写 `raw_d_k.json` 后继续（与 generate 脚本同） |
| Ck 清洗服务不可达 | 沿用清洗脚本策略：首请求拒连即快速退出整轮；零散错误 fail-open 保留 |
| Dk 训练 OOM | 直接抛错，提示减小 batch / 用 AutoBatch 等 |
| Ek 评估异常 | 视为该轮失败，summary 里 delta=null，进下一轮 |
| 中途 Ctrl+C | 保存当前 summary.json，下次重跑同一 run_dir 继续 |

## 第三部分：非目标与风险

### 非目标（不做）

- 每轮不调用 VLX-Seek 教师（那会变回方案 B，超出本期 scope）
- 不做跨 run 蒸馏 / multi-teacher 融合
- 不自动调超参（HPO）
- 不实现训练 epoch 级续训（整轮重训即可）
- 不实现 val 增强 / 数据混合

### 已知风险

1. **长尾级别联退化**：round N 后某些 rare 类 AP 可能跌到 0。缓解：`--ioup-per-class-alert` 告警 + 用户中断 or 教师兜底（plan B 留给后续）。
2. **数据单调递减**：VLM 清洗是"减法"，d_k 帧数会逐轮下降。这是自愈方向（剔除噪声）但需用户知情。
3. **val 只反映对剩余伪标签拟合**：不等价于真实精度。建议最终轮人工抽检 50-100 张。
4. **640² 下 small object 仍可能失败**：m_{k-1} 在 640 letterbox 下对 30×30 框可能检不出；这是 YOLO-World 自身多尺度能力问题，本期不解决（用户已明确用随机裁剪增强兜底）。

## 第四部分：交付物清单

1. `distill/clean_pseudo_labels.py` — 裁剪逻辑重写 + 画红框 + 新 prompt + 新参数
2. `distill/tests/test_clean_pseudo_labels.py` — 新增 `CropDecodeTest` 覆盖上述裁剪/prompt 场景
3. `distill/self_improve.py` — 新增；~300-500 行
4. `distill/coco_utils.py` — 视需要新增 `split_coco_by_image`（image-level 划分）
5. `distill/README.md` — 新增「步骤 4.5 改进（min-crop-size / 红框）」小节 + 新增「步骤 6：自改进迭代 (self_improve.py)」小节
6. `docs/superpowers/specs/2026-08-27-pseudo-label-iteration-design.md` — 本 spec

不修改：
- `distill/generate_pseudo_labels.py`（教师生成脚本，本期不动）
- `distill/finetune_yolo_world.py`（被 self_improve in-process 复用其函数，不改 API）
- `vlx_seek_worker.py` / `vllm_serve/*`（不在本期范围）

## 验收方式

1. **clean_pseudo_labels.py 单测**：新增 `CropDecodeTest` 全绿；旧 `CropEncodeTest` 重命名为 `CropDecodeTest` 或删除（不再覆盖）
2. **self_improve.py 离线冒烟**：用 `distill/examples`（2 张图 / 2 类）跑 1 轮：
   - 清洗不可达（mock 一个错的 base-url）→ 全部 fail-open；清洗→训练链路正确推进
   - 清洗可达（mock 一个返回"是/否"的服务，已有测试里的 mock harness 可复用）→ mAP 链路正确端到端跑通
   - 中断后重跑同一 run → 跳过已完成步骤
3. **真实小批量**：用户用 5 张图、5 个类别、2 轮在 vLLM 上验证每轮 eval 数值与 summary.json 折线
4. **py_compile + 无未使用 import/lint** 通过
