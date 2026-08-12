# 设计文档：提示词分批推理 + 图片编码器复用

**日期**: 2026-08-12
**状态**: 已确认，待实现

## 1. 背景与动机

当前 `distill/generate_pseudo_labels.py` 通过 `--categories` 传入所有检测类别（数百个），组成一条
超长 prompt 交给 VLX-Seek 推理。存在的问题：

1. **质量下降**：prompt 越长，VLX-Seek 模型的检测准确率越低。
2. **性能浪费**：对同一张图片（或同一裁剪块）用多组子提示词循环推理时，图片编码器
   （vision tower + mm_projector + object encoder）被重复执行 N 次，但输入图片完全相同。

本设计在 `generate_pseudo_labels.py` 中新增 `--prompt-batch-size` 参数，将长类别列表拆分为
多个子提示词循环推理；同时在模型层增加图片特征缓存机制，使同一张图的多次推理只编码一次图片。

## 2. 方案选择

### 方案 A：浅层分片（不改动模型层）

仅做类别拆分，每张图/裁剪块的每个子提示词都完整跑一次模型（含图片编码器）。

- 优点：实现简单，零风险
- 缺点：图片编码器被重复执行 N 次，在裁剪推理场景下浪费严重

### 方案 B：深层复用（改动模型层）— 已选定

在模型层增加缓存机制，预计算图片特征后缓存，后续 N 次推理复用。

- 优点：图片编码器只执行 1 次，裁剪推理场景下节省约 30-50% 图片编码时间
- 缺点：需改动模型 forward，有一定复杂度

**决策**：采用方案 B。

## 3. 架构设计

### 3.1 分层改动

```
┌─ generate_pseudo_labels.py ─────────────────────────────┐
│  --prompt-batch-size 30                                  │
│  类别拆分 → 子提示词列表 → 调用 detect_multi_prompt()     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌─ vlx_seek_worker.py ────────────────────────────────────┐
│  encode_image_cache()  →  预编码图片特征，写入模型缓存    │
│  detect_multi_prompt() →  循环调用 detect()  N 次        │
│  clear_image_cache()   →  清除缓存                       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌─ modeling_vlx_seek_1_5.py ──────────────────────────────┐
│  set_cached_image()   →  存储预计算的特征                │
│  clear_cached_image() →  清除                            │
│  forward() 分支        →  检测到缓存 → 跳过视觉编码       │
│  prepare_inputs_for_generation() → 缓存模式兼容          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 执行流程

1. 图片加载 → WeDetect 生成 proposals（一次）
2. `worker.encode_image_cache(image, boxes)` → 调用 `model.encode_images()` +
   `model.encode_objects()` → 写入 `model._cached_*`
3. 循环 N 次：`worker.detect(image, boxes, batch_N)` → 内部 `model.generate()` →
   `forward()` 检测缓存，直接使用预计算特征，跳过视觉编码
4. `worker.clear_image_cache()` 释放缓存

### 3.3 性能收益估算

假设 300 个类别，batch_size=30，每张图有 4 个裁剪块：

| 指标 | 不改（全量 prompt） | 浅层分片（A） | 深层复用（B） |
|---|---|---|---|
| 图片编码次数 | 4 | 4 × 10 = 40 | 4 |
| LLM 推理次数 | 4 | 4 × 10 = 40 | 4 × 10 = 40 |
| 图片编码耗时 | 1x | 10x | 1x |

图片编码器（ViT）在总推理中占比约 30-50%，裁剪推理场景下深层复用可比浅层分片
节省约 30-50% 的图片编码时间。

## 4. 详细设计

### 4.1 模型层：`vlx_seek/models/vlx_seek_1_5/language_model/modeling_vlx_seek_1_5.py`

#### 4.1.1 新增缓存方法

在 `VLXSeek1_5Model` 类中新增 `set_cached_image()` 和 `clear_cached_image()` 方法：

```python
def set_cached_image(
    self,
    image_embeds: torch.Tensor,
    image_grid_thws: list[torch.Tensor],
    vt_multi_level_features_list: Optional[list] = None,
    object_features: Optional[list[torch.Tensor]] = None,
) -> None:
    """缓存预计算的图片特征，后续 forward() 调用将跳过视觉编码。"""
    self._cached_image_embeds = image_embeds
    self._cached_image_grid_thws = image_grid_thws
    self._cached_vt_multi_level_features = vt_multi_level_features_list
    self._cached_object_features = object_features

def clear_cached_image(self) -> None:
    """清除图片特征缓存。"""
    self._cached_image_embeds = None
    self._cached_image_grid_thws = None
    self._cached_vt_multi_level_features = None
    self._cached_object_features = None
```

#### 4.1.2 `forward()` 缓存分支

在现有 `if images is not None and len(images) > 0:` 分支**之前**插入缓存检查分支：

```python
# --- 新增：缓存分支 ---
cached_embeds = getattr(self, '_cached_image_embeds', None)
if cached_embeds is not None and images is not None and len(images) > 0:
    # 跳过 self.encode_images()，直接用缓存
    image_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    image_grid_thws = self._cached_image_grid_thws

    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # 复用缓存的 object features
    object_features = self._cached_object_features
    if object_features is not None:
        valid_object_features = []
        obj_feat_idx = 0
        for i, input_id in enumerate(input_ids):
            num_obj_tokens = (input_id == VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX).sum().item()
            if num_obj_tokens == 0:
                continue
            if obj_feat_idx >= len(object_features):
                break
            feat = object_features[obj_feat_idx]
            obj_feat_idx += 1
            if feat is None:
                continue
            if feat.shape[0] != num_obj_tokens:
                feat = feat[:num_obj_tokens]
            valid_object_features.append(feat)

        if len(valid_object_features) > 0:
            all_object_features = torch.cat(valid_object_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            object_mask = (input_ids == VLX_SEEK_1_5_OBJECT_FEATURE_TOKEN_INDEX)
            if object_mask.sum() == all_object_features.shape[0]:
                inputs_embeds[object_mask] = all_object_features

    image_grid_thw = torch.cat(image_grid_thws, dim=0)

# --- 原有分支不变 ---
elif images is not None and len(images) > 0:
    image_embeds, image_grid_thws, vt_multi_level_features_list = self.encode_images(
        images, image_grid_thws
    )
    # ... 原有代码 ...
```

#### 4.1.3 `prepare_inputs_for_generation()` 兼容

当前代码在非首次迭代时设 `images=None`、`images_aux=None`。缓存模式下需要确保
`images` 参数在首次迭代时传递到 `forward()`，使缓存分支被触发。

现有 `prepare_inputs_for_generation()` 在 `is_first_iteration=True` 时保留 `images`，
因此**无需修改**。缓存模式下 `forward()` 的缓存分支会在首次迭代时命中，后续迭代
`images=None` 走正常 KV cache 路径。

### 4.2 Worker 层：`vlx_seek_worker.py`

#### 4.2.1 `encode_image_cache()`

预计算图片特征并写入模型缓存：

```python
def encode_image_cache(
    self,
    image: Image.Image,
    boxes: Optional[Sequence[Sequence[float]]],
) -> None:
    """预计算并缓存图片特征，供后续多次 detect() 复用。"""
    image = image.convert("RGB")
    caller_boxes = self._validate_boxes(boxes, image)
    boxes, _ = self._order_boxes(caller_boxes)

    images, image_grid_thws, images_aux = self._prepare_image_inputs(image, boxes)

    # 编码图片特征 → 语言空间投影
    image_embeds, _, vt_multi_level_features = self.model.encode_images(
        images, image_grid_thws
    )
    image_embeds = torch.cat(image_embeds, dim=0)

    # 编码 object features（如果有 bbox，用于 region-level 检测）
    object_features = None
    if images_aux and boxes:
        vision_tower = self.model.get_vision_tower()
        patch_size = vision_tower.config.patch_size
        vt_images_size = [thw[0][-2:] * patch_size for thw in image_grid_thws]
        tmp_images_aux = [aux.unsqueeze(0) for aux in images_aux]
        object_features = self.model.encode_objects(
            tmp_images_aux,
            [torch.tensor(boxes)],
            vt_multi_level_features,
            vt_images_size,
        )

    self.model.set_cached_image(
        image_embeds=image_embeds,
        image_grid_thws=image_grid_thws,
        vt_multi_level_features_list=vt_multi_level_features,
        object_features=object_features,
    )
```

#### 4.2.2 `clear_image_cache()`

```python
def clear_image_cache(self) -> None:
    """清除图片特征缓存。"""
    self.model.clear_cached_image()
```

#### 4.2.3 `detect_multi_prompt()`

对同一张图、同一组 proposals，用多组类别分片依次调用 `detect()`，合并结果：

```python
def detect_multi_prompt(
    self,
    image: Image.Image,
    bbox_list: Sequence[Sequence[float]],
    category_batches: list[list[str]],
    **kwargs,
) -> dict:
    """多组类别分批检测，合并结果。

    Returns:
        与 detect() 相同格式的 dict，result_bbox_list 为所有批次的并集。
    """
    merged = {
        "answer": "",
        "result_bbox_list": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for batch in category_batches:
        result = self.detect(image, bbox_list, batch, **kwargs)
        merged["result_bbox_list"].extend(result.get("result_bbox_list", []))
        if result.get("answer"):
            merged["answer"] += result["answer"] + "\n"
        merged["prompt_tokens"] += result.get("prompt_tokens", 0)
        merged["completion_tokens"] += result.get("completion_tokens", 0)
    return merged
```

### 4.3 应用层：`distill/generate_pseudo_labels.py`

#### 4.3.1 新增 CLI 参数

```python
parser.add_argument(
    "--prompt-batch-size",
    type=int,
    default=0,
    help="每个子提示词包含的类别数上限。0 表示不拆分（默认）。设为 30 则每 30 个类别一组循环推理。",
)
```

#### 4.3.2 新增工具函数：`_split_categories()`

```python
def _split_categories(categories: list[str], batch_size: int) -> list[list[str]]:
    """将类别列表按 batch_size 分批。batch_size<=0 或类别数<=batch_size 时返回单批。"""
    if batch_size <= 0 or len(categories) <= batch_size:
        return [categories]
    return [categories[i:i + batch_size] for i in range(0, len(categories), batch_size)]
```

#### 4.3.3 修改 `run_pipeline()` 推理逻辑

在非裁剪推理路径中（约第 264-282 行），增加分批判断。注意：`except` 块中也需要
调用 `clear_image_cache()`，防止 `detect_multi_prompt()` 抛异常时缓存泄漏到下一张图：

```python
try:
    if args.prompt_batch_size > 0 and len(categories) > args.prompt_batch_size:
        category_batches = _split_categories(categories, args.prompt_batch_size)
        boxes = load_proposals(image, args.detector_checkpoint)
        worker.encode_image_cache(image, boxes)
        result = worker.detect_multi_prompt(
            image, boxes, category_batches,
            lang=args.lang,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        worker.clear_image_cache()
        detections = [
            (rb["label"], rb["xmin"], rb["ymin"], rb["xmax"], rb["ymax"])
            for rb in result.get("result_bbox_list", [])
        ]
    else:
        # 原有逻辑不变
        boxes = load_proposals(image, args.detector_checkpoint)
        result = worker.detect(image, boxes, categories, ...)
        detections = [...]
except Exception as exc:
    # 确保异常时也清除缓存，防止泄漏到下一张图
    if args.prompt_batch_size > 0 and len(categories) > args.prompt_batch_size:
        worker.clear_image_cache()
    print(f"[{i + 1}/{total}] 失败 {img_path.name}: {exc}", file=sys.stderr)
    continue
```

#### 4.3.4 修改 `detect_with_crop()` callback

裁剪推理场景下，每个裁剪块也复用图片编码：

```python
def callback(slices) -> None:
    for slc in slices:
        crop = slc.image
        try:
            boxes = generator(crop)
            if args.prompt_batch_size > 0 and len(categories) > args.prompt_batch_size:
                category_batches = _split_categories(categories, args.prompt_batch_size)
                worker.encode_image_cache(crop, boxes)
                result = worker.detect_multi_prompt(
                    crop, boxes, category_batches,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                worker.clear_image_cache()
            else:
                result = worker.detect(crop, boxes, categories, ...)
        except Exception as exc:
            print(f"裁剪推理失败: {exc}", file=sys.stderr)
            worker.clear_image_cache()  # 确保异常时也清除缓存
            continue
        # ... 后续 shapes 构建逻辑不变 ...
```

#### 4.3.5 日志增强

在 `run_pipeline()` 开头输出分批信息：

```python
if args.prompt_batch_size > 0 and len(categories) > args.prompt_batch_size:
    n_batches = len(_split_categories(categories, args.prompt_batch_size))
    print(
        f"类别分批: {len(categories)} 个类别 → {n_batches} 批，"
        f"每批 ≤{args.prompt_batch_size} 个",
        file=sys.stderr,
    )
```

## 5. 错误处理

| 场景 | 处理方式 |
|---|---|
| 某个子提示词推理失败 | `detect_multi_prompt()` 中单批次异常会向上抛出，由 `run_pipeline()` 的 `except` 捕获，清除缓存后跳过该图 |
| 裁剪块推理失败 | `callback` 中 `except` 捕获，调用 `worker.clear_image_cache()` 清除缓存后 `continue` |
| 缓存未清除 | 下次 `encode_image_cache()` 会覆盖旧缓存，不会出错 |
| `--prompt-batch-size 0` | 完全走原有逻辑，零行为变化 |

## 6. 向后兼容性

- `--prompt-batch-size` 默认值为 0，不启用分批时行为与改动前完全一致
- 模型层缓存方法为纯新增，不修改任何现有方法签名
- `forward()` 缓存分支为纯新增 `if`，不影响没有缓存的正常推理路径
- Worker 层新增方法为纯新增，不修改现有 `detect()` / `predict()` 签名

## 7. 改动文件清单

| 文件 | 改动类型 | 改动量 |
|---|---|---|
| `vlx_seek/models/vlx_seek_1_5/language_model/modeling_vlx_seek_1_5.py` | 新增方法 + forward 分支 | ~60 行新增 |
| `vlx_seek_worker.py` | 新增 3 个方法 | ~60 行新增 |
| `distill/generate_pseudo_labels.py` | 新增参数 + 函数 + 修改推理逻辑 | ~40 行新增 + ~10 行修改 |

## 8. 测试验证

1. **基础功能**：`--prompt-batch-size 0`（默认），验证输出与改动前一致
2. **分批推理**：`--prompt-batch-size 30`，验证所有类别检测结果被正确合并到 COCO
3. **裁剪推理 + 分批**：`--crop-inference --prompt-batch-size 30`，验证裁剪块分批推理正常
4. **多卡 + 分批**：`--gpu-ids 0,1 --prompt-batch-size 30`，验证多卡场景下分批正常
5. **异常恢复**：模拟单批次推理失败，验证缓存被正确清除，后续图片不受影响
