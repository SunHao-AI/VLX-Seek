# VLX-Seek 1.5 推理指南

[English](inference.md) | 中文

本文介绍 VLX-Seek 1.5-10B 的命令行与 Python 推理方式。模型简介和安装方法请参阅[项目 README](../README_zh.md)。

## 环境要求

当前提供的推理流程需要：

- `requirements.txt` 中固定的依赖
- VLX-Seek 1.5-10B 模型权重：[omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B)
- WeDetect-Base-Uni 模型权重：[WeDetect](https://huggingface.co/fushh7/WeDetect)

可直接通过 Hugging Face 模型名加载 VLX-Seek 主模型：

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

未提供 `--bbox-list` 时，程序会在需要时自动把 `wedetect_base_uni.pth` 下载到 `resources/wedetect_base_uni.pth`。

也可以将权重下载到本地默认目录：

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` 既支持 Hugging Face 模型标识符，也支持本地 checkpoint 目录。模型结构由本仓库中的 `vlx_seek` 包提供。

也可以通过 `--model-path` 和 `--detector-checkpoint` 指定其他路径。

### 可选：加速推理

模型使用了 Linear Attention 层。如果缺少对应的加速内核，会看到类似如下警告，并回退到较慢的 PyTorch 实现：

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

安装 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) 和 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 可以启用快速路径，从而提升推理速度。

## 候选区域

大多数区域级任务需要先准备候选框。受公司政策限制，我们无法开源blog中使用的内部训练 OPN。本仓库集成开源、轻量的 **WeDetect-Base-Uni** 检测器，作为候选区域生成的替代方案。

有两种方式可以提供 proposals：

1. 不传 `--bbox-list`，由命令行程序加载 WeDetect-Uni 并生成 100 个候选框。
2. 使用任意检测器生成候选框，再通过 `--bbox-list` 传入。

自定义候选框应使用原图像素坐标，格式为 `[x1, y1, x2, y2]`：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

模型最多支持 100 个目标区域 token，因此 proposals 数量应不超过 100。

## 命令行用法

### 开放目标检测

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh
```

多个检测类别使用分号分隔。使用内置中文任务模板时，请添加 `--lang zh`。

### 指代表达定位

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "穿白色上衣的人" \
  --lang zh
```

### 目标计数

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "橘子" \
  --lang zh
```

### 推理检测

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "穿着红色衣服的女人" \
  --lang zh
```

### 区域描述

区域描述需要同时提供 proposals 和一个或多个从零开始的区域索引：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]' \
  --lang zh
```

索引对应 `--bbox-list` 中的输入顺序。

### 区域 OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]' \
  --lang zh
```

### 通用 VQA

VQA 直接处理整幅图像，不会加载候选区域模型：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "图片中有哪些水果？"
```

## 支持的任务

| 任务 | `--text` 内容 | 是否需要 proposals | 额外要求 |
| --- | --- | --- | --- |
| `detection` | 一个或多个以 `;` 分隔的类别或描述 | 是 | 无 |
| `grounding_single` | 指代表达 | 是 | 无 |
| `counting` | 目标描述 | 是 | 无 |
| `reasoning_detection` | 目标描述或类别 | 是 | 无 |
| `brief_region_caption` | 根据选中区域自动生成 | 是 | `--target-region-indexes` |
| `region_ocr` | 根据选中区域自动生成 | 是 | `--target-region-indexes` |
| `vqa` | 自由形式问题 | 否 | 无 |

对于依赖 proposals 的任务，如果省略 `--bbox-list`，程序会自动调用候选框检测器生成。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | 本地权重目录或 Hugging Face 模型标识符，例如 `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | 输入图像 |
| `--lang` | `en` | 内置任务模板语言：`en` 或 `zh` |
| `--bbox-list` | 无 | 像素坐标 proposals 的 JSON 列表 |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | 未提供 proposals 时使用的 WeDetect-Uni 权重；默认路径缺失时会自动下载 |
| `--target-region-indexes` | 无 | 区域描述或 OCR 选中的 proposal 索引 JSON 列表 |
| `--max-new-tokens` | `2048` | 最大生成长度 |
| `--temperature` | `0.0` | 采样温度；大于零时启用采样。建议检测相关任务均使用温度 `0.0` |
| `--no-visualize` | 关闭 | 禁用检测结果可视化 |
| `--visualization-output` | `<image_stem>_result.png` | 自定义可视化输出路径 |

运行 `python inference.py --help` 可以查看完整命令行参数。

## 输出格式

命令最终会以格式化 JSON 打印结果字典：

```json
{
  "answer": "<ground>橘子</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "橘子"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` 是模型的原始回答。
- `result_bbox_list` 将生成的 `<objN>` 引用解析为输入 proposal 坐标。
- `prompt_tokens` 和 `completion_tokens` 是输入与输出 token 数量。
- 对选中目标进行可视化时，结果中会包含 `visualization_path`。

VQA、区域描述或 OCR 可以返回自由文本，其 `result_bbox_list` 可能为空。

## Python API

`VLXSeekWorker` 可以在多次请求之间保持模型常驻。候选区域生成由命令行入口负责，因此 Python API 调用方需要为区域级任务自行准备候选框。

```python
from PIL import Image

from vlx_seek_worker import VLXSeekWorker

worker = VLXSeekWorker(
    model_path="omlab/VLX-Seek-1.5-10B",
    device="cuda",
)
image = Image.open("demo/demo_image.jpg").convert("RGB")
boxes = [
    [12, 20, 180, 210],
    [190, 25, 430, 250],
]

detection = worker.detect(
    image,
    boxes,
    ["橘子", "苹果"],
    lang="zh",
)

vqa = worker.predict(
    image,
    "图片中有哪些水果？",
)
```

其他任务封装包括 `ground`、`count`、`reasoning_detect`、`describe_region` 和 `read_region_text`。候选框会被裁剪到输入图像边界，裁剪后必须保持有效的正面积。
