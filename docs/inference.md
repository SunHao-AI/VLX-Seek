# VLX-Seek 1.5 Inference Guide

English | [中文](inference_zh.md)

This guide covers command-line and Python inference for VLX-Seek 1.5-10B. For the model overview and installation instructions, see the [project README](../README.md).

## Prerequisites

The provided pipeline currently requires:

- the packages pinned in `requirements.txt`;
- the VLX-Seek 1.5-10B checkpoint from [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B);
- the region detector checkpoint from [WeDetect](https://huggingface.co/fushh7/WeDetect)

You can load the VLX-Seek main model directly from Hugging Face:

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

When `--bbox-list` is omitted, `wedetect_base_uni.pth` is automatically downloaded to `resources/wedetect_base_uni.pth` if needed.

Alternatively, place local copies in the default directory:

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` accepts either a Hugging Face model identifier or a local checkpoint directory. The model architecture is provided by the local `vlx_seek` package in this repository.

Alternatively, set paths with `--model-path` and `--detector-checkpoint`.

### Optional: Faster Inference

The model uses Linear Attention layers. Without the corresponding acceleration kernels, you will see a warning like the following and fall back to a slower PyTorch implementation:

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

Installing [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) enables the fast path and speeds up inference.

## Region Proposals

Most region-level tasks first require candidate bounding boxes. Due to company policy, we are unable to release the internally trained OPN referenced in our blog. The repository integrates the open-source, lightweight **WeDetect-Base-Uni** detector as an alternative proposal generator.

There are two ways to provide proposals:

1. Omit `--bbox-list`. The CLI loads WeDetect-Uni and generates 100 proposals.
2. Use any detector of your choice and pass its proposals through `--bbox-list`.

Custom boxes must use original-image pixel coordinates in `[x1, y1, x2, y2]` format:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple" \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

The model supports up to 100 object-region tokens, so keep the proposal list at or below 100 boxes.

## Command-Line Usage

### Open-Vocabulary Detection

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

Multiple detection categories are separated by semicolons. Add `--lang zh` when using the built-in Chinese task templates.

### Referring Expression Grounding

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "the person in a white shirt"
```

### Object Counting

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "orange"
```

### Reasoning Detection

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "the woman wearing red clothes"
```

### Region Captioning

Region captioning requires both proposals and one or more zero-based proposal indexes:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]'
```

The indexes refer to the order of the boxes supplied through `--bbox-list`.

### Region OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]'
```

### General VQA

VQA operates on the full image and does not load a proposal model:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "What fruits are in the image?"
```

## Supported Tasks

| Task | `--text` | Proposals | Additional requirement |
| --- | --- | --- | --- |
| `detection` | One or more `;`-separated categories/descriptions | Required | None |
| `grounding_single` | A referring expression | Required | None |
| `counting` | A target description | Required | None |
| `reasoning_detection` | A target description or category | Required | None |
| `brief_region_caption` | Generated from selected regions | Required | `--target-region-indexes` |
| `region_ocr` | Generated from selected regions | Required | `--target-region-indexes` |
| `vqa` | A free-form question | Not used | None |

For proposal-dependent tasks, omitting `--bbox-list` automatically invokes the proposal detector.

## Common Options

| Option | Default | Description |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | Local checkpoint directory or Hugging Face model identifier, e.g. `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | Input image |
| `--lang` | `en` | Built-in task-template language: `en` or `zh` |
| `--bbox-list` | None | JSON list of pixel-coordinate proposal boxes |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | WeDetect-Uni checkpoint used when proposals are omitted; auto downloaded if the default path is missing |
| `--target-region-indexes` | None | JSON list of selected proposal indexes for captioning or OCR |
| `--max-new-tokens` | `2048` | Maximum completion length |
| `--temperature` | `0.0` | Sampling temperature; values above zero enable sampling. Temperature `0.0` is recommended for all detection-related tasks |
| `--no-visualize` | Off | Disable result-box visualization |
| `--visualization-output` | `<image_stem>_result.png` | Custom visualization path |

Run `python inference.py --help` for the complete CLI reference.

## Output

The command prints a result dictionary as formatted JSON:

```json
{
  "answer": "<ground>orange</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "orange"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` is the raw model response.
- `result_bbox_list` resolves generated `<objN>` references to the input proposal coordinates.
- `prompt_tokens` and `completion_tokens` report token counts.
- `visualization_path` is included when selected boxes are visualized.

VQA, captioning, or OCR answers may be free-form text and can have an empty `result_bbox_list`.

## Python API

`VLXSeekWorker` keeps the model loaded across requests. Proposal generation is handled by the CLI, so Python API callers should prepare their own boxes for region-level tasks.

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
    ["orange", "apple"],
    lang="en",
)

vqa = worker.predict(
    image,
    "What fruits are in the image?",
)
```

Other task wrappers are `ground`, `count`, `reasoning_detect`, `describe_region`, and `read_region_text`. Boxes are clipped to the input image bounds and must retain positive area after clipping.
