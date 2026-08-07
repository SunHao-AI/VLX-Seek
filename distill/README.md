# VLX-Seek → YOLO-World 蒸馏

用 VLX-Seek（teacher，细粒度感知 VLM）生成伪标签，蒸馏训练官方 YOLO-World（student，开放词汇检测器）。

## 目录结构

```
distill/
├── generate_pseudo_labels.py   # 步骤1：VLX-Seek → COCO 伪标签
├── finetune_yolo_world.py      # 步骤2：COCO → 官方 YOLO-World 微调
├── coco_utils.py               # 共享 COCO 工具（坐标转换/划分/转 YOLO txt）
└── examples/                   # 端到端示例数据
    ├── images/                 #   demo 图片（demo_image.jpg / demo_image2.jpg）
    └── pseudo_labels.json      #   示例 COCO 伪标签（orange / apple）
```

## 环境要求

- **步骤1（伪标签生成）**：需要 GPU + VLX-Seek 权重（`resources/VLX-Seek-1.5-10B`）与 WeDetect 权重（`resources/wedetect_base_uni.pth`，缺失时自动下载）。依赖见项目根 `requirements.txt`。
- **步骤2（微调）**：需要额外安装 `ultralytics`（`pip install ultralytics`），建议独立虚拟环境，避免与项目 `torch 2.10 / transformers 5.13` 冲突。训练需足够内存/GPU。

## 步骤1：生成伪标签

```bash
python distill/generate_pseudo_labels.py \
  --image-dir data/images \
  --categories "person; car; dog" \
  --output data/pseudo_labels.json \
  --model-path resources/VLX-Seek-1.5-10B \
  --device cuda
```

- 每张图先用 WeDetect 生成候选区域（proposals），再调 VLX-Seek 开放词汇检测，输出 `{label, bbox}` 写入 COCO。
- 支持 `--resume` 断点续跑、`--start/--end-index` 分片、`--min-area` 过滤小框。
- 类别按 `--categories` 精确匹配 VLX-Seek 输出的 label（忽略大小写），不匹配的框会被丢弃。

## 步骤2：微调 YOLO-World

```bash
python distill/finetune_yolo_world.py \
  --coco-json data/pseudo_labels.json \
  --image-dir data/images \
  --weights yolov8s-worldv2.pt \
  --epochs 50 --imgsz 640 --batch 16 --device 0
```

- 自动划分 train/val（或传 `--val-coco-json` 指定独立验证集），转成 YOLO txt + 生成 `dataset.yaml`，调用 `YOLOWorld.train()`。
- 首次运行会下载 YOLO-World 预训练权重与 CLIP 文本编码器权重。

## 端到端示例（examples/）

`examples/` 提供最小可运行示例：2 张 demo 图片 + 一份示例 COCO 伪标签（`orange` / `apple`）。

```bash
# 用示例伪标签微调（CPU 小规模验证流程）
python distill/finetune_yolo_world.py \
  --coco-json distill/examples/pseudo_labels.json \
  --image-dir distill/examples/images \
  --output-dir distill/examples/runs \
  --weights yolov8s-worldv2.pt \
  --epochs 1 --imgsz 320 --batch 1 --device cpu --workers 0 --val-ratio 0.5
```

> 说明：`examples/pseudo_labels.json` 是手工构造的示例，用于验证微调流程；真实场景请用步骤1 由 VLX-Seek 生成。

## 注意事项

1. **伪标签质量受 proposal 召回限制**：VLX-Seek 是"区域检索"而非"坐标回归"，只能从 WeDetect 提出的候选框中选择。若 WeDetect 漏检目标，伪标签会系统性漏检。可提高 `num_proposals` 或换更强的 proposal 生成器。
2. **CLIP 权重下载**：微调需下载 CLIP 文本编码器权重（约 337MB）。若自动下载校验失败（网络截断），可手动下载到 ultralytics 的 `weights/clip/` 目录。
3. **内存/GPU**：YOLO-World 训练与 VLX-Seek 推理都需要较大内存/GPU，CPU 小内存环境可能无法完成完整训练。
