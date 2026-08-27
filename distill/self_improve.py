# -*- coding: utf-8 -*-
"""步骤6: 自改进迭代循环(教师伪标签 → m0 → d1 → m1 → ... 自改进)。

Round 0:  d0(教师伪标签)        -> 训练 m0
Round 1..N:
    Bk.  m_{k-1} 整图推理 -> raw_d_k
    Ck.  VLM 清洗 raw_d_k    -> clean_d_k
    Dk.  clean 子集微调 m_{k-1}(热启动) -> m_k
    Ek.  固定 val 集上评估 m_k -> eval.json + summary.json

策略: student 推理 + 清洗(减法迭代) + 热启动微调; 不调 VLX-Seek 教师;
早停 mAP50 连续 N 轮无提升。用法详见 distill/README.md 步骤 6。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill.coco_utils import load_coco, save_coco, split_coco  # noqa: E402


def split_coco_by_image(
    coco,
    val_ratio=0.1,
    seed=42,
):
    """按 image 切 train/val, annotations 同图跟随。

    内部委托 `coco_utils.split_coco`(已按 image 切), 这里补 annotation id 各自
    0..n 连续化(原 split_coco 不改 ann id,这里 self_improve 的入口稳定 API)。
    """
    train_coco, val_coco = split_coco(coco, val_ratio=val_ratio, seed=seed)
    for c in (train_coco, val_coco):
        for i, a in enumerate(c["annotations"]):
            a["id"] = i
    return train_coco, val_coco


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="自改进迭代: m 整图推理 -> Qwen 清洗 -> 热启动微调(多轮自动)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--init-coco-json", required=True, help="d0 教师伪标签 COCO")
    p.add_argument("--image-dir", required=True, help="图像目录")
    p.add_argument("--category-map", required=True, help="category_prompts.json 路径")
    p.add_argument("--init-weights", default="yolov8s-worldv2.pt", help="m0 热起点(预训练 YOLO-World 权重路径)")
    p.add_argument("--max-rounds", type=int, default=3, help="self-improve 轮数(不含 round 0)")
    p.add_argument("--val-ratio", type=float, default=0.1, help="image-level 验证集比例(固定切, 永不参与训练)")
    p.add_argument("--imgsz", type=int, default=640, help="推理 / 训练 / 评估 一致分辨率")
    p.add_argument("--conf-thresh", type=float, default=0.30, help="推理 conf 阈值")
    p.add_argument("--nms-iou", type=float, default=0.50, help="推理 NMS IoU 阈值")
    p.add_argument("--epochs", type=int, default=30, help="单轮训练 epochs")
    p.add_argument("--batch", type=int, default=32, help="单轮训练 batch(DDP 下均分到各卡)")
    p.add_argument("--optimizer", default="auto", help="ultralytics optimizer(auto/sgd/AdamW/...)")
    p.add_argument("--lr0", type=float, default=None, help="初始学习率; None 走 ultralytics auto")
    p.add_argument("--train-device", default="0", help="训练 GPU(可 0 / 0,1 DDP / cpu)")
    p.add_argument("--infer-device", default="0", help="推理辅助 GPU(与 train-device 解耦, 可分队列)")
    p.add_argument("--patience", type=int, default=30, help="early-stop patience(epochs 数, 0 关)")

    # 清洗(透传 clean_pseudo_labels.parse_args)
    p.add_argument("--clean-base-url", default="http://127.0.0.1:8101/v1", help="VLM OpenAI 兼容 base url")
    p.add_argument("--model", default=None, help="VLM served-model-name, e.g. qwen3.8-vllm")
    p.add_argument("--api-key", default=None, help="VLM api key(本地 vLLM 可省; 默认取 OPENAI_API_KEY 环境变量)")
    p.add_argument("--clean-concurrency", type=int, default=16, help="VLM 清洗并发线程数")
    p.add_argument("--min-crop-size", type=int, default=640, help="清洗裁剪最小边(像素, 目标居中)")
    p.add_argument("--max-side", type=int, default=960, help="清洗裁剪最长边(像素, 只缩不放)")
    p.add_argument("--box-color", default="red", choices=["red", "yellow", "off"], help="裁剪图目标框颜色(off 回退旧 prompt)")

    # 调度
    p.add_argument("--run-dir", required=True, help="输出根目录(断点续跑 marker)")
    p.add_argument("--early-stop-no-improve", type=int, default=2, help="连续 N 轮(round>0) mAP50 无提升即提前停")
    p.add_argument("--ap-drop-alert", type=float, default=0.20, help="每类 AP 跌幅阈值(小数, 0.20 = 跌 20%%), 仅告警不终止")
    p.add_argument("--ap-drop-window", type=int, default=2, help="每类 AP 跌幅判定的连续轮数窗口")
    p.add_argument("--skip-clean", action="store_true", help="跳过清洗(调试用; 生产禁用)")
    p.add_argument("--model-provider", default=None, help='测试后门: "module.ClassName"; 不传走 ultralytics')
    return p.parse_args(argv)


def load_category_map(path):
    """加载 category_prompts.json, 返回 (names_list, train_to_cid, cn_to_train)。

    names_list 的 index = category_id = `coco["categories"]` 的 index(同 COCO 顺序)。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", {})
    names_list: list[str] = []
    cn_to_train: dict[str, str] = {}
    train_to_cid: dict[str, int] = {}
    for cid, (cn, entry) in enumerate(cats.items()):
        names_list.append(cn)
        train = str(entry.get("train_name", "")).strip()
        if not train:
            continue
        cn_to_train.setdefault(cn, train)
        train_to_cid.setdefault(train, cid)
    return names_list, train_to_cid, cn_to_train


def infer_one_image(model, image_path, imgsz, conf, iou, device="0"):
    """对单张图推理, 返回 [(cls_idx, x, y, w, h), ...](原图像素)。

    letterbox 反算: ultralytics `model.predict` 出的 `boxes.xyxy` 在 letterbox
    域; 输入 src_w × src_h → scale = min(imgsz/w, imgsz/h, 1.0), 居中 pad:
    pad_x = (imgsz - src_w*scale)/2, pad_y = (imgsz - src_h*scale)/2,
    原图 (x-pgx)/scale。越界裁剪到 [0, src] 内, 宽高 < 1px 丢。
    device 参数显式透传给 ultralytics predict, 与训练解耦。
    """
    import PIL.Image as _PIL

    with _PIL.open(image_path) as _im:
        src_w, src_h = _im.size
    scale = min(imgsz / src_w, imgsz / src_h, 1.0)
    pad_x = (imgsz - src_w * scale) / 2
    pad_y = (imgsz - src_h * scale) / 2

    def _f(v):
        try:
            return float(v.item())
        except AttributeError:
            return float(v)

    result = model.predict(image_path, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    boxes = result[0].boxes
    xyxy = getattr(boxes, "xyxy", None)
    n = getattr(xyxy, "shape", (None,))[0]
    if n is None:
        n = len(xyxy)
    if boxes is None or xyxy is None or n == 0:
        return []
    out = []
    for xy, cls in zip(boxes.xyxy, boxes.cls):
        cid = int(_f(cls))
        x1 = _f(xy[0])
        y1 = _f(xy[1])
        x2 = _f(xy[2])
        y2 = _f(xy[3])
        ox1 = max(0.0, (x1 - pad_x) / scale)
        oy1 = max(0.0, (y1 - pad_y) / scale)
        ox2 = min(src_w, (x2 - pad_x) / scale)
        oy2 = min(src_h, (y2 - pad_y) / scale)
        w = ox2 - ox1
        h = oy2 - oy1
        if w < 1 or h < 1:
            continue
        out.append((cid, ox1, oy1, w, h))
    return out


def build_round_coco(image_paths, preds_by_image, names_list):
    """纯函数: 图像组预测 → COCO。调用方负责 save_coco 落盘。

    images 同时填 width/height 从 PIL 读, 让下游 prepare_dataset 走
    coco_to_yolo_txt 不需要再做一次读尺寸(下游对缺 width/height 会 KeyError)。
    """
    import PIL.Image as _PIL

    images = []
    for i, p in enumerate(image_paths):
        with _PIL.open(p) as _im:
            _w, _h = _im.size
        images.append({"id": i, "file_name": p.name, "width": _w, "height": _h})
    coco: dict = {
        "images": images,
        "categories": [{"id": i, "name": n} for i, n in enumerate(names_list)],
        "annotations": [],
    }
    ann_id = 0
    for img_id, p in enumerate(image_paths):
        for cid, x, y, w, h in preds_by_image.get(p, []):
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco


def _resolve_model_provider_args(args):
    """返回 callable(checkpoint) -> model 实例。

    生产不传 --model-provider → 走 `ultralytics.YOLOWorld`。
    测试传 'distill.tests.test_self_improve.FakeYOLOWorld' → importlib 注入。
    """
    if args.model_provider:
        import importlib

        mod_name, _, cls_name = args.model_provider.rpartition(".")
        if not mod_name or not cls_name:
            raise ValueError(f'--model-provider {args.model_provider!r} 必须为 "module.ClassName"')
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return lambda ckpt: cls(ckpt)
    from ultralytics import YOLOWorld

    return YOLOWorld


def build_dataset_yaml(
    train_coco,
    val_coco,
    image_dir,
    dataset_root,
    category_map_path,
):
    """复用 distill.finetune_yolo_world.prepare_dataset, 传 train+val_coco。
    val_ratio/seed 形式上必须传(val_coco 非 None 时 split 分支不调)。"""
    from distill.finetune_yolo_world import prepare_dataset as _pd

    return _pd(
        coco=train_coco,
        image_dir=image_dir,
        output_dir=str(dataset_root),
        val_coco=val_coco,
        val_ratio=0.0,
        seed=42,
        category_map=category_map_path,
    )


def train_direct(model, dataset_yaml, epochs, batch, device, optimizer, lr0, imgsz, project, name, patience):
    """薄封装 ultralytics YOLOWorld.train()。"""
    kwargs = dict(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer=optimizer,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        seed=42,
    )
    if lr0 is not None:
        kwargs["lr0"] = lr0
    return model.train(**kwargs)


def collect_eval_metrics(model, dataset_yaml, imgsz, conf, iou, names):
    """YOLOWorld.val() → mAP / mAP50 / 每类 AP(ap50_95)。"""
    results = model.val(data=dataset_yaml, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    box = results.box
    ap5095 = list(box.ap50_95) if hasattr(box, "ap50_95") else [0.0] * len(names)
    return {
        "mAP": float(box.map),
        "mAP50": float(box.map50),
        "ap_per_class": {n: float(ap) for n, ap in zip(names, ap5095)},
    }


def _copy_path(src, dst):
    import shutil

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _store_config(args, run_dir):
    p = run_dir / "config.json"
    if p.is_file():
        return
    p.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_summary(run_dir):
    p = run_dir / "summary.json"
    if not p.is_file():
        return []
    return json.loads(p.read_text())["rounds"]


def _store_summary(run_dir, rounds, final_model, early_stopped):
    p = run_dir / "summary.json"
    p.write_text(
        json.dumps(
            {
                "rounds": rounds,
                "final_model": final_model,
                "early_stopped": early_stopped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _alert_per_class_drop(rounds, thr, window):
    """若某类连续 `window` 轮 AP 跌 > thr * 前值, 告警(不终止)。"""
    if len(rounds) < window + 1:
        return
    tail = rounds[-(window + 1) :]
    by_class: dict[str, list[float]] = {}
    for r in tail:
        for cname, ap in r.get("ap_per_class", {}).items():
            by_class.setdefault(cname, []).append(ap)
    for cname, series in by_class.items():
        base = series[0]
        if base <= 0.05:
            continue
        cur = series[-1]
        if cur < base * (1 - thr):
            print(f"!! [告警] 类别「{cname}」连续 {window} 轮 AP " f"{base:.4f} → {cur:.4f}(跌 {(1 - cur / base) * 100:.1f}%)")


def run_round(args, run_dir, k, prev_pt, store):
    """第 k 轮: k=0 仅 D/E(用教师 d0); k>0 B→C→D→E。幂等。"""
    round_dir = run_dir / f"round_{k}"
    round_dir.mkdir(parents=True, exist_ok=True)
    names = store["names_list"]

    # ---- B  推理(k=0 跳过)
    raw_path = round_dir / f"raw_d{k}.json"
    if k == 0:
        print(f"[round 0] B: 跳过(直接用教师 d0)")
    elif raw_path.is_file():
        print(f"[round {k}] B: raw_d{k} 已存在, 跳过推理")
    else:
        image_paths = store["image_paths"]
        model = store["model"](str(Path(run_dir) / prev_pt) if prev_pt else args.init_weights)
        preds = {}
        for p in image_paths:
            preds[p] = infer_one_image(model, p, args.imgsz, args.conf_thresh, args.nms_iou, device=args.infer_device)
        raw_coco = build_round_coco(image_paths, preds, names)
        save_coco(raw_coco, raw_path)
        print(f'[round {k}] B: {len(raw_coco["annotations"])} 框 → {raw_path.name}')

    # ---- C 清洗(k=0 跳过)
    clean_path = round_dir / f"clean_d{k}.json"
    if k == 0:
        print(f"[round 0] C: 跳过(直接用教师 d0)")
    elif args.skip_clean:
        if not clean_path.is_file():
            save_coco(load_coco(raw_path), clean_path)
    elif clean_path.is_file():
        print(f"[round {k}] C: clean_d{k} 已存在, 跳过清洗")
    else:
        decision_log = round_dir / f"decisions_d{k}.jsonl"
        from distill.clean_pseudo_labels import parse_args as _pa_clean, run_pipeline as _run_clean

        cargs = _pa_clean(
            [
                "--coco-json",
                str(raw_path),
                "--image-dir",
                str(store["image_dir"]),
                "--output",
                str(clean_path),
                "--decision-log",
                str(decision_log),
                "--model",
                args.model,
                "--base-url",
                args.clean_base_url,
                "--concurrency",
                str(args.clean_concurrency),
                "--min-crop-size",
                str(args.min_crop_size),
                "--max-side",
                str(args.max_side),
                "--box-color",
                args.box_color,
            ]
        )
        if args.api_key:
            cargs.api_key = args.api_key
        report = _run_clean(cargs, load_coco(raw_path))
        print(f'[round {k}] C: keep={report.get("kept")} ' f'delete={report.get("vlm_removed")} ' f'dedup={report.get("dedup_removed")} ' f'error_keep={report.get("error_keep")}')

    # ---- D 训练(从 prev 热启动; k>0 用 clean_d_k train 子集; k==0 用 split_train)
    model_out = round_dir / (f"m{k}.pt" if k > 0 else "m0.pt")
    if model_out.is_file():
        print(f"[round {k}] D: {model_out.name} 已存在, 跳过训练")
    else:
        if k == 0:
            train_coco_for_train = store["train_coco"]
        else:
            train_coco_for_train, _ = split_coco_by_image(load_coco(clean_path), val_ratio=args.val_ratio, seed=42)
        val_coco = load_coco(store["val_coco_path"])
        dataset_root = round_dir / "dataset_root"
        train_yaml = build_dataset_yaml(
            train_coco=train_coco_for_train,
            val_coco=val_coco,
            image_dir=str(store["image_dir"]),
            dataset_root=dataset_root,
            category_map_path=args.category_map,
        )
        src_weights = Path(run_dir) / prev_pt if prev_pt else Path(args.init_weights)
        model = store["model"](str(src_weights))
        train_direct(
            model=model,
            dataset_yaml=str(train_yaml),
            epochs=args.epochs,
            batch=args.batch,
            device=args.train_device,
            optimizer=args.optimizer,
            lr0=args.lr0,
            imgsz=args.imgsz,
            project=str(round_dir),
            name="yolo_world",
            patience=args.patience,
        )
        # 归档 best.pt(若缺回退 last.pt)
        best_pt = Path(round_dir) / "yolo_world" / "best.pt"
        if not best_pt.is_file():
            last_pt = Path(round_dir) / "yolo_world" / "last.pt"
            if not last_pt.is_file():
                raise FileNotFoundError(f'round {k}: 未找到 best.pt/last.pt 在 {round_dir / "yolo_world"}')
            print(f"[warn] round {k} 无 best.pt, 回退 last.pt")
            _copy_path(last_pt, model_out)
        else:
            _copy_path(best_pt, model_out)
        print(f"[round {k}] D: best.pt → {model_out.name}")

    # ---- E 评估(固定 val 集, 透传 D 的 dataset yaml)
    eval_path = round_dir / "eval.json"
    if eval_path.is_file():
        print(f"[round {k}] E: eval.json 已存在, 跳过评估")
        return json.loads(eval_path.read_text())
    model = store["model"](str(model_out))
    metrics = collect_eval_metrics(
        model,
        dataset_yaml=str(dataset_root / "dataset" / "dataset.yaml"),
        imgsz=args.imgsz,
        conf=args.conf_thresh,
        iou=args.nms_iou,
        names=list(store["names_list"]),
    )
    eval_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _store_config(args, run_dir)

    # ---- 初始数据
    d0 = load_coco(args.init_coco_json)
    store = {
        "image_dir": Path(args.image_dir),
        "image_paths": sorted([Path(args.image_dir) / fn for fn in (im["file_name"] for im in d0["images"])]),
        "train_coco_path": str(run_dir / "split_train.json"),
        "val_coco_path": str(run_dir / "split_val.json"),
    }
    names_list, _, cn_to_train = load_category_map(args.category_map)
    store["names_list"] = names_list
    store["cn_to_train"] = cn_to_train

    store["train_coco"], store["val_coco"] = split_coco_by_image(d0, val_ratio=args.val_ratio, seed=42)
    save_coco(store["train_coco"], store["train_coco_path"])
    save_coco(store["val_coco"], store["val_coco_path"])
    val_names = [im["file_name"] for im in store["val_coco"]["images"]]
    split_path = run_dir / "split.json"
    if not split_path.is_file():
        split_path.write_text(json.dumps({"val_file_names": val_names}, ensure_ascii=False), encoding="utf-8")

    store["model"] = _resolve_model_provider_args(args)
    rounds = _load_summary(run_dir)
    last_done = rounds[-1]["round"] if rounds else -1
    prev_pt = f"round_0/m0.pt" if last_done >= 0 and (run_dir / "round_0" / "m0.pt").is_file() else None
    early_stopped = False

    for k in range(max(0, last_done + 1), args.max_rounds + 1):
        print(f"======== Round {k}/{args.max_rounds}" f"(k=0 用教师 d0; k>0 B→C→D→E) ========")
        eval_d = run_round(args, run_dir, k, prev_pt, store)
        vmap50 = eval_d["mAP50"]
        base_map50 = rounds[-1]["mAP50"] if rounds else vmap50
        delta = 0.0 if k == 0 else round(vmap50 - base_map50, 5)
        rounds.append(
            {
                "round": k,
                "mAP50": vmap50,
                "mAP": eval_d.get("mAP"),
                "ap_per_class": eval_d.get("ap_per_class", {}),
                "delta_map50": delta,
            }
        )
        _store_summary(run_dir, rounds, f"round_{k}/m{k}.pt", False)
        print(f"[Round {k}] mAP50={vmap50:.4f} (Δ={delta:+.4f})")

        # 长尾呆类告警(不终止)
        if len(rounds) >= args.ap_drop_window + 1:
            _alert_per_class_drop(rounds, args.ap_drop_alert, args.ap_drop_window)

        # 早停(round 0 不计入"连续 N 轮无提升"; 只数 round>0 末 N 轮)
        if not early_stopped:
            recent = [r for r in rounds if r["round"] > 0]
            if len(recent) >= args.early_stop_no_improve:
                tail_deltas = [r["delta_map50"] for r in recent[-args.early_stop_no_improve :]]
                if all(d <= 0 for d in tail_deltas):
                    print(f"[Early-stop] 末 {args.early_stop_no_improve} 轮 mAP50 无提升, 提前结束")
                    early_stopped = True
                    break

        prev_pt = f"round_{k}/m{k}.pt"

    _store_summary(run_dir, rounds, prev_pt, early_stopped)
    print(f'完成: {run_dir / "summary.json"}')


if __name__ == "__main__":
    main()
