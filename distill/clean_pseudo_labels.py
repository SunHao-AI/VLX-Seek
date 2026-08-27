"""步骤4.5（可选）：VLM 清洗伪标签——本地 IoU-NMS 去重 + 多模态模型逐框验证。

用法示例:
    python distill/clean_pseudo_labels.py \
        --coco-json distill/data/pseudo_labels.json \
        --image-dir distill/data/images \
        --base-url http://localhost:8000/v1 \
        --model qwen3-vl-8b

流程: 阶段1 同图同类 IoU-NMS 去重（零 API 成本）→ 阶段2 每个标注框裁成局部小图
发给 OpenAI 兼容多模态服务判断「主要拍摄对象是否属于该类别」（「否」删除，
失败保守保留）→ 阶段3 写出清洗后 COCO 与统计报告。
决策日志 JSONL 即时落盘，支持中断后原命令重跑（已判定框从日志回放，不发请求）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill.coco_utils import load_coco, save_coco  # noqa: E402
from PIL import Image  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM 清洗伪标签：本地 IoU-NMS 去重 + 多模态模型逐框裁剪验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coco-json", required=True, help="输入伪标签 COCO JSON")
    p.add_argument("--image-dir", required=True, help="图像目录（COCO file_name 相对此目录）")
    p.add_argument("--output", default=None,
                   help="输出 COCO 路径，默认 <input>.cleaned.json，永不覆盖输入")
    p.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI 兼容服务地址")
    p.add_argument("--model", default=os.environ.get("CLEAN_VLM_MODEL"),
                   help="模型名（vLLM --served-model-name）；默认取环境变量 CLEAN_VLM_MODEL")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="API key（本地 vLLM 通常不需要）")
    p.add_argument("--concurrency", type=int, default=16, help="线程池大小")
    p.add_argument("--iou-threshold", type=float, default=0.55, help="同图同类 NMS 的 IoU 阈值")
    p.add_argument("--no-dedup", action="store_true", help="跳过本地去重阶段")
    p.add_argument("--max-side", type=int, default=960,
                   help="裁剪图最长边超过则等比缩小（只缩不放），默认 960")
    p.add_argument("--min-crop-size", type=int, default=640,
                   help="裁剪最小边（像素），目标居中；原图任一边小于该值时启动报错")
    p.add_argument("--box-color", choices=list(BOX_COLORS) + [BOX_COLOR_OFF],
                   default="red",
                   help="裁剪图上目标框颜色；off 关闭画框（回退旧 prompt）")
    p.add_argument("--no-draw-box", action="store_true",
                   help="等同 --box-color off；兼容入口")
    # --min-crop-pad 保留解析避免老命令 break，但不再消费（deprecated，-h 隐藏）
    p.add_argument("--min-crop-pad", type=float, default=0.12,
                   help=argparse.SUPPRESS)
    p.add_argument("--decision-log", default=None,
                   help="决策日志 JSONL 路径，默认 <output>.decisions.jsonl（断点续跑依据）")
    p.add_argument("--report", default=None, help="统计报告 JSON 输出路径（默认仅终端打印）")
    p.add_argument("--max-retries", type=int, default=3, help="单框请求最大重试次数")
    p.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数")
    return p.parse_args(argv)


def validate_refs(coco: dict) -> None:
    """校验每条标注引用的 image_id/category_id 均存在，缺失则报错退出。"""
    image_ids = {img["id"] for img in coco["images"]}
    category_ids = {c["id"] for c in coco["categories"]}
    bad = [
        a for a in coco["annotations"]
        if a["image_id"] not in image_ids or a["category_id"] not in category_ids
    ]
    if bad:
        sample = ", ".join(str(a["id"]) for a in bad[:10])
        sys.exit(f"错误：{len(bad)} 条标注引用了不存在的 image_id/category_id（示例 ann id: {sample}）")


class DecisionLog:
    """JSONL 决策日志：逐条判定即时落盘，支持断点续跑回放。"""

    def __init__(self, path: Path, meta: dict) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_meta = True
        if self.path.exists() and self.path.stat().st_size > 0:
            needs_meta = False
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":  # 上次中断留下的半行，补换行隔离
                    with open(self.path, "ab") as fb:
                        fb.write(b"\n")
        self._fh = open(self.path, "a", encoding="utf-8")
        if needs_meta:
            self.append({"_meta": meta})

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def load_previous_decisions(path: Path, meta: dict) -> tuple[dict[tuple[str, int], dict], bool]:
    """读取旧决策日志并以 (file_name, ann_id) 建索引。

    返回 (索引, 日志是否可续跑)：文件不存在视为可续跑（空索引）；
    首行 _meta 与当前参数不一致则告警并返回不可续跑（调用方应删除旧日志重来）。
    """
    path = Path(path)
    if not path.is_file():
        return {}, True
    records: list[dict] = []
    with open(path, "rb") as f:
        for raw in f:  # 二进制逐行读，避免撕裂的多字节 UTF-8 尾行在迭代解码时崩溃
            try:
                records.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # 中断产生的半行/撕裂尾行，跳过
    if not records or records[0].get("_meta") != meta:
        print(f"[warn] 旧决策日志 {path} 参数不一致或为空，将忽略并重新开始", file=sys.stderr)
        return {}, False
    index: dict[tuple[str, int], dict] = {}
    for rec in records[1:]:
        index[(rec["file_name"], rec["ann_id"])] = rec
    return index, True


def iou_xywh(a: list[float], b: list[float]) -> float:
    """计算两个 xywh 框的交并比（IoU），不相交返回 0.0。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedup_annotations(coco: dict, threshold: float) -> tuple[list[dict], list[dict]]:
    """同图同类 IoU-NMS：面积降序保留更大框（滑窗重复对中较大者边界通常更完整）。

    返回 (保留标注列表, dedup 决策记录列表)。保留列表保持原 COCO 顺序，
    保证同一输入多次运行产出一致。
    """
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    file_by_id = {img["id"]: img["file_name"] for img in coco["images"]}

    groups: dict[tuple[int, int], list[dict]] = {}  # (image_id, category_id) -> anns
    for ann in coco["annotations"]:
        groups.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)

    keep_ids: set[int] = set()
    dedup_records: list[dict] = []
    for (image_id, category_id), anns in groups.items():
        ordered = sorted(
            anns,
            key=lambda a: a.get("area") or a["bbox"][2] * a["bbox"][3],
            reverse=True,
        )
        kept_boxes: list[list[float]] = []
        for ann in ordered:
            if any(iou_xywh(ann["bbox"], kb) > threshold for kb in kept_boxes):
                dedup_records.append({
                    "file_name": file_by_id[image_id],
                    "ann_id": ann["id"],
                    "category_id": category_id,
                    "category_name": cat_names[category_id],
                    "verdict": "dedup",
                    "raw_reply": "",
                    "elapsed_ms": 0,
                })
            else:
                kept_boxes.append(list(ann["bbox"]))
                keep_ids.add(ann["id"])

    kept = [a for a in coco["annotations"] if a["id"] in keep_ids]
    return kept, dedup_records


BOX_COLORS = {'red': (255, 0, 0), 'yellow': (255, 255, 0)}
BOX_COLOR_OFF = 'off'


def crop_decode(
    image: Image.Image,
    bbox_xywh: list[float],
    min_crop_size: int = 640,
    max_side: int = 960,
    box_color: str = 'red',
) -> tuple[bytes, tuple[int, int, int, int]]:
    """以目标为中心裁剪, 边长取 max(原框, min_crop_size), 越界反推(借鉴 cv_utils.crop_rect)。

    送 VLM 前在裁剪图上画 box_color 矩形框(4px)引导注意力; 最长边 > max_side
    等比缩小(只缩不放, 坐标同步换算, 钳制到裁剪图内)。

    Returns:
        (JPEG 字节流, 目标框在裁剪图局部坐标系的 xywh 像素整数)。

    Raises:
        ValueError: 当 image 任一边 < min_crop_size, 或 box_color 非法。
    """
    if box_color not in BOX_COLORS and box_color != BOX_COLOR_OFF:
        raise ValueError(f'非法 box_color: {box_color!r}, 可选 {list(BOX_COLORS) + [BOX_COLOR_OFF]}')
    x, y, w, h = bbox_xywh
    img_w, img_h = image.size
    if img_w < min_crop_size or img_h < min_crop_size:
        raise ValueError(
            f'图像 {img_w}x{img_h} 小于 min_crop_size={min_crop_size}, 无法保证 '
            f'{min_crop_size}x{min_crop_size} 上下文; 请上扬图源或减小 --min-crop-size'
        )
    # 1) 计算裁剪窗口: 撑到 min_crop_size, 越界反推(借鉴 cv_utils.crop_rect)
    cw = max(int(round(w)), min_crop_size)
    ch = max(int(round(h)), min_crop_size)
    cx, cy = x + w / 2, y + h / 2
    left = max(0, int(round(cx - cw / 2)))
    top = max(0, int(round(cy - ch / 2)))
    right = min(img_w, left + cw)
    bottom = min(img_h, top + ch)
    left = right - cw
    top = bottom - ch
    crop = image.crop((left, top, right, bottom))
    # 2) 只缩不放: 最长边超 max_side 等比缩
    scale = 1.0
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        )
    # 3) 目标框局部坐标(换算 + 钳制) + 画框
    bx = (x - left) * scale
    by = (y - top) * scale
    bw = w * scale
    bh = h * scale
    crop_w, crop_h = crop.size
    bx = max(0, min(crop_w, round(bx)))
    by = max(0, min(crop_h, round(by)))
    bw = max(1, min(crop_w - bx, round(bw)))
    bh = max(1, min(crop_h - by, round(bh)))
    if box_color != BOX_COLOR_OFF:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(crop)
        outline = BOX_COLORS[box_color]
        for i in range(4):  # 4px 线宽
            draw.rectangle([bx + i, by + i, bx + bw - i - 1, by + bh - i - 1], outline=outline)
    # 4) JPEG 编码
    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=85)
    return buf.getvalue(), (int(bx), int(by), int(bw), int(bh))


SYSTEM_PROMPT = '你是严格的图像内容审核助手，只回答"是"或"否"。'
# 裁剪图上已画 box_color 矩形框；prompt 不再注入数值坐标，避免注意力分歧
USER_PROMPT = ('这张图中{box_word}矩形框已标注了待审核目标。'
               '请判断框内的主要拍摄对象是否属于类别「{name}」。'
               '只回答"是"或"否"。')
LEGACY_USER_PROMPT = '这张从大图裁出的局部区域中，主要拍摄对象是否属于类别「{name}」？只回答"是"或"否"。'


def _box_word(box_color: str) -> str:
    """按 box_color 返回 prompt 中的颜色词。"""
    return {'red': '红色', 'yellow': '黄色'}.get(box_color, '')


def classify_reply(reply: str) -> str:
    """把模型回复归类为 keep / delete / error_keep；无法归类返回空串（交由上层重试）。

    兼容真实模型自然表述：「属于/对/符合」→ keep；「不属于/不是/否」→ delete；
    「无法判断/不确定/看不清/多个目标」等 → error_keep（fail-open 保守保留）。
    """
    # 去掉首尾空白与常见中英文标点，避免 "是。"、"「否」"、"不确定……" 被误判
    r = reply.strip().strip(" \t\n\r。．！？!?，,、;；:：'\"“”‘’（）()[]【】《》")
    if not r:
        return ""
    # 模型明确表示无法判断 → 直接保守保留（fail-open），不再盲目重试
    UNCERTAIN = (
        "无法判断", "无法确定", "无法回答", "无法分辨", "无法识别",
        "不确定", "不能确定", "不能判断", "不清楚", "看不清", "看不太清",
        "信息不足", "不知道", "判断不出", "无法", "多个目标", "多个物体",
        "不只一个", "可能是", "也许",
    )
    if any(k in r for k in UNCERTAIN):
        return "error_keep"
    # 否定优先（"不属于/不是" 含 "属于/是"，必须先判否定）
    NEGATIVE = ("不属于", "不是", "并非", "并不属于", "不属于该类别", "否")
    if r.startswith(("不", "否", "非", "没")) or any(k in r for k in NEGATIVE):
        return "delete"
    # 肯定
    POSITIVE = ("属于", "是", "对", "符合", "就是", "确属", "正确", "应该", "确实是", "属于该类别")
    if any(k in r for k in POSITIVE):
        return "keep"
    return ""


class ServiceUnreachable(Exception):
    """服务不可达（连接拒绝/域名解析失败），仅限首个请求触发快速退出。"""


class VLMVerifier:
    """OpenAI 兼容 chat/completions 逐框验证客户端（temperature=0 强制短回答）。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 120,
        backoff_base: float = 2.0,
    ) -> None:
        import requests  # 延迟导入，与仓库其他脚本一致

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.calls = 0      # HTTP 尝试总数
        self.failures = 0   # 失败尝试数
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.headers = headers
        self.got_any_response = False  # 是否收到过任何 HTTP 响应
        self._post = requests.post
        self._conn_error_cls = requests.exceptions.ConnectionError

    def verify(
        self,
        image_bytes: bytes,
        category_name: str,
        target_box: tuple[int, int, int, int] | None = None,
        box_color: str = "red",
    ) -> tuple[str, str, int]:
        """验证单框。返回 (verdict, raw_reply, elapsed_ms)，失败耗尽重试后 fail-open。

        ``target_box`` 为目标框在裁剪图局部坐标系的 xywh（可空）；有值时画红框+
        注入颜色说明后发问；无值则退化为旧通用询问。``box_color`` 不能是 'off'
        （off 时调用方必须传 target_box=None）。
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        if target_box is not None:
            word = _box_word(box_color)
            text = USER_PROMPT.format(box_word=word, name=category_name)
        else:
            text = LEGACY_USER_PROMPT.format(name=category_name)
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }
        t0 = time.perf_counter()
        last_err = ""
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            try:
                resp = self._post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                self.got_any_response = True  # 收到过响应后不再触发快速退出
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                reply = resp.json()["choices"][0]["message"]["content"].strip()
                verdict = classify_reply(reply)
                if verdict in ("keep", "delete"):
                    return verdict, reply, int((time.perf_counter() - t0) * 1000)
                if verdict == "error_keep":
                    # 模型明确表示无法判断 → 保守保留（fail-open），不再盲目重试
                    return "error_keep", reply, int((time.perf_counter() - t0) * 1000)
                raise ValueError(f"无法解析回复: {reply!r}")
            except Exception as exc:  # noqa: BLE001 网络/HTTP/解析错误统一重试
                last_err = f"{type(exc).__name__}: {exc}"
                self.failures += 1
                if isinstance(exc, self._conn_error_cls) and not self.got_any_response:
                    raise ServiceUnreachable(
                        f"服务不可达（{last_err}）。请检查 --base-url 是否正确、"
                        f"vLLM 服务是否已启动。"
                    ) from exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base ** attempt * 0.5)
        return "error_keep", last_err, int((time.perf_counter() - t0) * 1000)


def write_output(coco: dict, kept_anns: list[dict], output_path: Path) -> None:
    """写出清洗后的 COCO：图像全量保留（0 标注图作负样本），ann id 连续重编号。"""
    out: dict = {
        "images": coco["images"],
        "categories": coco["categories"],
        "annotations": [],
    }
    if "info" in coco:
        out["info"] = coco["info"]
    for new_id, ann in enumerate(kept_anns):
        item = dict(ann)
        item["id"] = new_id
        out["annotations"].append(item)
    save_coco(out, output_path)


def run_pipeline(args: argparse.Namespace, coco: dict) -> dict:
    """三阶段流水线：本地去重 → 并发逐框验证 → 写出与统计报告。"""
    t0 = time.time()
    image_dir = Path(args.image_dir)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    file_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    total_orig = len(coco["annotations"])

    meta = {
        "model": args.model,
        "coco_json": args.coco_json,
        "iou_threshold": args.iou_threshold,
        "min_crop_size": args.min_crop_size,
        "max_side": args.max_side,
        "box_color": args.box_color,
        "no_draw_box": args.no_draw_box,
    }
    log_path = Path(args.decision_log)
    prev, reusable = load_previous_decisions(log_path, meta)
    if not reusable and log_path.exists():
        log_path.unlink()
        print("[warn] 已删除参数不一致的旧决策日志，重新开始", file=sys.stderr)
    log = DecisionLog(log_path, meta)

    # ---- 阶段1：本地 IoU-NMS 去重（零 API 成本）----
    if args.no_dedup:
        working = list(coco["annotations"])
        n_dedup = 0
    else:
        working, dedup_records = dedup_annotations(coco, args.iou_threshold)
        n_dedup = len(dedup_records)
        known_keys = set(prev)
        for rec in dedup_records:
            # 续跑时同一重复对可能已记录过，避免日志冗余
            if (rec["file_name"], rec["ann_id"]) not in known_keys:
                log.append(rec)
    print(f"[阶段1] 去重删除 {n_dedup} 框，待验证 {len(working)} 框", file=sys.stderr)

    by_image: dict[int, list[dict]] = {}
    for ann in working:
        by_image.setdefault(ann["image_id"], []).append(ann)

    # 预扫描：需真实调用 VLM 的框数（进度条 total / ETA 依据），并标记缺图
    missing_images: set[int] = set()
    pending_total = 0

    def needs_verify(file_name: str, ann: dict) -> bool:
        """与回放分支同口径：无记录或记录仅为 dedup 判定的框都需要真实送验。"""
        rec = prev.get((file_name, ann["id"]))
        return rec is None or rec.get("verdict") == "dedup"

    for image_id, anns in by_image.items():
        file_name = file_by_id[image_id]
        if not (image_dir / file_name).is_file():
            missing_images.add(image_id)
        # 缺图框同样计入 total：处理时瞬时落盘并推进进度，使 postfix 计数与进度条口径一致
        pending_total += sum(1 for a in anns if needs_verify(file_name, a))

    verifier = VLMVerifier(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    from tqdm import tqdm

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    inflight: deque = deque()  # 有界 future 队列，控制内存峰值
    max_inflight = max(args.concurrency * 4, 8)
    counters = {"keep": 0, "delete": 0, "error_keep": 0}
    verdicts: dict[int, str] = {}
    n_replay = 0

    outer = tqdm(total=len(by_image), desc="读取图像", unit="图",
                 file=sys.stderr, disable=len(by_image) <= 1)
    inner = tqdm(total=pending_total, desc="VLM 验证", unit="框",
                 file=sys.stderr, disable=pending_total <= 1)

    def drain(item) -> None:
        """收割单个 future：落盘决策、更新计数与内层进度。"""
        future, ann, fname = item
        try:
            verdict, raw, elapsed_ms = future.result()
        except ServiceUnreachable as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            log.close()  # 退出前关闭日志句柄，避免 ResourceWarning（数据均已 flush 无丢失）
            sys.exit(f"错误：{exc}")
        except Exception as exc:  # noqa: BLE001 线程内意外异常同样 fail-open
            verdict, raw, elapsed_ms = "error_keep", f"{type(exc).__name__}: {exc}", 0
        verdicts[ann["id"]] = verdict
        log.append({
            "file_name": fname,
            "ann_id": ann["id"],
            "category_id": ann["category_id"],
            "category_name": cat_names[ann["category_id"]],
            "verdict": verdict,
            "raw_reply": raw,
            "elapsed_ms": elapsed_ms,
        })
        counters[verdict] += 1
        inner.update(1)
        inner.set_postfix_str(f"留{counters['keep']} 删{counters['delete']} 错{counters['error_keep']}")

    def record_error(ann: dict, fname: str, raw_reply: str) -> None:
        """单框本地失败（图片不可读/裁剪异常）的 fail-open 落盘。"""
        verdicts[ann["id"]] = "error_keep"
        log.append({
            "file_name": fname,
            "ann_id": ann["id"],
            "category_id": ann["category_id"],
            "category_name": cat_names[ann["category_id"]],
            "verdict": "error_keep",
            "raw_reply": raw_reply,
            "elapsed_ms": 0,
        })
        counters["error_keep"] += 1
        inner.update(1)
        inner.set_postfix_str(f"留{counters['keep']} 删{counters['delete']} 错{counters['error_keep']}")

    # ---- 阶段2：逐图读图裁剪 + 并发验证（流水线）----
    for image_id, anns in by_image.items():
        fname = file_by_id[image_id]
        outer.update(1)

        todo: list[dict] = []
        for ann in anns:
            if needs_verify(fname, ann):
                # 无记录或记录仅为 dedup 判定（如 --no-dedup 重跑）→ 重新走
                # 裁剪 + VLM 验证流程，verdict 会以 keep/delete/error_keep 覆盖落盘
                todo.append(ann)
            else:  # 断点回放：直接复用判定，不发请求
                rec = prev[(fname, ann["id"])]
                verdicts[ann["id"]] = rec["verdict"]
                counters[rec["verdict"]] += 1
                n_replay += 1

        if image_id in missing_images:
            print(f"[warn] 图片缺失，{len(todo)} 框记 error_keep: {fname}", file=sys.stderr)
            for ann in todo:
                verdicts[ann["id"]] = "error_keep"
                log.append({
                    "file_name": fname,
                    "ann_id": ann["id"],
                    "category_id": ann["category_id"],
                    "category_name": cat_names[ann["category_id"]],
                    "verdict": "error_keep",
                    "raw_reply": "image file missing",
                    "elapsed_ms": 0,
                })
                counters["error_keep"] += 1
                inner.update(1)  # 缺图框推进内层进度，避免 postfix 计数与进度条脱节
            continue

        if not todo:
            continue
        try:
            image = Image.open(image_dir / fname).convert("RGB")
        except Exception as exc:  # noqa: BLE001 图片损坏等读取失败 → 该图全部框 fail-open
            print(f"[warn] 图片读取失败，{len(todo)} 框记 error_keep: {fname}（{exc}）",
                  file=sys.stderr)
            for ann in todo:
                record_error(ann, fname, f"image open failed: {type(exc).__name__}: {exc}")
            continue
        for ann in todo:
            if args.box_color == BOX_COLOR_OFF or args.no_draw_box:
                # 不画框 → 走 legacy prompt，target_box=None
                try:
                    data, _ = crop_decode(
                        image, ann["bbox"],
                        min_crop_size=args.min_crop_size,
                        max_side=args.max_side,
                        box_color=BOX_COLOR_OFF,
                    )
                except Exception as exc:  # noqa: BLE001
                    record_error(ann, fname, f"crop failed: {type(exc).__name__}: {exc}")
                    continue
                future = executor.submit(verifier.verify, data, cat_names[ann["category_id"]])
            else:
                try:
                    data, box = crop_decode(
                        image, ann["bbox"],
                        min_crop_size=args.min_crop_size,
                        max_side=args.max_side,
                        box_color=args.box_color,
                    )
                except Exception as exc:  # noqa: BLE001
                    record_error(ann, fname, f"crop failed: {type(exc).__name__}: {exc}")
                    continue
                future = executor.submit(
                    verifier.verify, data, cat_names[ann["category_id"]], box, args.box_color
                )
            inflight.append((future, ann, fname))
            while len(inflight) >= max_inflight:
                drain(inflight.popleft())

    while inflight:
        drain(inflight.popleft())
    executor.shutdown()
    outer.close()
    inner.close()
    log.close()

    # ---- 阶段3：写出与统计报告 ----
    kept_anns = [a for a in working if verdicts.get(a["id"]) != "delete"]
    kept_imgs = {a["image_id"] for a in kept_anns}
    emptied = sum(1 for iid in by_image if iid not in kept_imgs)

    del_per_cat: dict[str, int] = {}
    for a in working:
        if verdicts.get(a["id"]) == "delete":
            name = cat_names[a["category_id"]]
            del_per_cat[name] = del_per_cat.get(name, 0) + 1
    top = sorted(del_per_cat.items(), key=lambda kv: kv[1], reverse=True)[:10]

    elapsed = max(time.time() - t0, 1e-6)
    judged = len(working)
    report = {
        "total_annotations": total_orig,
        "dedup_removed": n_dedup,
        "vlm_removed": counters["delete"],
        "kept": counters["keep"],
        "error_keep": counters["error_keep"],
        "final_annotations": len(kept_anns),
        "replayed_from_log": n_replay,
        "per_category_deleted_top": [
            {"category_name": k, "deleted": v} for k, v in top
        ],
        "elapsed_sec": round(elapsed, 1),
        "throughput_boxes_per_sec": round(judged / elapsed, 2),
        "vlm_calls": verifier.calls,
        "vlm_failures": verifier.failures,
        "vlm_failure_rate": round(verifier.failures / verifier.calls, 4) if verifier.calls else 0.0,
        "images_total": len(coco["images"]),
        "images_emptied": emptied,
        "missing_images": sorted(file_by_id[i] for i in missing_images),
    }

    write_output(coco, kept_anns, Path(args.output))
    return report


def print_report(report: dict) -> None:
    """把统计报告打印为终端友好的中文摘要。"""
    print("\n===== 清洗统计 =====")
    print(f"输入总框数: {report['total_annotations']}")
    print(
        f"去重删除 {report['dedup_removed']} | VLM 删除 {report['vlm_removed']} | "
        f"保留 {report['kept']} | 失败保留 {report['error_keep']} | "
        f"日志回放 {report['replayed_from_log']}"
    )
    print(
        f"输出标注 {report['final_annotations']} 条；图像 {report['images_total']} 张"
        f"（清零保留 {report['images_emptied']} 张作负样本）"
    )
    if report["missing_images"]:
        print(f"缺失图片 {len(report['missing_images'])} 张（已 fail-open）")
    if report["per_category_deleted_top"]:
        print("各类别删除 Top:")
        for item in report["per_category_deleted_top"]:
            print(f"  {item['category_name']}: {item['deleted']}")
    print(
        f"耗时 {report['elapsed_sec']}s | 吞吐 {report['throughput_boxes_per_sec']} 框/s | "
        f"VLM 调用 {report['vlm_calls']} 次（失败率 {report['vlm_failure_rate']}）"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.coco_json)
    if not input_path.is_file():
        sys.exit(f"错误：输入文件不存在: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".cleaned.json")
    if output_path.resolve() == input_path.resolve():
        sys.exit(f"错误：--output 不能与输入文件相同: {output_path}")
    decision_log_path = (
        Path(args.decision_log) if args.decision_log else Path(f"{output_path}.decisions.jsonl")
    )
    if not args.model:
        sys.exit("错误：未指定 --model，且环境变量 CLEAN_VLM_MODEL 未设置")
    if args.min_crop_size < 1:
        sys.exit("错误：--min-crop-size 必须 >= 1")
    if args.max_side < 1:
        sys.exit("错误：--max-side 必须 >= 1")

    # 解析出的具体路径写回 args，供 run_pipeline 直接消费（CLI 未显式指定时为默认值）
    args.output = str(output_path)
    args.decision_log = str(decision_log_path)

    coco = load_coco(input_path)
    validate_refs(coco)
    report = run_pipeline(args, coco)
    print_report(report)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"统计报告已写入 {report_path}")


if __name__ == "__main__":
    main()
