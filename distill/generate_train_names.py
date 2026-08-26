"""为 category_prompts.json 中缺失 train_name 的类别增量生成英文训练名。

背景：
    YOLO-World 训练时用 CLIP 文本编码器对 dataset.yaml 的类别名编码生成分类器
    权重，中文类名无法被 CLIP 正确编码（表现为 cls_loss 不下降、mAP≈0）。
    因此每个类别需要一个 CLIP 友好的自然英文短语作为训练/推理类别名，
    存储在 category_prompts.json 每个条目的 ``train_name`` 字段。

行为：
    1. 读取 category_prompts.json，找出缺少 train_name 的条目；全部齐全则直接退出。
    2. 分批调用 OpenAI 兼容 Chat Completions 接口翻译，要求输出严格 JSON。
    3. 校验：非空、仅 ASCII、不含下划线、大小写不敏感全局唯一；
       已存在的 train_name 与人工修订永不覆盖。
    4. 写回原文件。

用法示例：
    export OPENAI_API_KEY=sk-xxx            # PowerShell: $env:OPENAI_API_KEY="sk-xxx"
    python distill/generate_train_names.py \
        --category-prompts distill/data/category_prompts.json \
        --model gpt-4o-mini --batch-size 40
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATEGORY_PROMPTS = ROOT / "distill/data/category_prompts.json"

SYSTEM_PROMPT = """You are a professional translator for computer-vision detection labels.
For each Chinese category you receive, produce an English class name for the
CLIP text encoder of an open-vocabulary detector (YOLO-World).

Rules:
1. Lowercase natural English noun phrases separated by single spaces.
2. ASCII only; no underscores, hyphens, punctuation, or articles.
3. 2-7 words; faithful to the full Chinese description (scene, state, location).
4. Names must stay mutually distinguishable across all categories.
5. Do NOT reuse any name from the existing-names list.

Output STRICT JSON only: {"<chinese category>": "<english name>", ...}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量补全 category_prompts.json 的 train_name 字段")
    parser.add_argument("--category-prompts", default=str(DEFAULT_CATEGORY_PROMPTS), help="category_prompts.json 路径")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="聊天模型名")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI 兼容接口地址")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API Key（缺省读环境变量 OPENAI_API_KEY）")
    parser.add_argument("--batch-size", type=int, default=40, help="每次请求翻译的类别数")
    parser.add_argument("--max-retries", type=int, default=3, help="每批最大重试次数")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒数")
    return parser.parse_args()


def load_categories(path: Path) -> tuple[dict, dict[str, dict]]:
    """返回 (原始 JSON, 缺失 train_name 的子集)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    categories = data["categories"]
    missing = {zh: e for zh, e in categories.items() if not str(e.get("train_name", "")).strip()}
    return data, missing


def validate_names(names: dict[str, str], taken: set[str]) -> list[str]:
    """校验生成的名字，返回错误列表。"""
    errors = []
    seen = {n.casefold() for n in taken}
    for zh, en in names.items():
        en = en.strip()
        if not en:
            errors.append(f"{zh}: 名称为空")
        elif not en.isascii():
            errors.append(f"{zh}: 含非 ASCII 字符 -> {en}")
        elif "_" in en:
            errors.append(f"{zh}: 含下划线 -> {en}")
        elif en.casefold() in seen:
            errors.append(f"{zh}: 英文名重复 -> {en}")
        else:
            seen.add(en.casefold())
    return errors


def chat(base_url: str, api_key: str, model: str, messages: list[dict], timeout: float) -> str:
    import requests

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": 0.2},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_reply(text: str) -> dict[str, str]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or not all(isinstance(v, str) for v in obj.values()):
        raise ValueError("回复不是 {str: str} 形式的 JSON 对象")
    return obj


def translate_batch(
    batch: dict[str, dict],
    taken: set[str],
    args: argparse.Namespace,
) -> dict[str, str]:
    """请求翻译一批类别，重试直到全部通过校验或达到上限。"""
    items = "\n".join(f"- {zh} | 描述: {e['prompt']}" for zh, e in batch.items())
    user_prompt = (
        f"Existing names (must not reuse):\n{json.dumps(sorted(taken), ensure_ascii=False)}\n\n"
        f"Translate these {len(batch)} categories:\n{items}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(1, args.max_retries + 1):
        try:
            reply = chat(args.base_url, args.api_key, args.model, messages, args.timeout)
            names = parse_json_reply(reply)
            absent = set(batch) - set(names)
            if absent:
                raise ValueError(f"缺少 {len(absent)} 条: {sorted(absent)[:5]}")
            errors = validate_names({zh: names[zh] for zh in batch}, taken)
            if errors:
                raise ValueError("; ".join(errors[:5]))
            return {zh: names[zh].strip() for zh in batch}
        except Exception as exc:  # noqa: BLE001 网络/解析/校验失败统一重试
            print(f"  第 {attempt}/{args.max_retries} 次尝试失败: {exc}", file=sys.stderr)
            if attempt < args.max_retries:
                time.sleep(2 * attempt)
    raise SystemExit("该批次多次重试后仍失败，请检查模型/网络或手动补充 train_name 后重跑")


def main() -> None:
    args = parse_args()

    path = Path(args.category_prompts)
    data, missing = load_categories(path)
    total = len(data["categories"])
    if not missing:
        print(f"全部 {total} 个类别均已有 train_name，无需处理")
        return
    if not args.api_key:
        sys.exit(f"有 {len(missing)} 个类别缺失 train_name，但未配置 API Key；请设置环境变量 OPENAI_API_KEY 或传 --api-key")
    print(f"共 {total} 个类别，其中 {len(missing)} 个缺失 train_name，开始增量翻译")

    # 已有名字进入占用集合，保证新增名字与现有名字互异
    taken = {
        str(e["train_name"]).strip()
        for e in data["categories"].values()
        if str(e.get("train_name", "")).strip()
    }

    entries = list(missing.items())
    for start in range(0, len(entries), args.batch_size):
        batch = dict(entries[start : start + args.batch_size])
        print(f" translating {start + 1}~{start + len(batch)} / {len(entries)} ...")
        names = translate_batch(batch, taken, args)
        for zh, en in names.items():
            data["categories"][zh]["train_name"] = en
        taken.update(n.casefold() for n in names.values())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已补全 {len(entries)} 条并写回 {path}")


if __name__ == "__main__":
    main()
