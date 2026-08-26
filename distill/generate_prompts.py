"""从检测服务获取类别信息，生成 VLX-Seek 推理用的类别 prompt 映射。

接口 GET /v2/detect/all_class 返回的 ``all_cls`` 格式为：
    {
        f"{任务名称}&{中文类别名称}": f"{任务名称}:{英文类别名称}",
        ...
    }
本脚本解析出全部 中文类别名称 <=> 英文类别名称 映射，并为每个中文类别
生成 VLX-Seek 推理用的 prompt。脚本仅依赖 requests，可独立运行。

输出 JSON 格式：
    {
        "all_prompt": 拼接全部类别 prompt 的 VLX-Seek 检测提示词,
        "categories": {
            "中文类别名称": {
                "en_label": 英文类别名称,
                "prompt": 用于 VLX-Seek 推理的文本,
                "models": 该类别所属的任务（模型）列表
            },
            ...
        },
        "prompt_to_category": {推理 prompt: 真实中文类别名} 的反向映射,
        供 generate_pseudo_labels.py 把 COCO categories.name 还原为真实中文类别名。
    }

prompt 默认取中文类别名称，可直接用于 VLX-Seek 推理；如需更精确的语义，
可手动把 prompt 改成描述性文本（如 "卫星锅" -> "接收电视信号的卫星天线"）。

all_prompt 用 VLX-Seek 的 detection 模板（"Detect all the instances of: {}."）
把全部类别的 prompt 用 '; ' 拼接而成，可直接用于整图开放词汇检测推理。

说明：若同一中文类别出现在多个任务下且对应不同英文名，en_label 用 '; '
连接全部英文名（去重），prompt 保持中文类别名称，models 记录全部任务名。

保留手动修改：再次运行时，若输出文件已存在，已存在类别的手动优化 prompt
会被保留（仅当与默认中文名不同时），en_label / models 仍按最新接口更新。

用法示例：
    python distill/generate_prompts.py --output data/category_prompts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_URL = "http://192.168.10.102:8000/v2/detect/all_class"
DEFAULT_TIMEOUT = 30
# 默认输出到脚本所在目录的 data/ 下，避免受运行 cwd 影响
DEFAULT_OUTPUT = str(Path(__file__).resolve().parent / "data" / "category_prompts.json")

# VLX-Seek 开放词汇检测模板（与 vlx_seek/task_templates.py 中 VLX_SEEK_1_5_EN 的 detection 一致）
DETECTION_TEMPLATE = "Detect all the instances of: {}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 VLX-Seek 类别 prompt 映射")
    parser.add_argument("--url", default=DEFAULT_URL, help="类别信息接口地址")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="请求超时（秒）")
    return parser.parse_args()


def fetch_all_cls(url: str, timeout: int) -> dict[str, str]:
    """请求类别接口，返回 all_cls 字典，格式 {任务&中文名: 任务:英文名}。"""
    resp = requests.get(url, headers={"accept": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not data.get("all_cls"):
        raise ValueError(f"接口响应中缺少非空的 all_cls 字段: {str(data)[:200]}")
    return data["all_cls"]


def parse_mapping(all_cls: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """解析 all_cls 为 中文类别名称 -> {en_names, models}。

    key 格式 ``任务&中文名``，value 格式 ``任务:任务&英文名``（英文名部分
    也带任务前缀）。解析时统一去掉 ``任务&`` 前缀，得到纯英文名；任务名
    记录到 models。同一中文类别对应多个英文名/任务时全部保留（去重）。
    """
    zh_to_info: dict[str, dict[str, list[str]]] = {}
    for key, value in all_cls.items():
        task, zh_name = key.split("&", 1)  # key: 任务&中文名
        en_name = value.split(":", 1)[-1].split("&", 1)[-1]  # value: 任务:任务&英文名
        info = zh_to_info.setdefault(zh_name, {"en_names": [], "models": []})
        if zh_name in zh_to_info and en_name not in info["en_names"]:
            print(
                f"提示: 中文类别 '{zh_name}' 已存在，追加英文名 '{en_name}'",
                file=sys.stderr,
            )
        if en_name not in info["en_names"]:
            info["en_names"].append(en_name)
        if task not in info["models"]:
            info["models"].append(task)
    return zh_to_info


def build_prompts(
    zh_to_info: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, str | list[str]]]:
    """为每个中文类别生成 {en_label, prompt, models}。

    en_label 为全部英文名（'; ' 连接，去重）；prompt 取中文类别名称，
    可直接用于 VLX-Seek 推理，也可手动改成描述性文本；models 记录该类别
    所属的任务（模型）列表。
    """
    prompts: dict[str, dict[str, str | list[str]]] = {}
    for zh_name, info in zh_to_info.items():
        prompts[zh_name] = {
            "en_label": "; ".join(info["en_names"]),
            "prompt": zh_name,
            "models": info["models"],
        }
    return prompts


def load_existing_categories(output: Path) -> dict[str, dict]:
    """读取已存在的输出文件，返回 中文类别 -> 原条目（用于保留手动修改）。

    兼容两种结构：新结构 ``{"all_prompt": ..., "categories": {...}}`` 与
    旧结构（顶层直接是类别字典）。
    """
    if not output.is_file():
        return {}
    try:
        with open(output, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    categories = data.get("categories")
    if isinstance(categories, dict):
        return categories
    return data


def build_all_prompt(categories: dict[str, dict]) -> str:
    """把全部类别的 prompt 用 '; ' 拼接，填入 VLX-Seek 检测模板。"""
    labels = "; ".join(entry["prompt"] for entry in categories.values())
    return DETECTION_TEMPLATE.format(labels)


def build_prompt_to_category(categories: dict[str, dict]) -> dict[str, str]:
    """反向映射：推理 prompt -> 真实中文类别名。

    供 generate_pseudo_labels.py 把 COCO categories.name 从 prompt 还原为
    真实中文类别名使用（prompt 可能被手动改成描述性文本）。
    """
    return {entry["prompt"]: zh_name for zh_name, entry in categories.items()}


def main() -> None:
    args = parse_args()
    all_cls = fetch_all_cls(args.url, args.timeout)
    zh_to_info = parse_mapping(all_cls)
    categories = build_prompts(zh_to_info)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 保留用户手动优化的字段：已存在类别沿用原 prompt / train_name，仅更新 en_label/models
    existing = load_existing_categories(output)
    kept_prompt = 0
    kept_train_name = 0
    for zh_name, entry in categories.items():
        old = existing.get(zh_name)
        if not old:
            continue
        if isinstance(old.get("prompt"), str) and old["prompt"] != zh_name:
            entry["prompt"] = old["prompt"]
            kept_prompt += 1
        old_train_name = old.get("train_name")
        if isinstance(old_train_name, str) and old_train_name.strip():
            entry["train_name"] = old_train_name.strip()
            kept_train_name += 1

    result = {
        "all_prompt": build_all_prompt(categories),
        "categories": categories,
        "prompt_to_category": build_prompt_to_category(categories),
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"类别总数: {len(categories)}，保留手动优化 prompt: {kept_prompt}，保留 train_name: {kept_train_name}")
    print(f"已保存到 {output}")


if __name__ == "__main__":
    main()
