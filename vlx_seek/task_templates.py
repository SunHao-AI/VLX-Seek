"""Versioned bilingual task prompt templates for VLX-Seek.

Usage:
    from vlx_seek.task_templates import get_task_template, build_prompt

    build_prompt("vlx_seek_1_5", "detection", "person", lang="en")
    # -> "Detect every: person."

    build_prompt("vlx_seek_1_5", "detection", ["人", "车"], lang="zh")
    # -> "检测每一个：人; 车。"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# VLX-Seek 1.5  — each template has a single ``{}`` input slot.
# ---------------------------------------------------------------------------

VLX_SEEK_1_5_EN: dict[str, str] = {
    "detection": "Detect all the instances of: {}.",
    "grounding_single": "Detect a single instance of: {}.",
    "counting": "Find and count all {} in the image. Provide the object indexes along with the total number.",
    "brief_region_caption": "Give a brief description of {}.",
    "region_ocr": "Please provide the ocr results of {} in the image.",
    "reasoning_detection": "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\nFor the reasoning process, analyze each object in the image related to the question and identify the correct objects that correspond to the target label.\nFind the following in this picture: {}.",
}

VLX_SEEK_1_5_ZH: dict[str, str] = {
    "detection": "识别每个：{}。",
    "grounding_single": "检测单个目标：{}。",
    "counting": "在图片中识别并计数： {}。返回目标数量以及索引。",
    "brief_region_caption": "给出这些区域的简要描述：{}。",
    "region_ocr": "请提供该区域内的 OCR 结果：{}。",
    "reasoning_detection": "首先思考思维过程，然后向用户提供答案。思维过程和答案分别用 <think>   </think> 和 <answer>   </answer> 标签括起来，即 <think> 在这里写推理过程 </think><answer> 在这里写答案 </answer>。 对于推理过程，检查图像中与问题相关的所有对象，并确定符合目标标签的对象。在图像中识别{}。",
}

VLX_SEEK_1_5: dict[str, dict[str, str]] = {
    "en": VLX_SEEK_1_5_EN,
    "zh": VLX_SEEK_1_5_ZH,
}


TASK_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "vlx_seek_1_5": VLX_SEEK_1_5
}

DEFAULT_VERSION = "vlx_seek_1_5"
DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "zh")


def list_versions() -> list[str]:
    return list(TASK_TEMPLATES.keys())


def list_langs(version: str = DEFAULT_VERSION) -> list[str]:
    if version not in TASK_TEMPLATES:
        raise KeyError(
            f"Unknown version '{version}'. Available: {list_versions()}"
        )
    return list(TASK_TEMPLATES[version].keys())


def list_tasks(version: str = DEFAULT_VERSION, lang: str = DEFAULT_LANG) -> list[str]:
    templates = _get_lang_templates(version, lang)
    return list(templates.keys())


def _get_lang_templates(version: str, lang: str) -> dict[str, str]:
    if version not in TASK_TEMPLATES:
        raise KeyError(
            f"Unknown version '{version}'. Available: {list_versions()}"
        )
    version_templates = TASK_TEMPLATES[version]
    if lang not in version_templates:
        raise KeyError(
            f"Unknown lang '{lang}' for version '{version}'. "
            f"Available: {list(version_templates.keys())}"
        )
    return version_templates[lang]


def get_task_template(
    version: str,
    task: str,
    lang: str = DEFAULT_LANG,
) -> str:
    """Return the raw template string for ``version`` / ``lang`` / ``task``."""
    templates = _get_lang_templates(version, lang)
    if task not in templates:
        raise KeyError(
            f"Unknown task '{task}' for version '{version}', lang '{lang}'. "
            f"Available: {list(templates.keys())}"
        )
    return templates[task]


def build_prompt(
    version: str,
    task: str,
    text: str | list[str],
    lang: str = DEFAULT_LANG,
) -> str:
    """Fill the single ``{}`` slot and return the final prompt.

    If ``text`` is a list, items are joined with ``'; ``.
    """
    if isinstance(text, list):
        text = "; ".join(text)
    return get_task_template(version, task, lang=lang).format(text)
