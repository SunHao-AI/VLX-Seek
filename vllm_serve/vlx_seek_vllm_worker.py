# -*- coding: utf-8 -*-
"""VLX-Seek vLLM 后端推理 worker（与 vlx_seek_worker.VLXSeekWorker 同接口）。

使用 vLLM 引擎实现 detect / detect_multi_prompt / predict 等接口，
供 distill/generate_pseudo_labels.py 通过 --backend vllm 切换。
核心收益：
- 引擎常驻，图像编码 + LLM 权重一次加载
- 同 crop 的多个 prompt 批量提交，共享图像/object 前缀 KV
  （continuous batching 在 scheduler 层自动复用，无需 APC）
"""
from __future__ import annotations

import sys
import time
from typing import Optional, Sequence, Union

import torch
from PIL import Image

from vlx_seek.task_templates import build_prompt
from vlx_seek_worker import VLXSeekWorker  # 复用静态解析工具（_validate_boxes 等）


class VLXSeekVLLMWorker:
    """vLLM 版 VLX-Seek worker。

    ``bbox_list`` 坐标为原图像素坐标 ``[[x1, y1, x2, y2], ...]``。
    所有返回 dict 与 VLXSeekWorker 一致（answer / result_bbox_list /
    prompt_tokens / completion_tokens / elapsed）。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.7,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
    ):
        import vllm_serve.plugin
        vllm_serve.plugin.init()

        from vllm import LLM

        self.model_path = model_path
        self.device = torch.device(device)
        self.log_timing = False  # True 时每次 generate 输出耗时日志

        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,  # hybrid mamba 架构下 CUDA graph 捕获不匹配，必须 eager
            max_model_len=max_model_len,
        )

        # aux image processor（C-RADIOv4 硬编码配置，与 VLXSeekWorker 一致）
        from transformers import CLIPImageProcessor

        self._aux_processor = CLIPImageProcessor(
            do_resize=False,
            do_center_crop=False,
            do_rescale=True,
            do_normalize=False,
            do_convert_rgb=True,
            resample=3,
        )

        # mm_bbox_order_mode（VLXSeekWorker._order_boxes 依赖模型 config）
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self._order_mode = getattr(config, "mm_bbox_order_mode", "none")

    # ------------------------------------------------------------------
    # 内部工具（与 VLXSeekWorker 语义一致）
    # ------------------------------------------------------------------

    def _order_boxes(
        self, boxes: Optional[list[list[float]]]
    ) -> tuple[Optional[list[list[float]]], list[int]]:
        """Order boxes internally and retain the mapping to caller indices."""
        if not boxes:
            return boxes, []
        order_mode = self._order_mode
        if order_mode == "none":
            return boxes, list(range(len(boxes)))
        if order_mode == "raster":
            sort_key = lambda item: (item[1][0], item[1][1], item[0])
        elif order_mode == "area_asc":
            sort_key = lambda item: (
                (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
                item[0],
            )
        elif order_mode == "area_desc":
            sort_key = lambda item: (
                -((item[1][2] - item[1][0]) * (item[1][3] - item[1][1])),
                item[0],
            )
        else:
            raise ValueError(f"Unsupported mm_bbox_order_mode: {order_mode}")
        ordered = sorted(enumerate(boxes), key=sort_key)
        return [box for _, box in ordered], [original_index for original_index, _ in ordered]

    def _build_prompt(self, question: str, boxes) -> str:
        """vLLM 版 prompt：<image> 用 <|image_pad|>（vLLM processor 自动展开）。"""
        image_part = "<|vision_start|><|image_pad|><|vision_end|>"
        if boxes:
            object_tokens = "".join(
                f"<obj{index}><objfeat>" for index in range(len(boxes))
            )
            image_part = f"{image_part}\n{object_tokens}"
        return (
            f"<|im_start|>user\n{image_part}\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _make_request(self, image: Image.Image, boxes, prompt: str) -> dict:
        """构建单个 vLLM 请求（含自定义 mm_processor_kwargs）。"""
        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        if boxes:
            aux = self._aux_processor.preprocess(image, return_tensors="pt")
            request["mm_processor_kwargs"] = {
                "bbox_list": torch.tensor([boxes], dtype=torch.float32),
                "images_aux": aux["pixel_values"],
            }
        return request

    def _run_generate(
        self,
        requests: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ):
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=["<|im_end|>"],
        )
        start = time.perf_counter()
        outs = self.llm.generate(requests, params)
        elapsed = time.perf_counter() - start
        if self.log_timing:
            for o in outs:
                n_in = len(o.prompt_token_ids)
                n_out = len(o.outputs[0].token_ids)
                tok_s = n_out / elapsed if elapsed > 0 else float("inf")
                print(
                    f"[timing] prompt={n_in} completion={n_out} {elapsed:.2f}s "
                    f"({tok_s:.1f} tok/s)",
                    file=sys.stderr,
                )
        return outs, elapsed

    # ------------------------------------------------------------------
    # 公共接口（与 VLXSeekWorker 一致）
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        question: str,
        bbox_list: Optional[Sequence[Sequence[float]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> dict:
        """Answer one image question, optionally using object-region prompts."""
        image = image.convert("RGB")
        caller_boxes = VLXSeekWorker._validate_boxes(bbox_list, image)
        boxes, sorted_to_original = self._order_boxes(caller_boxes)
        question = VLXSeekWorker._remap_prompt_object_tokens(
            question, sorted_to_original
        )
        prompt = self._build_prompt(question, boxes)
        request = self._make_request(image, boxes, prompt)

        (out,), elapsed = self._run_generate(
            [request], max_new_tokens, temperature, top_p, repetition_penalty
        )
        answer = out.outputs[0].text.strip()
        answer = VLXSeekWorker._remap_object_tokens(answer, sorted_to_original)
        result_bbox_list = VLXSeekWorker._build_result_bbox_list(answer, caller_boxes)
        return {
            "answer": answer,
            "result_bbox_list": result_bbox_list,
            "prompt_tokens": len(out.prompt_token_ids),
            "completion_tokens": len(out.outputs[0].token_ids),
            "elapsed": elapsed,
        }

    def predict_batch(
        self,
        requests: list[Union[tuple[Image.Image, str], dict]],
        **kwargs,
    ) -> list[dict]:
        """批量提交（一次 llm.generate），返回与 predict 相同格式的 list。"""
        vllm_requests: list[dict] = []
        metas: list[tuple[list[int], list[list[float]]]] = []
        for request in requests:
            if isinstance(request, dict):
                image = request["image"]
                question = request.get("question", request.get("prompt", ""))
                boxes = request.get("bbox_list")
            else:
                image, question = request
                boxes = None
            image = image.convert("RGB")
            caller_boxes = VLXSeekWorker._validate_boxes(boxes, image)
            ordered_boxes, sorted_to_original = self._order_boxes(caller_boxes)
            question = VLXSeekWorker._remap_prompt_object_tokens(
                question, sorted_to_original
            )
            prompt = self._build_prompt(question, ordered_boxes)
            vllm_requests.append(self._make_request(image, ordered_boxes, prompt))
            metas.append((sorted_to_original, caller_boxes))

        outs, elapsed = self._run_generate(
            vllm_requests,
            kwargs.get("max_new_tokens", 512),
            kwargs.get("temperature", 0.2),
            kwargs.get("top_p", 1.0),
            kwargs.get("repetition_penalty", 1.0),
        )
        results = []
        for out, (sorted_to_original, caller_boxes) in zip(outs, metas):
            answer = out.outputs[0].text.strip()
            answer = VLXSeekWorker._remap_object_tokens(answer, sorted_to_original)
            results.append(
                {
                    "answer": answer,
                    "result_bbox_list": VLXSeekWorker._build_result_bbox_list(
                        answer, caller_boxes
                    ),
                    "prompt_tokens": len(out.prompt_token_ids),
                    "completion_tokens": len(out.outputs[0].token_ids),
                    "elapsed": elapsed,
                }
            )
        return results

    def run_task(
        self,
        image: Image.Image,
        task: str,
        text: str | list[str],
        lang: str = "en",
        bbox_list: Optional[Sequence[Sequence[float]]] = None,
        **kwargs,
    ) -> dict:
        """Build a VLX-Seek 1.5 task-template prompt and run inference."""
        prompt = build_prompt("vlx_seek_1_5", task, text, lang=lang)
        return self.predict(image, prompt, bbox_list=bbox_list, **kwargs)

    def detect(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        categories: str | list[str],
        **kwargs,
    ) -> dict:
        """Detect categories among the candidate regions in ``bbox_list``."""
        return self.run_task(
            image, "detection", categories, bbox_list=bbox_list, **kwargs
        )

    def encode_image_cache(
        self,
        image: Image.Image,
        boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """vLLM 版 no-op：引擎常驻，图像编码按请求自动进行，无需显式缓存。"""
        return None

    def clear_image_cache(self) -> None:
        """vLLM 版 no-op。"""
        return None

    def detect_multi_prompt(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        category_batches: list[list[str]],
        **kwargs,
    ) -> dict:
        """多组类别批量检测：一次提交所有批次请求（同图共享前缀 KV）。

        Returns:
            与 detect() 相同格式的 dict，result_bbox_list 为所有批次的并集。
        """
        image = image.convert("RGB")
        caller_boxes = VLXSeekWorker._validate_boxes(bbox_list, image)
        boxes, sorted_to_original = self._order_boxes(caller_boxes)

        lang = kwargs.get("lang", "en")
        requests = []
        for batch in category_batches:
            question = build_prompt("vlx_seek_1_5", "detection", batch, lang=lang)
            question = VLXSeekWorker._remap_prompt_object_tokens(
                question, sorted_to_original
            )
            prompt = self._build_prompt(question, boxes)
            requests.append(self._make_request(image, boxes, prompt))

        outs, _ = self._run_generate(
            requests,
            kwargs.get("max_new_tokens", 512),
            kwargs.get("temperature", 0.2),
            kwargs.get("top_p", 1.0),
            kwargs.get("repetition_penalty", 1.0),
        )
        merged: dict = {
            "answer": "",
            "result_bbox_list": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        for out in outs:
            answer = out.outputs[0].text.strip()
            merged["result_bbox_list"].extend(
                VLXSeekWorker._build_result_bbox_list(answer, caller_boxes)
            )
            if answer:
                merged["answer"] += answer + "\n"
            merged["prompt_tokens"] += len(out.prompt_token_ids)
            merged["completion_tokens"] += len(out.outputs[0].token_ids)
        return merged

    def ground(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        description: str,
        **kwargs,
    ) -> dict:
        """Locate one matching instance among the candidate ``bbox_list``."""
        return self.run_task(
            image, "grounding_single", description, bbox_list=bbox_list, **kwargs
        )

    def count(
        self,
        image: Image.Image,
        bbox_list: Sequence[Sequence[float]],
        description: str,
        **kwargs,
    ) -> dict:
        """Count instances matching ``description`` among ``bbox_list``."""
        return self.run_task(
            image, "counting", description, bbox_list=bbox_list, **kwargs
        )
