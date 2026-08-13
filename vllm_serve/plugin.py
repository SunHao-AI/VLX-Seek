# -*- coding: utf-8 -*-
"""vLLM 插件入口（vllm.general_plugins）。

用法（未作为 entry point 安装时，在构造 LLM/Engine 之前手动调用）：
    import vllm_serve.plugin
    vllm_serve.plugin.init()

已安装为 entry point 后（pyproject 中声明 vllm.general_plugins 组），
vLLM 会在引擎初始化前自动调用 init()，覆盖所有 worker 子进程。
"""
from __future__ import annotations


def init() -> None:
    """注册 VLX-Seek config 与 vLLM 模型类（幂等）。"""
    from .vlx_seek_vlm import VLXSeek1_5ForCausalLM, register_config
    from vllm.model_executor.models.registry import ModelRegistry

    register_config()

    # 懒加载字符串：避免在 fork 子进程中重复执行本模块 import 链（CUDA 重初始化风险）
    ModelRegistry.register_model(
        "VLXSeek1_5ForCausalLM",
        "vllm_serve.vlx_seek_vlm:VLXSeek1_5ForCausalLM",
    )
