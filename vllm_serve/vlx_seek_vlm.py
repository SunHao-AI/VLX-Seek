# -*- coding: utf-8 -*-
"""vLLM 0.17.0 自定义多模态模型注册：VLXSeek1_5ForCausalLM。

策略（依据 v0.17.0 源码调研，见 docs/superpowers/plans/2026-08-13-vllm-prefix-kv-migration.md）：
- vLLM 0.17 原生支持 Qwen3.5（vllm/model_executor/models/qwen3_5.py），本类继承
  Qwen3_5ForConditionalGeneration，语言主干 / 采样 / KV 缓存 / 引擎集成全部复用；
- 仅替换视觉栈：vision_tower（Qwen3_5_VlVisionTower）+ mm_projector +
  vision_tower_aux（CRadioV4AuxEncoder）+ object_vp_extractor（HFRE）+ mm_projector_aux，
  全部复用项目的 nn.Module，属性名与 HF 保持完全一致，便于 encode_images/encode_objects 逐字移植；
- 权重加载：AutoWeightsLoader + WeightsMapper（键名前缀改写，照抄 Qwen3VL 的 mapper 模式）。

里程碑：
  1a. 本文件 = 模型类 + embed_multimodal 完整移植（图像/object 双通道）——先做 text-only 验证；
  1b. 自定义 MultiModalProcessor（处理 images_aux / image_grid_thws / bbox_list，并让
      objfeat 占位符进入 is_multimodal 掩码）——下个迭代，当前继承 Qwen3.5 处理器。
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Set

import torch

from transformers import AutoConfig

_VLX_DEBUG = os.environ.get("VLX_DEBUG") == "1"


def _dbg(*args) -> None:
    if _VLX_DEBUG:
        print("[VLX-DEBUG]", *args, flush=True)

from vllm.config import VllmConfig
# 必须继承 vLLM 自己的 Qwen3_5Config（vllm.transformers_utils.configs.qwen3_5），
# 否则 Qwen3_5ProcessingInfo.get_hf_config 的 isinstance 校验失败
# （报错：Expected type vllm...Qwen3_5Config, found vlx_seek_vlm.VLXSeek1_5Config）
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config as _VllmQwen3_5Config

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _VllmQwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration as _VllmQwen3_5VLM,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLMultiModalProcessor,
    Qwen3VLDummyInputsBuilder,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

# 项目视觉栈（属性名与 HF modeling 保持一致）
from vlx_seek.models.vlx_seek_1_5.multimodal_encoder.builder import (
    build_vision_tower,
    build_vision_tower_aux,
)
from vlx_seek.models.vlx_seek_1_5.multimodal_projector.builder import (
    build_vision_projector,
    build_vision_projector_aux,
)
from vlx_seek.models.vlx_seek_1_5.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import (
    HybridFineGrainedRegionEncoder,
)
from vlx_seek.models.vlx_seek_1_5.multimodal_encoder.qwen3_5_vl_encoder import Qwen3_5_VlVisionTower
from vlx_seek.models.vlx_seek_1_5.omchat_arch import _infer_aux_spatial_scale


class VLXSeek1_5Config(_VllmQwen3_5Config):
    """最小 config 类：仅用于让 transformers AutoConfig 认识 vlx_seek_1_5。

    from_pretrained 会把 config.json 的全部字段注入该类，自定义字段
    （mm_vision_tower、vision_config 等）由 JSON 提供，无需在此声明。
    """

    model_type = "vlx_seek_1_5"

    def __init__(self, **kwargs):
        # transformers 5.x 的 from_dict 会先处理 sub_configs：把嵌套的
        # text_config/vision_config 字典转成对象再传入 __init__；而 vLLM 的
        # Qwen3_5Config.__init__ 只处理 dict/None 两种输入，对象会被静默丢弃，
        # 导致属性缺失（报错：text_config does not have num_attention_heads）。
        # 这里统一兜底：dict → 用 vLLM 的 sub_configs 解析成对象；对象 → 直接
        # 透传；None → 沿用 vLLM 默认值；最后无论如何强制写回属性。
        from vllm.transformers_utils.configs.qwen3_5 import (
            Qwen3_5TextConfig,
            Qwen3_5VisionConfig,
        )

        tc = kwargs.pop("text_config", None)
        vc = kwargs.pop("vision_config", None)
        if isinstance(tc, dict):
            tc = Qwen3_5TextConfig(**tc)
        if isinstance(vc, dict):
            vc = Qwen3_5VisionConfig(**vc)
        super().__init__(text_config=tc, vision_config=vc, **kwargs)
        self.text_config = tc if tc is not None else self.text_config
        self.vision_config = vc if vc is not None else self.vision_config

        # vLLM 的 Qwen3_5Config 只认声明过的字段，VLX-Seek 自定义字段
        # （mm_vision_tower / mm_vision_tower_aux / mm_object_hidden_size 等）
        # 在超类构造时会被丢弃；这里从传入 kwargs 找回并写回 self，
        # 否则视觉栈构建条件（getattr(config, "mm_vision_tower")）会失配。
        for key, value in kwargs.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# 1b-2: 自定义 MultiModalProcessor
# 让 <objfeat>(token 248181) 进入 is_multimodal 掩码 + 透传 images_aux/bbox_list
# ---------------------------------------------------------------------------

# <objfeat> 在 tokenizer vocab 中是单 token（248181），processor 无需展开
_OBJFEAT_TOKEN_ID = 248181


class VLXSeekProcessingInfo(Qwen3_5ProcessingInfo):
    """VLX-Seek processing info: 复用 Qwen3.5，无额外配置。"""
    pass


class VLXSeekDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """VLX-Seek dummy inputs: profile run 复用 Qwen3VL（无 object features）。"""
    pass


class VLXSeekMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """VLX-Seek processor:
    - 添加 ``<objfeat>`` PromptReplacement（1:1 替换为 248181，进入 is_multimodal 掩码）
    - 从 ``mm_processor_kwargs`` 透传 ``images_aux`` / ``bbox_list`` 到 mm_kwargs
    """

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        # 基类的 image/video replacement（<|image_pad|> 展开等）
        updates = list(super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        ))
        _dbg(
            "[_get_prompt_updates] base updates:",
            [(u.modality, getattr(u, "target", None)) for u in updates],
        )
        # 移除基类的 image replacement：vLLM 0.17 对同一 item 的多个 prompt updates
        # 只应用第一个（_find_matches 中 "Already found a match for this item" 直接
        # break），追加独立的 <objfeat> replacement 永远不会生效，导致 248181 不进
        # is_multimodal 掩码、对象嵌入在 _merge_multimodal_embeddings 的
        # masked_scatter_ 中被静默丢弃（输出发散为 '小汽车None'）。
        # 这里改为「单个合并 replacement」：target 覆盖 image_pad 段 + objfeat 行，
        # replacement 用 PromptUpdateDetails.is_embed 同时标记图像与 <objfeat> 位置。
        updates = [u for u in updates if u.modality != "image"]

        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        merge_length = image_processor.merge_size ** 2

        image_token_id = hf_processor.image_token_id
        vision_end_id = hf_config.vision_end_token_id
        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_tokens[0] if newline_tokens else 198

        # bbox 数量：与 vlx_seek_vllm_worker._build_prompt 的 <objN> 行一致
        bbox_list = hf_processor_mm_kwargs.get("bbox_list")
        n_boxes = 0
        if isinstance(bbox_list, torch.Tensor):
            n_boxes = bbox_list.shape[-2] if bbox_list.ndim >= 2 else 0
        elif isinstance(bbox_list, (list, tuple)) and len(bbox_list) > 0:
            first = bbox_list[0]
            if isinstance(first, (list, tuple)):
                n_boxes = len(first)
            elif hasattr(first, "shape") and first.ndim >= 1:
                n_boxes = first.shape[0]
        _dbg(
            "[_get_prompt_updates] bbox_list type/shape:",
            type(bbox_list).__name__,
            getattr(bbox_list, "shape", None),
            "n_boxes:",
            n_boxes,
            "mm_items:",
            len(mm_items),
        )

        # <obj0>, <obj1>, ... 为训练时加入词表的特殊 token（应为单 token）
        obj_token_ids = []
        for i in range(n_boxes):
            tokens = tokenizer.encode(f"<obj{i}>", add_special_tokens=False)
            obj_token_ids.append(tokens[0] if tokens else _OBJFEAT_TOKEN_ID)

        def build_target(item_idx: int):
            # 文本形式 target：image_pad 段 + objfeat 行（与 worker _build_prompt 一致，
            # 不含 <|vision_start|>）。文本匹配对 <objN> 的分词更宽容，失败时
            # _apply_prompt_updates 会自动回退到字符串匹配。
            # 注意 dummy/profile 输入（Qwen3VLDummyInputsBuilder）的 prompt 是
            # "<|vision_start|><|image_pad|><|vision_end|>"（无 \n、无 objfeat 行），
            # 故 n_boxes=0 时 target 不含 "\n"，否则会匹配失败。
            if n_boxes == 0:
                return "<|image_pad|><|vision_end|>"
            parts = ["<|image_pad|><|vision_end|>\n"]
            for i in range(n_boxes):
                parts.append(f"<obj{i}><objfeat>")
            return "".join(parts)

        def build_replacement(item_idx: int):
            # 展开后 token 序列：图像 tokens + vision_end（+\n + obj 行）
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            num_tokens = int(grid_thw.prod()) // merge_length

            full = [image_token_id] * num_tokens
            full.append(vision_end_id)
            if n_boxes > 0:
                full.append(newline_id)
                for i in range(n_boxes):
                    full.extend([obj_token_ids[i], _OBJFEAT_TOKEN_ID])

            # is_embed 掩码：标记图像 token 与 <objfeat>（248181）位置，
            # 与 embed_multimodal 返回的「图像行 + 对象行」顺序一一对应。
            details = PromptUpdateDetails.select_token_ids(
                full, [image_token_id, _OBJFEAT_TOKEN_ID]
            )
            _dbg(
                f"[_get_prompt_updates] item {item_idx} num_tokens={num_tokens} "
                f"full_len={len(full)} embed_mask_sum={int(details.is_embed.sum()) if hasattr(details.is_embed, 'sum') else None}"
            )
            return details

        updates.append(PromptReplacement(
            modality="image",
            target=build_target,
            replacement=build_replacement,
        ))
        return updates

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        # 自定义字段：按 image modality batched 切分
        if "images_aux" in hf_inputs:
            fields["images_aux"] = MultiModalFieldConfig.batched("image")
        if "bbox_list" in hf_inputs:
            fields["bbox_list"] = MultiModalFieldConfig.batched("image")
        return fields

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        # 提取自定义参数（不传给 HF processor，避免报错）
        kwargs_copy = dict(mm_kwargs)
        bbox_list = kwargs_copy.pop("bbox_list", None)
        images_aux = kwargs_copy.pop("images_aux", None)

        # 调用基类处理标准 image/video
        result = super()._call_hf_processor(prompt, mm_data, kwargs_copy, tok_kwargs)

        # 合并自定义字段到 BatchFeature
        if bbox_list is not None:
            result["bbox_list"] = bbox_list
        if images_aux is not None:
            result["images_aux"] = images_aux
        return result


@MULTIMODAL_REGISTRY.register_processor(
    VLXSeekMultiModalProcessor,
    info=VLXSeekProcessingInfo,
    dummy_inputs=VLXSeekDummyInputsBuilder,
)
class VLXSeek1_5ForCausalLM(_VllmQwen3_5VLM):
    """vLLM 0.17 实现：Qwen3.5 语言主干 + VLX-Seek 视觉栈。"""

    # HF 权重键 → vLLM 参数键。HF 键形如 model.language_model.* / model.vision_tower.* / lm_head.*
    # 注意 dict 按插入顺序遍历，长前缀必须先于其超集（"model.vision_tower." 在 "model." 之前）。
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # 视觉塔：HF model.vision_tower.visual.blocks.*（vision_tower 属性 → 项目塔）
            #  → vLLM self.visual.visual.blocks.*（只需替换第一段 vision_tower → visual）
            "model.vision_tower.": "visual.",
            # 嵌套 language_model 需要先替换（更长前缀优先），映射到 vLLM 的 language_model.model.*
            "model.language_model.": "language_model.model.",
            # 其余视觉子模块去 model. 前缀即可（本类直接挂在这些属性上）
            "model.": "",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # 注意：super().__init__ 已用 vLLM 的 Qwen3_VisionTransformer 构建 self.visual 并
        # 构建了 self.language_model；这里替换 self.visual 为项目视觉塔并挂载其余模块。
        config = vllm_config.model_config.hf_config
        self.config = config  # HF 代码路径依赖 self.config

        with self._mark_tower_model(vllm_config, {"image"}):
            self.visual = build_vision_tower(config, delay_load=False)
            if getattr(config, "mm_vision_tower_aux", None) is not None:
                self.vision_tower_aux = build_vision_tower_aux(config, delay_load=False)

        # 与项目 builder 一致：mm_vision_tower 或 vision_tower 任一存在即构建投影器
        if getattr(config, "mm_vision_tower", None) is not None or getattr(config, "vision_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)

        if getattr(config, "mm_vision_tower_aux", None) is not None:
            vision_tower = self.get_vision_tower()
            aux_tower = self.get_vision_tower_aux()
            self.object_vp_extractor = HybridFineGrainedRegionEncoder(
                output_size=getattr(config, "mm_roi_output_size", 7),
                spatial_scale=_infer_aux_spatial_scale(aux_tower),
                add_pos_embedding=getattr(config, "mm_add_pos_embed", True),
                pos_embedding_dim=config.mm_object_hidden_size,
                use_vision_tower_object_feature=getattr(config, "mm_use_vision_tower_object_feature", False),
                vision_tower_object_feature_dim=(vision_tower.config.hidden_size * 4 if not getattr(config, "mm_use_simpleFPN_for_vt", False) else vision_tower.config.out_hidden_size),
                vision_tower_spatial_scale=1 / (vision_tower.config.patch_size * vision_tower.config.spatial_merge_size),
                object_feature_combination=getattr(config, "mm_object_feature_combination", "mean"),
                use_vt_object_feature_only=getattr(config, "mm_use_vt_object_feature_only", False),
                use_simpleFPN_for_vt=getattr(config, "mm_use_simpleFPN_for_vt", False),
                use_separate_mlp_for_object=getattr(config, "mm_use_separate_mlp_for_object", False),
                obj_pooling_type=getattr(config, "mm_obj_pooling_type", "mean"),
                use_multi_scale_roi_align=getattr(config, "mm_use_multi_scale_roi_align", False),
                apply_object_layer_norm=getattr(config, "mm_apply_object_layer_norm", False),
                roi_algined=getattr(config, "mm_roi_algined", False),
                use_simpleFPN_for_vt_aux=getattr(config, "mm_use_simpleFPN_for_vt_aux", False),
                aux_feature_dims=[getattr(aux_tower.config, "hidden_size", None)],
                simpleFPN_out_channels_for_vt=getattr(config, "mm_simpleFPN_out_channels_for_vt", 512),
            )
            self.mm_projector_aux = build_vision_projector_aux(config)

    # ------------------------------------------------------------------
    # 视觉编码（逐字移植自 HF modeling_vlx_seek_1_5.py，属性名一致）
    # ------------------------------------------------------------------

    def encode_images(self, images, image_grid_thws=None):
        if isinstance(self.get_vision_tower(), Qwen3_5_VlVisionTower):
            image_features, image_grid_thws, multi_level_features_list = self.get_vision_tower()(images, image_grid_thws)
            if type(image_features) is list:
                token_length_list = [i.shape[1] for i in image_features]
                image_features = torch.cat(image_features, dim=1)
        else:
            image_features = self.get_vision_tower()(images)
            image_grid_thws = None
            multi_level_features_list = None

        image_features = self.mm_projector(image_features)

        if isinstance(self.get_vision_tower(), Qwen3_5_VlVisionTower):
            start = 0
            new_image_features = []
            for length in token_length_list:
                end = start + length
                new_image_features.append(image_features[:, start:end, :].squeeze(0))
                start = end
            image_features = new_image_features

        return image_features, image_grid_thws, multi_level_features_list

    def encode_objects(self, images, bbox_list, vt_multi_level_features=None, vt_images_size=None, add_pos_embed=True):
        aux_image_features_list = self.get_vision_tower_aux()(images)
        object_features = []

        if getattr(self.config, "mm_use_vision_tower_object_feature", False):
            image_features_list = vt_multi_level_features
            for batch_idx, (image_features, aux_image_features, boxes) in enumerate(zip(image_features_list, aux_image_features_list, bbox_list)):
                if boxes is None or len(boxes) == 0:
                    continue
                if getattr(self.config, "mm_use_simpleFPN_for_vt", False):
                    multilevel_visual_feats = image_features[-1]
                else:
                    multilevel_visual_feats = image_features
                if getattr(self.config, "mm_use_simpleFPN_for_vt_aux", False):
                    multilevel_aux_visual_feats = [aux_image_features["image_features"][-1]]
                else:
                    multilevel_aux_visual_feats = aux_image_features["image_features"]
                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_aux_visual_feats[0].device, dtype=torch.float32)
                current_image_height, current_image_width = images[batch_idx].shape[-2:]
                boxes = boxes.to(torch.float32).to(multilevel_aux_visual_feats[0].device)
                original_height, original_width = vt_images_size[batch_idx]
                scale_height = original_height / current_image_height
                scale_width = original_width / current_image_width
                scale_tensor = torch.tensor([scale_width, scale_height, scale_width, scale_height], device=boxes.device)
                vt_boxes = boxes * scale_tensor
                extracted_object_feat = (
                    self.object_vp_extractor(
                        multi_level_features=multilevel_aux_visual_feats,
                        vt_multi_level_features=multilevel_visual_feats,
                        boxes=[boxes],
                        vt_boxes=[vt_boxes],
                        add_pos_embed=add_pos_embed,
                    )
                    .squeeze(0)
                    .to(dtype=next(self.mm_projector_aux.parameters()).dtype)
                )
                object_feat = self.mm_projector_aux(extracted_object_feat)
                object_features.append(object_feat)
        else:
            for batch_idx, image_features in enumerate(aux_image_features_list):
                multilevel_visual_feats = image_features["image_features"]
                boxes = bbox_list[batch_idx]
                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_visual_feats[0].device, dtype=torch.float32)
                current_image_height, current_image_width = images[batch_idx].shape[-2:]
                boxes = boxes.to(torch.float32).to(multilevel_visual_feats[0].device)
                extracted_object_feat = self.object_vp_extractor(multilevel_visual_feats, [boxes], add_pos_embed=add_pos_embed).squeeze(0).to(dtype=next(self.mm_projector_aux.parameters()).dtype)
                object_feat = self.mm_projector_aux(extracted_object_feat)
                object_features.append(object_feat)

        return object_features

    def get_vision_tower(self):
        return self.visual

    def get_vision_tower_aux(self):
        return getattr(self, "vision_tower_aux", None)

    # ------------------------------------------------------------------
    # 多模态嵌入（vLLM 协议：覆盖 embed_multimodal 处理图像 + object features）
    # ------------------------------------------------------------------

    def embed_multimodal(self, **kwargs: object) -> torch.Tensor | None:
        """覆盖基类方法：图像编码 + object features 编码。

        vLLM 要求每个 mm item 恰好返回 1 个 embedding（sanity_check_mm_encoder_outputs
        校验 len(mm_embeddings) == num_items）。图像 + object 属于同一个 image item，
        必须合并为单个 tensor。

        合并后的行序必须与 prompt 中占位符出现顺序一致：
        先 <|image_pad|>(248056) × N，再 <objfeat>(248181) × M。
        ``_merge_multimodal_embeddings`` 按 is_multimodal 掩码顺序 scatter。
        """
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        _dbg("[embed_multimodal] kwargs keys:", list(kwargs.keys()))
        for modality, items in mm_input_by_modality.items():
            _dbg(f"[embed_multimodal] modality={modality} items={len(items)}")

        all_embeds: list[torch.Tensor] = []
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                embeds = self._process_image_and_object(multimodal_input, kwargs)
                all_embeds.extend(embeds)

        if not all_embeds:
            return None
        # 返回 tuple（元素数 = mm item 数），每个元素是单个 tensor：
        # 图像 token + object token 合并。sanity_check_mm_encoder_outputs 校验
        # len(mm_embeddings) == expected_num_items（1 个 image item → 1 个元素）。
        _dbg(
            "[embed_multimodal] returning rows:",
            [tuple(t.shape) for t in all_embeds],
        )
        return (torch.cat(all_embeds, dim=0),)

    def _process_image_and_object(self, image_input, kwargs) -> tuple[torch.Tensor, ...]:
        """编码图像 + object features，返回有序 embeddings tuple。

        顺序：[image_embeds..., object_embeds...]（与 prompt 中占位符顺序一致）。
        """
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input.get("type") == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
            merge_size = self.visual.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            image_embeds = tuple(image_embeds.split(sizes))
            # embeds 路径不提取 multi_level_features（object 不支持 precomputed embeds）
            vt_multi_level_features = None
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            # get_image_features 返回 (split_embeds, vision_output)
            image_embeds, vision_output = self.visual.get_image_features(pixel_values, grid_thw)
            # 提取 multi_level_features（object 编码需要）
            vt_multi_level_features = self._extract_multi_level_features(image_embeds, grid_thw)

        # mm_projector（当前 identity；如改为 MLP 也兼容 2D 输入）
        projected = [self.mm_projector(e) for e in image_embeds]

        # object features 编码（如果有 bbox_list + images_aux）
        object_embeds = self._process_object_input(kwargs, vt_multi_level_features, grid_thw)
        if object_embeds:
            projected.extend(object_embeds)

        return tuple(projected)

    def _extract_multi_level_features(self, image_embeds, grid_thw):
        """从 get_image_features 的输出提取 multi_level_features（object 编码用）。

        image_embeds: tuple of [tokens, hidden]（每张图一个）
        grid_thw: [N, 3] tensor
        """
        if not isinstance(self.visual, Qwen3_5_VlVisionTower):
            return None
        grid_list = grid_thw.tolist()
        if len(grid_list) == 1 and isinstance(grid_list[0], list):
            grid_list = [grid_list]
        multi_level_features_list = []
        for i, embed in enumerate(image_embeds):
            grid_single = grid_thw[i:i + 1]
            mlv = self.visual.get_multi_level_features(embed, grid_single)
            multi_level_features_list.append(mlv)
        return multi_level_features_list

    def _process_object_input(self, kwargs, vt_multi_level_features, grid_thw):
        """编码 object features（bbox_list + images_aux → object embeddings）。

        返回 list of [num_bbox, hidden] tensor（与 prompt 中 <objfeat> 顺序一致）。
        """
        images_aux = kwargs.get("images_aux")
        bbox_list = kwargs.get("bbox_list")

        _dbg(
            "[_process_object_input] images_aux:",
            None if images_aux is None else type(images_aux).__name__,
            None if images_aux is None else getattr(images_aux, "shape", None),
            "bbox_list:",
            None if bbox_list is None else type(bbox_list).__name__,
            None if bbox_list is None else getattr(bbox_list, "shape", None),
        )

        if images_aux is None or bbox_list is None:
            return None

        vision_tower_aux = self.get_vision_tower_aux()
        if vision_tower_aux is None:
            _dbg("[_process_object_input] vision_tower_aux is None")
            return None

        # 检查是否有有效 bbox
        has_bbox = any(b is not None and len(b) > 0 for b in bbox_list)
        _dbg("[_process_object_input] has_bbox:", has_bbox)
        if not has_bbox:
            return None

        if vt_multi_level_features is None:
            _dbg("[_process_object_input] vt_multi_level_features is None")
            return None

        # 计算 vt_images_size（从 grid_thw 推断原图尺寸）
        patch_size = self.visual.config.patch_size
        image_grid_thws = [grid_thw[i:i + 1] for i in range(len(grid_thw))]
        vt_images_size = [thw[0][-2:] * patch_size for thw in image_grid_thws]

        # encode_objects 期望 images_aux 是 list of [C, H, W] tensor。
        # vLLM batched 字段切分后可能是 list of [C,H,W] / [1,C,H,W] / 单个 tensor。
        if isinstance(images_aux, torch.Tensor):
            images_aux = [images_aux]
        # 逐项规整：去掉多余的 batch 维（[1,C,H,W] -> [C,H,W]）
        normalized_aux = []
        for item in images_aux:
            if isinstance(item, torch.Tensor) and item.ndim == 4 and item.shape[0] == 1:
                item = item[0]
            normalized_aux.append(item)
        images_aux = normalized_aux

        # bbox_list 期望是 list of [N, 4] tensor。
        if isinstance(bbox_list, torch.Tensor):
            if bbox_list.ndim == 3:
                # [1, N, 4] -> list of [N, 4]
                bbox_list = [bbox_list[i] for i in range(len(bbox_list))]
            else:
                bbox_list = [bbox_list]
        elif isinstance(bbox_list, list) and len(bbox_list) > 0 and isinstance(bbox_list[0], torch.Tensor) and bbox_list[0].ndim == 3:
            # list of [1, N, 4] -> list of [N, 4]
            bbox_list = [b[0] if b.shape[0] == 1 else b for b in bbox_list]

        object_features = self.encode_objects(
            images_aux, bbox_list, vt_multi_level_features, vt_images_size
        )
        _dbg(
            "[_process_object_input] object_features:",
            None if object_features is None else [tuple(f.shape) for f in object_features],
        )
        return object_features

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        """覆盖基类方法（兼容路径，不含 object features）。

        基类 ``_parse_and_validate_image_input`` 把 kwargs 解析成
        ``Qwen2_5_VLImageInputs``（含 pixel_values / image_grid_thw / type），
        然后调用本方法做实际视觉编码。
        """
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input.get("type") == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
            merge_size = self.visual.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            image_embeds = tuple(image_embeds.split(sizes))
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            image_embeds, _ = self.visual.get_image_features(pixel_values, grid_thw)

        return tuple(self.mm_projector(e) for e in image_embeds)

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=[
                # input_conditioner 的 norm_mean/norm_std 与 radio_model.summary_idxs
                # 是 buffer（非 nn.Parameter），AutoWeightsLoader 不识别普通 buffer，
                # 由 CRadioV4AuxEncoder.load_weights 单独加载，这里忽略递归下钻。
                "vision_tower_aux.image_tower.radio_model.input_conditioner.",
                "vision_tower_aux.image_tower.radio_model.summary_idxs",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


_CONFIG_REGISTERED = False


def register_config() -> None:
    """让 transformers AutoConfig 认识 vlx_seek_1_5（vLLM ModelConfig 校验用）。"""
    global _CONFIG_REGISTERED
    if _CONFIG_REGISTERED:
        return
    try:
        AutoConfig.register("vlx_seek_1_5", VLXSeek1_5Config)
    except ValueError:
        # 已注册过（例如项目环境已 import 过 modeling_vlx_seek_1_5）
        pass
    _CONFIG_REGISTERED = True
