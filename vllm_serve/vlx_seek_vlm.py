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

from typing import Iterable, Optional, Set

import torch

from transformers import AutoConfig

from vllm.config import VllmConfig
# 必须继承 vLLM 自己的 Qwen3_5Config（vllm.transformers_utils.configs.qwen3_5），
# 否则 Qwen3_5ProcessingInfo.get_hf_config 的 isinstance 校验失败
# （报错：Expected type vllm...Qwen3_5Config, found vlx_seek_vlm.VLXSeek1_5Config）
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config as _VllmQwen3_5Config

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _VllmQwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration as _VllmQwen3_5VLM,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

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


class VLXSeek1_5ForCausalLM(_VllmQwen3_5VLM):
    """vLLM 0.17 实现：Qwen3.5 语言主干 + VLX-Seek 视觉栈。"""

    # HF 权重键 → vLLM 参数键。HF 键形如 model.language_model.* / model.vision_tower.* / lm_head.*
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
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

        if getattr(config, "mm_vision_tower", None) is not None:
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
    # 多模态嵌入（vLLM 协议：embed_multimodal 返回待合并的嵌入序列）
    # ------------------------------------------------------------------

    def embed_multimodal(self, **kwargs: object):
        images = kwargs.get("images")
        if images is None or len(images) == 0:
            return None

        images_aux = kwargs.get("images_aux")
        bbox_list = kwargs.get("bbox_list")
        image_grid_thws = kwargs.get("image_grid_thws")

        image_embeds, image_grid_thws, vt_multi_level_features_list = self.encode_images(images, image_grid_thws)
        image_embeds = torch.cat(image_embeds, dim=0)

        has_bbox = False
        if bbox_list is not None:
            for bbox in bbox_list:
                if bbox is not None and len(bbox) > 0:
                    has_bbox = True
                    break

        object_features = []
        if images_aux is not None and self.get_vision_tower_aux() is not None and has_bbox:
            patch_size = self.get_vision_tower().config.patch_size
            vt_images_size_minibatch = [g[0][-2:] * patch_size for g in image_grid_thws]
            tmp_images_aux = [images_aux[i].unsqueeze(0) for i in range(len(images_aux))]
            object_features = self.encode_objects(tmp_images_aux, bbox_list, vt_multi_level_features_list, vt_images_size_minibatch)

        if object_features:
            # 顺序与占位符一致：图像 token 在前，object token 在后
            return (image_embeds, torch.cat(object_features, dim=0))
        return image_embeds

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
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
