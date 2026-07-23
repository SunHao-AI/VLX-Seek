from .c_radio_v4_aux_encoder import CRadioV4AuxEncoder
from .qwen3_5_vl_encoder import Qwen3_5_VlVisionTower


def build_vision_tower(config, delay_load=False, **kwargs):
    vision_tower = getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)
    if vision_tower is None:
        raise ValueError("mm_vision_tower is not set in the model config.")

    vision_tower_name = vision_tower.lower()
    vision_model_type = getattr(getattr(config, "vision_config", None), "model_type", "").lower()
    if "qwen3.5" in vision_tower_name or "qwen3_5" in vision_tower_name or vision_model_type == "qwen3_5":
        return Qwen3_5_VlVisionTower(vision_tower, args=config, delay_load=delay_load)

    raise ValueError(f"Unsupported vision tower for VLX-Seek 1.5: {vision_tower}")


def build_vision_tower_aux(config, delay_load=False, **kwargs):
    vision_tower_aux = getattr(config, "mm_vision_tower_aux", None) or getattr(config, "vision_tower_aux", None)
    if vision_tower_aux is None:
        return None

    aux_name = vision_tower_aux.lower()
    if "c-radio" in aux_name or "radio" in aux_name:
        return CRadioV4AuxEncoder(
            vision_tower_aux,
            args=config,
            delay_load=delay_load,
            image_size=kwargs.get("image_size", getattr(config, "aux_image_size", 1024)),
            aspect_ratio=kwargs.get("aspect_ratio", getattr(config, "aux_image_aspect_ratio", "squash")),
        )

    raise ValueError(f"Unsupported auxiliary vision tower for VLX-Seek 1.5: {vision_tower_aux}")
