"""timm registration for the C-RADIOv4-SO400M inference backbone."""

from timm.models import PretrainedCfg, register_model
from timm.models.vision_transformer import (
    Block,
    LayerScale as TIMMLayerScale,
    VisionTransformer,
    _create_vision_transformer,
)

from .layer_scale import LayerScale


@register_model
def vit_so400m_patch16_224(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    if pretrained:
        raise ValueError("C-RADIOv4-SO400M weights must be loaded from VLX-Seek.")
    kwargs.setdefault("pretrained_cfg", PretrainedCfg())

    model = _create_vision_transformer(
        "vit_so400m_patch16_224",
        pretrained=False,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4304 / 1152,
        **kwargs,
    )
    _patch_layer_scale(model)
    return model


def _patch_layer_scale(model: VisionTransformer) -> None:
    """Use the checkpoint-compatible ``grandma`` parameter name."""

    def replace(layer_scale: TIMMLayerScale) -> LayerScale:
        patched = LayerScale(layer_scale.gamma.shape[0], inplace=layer_scale.inplace)
        patched.load_state_dict(layer_scale.state_dict())
        return patched

    for block in model.modules():
        if isinstance(block, Block):
            if isinstance(block.ls1, TIMMLayerScale):
                block.ls1 = replace(block.ls1)
            if isinstance(block.ls2, TIMMLayerScale):
                block.ls2 = replace(block.ls2)
