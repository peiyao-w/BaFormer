from detectron2.utils.registry import Registry


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """Registry for backbones"""

def build_backbone(cfg):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    # name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    name = cfg.model.action_seg.backbone.name
    model = BACKBONE_REGISTRY.get(name)(cfg)
    return model