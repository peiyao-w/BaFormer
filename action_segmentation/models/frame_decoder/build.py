from detectron2.utils.registry import Registry

FRAME_DECODER_REGISTRY = Registry("FRAME_DECODER_MODULE")
# FRAME_DECODER_REGISTRY.__doc__ = """
# Registry for transformer module in MaskFormer.
# """

def build_frame_decoder(cfg):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    # name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    name = cfg.model.action_seg.frame_decoder.name
    model = FRAME_DECODER_REGISTRY.get(name)(cfg)
    return model