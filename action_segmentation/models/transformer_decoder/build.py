from detectron2.utils.registry import Registry

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
# FRAME_DECODER_REGISTRY.__doc__ = """
# Registry for transformer module in MaskFormer.
# """

def build_transformer_decoder(cfg):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.model.action_seg.transformer_decoder.name
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg)