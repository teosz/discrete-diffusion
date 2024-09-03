import torch
from .dit import DiT
from .ar_transformer import ARTransformer


def get_backbone(config, vocab_size) -> torch.nn.Module:
    # set backbone
    mtype = config.model.type
    if mtype == "ddit":
        backbone = DiT(config, vocab_size=vocab_size, adaptive=config.time_conditioning)
    elif mtype == "ar_transformer":
        # Adaptive means there is conditional information -> not the case with AR modeling
        backbone = ARTransformer(config, vocab_size=vocab_size)
    # elif self.config.backbone == 'dimamba':
    #  self.backbone = models.dimamba.DiMamba(
    #    self.config,
    #    vocab_size=self.vocab_size,
    #    pad_token_id=pad_token_id
    #  )
    # elif self.config.backbone == 'ar':
    #  self.backbone = models.autoregressive.AR(
    #    self.config,
    #    vocab_size=self.vocab_size,
    #    mask_index=self.mask_index
    #  )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    return backbone
