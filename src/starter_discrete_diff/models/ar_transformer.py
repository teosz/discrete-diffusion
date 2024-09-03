import torch
import torch.nn.functional as F
import huggingface_hub
from .modules.transformer import Transformer
from .modules.embeddings.time import TimestepEmbedder


class ARTransformer(Transformer, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__(config, vocab_size, adaptive=False)
        assert self.causal
        assert not config.time_conditioning

    def forward(self, indices):
        x = self.vocab_embed(indices)
        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)
            x = self.output_layer(x)

        return x
