import core
from core import diffusion
from core.ar import ARCore


def get_diffusion(config, tokenizer):
    mode = config.parameterization.name
    if mode == "sedd":
        return diffusion.SEDD(config, tokenizer)
    elif mode == "mdlm":
        return diffusion.MDLM(config, tokenizer)
    elif mode == "ar":
        return ARCore(config, tokenizer)
    else:
        raise ValueError(f"Unknown parameterization `{mode}`")


def get_diffusion_module(config):
    mode = config.parameterization.name
    if mode == "mdlm":
        return diffusion.mdlm
    elif mode == "ar":
        return core.ar
    elif mode == "sedd":
        return diffusion.sedd
    else:
        raise ValueError(f"Unknown parameterization `{mode}`")
