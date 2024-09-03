import torch

from .absorbing import DiffusionCore
from functools import partial
from core.sampling import AncestralSampler, AnalyticSampler
from tqdm import trange
from data.utils import params2key
from pathlib import Path
import os

from loguru import logger
import os

import torch

from pathlib import Path
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer

from data import dataloader
from lightning.fabric import Fabric
from utils import str_to_dtype
from tqdm import trange
import numpy as np
from einops import rearrange
import lightning as L

@torch.jit.script
def post_process_model_output(logits: torch.Tensor, xt: torch.Tensor, mask_index: int, neg_infty: float) -> torch.Tensor:
    unmasked_indices = xt != mask_index
    logits[unmasked_indices] = neg_infty
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    logits = torch.log_softmax(logits, dim=-1)

    return logits

class MDLM(DiffusionCore, AncestralSampler, AnalyticSampler):
    def __init__(self, config, tokenizer):
        DiffusionCore.__init__(self, config, tokenizer)
        self.validate_config()
        self.log_loss_buckets = self.config.parameterization.log_loss_buckets

        self.neg_infinity = -1000000.0
        self._post_process_outputs = partial(
            post_process_model_output, mask_index=self.mask_index, neg_infty=-1000000.0
        )

    def forward(self, xt, cond):
        if not self.time_conditioning:
            cond = torch.zeros_like(cond)
        with torch.amp.autocast("cuda",dtype=torch.float32):
            logits = self.backbone(xt, cond)
        logits = self._post_process_outputs(logits, xt)
        return logits

    def validate_config(self):
        assert self.T == 0, "Only continuous mode implemented"

    def diffusion_elbo(self, x0, t=None):
        if t is None:
            t = self._sample_t(x0.shape[0], x0.device)

        sigma, move_chance, dsigma = self._t_to_sigma(t)

        xt = self.q_xt(x0, move_chance)

        sigma = sigma.squeeze(-1)  # Original shape [bs, 1]
        logits = self.forward(xt, sigma)

        log_p_theta = torch.gather(input=logits, dim=-1, index=x0[:, :, None]).squeeze(
            -1
        )

        # TODO: not sure if it's good?
        if self.change_of_variables or self.importance_sampling:
            raise ValueError
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        elbo = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
        if self.trainer.training or self.trainer.validating:
            mode = "train" if self.trainer.training else "valid"
            # Log loss without scaling
            to_log = (-log_p_theta).mean(-1)
            key = "raw_loss"
            self._log_buckets(to_log, t, mode=mode, key=key)
            # Log elbo with scaling
            to_log = elbo.mean(-1)
            key = "scaled_loss"
            self._log_buckets(to_log, t, mode=mode, key=key)
        return elbo

    def loss(self, x, t=None):
        elbo = self.diffusion_elbo(x, t)
        return elbo.mean()

    @torch.no_grad()
    def sample(
        self,
        n_samples=8,
        num_steps=256,
        seq_len=1024,
        sampler="ancestral",
        cache_preds=False,
        verbose=False,
        add_bos=False,
        add_eos=False,
        project_fn=lambda x: x,
    ):
        assert not cache_preds, "Not implemented"
        if cache_preds:
            assert (
                not self.config.time_conditioning
            ), "Cannot use caching with time-conditional network"

        assert sampler in ("ancestral", "analytic")
        if seq_len is None:
            seq_len = self.config.model.length

        batch = self._sample_prior(n_samples, seq_len)
        batch = project_fn(batch)

        if add_bos:
            batch[:, 0] = self.tokenizer.bos_token_id

        if add_eos:
            batch[:, -1] = self.tokenizer.eos_token_id

        # +1 because we use the last value for denoising
        ts = torch.linspace(1.0, self.sampling_eps, steps=num_steps + 1)
        dt = (1 - self.sampling_eps) / num_steps

        for i in trange(num_steps, desc="sampling...", disable=not verbose):
            t = ts[i] * torch.ones(n_samples, 1, device=self.device)
            if sampler == "ancestral":
                _, new_batch = self._ddpm_update(batch, t, dt)
            elif sampler == "analytic":
                _, new_batch = self._analytic_update(batch, t, dt)
            new_batch = project_fn(new_batch)
            # If no caching or an update was made, remove cache
            # if not cache_preds or not torch.allclose(new_batch, batch):
            #    cache = None
            batch = new_batch

        # Denoise
        if (batch == self.mask_index).any():
            t = ts[-1] * torch.ones(n_samples, 1, device=self.device)
            _, batch = self._ddpm_update(
                batch, t, dt, denoise=True, mask_idx=self.mask_index
            )
            batch = project_fn(batch)

        return batch


def sample_uncond(module):
    logger.info("Starting unconditional sampling.")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    uncond_cfg = sampling_cfg.uncond

    metadata = dict(
        checkpoint_name=Path(config.checkpointing.resume_ckpt_path).name,
        num_samples=uncond_cfg.num_samples,
        from_ema=uncond_cfg.from_ema,
        num_steps=uncond_cfg.num_steps,
        seq_len=uncond_cfg.seq_len,
        sampler=uncond_cfg.sampler,
        add_bos=uncond_cfg.add_bos,
        add_eos=uncond_cfg.add_eos,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "uncond" / save_fname
    assert not save_path.exists(), save_fname

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(100 + fabric.global_rank)
    # Note: the next line creates a bug when calling functions from the module
    # pl_module = fabric.setup(module)
    pl_module = module
    fabric.to_device(pl_module)

    bs = uncond_cfg.batch_size
    num_steps = uncond_cfg.num_steps
    seq_len = uncond_cfg.seq_len
    target_num_samples = uncond_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert target_num_samples % (tot_num_device * bs) == 0
    n_sampling_rounds = target_num_samples // (tot_num_device * bs)

    if uncond_cfg.from_ema:
        pl_module.store_ema()

    all_samples = []
    for _ in trange(
        n_sampling_rounds,
        desc=f"Sampling with n_steps={num_steps}, seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        with fabric.autocast():
            out = pl_module.sample(
                n_samples=bs,
                num_steps=num_steps,
                seq_len=seq_len,
                sampler=uncond_cfg.sampler,
                add_bos=uncond_cfg.add_bos,
                add_eos=uncond_cfg.add_eos,
                cache_preds=uncond_cfg.cache_preds,
            )
        out = fabric.all_gather(data=out)
        if fabric.global_rank == 0:
            if out.ndim == 3:  # ndim == 2 when running on one device
                out = rearrange(out, "dev bs l -> (dev bs) l")
            all_samples.append(out.cpu())
        del out

    # Join and save to disk
    if fabric.global_rank == 0:
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_samples = all_samples[:target_num_samples]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(save_path, samples=all_samples, metadata=metadata)
        logger.info(f"Saved {len(all_samples)} samples in {save_path}")

    # Restore orig model weights
    if uncond_cfg.from_ema:
        pl_module.restore_ema()


def sample_cond_prefix(module):
    logger.info("Starting conditional sampling (cond on prefix).")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    cond_cfg = sampling_cfg.cond_prefix

    metadata = dict(
        checkpoint_name=Path(config.checkpointing.resume_ckpt_path).name,
        num_samples=cond_cfg.num_samples,
        from_ema=cond_cfg.from_ema,
        dataset=cond_cfg.dataset,
        seq_len=cond_cfg.seq_len,
        prefix_len=cond_cfg.prefix_len,
        num_cont_per_prefix=cond_cfg.num_cont_per_prefix,
        min_seq_len=cond_cfg.min_seq_len,
        num_steps=cond_cfg.num_steps,
        sampler=cond_cfg.sampler,
        add_bos=cond_cfg.add_bos,
        add_eos=cond_cfg.add_eos,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "cond" / save_fname
    assert not save_path.exists(), save_fname
    # Extract args from cfg
    bs = cond_cfg.batch_size
    prefix_len = cond_cfg.prefix_len
    num_steps = cond_cfg.num_steps
    seq_len = cond_cfg.seq_len
    target_num_samples = cond_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert target_num_samples % (tot_num_device * bs) == 0
    # Load prefix dataset
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    dataset = dataloader.get_dataset(
        cond_cfg.dataset,
        tokenizer,
        mode="valid",
        cache_dir=config.data_preprocess.data_cache,
        num_proc=config.trainer.devices * config.loader.num_workers,
        min_seq_len=cond_cfg.min_seq_len,
        seq_len=seq_len,
        group_text=False,
        remove_text=True,
        add_bos=cond_cfg.add_bos,
        add_eos=cond_cfg.add_eos,
    )

    assert len(dataset) >= target_num_samples
    dataset = dataset.select(range(cond_cfg.num_samples))

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(200 + fabric.global_rank)

    pl_module = module
    fabric.to_device(pl_module)

    if cond_cfg.from_ema:
        pl_module.store_ema()

    all_samples = []
    start = fabric.global_rank * bs
    stop = target_num_samples
    end = fabric.world_size * bs
    for idx in trange(
        start,
        stop,
        end,
        desc=f"Sampling with n_steps={num_steps}, seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        docs = dataset[idx : idx + bs]["input_ids"]
        prefixes = docs[:, :prefix_len]

        def project_fn(batch):
            batch[:, :prefix_len] = prefixes
            return batch

        # Generate potentially multiple continuations per prefix (typically 5)
        for _ in range(cond_cfg.num_cont_per_prefix):
            with fabric.autocast():
                out = pl_module.sample(
                    n_samples=bs,
                    num_steps=num_steps,
                    seq_len=seq_len,
                    sampler=cond_cfg.sampler,
                    add_bos=cond_cfg.add_bos,
                    add_eos=cond_cfg.add_eos,
                    cache_preds=cond_cfg.cache_preds,
                    project_fn=project_fn,
                )
            out = fabric.all_gather(data=out)
            if fabric.global_rank == 0:
                # unstack after all_gather
                if out.ndim == 3:
                    out = rearrange(out, "dev bs l -> (dev bs) l")
                all_samples.append(out.cpu())
            del out

    # Join and save to disk
    if fabric.global_rank == 0:
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_samples = all_samples[:target_num_samples]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        references = dataset[:target_num_samples]["input_ids"].numpy()
        np.savez(
            save_path, samples=all_samples, references=references, metadata=metadata
        )
        logger.info(f"Saved samples in {save_path}")

    if cond_cfg.from_ema:
        pl_module.restore_ema()
