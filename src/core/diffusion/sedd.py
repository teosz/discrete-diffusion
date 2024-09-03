import torch
import numpy as np

from .absorbing import DiffusionCore
from tqdm import trange
from core.sampling import AncestralSampler, AnalyticSampler
from .mdlm import MDLM, sample_uncond, sample_cond_prefix


class SEDD(DiffusionCore, AncestralSampler, AnalyticSampler):
    def __init__(self, config, tokenizer):
        DiffusionCore.__init__(self, config, tokenizer)
        self.normalization = config.parameterization.normalization
        self.validate_config()
        # TODO: add option for the correct scaling

    def forward(self, xt, sigma):
        with torch.amp.autocast("cuda", dtype=torch.float32):
            logits = self.backbone(xt, sigma)

        # Next: sedd parametrization of the output (scale + zero out predictions for input word)
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(
            logits, -1, xt[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits

    def validate_config(self):
        assert not self.change_of_variables
        assert self.normalization in ("mean", "sum", "mean-mask")

    def diffusion_elbo(self, x0, t=None, normalize=True):
        if t is None:
            t = self._sample_t(x0.shape[0], x0.device)

        sigma, move_chance, dsigma = self._t_to_sigma(t)

        xt = self.q_xt(x0, move_chance)

        sigma = sigma.squeeze(-1)  # Original shape [bs, 1]
        logscore = self.forward(xt, sigma)
        se = self._score_entropy(logscore, sigma[:, None], xt, x0)
        elbo = dsigma[:, None] * se

        # Normalize
        if normalize:
            if self.normalization == "mean":
                elbo = elbo.mean()
            elif self.normalization == "sum":
                elbo = elbo.sum()
            elif self.normalization == "mean-mask":
                counts = torch.sum(xt == self.mask_index)
                if counts.item() == 0:  # unlikely but possible
                    elbo = 0
                else:
                    elbo = elbo.sum() / counts
            else:
                raise ValueError(f"Unknown normalization mode `{self.normalization}`")

        return elbo

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
            log_score: float torch.Tensor with shape (batch_size,
                diffusion_model_input_length, vocab_size),
                log score, output of the denoising network.
            xt: int torch.Tensor with shape (batch_size,
                diffusion_model_input_length), input.
            x0: int torch.Tensor with shape (batch_size,
                diffusion_model_input_length), input.
            sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
            loss with shape (batch_size, diffusion_model_input_length)
        """
        masked_indices = xt == self.mask_index

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_score[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_score[masked_indices].exp()

        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(dim=-1) + score[
                :, self.mask_index + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy

    def loss(self, x, t=None):
        elbo = self.diffusion_elbo(x, t)
        return elbo
    
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
        assert not cache_preds, "sedd does not support prediction caching"
        out = MDLM.sample(self, n_samples, num_steps, seq_len, sampler, cache_preds=False, verbose=verbose, add_bos=add_bos, add_eos=add_eos, project_fn=project_fn)
        return out
    