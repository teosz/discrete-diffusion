import abc
import torch
from .utils import sample_categorical
from torch.nn import functional as F

def sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


class AnalyticSampler(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def forward(self, x, cond):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _t_to_sigma(self, t):
        raise NotImplementedError

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
        i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge
    
    def _analytic_update(self, x, t, step_size, forward=None):
        if forward is None:
            forward = self.forward

        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        log_score = self.get_log_score(x, curr_sigma, forward)
        score = log_score.exp()
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return log_score, sample_categorical(probs)

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score
    
    def get_log_score(self, x, sigma, forward):
        if sigma.ndim > 1:
            cond = sigma.squeeze(-1)
        else:
            cond = sigma

        cond = torch.zeros_like(cond)
        model_output = forward(x, cond)
        assert self.config.parameterization.name in ('mdlm',)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)
        
        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x 
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        
        log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1
        
        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = -10_000_000 * torch.ones_like(
            model_output)
        unmasked_score = torch.scatter(
            unmasked_score,
            -1,
            x[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = - (
            log_k[:, None] * torch.ones_like(x))
        
        masked_indices = (x == self.mask_index).to(
            model_output.dtype)[:, :, None]
        model_output = (
            masked_score * masked_indices
            + unmasked_score * (1 - masked_indices))
        return model_output
