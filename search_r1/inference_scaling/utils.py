from typing import Dict, Union, List, Optional

import os
import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def kl_uniform_from_logits(logits: torch.Tensor):
    """
    Compute KL(U || p) where U is uniform over vocabulary.
    logits: Tensor of shape (..., V)
    returns: KL divergence, shape (...)
    """
    V = logits.size(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # term: log(V * p(j)) = log_probs + log(V)
    term = log_probs + torch.log(torch.tensor(V, device=logits.device, dtype=logits.dtype))
    kl = -term.mean(dim=-1)   # average over vocab
    return kl

def gini_impurity_from_logits(logits: torch.Tensor):
    # logits shape: (..., V)
    lse = torch.logsumexp(logits, dim=-1)
    lse2 = torch.logsumexp(2 * logits, dim=-1)
    log_sum_p2 = lse2 - 2 * lse
    return 1.0 - torch.exp(log_sum_p2)

