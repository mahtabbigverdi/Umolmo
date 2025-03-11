import dataclasses
from typing import Optional, Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass
class TokenSelectionConfig:
    hard: bool = False
    noise: float = 0.0
    loss_weight: float = 0.01
    target_scale: float = 0.7 
    loss_pow: float = 1
    dropout: float = 0
    loss: str = "batch-mean"
    scale_at_test: bool = False
    rescale: float = 1  
    offset: float = 0.2
    attention_scaling: Optional[float]=None

    def build(self) -> 'TokenSelector':
        return TokenSelector(self)


@dataclasses.dataclass
class SelectionOutput:
    selection: torch.Tensor
    """[batch, n_tokens] Indices of tokens to select"""

    token_importance_mask: Optional[torch.Tensor]=None
    """[batch, n_tokens] mask over selected tokens"""

    token_importance: Optional[torch.Tensor]=None
    """[batch, n_tokens] Importance scores of selected tokens"""

    loss: Optional[torch.Tensor]=None
    """Losses to train the selector on"""

    metrics: Optional[Dict[str, torch.Tensor]]=None
    """Other metrics to log for the selector"""


class TokenSelector(nn.Module):
    MIN = torch.finfo(torch.bfloat16).min

    def __init__(self, config: TokenSelectionConfig):
        super().__init__()
        self.config = config

    def forward(self, scores, input_mask, topk_mask) -> SelectionOutput:
        cfg = self.config
        dev = scores.device
        batch, dim = scores.size()
        if cfg.rescale != 1:
            scores = scores / cfg.rescale
        if cfg.offset:
            scores = scores + cfg.offset

        # scores = scores - (scores * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        if input_mask is not None:
            scores = torch.where(input_mask, scores, self.MIN)
        k = topk_mask.shape[-1]

        metrics = {}
        if not self.training:
            top_k, selection = torch.topk(scores, k=k, sorted=True)
            valid = top_k > self.MIN
            if topk_mask is not None:
                valid = valid & (topk_mask > 0)
            importance_scores = torch.sigmoid(top_k) if cfg.scale_at_test else None
            loss = None
            metrics["HighResSelection"] = torch.sigmoid(top_k).flatten()[valid.flatten()]
        else:
            if cfg.noise:
                noise = torch.normal(mean=cfg.offset, std=cfg.noise, size=scores.shape, device=scores.device)
                top_k, top_k_ixs = torch.topk(scores + noise, k=k, sorted=True)
                batch_idx = torch.tile(torch.arange(batch, device=scores.device).unsqueeze(1), [1, cfg.n_features])
                top_k = scores[batch_idx, top_k_ixs]
                # hmm, how should this interact with the loss?
                raise NotImplementedError()
            else:
                top_k, top_k_ixs = torch.topk(scores, k=k, sorted=True)

            valid = (top_k > self.MIN)
            if topk_mask is not None:
                valid = valid & (topk_mask > 0)

            selection = top_k_ixs
            importance_scores = F.sigmoid(top_k)

            if cfg.dropout is not None:
                mask = torch.empty_like(importance_scores).uniform_(0, 1)
                mask = (mask > cfg.dropout).float()
                importance_scores = mask * importance_scores + (1 - mask)
                loss_mask = valid * mask
            else:
                loss_mask = valid.float()
            n_valid = loss_mask.sum()
            importance_score_mean = (importance_scores*loss_mask).sum() / n_valid

            if cfg.loss == "batch-mean":
                loss = torch.pow(torch.abs(importance_score_mean - cfg.target_scale), cfg.loss_pow) * cfg.loss_weight
            elif cfg.loss == "example-mean":
                # Compute loss on each example individually
                _importance_score_mean = ((importance_scores*loss_mask).sum(-1) / loss_mask.sum(-1))
                loss = torch.pow(torch.abs(
                    _importance_score_mean - cfg.target_scale), cfg.loss_pow).mean() * cfg.loss_weight
            elif cfg.loss == "loss-mean":
                _importance_score_mean = ((importance_scores*loss_mask).sum(-1) / loss_mask.sum(-1)).mean()
                loss = torch.pow(torch.abs(
                    _importance_score_mean - cfg.target_scale), cfg.loss_pow) * cfg.loss_weight
            else:
                raise NotImplementedError(cfg.loss)

            metrics["TokenScaleMean"] = importance_score_mean
            metrics["TokenScaleVar"] = (importance_scores - importance_score_mean).square().sum() / n_valid
            metrics["TokenScale80"] = ((importance_scores > 0.80).float()*loss_mask).sum() / n_valid
            metrics["TokenScale20"] = ((importance_scores < 0.20).float()*loss_mask).sum() / n_valid
            metrics["HighResSelection"] = importance_scores.flatten()[valid.flatten()]

        return SelectionOutput(
            selection=selection,
            token_importance=importance_scores,
            metrics=metrics,
            loss=loss,
            token_importance_mask=valid
        )