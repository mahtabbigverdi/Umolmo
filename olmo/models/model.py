from __future__ import annotations

from typing import NamedTuple, Optional, List, Tuple, Dict, Union, Iterator

import torch
import torchmetrics

from olmo.config import StrEnum
from olmo.nn.beam_search import Constraint, FinalSequenceScorer, Sampler


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """Wrap each OLMo block with its own FSDP instance."""

    by_block_and_size = "by_block_and_size"
    """Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well."""

    size_based = "size_based"
    """Used PyTorch's default size-based auto wrap policy."""


class OLMoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """
    image_output_features: Optional[torch.FloatTensor]
    """
    These are features after passing last hidden states to the visual decoder head
    """
   
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """

    metrics: Optional[Dict[str, Union[torch.Tensor, torchmetrics.Metric]]] = None
    """
    Model-specific metrics and losses
    """

    internal: Optional[Dict[str, torch.Tensor]] = None
    """
    Internal data the might be used for visualizations
    """
    


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """

    internal: Optional[Dict] = None
    """
    Internal data the might be used for visualizations
    """

    image_output_features: Optional[torch.FloatTensor] = None


class ModelBase(torch.nn.Module):

    def reset_parameters(self):
        """Re-initialize the weights from scratch"""
        raise NotImplementedError()

    def reset_with_pretrained_weights(self):
        """Re-initialize the weights, possibly loading pretrained weights for the LLM and ViT"""
        raise NotImplementedError()

    def apply_activation_checkpointing(self):
        """Enable activation checkpointing"""
        raise NotImplementedError()

    def apply_compile(self, **compile_kwargs):
        """Compile the model with `torch.compile`"""
        raise NotImplementedError()

    def warmup_cache(self, device):
        """Pre-fill the buffer-cache"""
        raise NotImplementedError()

    def apply_fsdp2(self, **fully_shard_kwargs):
        """Fully shard this model using `fully_shard`"""
        raise NotImplementedError()

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        """Get a FSDP1 wrap policy for this model."""
        raise NotImplementedError()

    def get_connector_parameters(self) -> Iterator[torch.Tensor]:
        raise NotImplementedError()

    def get_vit_parameters(self) -> Iterator[torch.Tensor]:
        raise NotImplementedError()

    def get_llm_parameters(self) -> Iterator[torch.Tensor]:
        raise NotImplementedError()

    def get_non_weight_decay_params(self) -> Iterator[torch.Tensor]:
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        raise NotImplementedError()

    def num_params(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> OLMoOutput:
        raise NotImplementedError()

    def generate(
        self,
        batch,
        max_steps: int = 10,
        is_distributed: bool=False,
    ) -> OLMoGenerateOutput:
        raise NotImplementedError()
    
    
    
    
    # def generate(
    #     self,
    #     batch,
    #     attention_bias: Optional[torch.Tensor] = None,
    #     max_steps: int = 10,
    #     beam_size: int = 1,
    #     per_node_beam_size: Optional[int] = None,
    #     sampler: Optional[Sampler] = None,
    #     min_steps: Optional[int] = None,
    #     final_sequence_scorer: Optional[FinalSequenceScorer] = None,
    #     constraints: Optional[List[Constraint]] = None,
    #     is_distributed: bool=False,
    #     return_prefill_output: bool = False,
    # ) -> OLMoGenerateOutput:
    #     raise NotImplementedError()
