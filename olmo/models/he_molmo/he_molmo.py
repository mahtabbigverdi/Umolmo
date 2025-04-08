import dataclasses
import logging
import math
from dataclasses import field
from typing import Optional, Iterator, Sequence, Tuple, List, Dict, ClassVar
import numpy as np

import torch
from einops import einops
from torch import nn
from torch.distributed.fsdp import fully_shard

from torch.nn import functional as F
from olmo import tokenizer
from olmo.config import BaseConfig, D
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.model_preprocessor import Preprocessor
from olmo.models.he_molmo.he_collator import HeMMCollator
from olmo.models.he_molmo.he_preprocessor import HePreprocessorConfig
from olmo.models.he_molmo.token_selector import TokenSelectionConfig, SelectionOutput
from olmo.nn.beam_search import FinalSequenceScorer, Constraint, Sampler, BeamSearch
from olmo.nn.image_vit import ResidualAttentionBlock, DinoResidualAttentionBlock, VisionTransformer, SiglipVisionTransformer, DinoVisionTransformer
from olmo.nn.legacy_config import convert_legacy_config
from olmo.nn.llm import LlmConfig, Llm, OLMoBlock, llm_activation_checkpoint_function, LayerNormBase
from olmo.models.model import FSDPWrapStrategy, OLMoOutput, OLMoGenerateOutput, ModelBase
from olmo.models.model_config import BaseModelConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig, MolmoVisionBackbone
from olmo.torch_util import BufferCache, get_default_device
from olmo.util import flatten_list


log = logging.getLogger(__name__)


@dataclasses.dataclass
class TokenScorerConfig(BaseConfig):
    selection_model: str = "linear"
    normalize_importance_scores: bool = False
    four_scores_per_low_res_patch: bool = True
    source: Optional[str] = "low_res_embeddings"
    low_res_features_drop: float = 0.1
    bp_low_res_end: int = -1
    bp_low_res_start: int = 0
    bp_patch_prior: bool = False
    bp_low_res_embed: bool = False
    high_res_patch_prior: bool = True
    high_res_patch_prior_drop: Optional[float] = 0.1
    high_res_col_tokens: bool = True
    unconditioned: bool = False
    n_low_res_layers: Optional[int] = None
    learned_rescaling: Optional[str] = None
    attention_scaling: Optional[float] = None
    vector_query: Optional[str] = None
    vector_query_scaling: Optional[str] = None
    importance_norm: bool = False
    low_to_high_interpolation_factor: int = 2


@dataclasses.dataclass
class HeMolmoConfig(BaseModelConfig):
    """Molmo model configuration"""

    _model_name: ClassVar[str] = "he_molmo"

    llm: LlmConfig = field(default_factory=LlmConfig)
    """LLM to use for generation"""

    shared_low_high_embedding: bool = True

    vision_backbone: Optional[MolmoVisionBackboneConfig] = field(default_factory=MolmoVisionBackboneConfig)
    """Vision embedding module to get image features"""

    data_formatter: DataFormatter = field(default_factory=lambda: DataFormatter(image_last=True))
    """How to prompt the model for different tasks"""

    token_scorer: TokenScorerConfig = field(default_factory=TokenScorerConfig)
    """"How to get token scores"""

    token_selector: TokenSelectionConfig = field(default_factory=TokenSelectionConfig)
    """"How to select tokens using the scores"""

    mm_preprocessor: HePreprocessorConfig = field(default_factory=HePreprocessorConfig)
    """How to crop images and encoding jointly with text"""

    bi_directional_attn: str = "within_image"

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "llm" not in config:
            # Old v1 style config
            config = convert_legacy_config(config)
        config.llm = LlmConfig.update_legacy_settings(config.llm)
        if config.vision_backbone is not None:
            config.vision_backbone = MolmoVisionBackboneConfig.update_legacy_settings(config.vision_backbone)
        config.data_formatter = DataFormatter.update_legacy_settings(config.data_formatter)
        config.mm_preprocessor = HePreprocessorConfig.update_legacy_settings(config.mm_preprocessor)
        return config

    def build_tokenizer(self):
        """Tokenizer this model uses"""
        return self.llm.build_tokenizer()

    def build_preprocessor(
        self,
        for_inference,
        is_training=True,
        include_image: bool = False,
        max_seq_len: Optional[int] = None,
    ) -> Preprocessor:
        """
        Build a preprocessor that converts 'raw' image/text data from various tasks into tensors
        inputs/targets that can be passed to the model's forward/generate methods
        """
        return Preprocessor(
            self.data_formatter,
            self.mm_preprocessor.build(self.build_tokenizer(), self.vision_backbone, max_seq_len=max_seq_len, low_to_high_interpolation_factor=self.token_scorer.low_to_high_interpolation_factor),
            for_inference=for_inference,
            is_training=is_training,
            include_image=include_image,
        )

    def build_collator(self, sequence_length, pad_mode: str, include_metadata=True) -> HeMMCollator:
        """Collators for tensors from the preprocessor produces"""
        padding_lens = self.mm_preprocessor.get_padding_lens(self.vision_backbone)
        if pad_mode:
            assert self.max_sequence_length >= sequence_length
            log.info(f"Building collator, pad={pad_mode} seq_len={sequence_length} " +
                     " ".join(f"{k}={v}" for k, v in padding_lens.items()))
        return HeMMCollator(
            self.build_tokenizer(),
            sequence_length,
            padding_lens,
            include_metadata=include_metadata,
            pad=pad_mode,
        )

    def build_model(self, device=None):
        return HeMolmo(self, device)

    @property
    def max_sequence_length(self):
        return self.llm.max_sequence_length


class MLP(nn.Module):

    def __init__(self, in_fe, hidden_size, out_dim, scale: float=None):
        super().__init__()
        self.scale = scale

        self.w1 = nn.Linear(
            in_fe,
            hidden_size*2,
            bias=False,
            )
        self.w2 = nn.Linear(
            hidden_size,
            out_dim,
            bias=True,
        )

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1.weight, std=0.02, a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=0.02, a=-2.0, b=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.w2(x)
        if self.scale:
            x *= self.scale
        return x


class HeMolmo(ModelBase):

    def __init__(self, config: HeMolmoConfig, device=None):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        self.transformer: Llm = self.config.llm.build(self.__cache, device)
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        self.vision_backbone = self.config.vision_backbone.build(self.config.llm, device)
        self.token_selector = config.token_selector.build()
        if self.config.bi_directional_attn:
            self.special_ids = tokenizer.get_special_token_ids(self.config.build_tokenizer())
            self.__cache["image_tokens"] = torch.as_tensor([self.special_ids[x] for x in [
                tokenizer.IM_START_TOKEN,
                tokenizer.IMAGE_PATCH_TOKEN,
                tokenizer.IM_COL_TOKEN,
                tokenizer.IM_END_TOKEN,
                tokenizer.IMAGE_LOW_RES_TOKEN,
            ]], dtype=torch.long, device=get_default_device())
            ts_config = config.token_selector
        self._image_end_token_id = self.special_ids[tokenizer.IM_END_TOKEN]
        self._image_start_token_id = self.special_ids[tokenizer.IM_START_TOKEN]
        self._low_res_patch_id = self.special_ids[tokenizer.IMAGE_LOW_RES_TOKEN]
        self._high_res_patch_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]
        self._block_checkpoint_fn = None

        ts_config = self.config.token_scorer
        llm_cfg = self.config.llm
        if ts_config.learned_rescaling in ["norm-with-bias", "rescale-with-bias"]:
            self.rescale = nn.Linear(1, 1, True)
            torch.nn.init.ones_(self.rescale.weight)
            torch.nn.init.zeros_(self.rescale.bias)
        elif ts_config.learned_rescaling is not None:
            raise NotImplementedError(ts_config.learned_rescaling)

        assert ts_config.source is None or (ts_config.source in ["all_layers"])
        if ts_config.source in ["all_layers"]:
            n_layers = ts_config.n_low_res_layers or llm_cfg.n_layers
            n_low_features = llm_cfg.d_model*n_layers
        else:
            n_low_features = llm_cfg.d_model

        if ts_config.vector_query:
            out_dim = llm_cfg.d_model
            if ts_config.vector_query_scaling:
                out_dim += 2
            self.image_query_ln = nn.Linear(n_low_features, out_dim, bias=True)

        if ts_config.selection_model in ["prior_only", "random"]:
            pass
        elif ts_config.selection_model == "linear":
            self.importance_ln = nn.Linear(n_low_features, ts_config.low_to_high_interpolation_factor**2, bias=False)
        else:
            self.importance_ln = MLP(n_low_features, 1024, ts_config.low_to_high_interpolation_factor**2,
                                     scale=1024**(-.5) if ts_config.normalize_importance_scores else None)

        if ts_config.importance_norm:
            self.importance_norm = LayerNormBase.build(
                llm_cfg, size=n_low_features,
                elementwise_affine=llm_cfg.attention_layer_norm_with_affine
            )

        if ts_config.high_res_patch_prior:
            self.patch_importance_ln = MLP(llm_cfg.d_model, 1024, 1,
                                           scale=1024**(-.5) if ts_config.normalize_importance_scores else None)
            if ts_config.importance_norm:
                self.patch_importance_norm = LayerNormBase.build(
                    llm_cfg, size=llm_cfg.d_model,
                    elementwise_affine=llm_cfg.attention_layer_norm_with_affine
                )

    def get_token_scoring_modules(self) -> Iterator[nn.Module]:
        for k in ["image_query_ln", "importance_ln", "patch_importance_ln", "low_res_features_ln",
                  "importance_norm", "patch_importance_norm"]:
            if hasattr(self, k):
                yield getattr(self, k)

    def reset_non_pretrained(self):
        for module in self.get_token_scoring_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.llm.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.normal_(module.bias, std=self.config.llm.initializer_range)
            elif isinstance(module, (MLP, LayerNormBase)):
                module.reset_parameters()
            else:
                raise NotImplementedError()

    def reset_parameters(self):
        """Re-initialize the weights from scratch"""
        self.reset_non_pretrained()
        self.transformer.reset_parameters()
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()

    def reset_with_pretrained_weights(self):
        """Re-initialize the weights, possibly loading pretrained weights for the LLM and ViT"""
        self.transformer.reset_with_pretrained_weights()
        self.reset_non_pretrained()
        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights()

    def apply_activation_checkpointing(self):
        """Enable activation checkpointing"""
        # self._block_checkpoint_fn = llm_activation_checkpoint_function(self.config.llm)
        self.transformer.apply_activation_checkpointing()
        if self.vision_backbone is not None:
            self.vision_backbone.apply_activation_checkpointing()

    def apply_compile(self, **compile_kwargs):
        """Compile the model with `torch.compile`"""
        self.transformer.apply_compile(**compile_kwargs)
        self.vision_backbone.apply_compile(**compile_kwargs)

    def warmup_cache(self, device):
        """Pre-fill the buffer-cache"""
        for val in self.__cache.values():
            val.to(device)
        if self.transformer.blocks[0].rotary_emb is not None:
            self.transformer.blocks[0].rotary_emb.warmup_cache(device)

    def apply_fsdp2(self, **fully_shard_kwargs):
        """Fully shard this model using `fully_shard`"""
        if self.vision_backbone is not None:
            self.vision_backbone.apply_fsdp2(**fully_shard_kwargs)
        self.transformer.apply_fsdp2(**fully_shard_kwargs)
        fully_shard(list(self.get_token_scoring_modules()), **fully_shard_kwargs)
        fully_shard(self, **fully_shard_kwargs)

    def get_fsdp_wrap_policy(self, wrap_strategy):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)
        if hasattr(self.transformer, "ln_f"):
            size_based_module_to_wrap.add(self.transformer.ln_f)
        if self.vision_backbone is not None:
            size_based_module_to_wrap.add(self.vision_backbone.image_pooling_2d)
            size_based_module_to_wrap.add(self.vision_backbone.image_projector)
        for module in self.get_token_scoring_modules():
            size_based_module_to_wrap.add(module)

        wrap_layer_names = (OLMoBlock, ResidualAttentionBlock, DinoResidualAttentionBlock, MolmoVisionBackbone, VisionTransformer, SiglipVisionTransformer, DinoVisionTransformer)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn

        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        else:
            raise NotImplementedError(wrap_strategy)

    def get_connector_parameters(self) -> Iterator[torch.Tensor]:
        parameters = list(self.vision_backbone.get_connector_parameters())
        if self.config.llm.additional_vocab_size:
            parameters.append(self.transformer.wte.new_embedding)
        return parameters + flatten_list(x.parameters() for x in self.get_token_scoring_modules())

    def get_vit_parameters(self) -> Iterator[torch.Tensor]:
        if self.vision_backbone is None:
            return []
        else:
            return self.vision_backbone.image_vit.parameters()

    def get_llm_parameters(self) -> Iterator[torch.Tensor]:
        if self.config.llm.additional_vocab_size:
            return (
                param for param in self.transformer.parameters() if
                param is not self.transformer.wte.new_embedding
            )
        else:
            return self.llm.parameters()

    def get_non_weight_decay_params(self) -> Iterator[torch.Tensor]:
        exclude_list = {
            "wte", "attn_norm", "ff_norm",
            "pre_attn_norm", "post_attn_norm",
            "pre_ff_norm", "post_ff_norm",
            "ln_f",
            "pre_ln",
            "attention_norm", "ffn_norm",
            "lambda1", "lambda2",
            "positional_embedding", "class_embedding", "patch_embedding",
        }
        return (param for name, param in self.named_parameters() if
                any(part in exclude_list for part in name.split(".")))

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device

    def num_params(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        """Get the total number of parameters."""
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        if not include_inactive_params:
            # Need to reduce blocks to the number of experts that are selected
            # If not dropless 'transformer.blocks.0.ffn.experts.mlp.w1' has shape (total_experts, in_dim, out_dim)
            # change to 'transformer.blocks.0.ffn.experts.mlp.w1' with shape (selected_experts, in_dim, out_dim)
            # If dropless, the total_experts & out_dim are combined into one dimension
            idx = self.config.llm.moe_top_k
            if self.config.llm.moe_dropless:
                idx *= self.transformer.blocks[1].moe_args.ffn_hidden_size
            params = [(np[0], np[1][:idx]) if "experts.mlp" in np[0] else np for np in params]  # type: ignore
        return sum(p.numel() for _, p in params)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        patch_ids: Optional[torch.Tensor] = None,

        # Data for token selection
        high_res_tokens_idx: Optional[torch.Tensor] = None,
        low_res_tokens_idx: Optional[torch.Tensor] = None,
        low_to_high: Optional[torch.Tensor] = None,
        high_res_features_weights: Optional[torch.Tensor] = None,
        high_res_pos_ids: Optional[torch.Tensor] = None,

        # Generation/output args
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None
    ):
        has_image = images is not None
        use_low_res_cache = False
        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        dim = self.config.llm.d_model

        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if attention_mask is None:
            attention_mask = input_ids != -1

        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = subsegment_mask & attention_mask[:, :, None] & attention_mask[:, None, :]
            attention_mask = attention_mask | torch.eye(seq_len, device=attention_mask.device, dtype=torch.bool)[None, :, :]
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
            position_ids = torch.clamp(position_ids, 0)
        else:
            if position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                    ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

        if input_embeddings is not None:
            x = input_embeddings
        elif self.config.shared_low_high_embedding:
            x = self.transformer.wte(torch.where(input_ids == self._low_res_patch_id, self._high_res_patch_id, input_ids))
        else:
            x = self.transformer.wte(input_ids)

        num_image: Optional[int] = None

        # Transform the attention mask into the 4D tensor blocks expect.
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, :past_length + seq_len]
            attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = attention_mask.unsqueeze(1)

        # Integrate the masking inherent to the LLM into `attention_mask`
        casual_mask_len = past_length + seq_len
        if "casual_mask" not in self.__cache or self.__cache["casual_mask"].shape[-1] < casual_mask_len:
            self.__cache["casual_mask"] = torch.tril(torch.ones(
                casual_mask_len, casual_mask_len,
                device=x.device, dtype=torch.bool))[None, None, :, :]
        casual_mask = self.__cache["casual_mask"].to(x.device)[:, :, :casual_mask_len, :casual_mask_len]

        # Possibly turn on bidirectional attention for image tokens
        if self.config.bi_directional_attn:
            image_tokens = self.__cache["image_tokens"].to(input_ids.device)
            is_image_token = torch.any(input_ids[:, :, None] == image_tokens, -1)
            can_attend_bk = (is_image_token[:, :, None] & is_image_token[:, None, :])
            if self.config.bi_directional_attn == "within_image":
                # images cannot attend to one another
                image_starts = input_ids == self._image_start_token_id
                image_segment_ids = torch.cumsum(image_starts, -1)
                bi_dir_mask = can_attend_bk & (image_segment_ids[:, None, :] == image_segment_ids[:, :, None])
            elif self.config.bi_directional_attn == "within_resolution":
                # low-res to low-res, high-res to high-res
                is_high_res = torch.cumsum((input_ids == self._image_end_token_id).to(torch.int32), -1) > 0
                is_high_res = F.pad(is_high_res[:, :-1], [1, 0, 0, 0])
                bi_dir_mask = can_attend_bk & (is_high_res[:, None, :] == is_high_res[:, :, None])
            elif self.config.bi_directional_attn == "image-to-question":
                raise NotImplementedError()
                # bi_dir_mask = (is_image_token[:, :, None] & response_mask[:, None, :])
                # image_starts = input_ids == self._image_start_token_id
                # image_segment_ids = torch.cumsum(image_starts, -1)
                # bi_dir_mask = bi_dir_mask & (image_segment_ids[:, None, :] == image_segment_ids[:, :, None])
            else:
                raise NotImplementedError()
            casual_mask = casual_mask | bi_dir_mask[:, None, :, :]

        mask_len = seq_len
        if attention_mask is not None:
            mask_len = attention_mask.shape[-1]
        elif past_key_values is not None:
            mask_len = past_key_values[0][0].shape[-2] + seq_len
        casual_mask = casual_mask[:, :, :mask_len, :mask_len]
        if attention_mask is not None:
            attention_mask = attention_mask & casual_mask
        else:
            attention_mask = casual_mask

        # Convert `attention_mask` to float mask `attention_bias`, possibly combine with `attention_bias`
        if attention_bias is not None:
            attention_bias = torch.where(attention_mask, attention_bias, torch.finfo(x.dtype).min)
        else:
            attention_bias = torch.where(attention_mask, 0, torch.finfo(x.dtype).min)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        selection_out = None

        # Add low res features to x
        ts_cfg = self.config.token_scorer
        if images is not None:
            image_features = self.vision_backbone(
                images, image_masks, [low_res_tokens_idx, high_res_tokens_idx])
            low_image_features, low_image_mask = image_features[0]
            high_image_features, high_res_mask = image_features[1]
            is_low_res_patch = (input_ids == self._low_res_patch_id)
            is_high_res_patch = (input_ids == self._high_res_patch_id)

            if self.config.token_scorer.source == "2llm":
                raise NotImplementedError()

            else:
                low_res_end_ixs = torch.argmax((input_ids == self._image_end_token_id).to(torch.int32), dim=-1) + 1
                low_res_end = torch.max(low_res_end_ixs)
                x = x.clone()
                x.view(-1, x.shape[-1])[is_low_res_patch.view(-1)] += (
                    low_image_features.reshape(-1, low_image_features.shape[-1])[low_image_mask.view(-1)])

                low_res_x = x[:, :low_res_end]
                if not ts_cfg.bp_low_res_embed:
                    low_res_x = low_res_x.detach()

                low_res_bias = attention_bias[:, :, :low_res_end, :low_res_end].contiguous()
                low_res_pos_ids = position_ids[:, :low_res_end].contiguous()

                low_res_x = self.transformer.emb_drop(low_res_x)  # type: ignore
                if self.config.llm.normalize_input_embeds:
                    low_res_x = low_res_x * (self.config.llm.d_model ** 0.5)

                if response_mask is not None:
                    low_res_response_mask = response_mask[:, :low_res_end]
                else:
                    low_res_response_mask = None

                # Do low-res pass
                low_res_layer_outputs = []
                low_res_cache = []

                if ts_cfg.unconditioned:
                    low_res_x = low_res_x * is_image_token[:, :low_res_end, None].float()
                    low_res_bias = torch.where(is_image_token[:, None, None, :low_res_end], low_res_bias, torch.finfo(attention_bias.dtype).min)

                low_res_x_in = low_res_x
                if ts_cfg.selection_model != "prior-only":
                    low_res_layer_outputs = []
                    for block_idx, block in enumerate(self.transformer.blocks[:ts_cfg.n_low_res_layers]):
                        enable_grad = ts_cfg.bp_low_res_start <= block_idx < ts_cfg.bp_low_res_end and self.training
                        with torch.set_grad_enabled(enable_grad), torch.compiler.set_stance("force_eager"):
                            low_res_x, cache = block(
                                low_res_x, drop_mask=low_res_response_mask, attention_bias=low_res_bias, position_ids=low_res_pos_ids,
                                use_cache=use_low_res_cache)
                        if use_low_res_cache:
                            low_res_cache.append(cache)
                        if ts_cfg.source and "all_layers" in ts_cfg.source:
                            low_res_layer_outputs.append(low_res_x)
                        if ts_cfg.source and ts_cfg.source == "layer1" and block_idx == 0:
                            low_res_layer_outputs.append(low_res_x)
                else:
                    low_res_layer_outputs = [low_res_x]

                n_low_res = low_res_tokens_idx.shape[1]

                # [batch, n_crops, n_patches, 576]
                # Mask patches (could padding, overlap, or bath-padding) get zero attention
                if ts_cfg.selection_model in ["cross-attend-v1"]:
                    assert ts_cfg.source is None
                    high_res_importance = self.importance_ln(
                        low_res_x,
                        low_res_bias,
                        high_image_features.reshape([batch_size, -1, dim]),
                        mask
                    )
                    low_res_importance_flat = None
                elif ts_cfg.selection_model in ["prior_only", "random"]:
                    low_res_importance_flat = None
                    high_res_importance = None
                elif ts_cfg.selection_model in ["transformer-selector"]:
                    raise NotImplementedError()
                else:
                    if ts_cfg.source in ["all_layers"]:
                        low_res_x_features = torch.cat(low_res_layer_outputs, -1)
                        if not ts_cfg.normalize_importance_scores:
                            low_res_x_features /= np.sqrt(len(low_res_layer_outputs))
                    else:
                        low_res_x_features = low_res_x

                    low_res_dim = low_res_x_features.shape[-1]
                    low_res_features_sparse = low_res_x_features.view(-1, low_res_dim)[is_low_res_patch[:, :low_res_end].flatten()]
                    low_res_features = torch.zeros(([batch_size, n_low_res, low_res_dim]), device=x.device, dtype=x.dtype)
                    low_res_features.view(-1, low_res_dim)[low_image_mask.view(-1)] = low_res_features_sparse
                    if ts_cfg.importance_norm:
                        low_res_features = self.importance_norm(low_res_features)
                    if ts_cfg.normalize_importance_scores:
                        low_res_features = low_res_features / np.sqrt(low_res_features.shape[-1])
                    if ts_cfg.low_res_features_drop:
                        low_res_features = F.dropout(low_res_features, ts_cfg.low_res_features_drop, self.training)
                    low_res_importance_flat = self.importance_ln(low_res_features)

                    # features are for the low-res patches and need to interpolates
                    if not ts_cfg.four_scores_per_low_res_patch:
                        low_res_importance_flat = torch.tile(torch.mean(low_res_importance_flat, dim=-1, keepdim=True), [1, 1, ts_cfg.low_to_high_interpolation_factor**2])
                    # if not ts_cfg.mean_score_per_low_res_patch:
                    #     low_res_importance_flat + low_res_importance_flat + torch.mean(low_res_importance_flat, dim=-1, keepdim=True)
                    high_res_importance = torch.matmul(
                        low_res_importance_flat.reshape(batch_size, 1, -1),
                        low_to_high
                    ).squeeze(1)

                if ts_cfg.selection_model in ["with-vector-query-v1", "with-vector-query-v2"]:
                    low_res_mask = (image_segment_ids[:, :low_res_end] <= 1)
                    high_res_importance = high_res_importance + self.low_to_high_scorer(
                        low_res_x, low_res_mask, high_image_features, mask)

                if ts_cfg.selection_model == "random":
                    high_res_importance = torch.zeros(high_image_features.shape[:2], device=x.device)
                    high_res_importance.random_(0, 1)
                elif ts_cfg.high_res_patch_prior or ts_cfg.vector_query:
                    features = high_image_features
                    if not ts_cfg.bp_patch_prior:
                        features = features.detach()
                    if ts_cfg.importance_norm:
                        features = self.patch_importance_norm(features)
                    if ts_cfg.high_res_patch_prior_drop:
                        features = F.dropout(features, ts_cfg.high_res_patch_prior_drop, training=self.training)

                    if ts_cfg.vector_query == "from_image_end_token":
                        first_image_end_idx = torch.argmax((input_ids == self._image_end_token_id).float(), -1) - 1
                        batch_idx = torch.arange(0, low_res_x_features.shape[0], device=low_res_x_features.device)
                        first_image_end_embed = low_res_x_features[batch_idx, first_image_end_idx]
                        query = self.image_query_ln(first_image_end_embed)
                        if ts_cfg.vector_query_scaling:
                            scale, bias = query[:, 0:1], query[:, 1:2]
                            scale = scale / np.sqrt(features.shape[-1])
                            bias = bias / np.sqrt(features.shape[-1])
                            query = query[:, 2:]
                        else:
                            scale, bias = None, None
                        # Note root scaling factor is very important to keep learning stable
                        high_res_importance = high_res_importance + torch.einsum("bd,bsd->bs", query.squeeze(1), features) / np.sqrt(features.shape[-1])
                        if scale is not None:
                            high_res_importance = high_res_importance * scale + bias

                    elif ts_cfg.vector_query is not None:
                        raise NotImplementedError(ts_cfg.vector_query)

                    if ts_cfg.selection_model == "prior_only":
                        high_res_importance = self.patch_importance_ln(features).squeeze(-1)
                    elif ts_cfg.high_res_patch_prior:
                        high_res_importance = high_res_importance + self.patch_importance_ln(features).squeeze(-1)

            if ts_cfg.learned_rescaling in ["norm-with-bias", "rescale-with-bias"]:
                if ts_cfg.learned_rescaling == "norm-with-bias":
                    masked = (high_res_importance * mask.float()).sum(-1) / mask.float().sum(-1)
                    high_res_importance = high_res_importance - masked.unsqueeze(-1)
                high_res_importance = self.rescale(high_res_importance.unsqueeze(-1)).squeeze(-1)
            elif ts_cfg.learned_rescaling is not None:
                raise NotImplementedError(ts_cfg.learned_rescaling)

            selection_mask = high_res_features_weights > 0
            selection_out: SelectionOutput = self.token_selector(
                high_res_importance, high_res_mask, selection_mask)
            if ts_cfg.selection_model == "random":
                selection_out.token_importance = None

            selection = selection_out.selection

            batch_ix = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_ix[:, None], [1, selection_mask.shape[1]])

            # To [batch, n_high_res, dim] -> [batch, n_selected, dim]
            selected_features = high_image_features[batch_idx, selection].to(dtype=x.dtype)
            selection_valid = (high_res_features_weights > 0).flatten()
            valid_selected_features = selected_features.reshape([-1, dim])[selection_valid]

            x = x.clone()
            x.view(-1, dim)[is_high_res_patch.view(-1)] += valid_selected_features
            selected_pos_ids = high_res_pos_ids[batch_idx, selection]
            position_ids.view(-1)[is_high_res_patch.view(-1)] += selected_pos_ids.view(-1)[selection_valid]

            if selection_out.token_importance is not None:
                token_scores = torch.ones([batch_size, x.shape[1]], dtype=x.dtype, device=x.device)
                token_scores.view(-1)[is_high_res_patch.view(-1)] = selection_out.token_importance.to(x.dtype).view(-1)[selection_valid]
            else:
                token_scores = None
        else:
            assert len(attention_mask.shape) == 4
            selection_out = None
            token_scores = None
            selection = None

        # Sanity check: valid position IDs should be unique
        # for i in range(len(position_ids)):
        #     assert np.unique(position_ids[i][(position_ids[i] >= 0) & selected_valid[i]].cpu().numpy(), return_counts=True)[1].max() == 1
        # logging.info("Santiy check pass")

        x = self.transformer.emb_drop(x)  # type: ignore
        if self.config.llm.normalize_input_embeds:
            x = x * (self.config.llm.d_model ** 0.5)

        elif use_low_res_cache:
            high_res_start = low_res_end_ixs.min()
            past_key_values = [(k[:, :, :high_res_start], v[:, :, :high_res_start]) for (k, v) in low_res_cache]
            x = x[:, high_res_start:]
            position_ids = position_ids[:, high_res_start:]
            response_mask = response_mask[:, high_res_start:]

        if self.config.token_scorer.attention_scaling and token_scores is not None:
            ascale = self.config.token_scorer.attention_scaling
            self_atten = (1 - torch.eye(x.shape[1], device=attention_bias.device, dtype=attention_bias.dtype))[None, None, :, :]
            attention_bias = attention_bias + torch.log(token_scores*ascale+(1-ascale))[:, None, None, :] * self_atten
            value_scaling = None
        else:
            value_scaling = token_scores

        all_hidden_states = []
        for block_idx, block in enumerate(self.transformer.blocks):
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            x, cache = block(
                x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask,
                layer_past=layer_past, use_cache=use_cache, value_scaling=value_scaling,
            )

            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)
        elif use_low_res_cache:
            x = torch.concatenate([low_res_x[:, :high_res_start], x], 1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.llm.weight_tying:
            # logits = F.linear(x, self.transformer.wte.embedding, None)  # type: ignore
            logits = self.transformer.wte(x, logits=True)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.llm.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.llm.d_model))

        if not last_logits_only and append_last_valid_logits is not None:
            last_valid_logit = logits[
                torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
            logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        metrics = dict()
        if selection_out is not None:
            if selection_out.loss is not None:
                metrics["AuxLoss"] = selection_out.loss

            if selection_out.metrics is not None:
                metrics.update(selection_out.metrics)

            return OLMoOutput(logits=logits,
                              attn_key_values=attn_key_values,
                              metrics=metrics,
                              internal=dict(
                                selected=selection_out.selection,
                                high_res_importance=high_res_importance.detach(),
                                low_res_importance=None if low_res_importance_flat is None else low_res_importance_flat.detach()
                                # high_res_pos_ids=high_res_pos_ids,
                                # low_to_high=low_to_high,
                              ),
                              hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]
        else:
            return OLMoOutput(logits=logits,
                              attn_key_values=attn_key_values,
                              metrics=metrics,
                              hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]


    def generate(
        self,
        batch,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
        is_distributed: bool=False,
        return_prefill_output=False
    ) -> OLMoGenerateOutput:
        llm_cfg = self.config.llm
        beam_search = BeamSearch(
            llm_cfg.build_tokenizer().eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
            distributed_model=is_distributed
        )

        input_ids: torch.LongTensor = batch["input_ids"]
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask")
        prefill_output = []
        position_ids: Optional[torch.Tensor] = batch.get("position_ids")

        image_data = dict(
            images=batch.get("images"),
            image_masks=batch.get("image_masks"),
            low_res_tokens_idx=batch.get("low_res_tokens_idx"),
            high_res_tokens_idx=batch.get("high_res_tokens_idx"),
            low_to_high=batch.get("low_to_high"),
            high_res_features_weights=batch.get("high_res_features_weights"),
            high_res_pos_ids=batch.get("high_res_pos_ids"),
        )

        batch_size, seq_len = input_ids.shape
        mask_len = seq_len + max_steps
        append_last_valid_logits: Optional[torch.Tensor] = None

        if attention_mask is None:
            attention_mask = input_ids != -1
        if position_ids is None:
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
        else:
            # Beam search code will use the last index of the position_ids to compute the position
            # id for the next generation, we need to ensure its set correctly even if
            # the position_ids were padded during batching
            position_ids = torch.where(position_ids >= 0, position_ids, position_ids.max(dim=-1, keepdim=True)[0])
        append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((batch_size, max_steps))],
            dim=1,
        )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(llm_cfg.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated
            nonlocal position_ids
            nonlocal image_data
            nonlocal append_last_valid_logits
            nonlocal prefill_output

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                position_ids = position_ids[:, -1:] + 1
                _, *last_dims = position_ids.size()
                _position_ids = (
                    position_ids.unsqueeze(1)
                    .expand(batch_size, beam_size, *last_dims)
                    .reshape(batch_size * beam_size, *last_dims)
                    )
                _image_data = {}
                _append_last_valid_logits = None
                _high_res_patch_data = None
            else:
                past_key_values = None
                input_ids = state["input_ids"]
                _image_data = image_data
                _position_ids = position_ids
                _append_last_valid_logits = append_last_valid_logits

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
                append_last_valid_logits=_append_last_valid_logits,
                **_image_data
            )
            if tokens_generated == 1 and return_prefill_output:
                prefill_output.append(output)
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        with torch.inference_mode():
            with torch.compiler.set_stance("eager_on_recompile"):
                token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
            internal=prefill_output[0].internal if return_prefill_output else None
        )
