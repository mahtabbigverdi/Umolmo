import dataclasses
from dataclasses import field

from typing import ClassVar, Optional, Sequence, Tuple, Iterator, List, Dict

import math
import torch
import torch.nn.functional as F

from torch.distributed.fsdp import fully_shard

import os
from transformers import AutoModel
from olmo.data.dataset import VIDEO_DATA_HOME

from olmo.config import D
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmo.data_formatter import DataFormatter

from olmo.models.video_olmo.video_preprocessor import VideoTextPreprocessor, \
    MultiModalVideoPreprocessorConfig, VideoPreprocessor

from olmo import tokenizer
from olmo.nn.llm import LlmConfig, Llm, OLMoBlock
from olmo.nn.legacy_config import convert_legacy_config
from olmo.nn.image_vit import ResidualAttentionBlock, VisionTransformer
from olmo.nn.vision_backbone import MolmoVisionBackbone, MolmoVisionBackboneConfig

from olmo.torch_util import BufferCache, get_default_device
from olmo.models.model import FSDPWrapStrategy, OLMoOutput, OLMoGenerateOutput, ModelBase

from olmo.nn.beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler


@dataclasses.dataclass
class QueryBasedVideoOlmoConfig(MolmoConfig):
    """VideoOlmo model configuration"""
    _model_name: ClassVar[str] = "video_olmo"

    @classmethod
    def get_default_model_name(cls):
        return "video_olmo"

    vision_backbone: Optional[MolmoVisionBackboneConfig] = field(default_factory=MolmoVisionBackboneConfig)
    """Vision embedding module to get image features"""

    mm_preprocessor: MultiModalVideoPreprocessorConfig = field(default_factory=MultiModalVideoPreprocessorConfig)
    """How to crop images and encoding jointly with text"""

    shared_low_high_embedding: bool = True

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "llm" not in config:
            # Old v1 style config
            config = convert_legacy_config(config)
        config.llm = LlmConfig.update_legacy_settings(config.llm)
        if config.vision_backbone is not None:
            config.vision_backbone = MolmoVisionBackboneConfig.update_legacy_settings(config.vision_backbone)
        config.data_formatter = DataFormatter.update_legacy_settings(config.data_formatter)
        config.mm_preprocessor = MultiModalVideoPreprocessorConfig.update_legacy_settings(config.mm_preprocessor)
        return config

    def build_preprocessor(
        self,
        for_inference,
        is_training=True,
        include_image=False,
        max_seq_len: Optional[int] = None,
    ) -> VideoPreprocessor:
        """
        Build a preprocessor that converts 'raw' image/text data from various tasks into tensors
        inputs/targets that can be passed to the model's forward/generate methods
        """

        return VideoPreprocessor(
            self.data_formatter,
            self.mm_preprocessor.build(self.build_tokenizer(), self.vision_backbone, max_seq_len),
            for_inference=for_inference,
            is_training=is_training,
            frame_sample_mode=self.mm_preprocessor.frame_sample_mode,
            include_image=include_image,
            max_frames=self.mm_preprocessor.max_frames,
            candidate_sampling_fps=self.mm_preprocessor.candidate_sampling_fps,
        )

    def build_model(self, device=None):
        return QueryBasedVideoOlmo(self, device)

    @property
    def max_sequence_length(self):
        return self.llm.max_sequence_length


class QueryBasedVideoOlmo(ModelBase):
    def __init__(self, config: QueryBasedVideoOlmoConfig, device=None):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        self.transformer: Llm = self.config.llm.build(self.__cache, device)
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if self.config.vision_backbone is not None:
            self.vision_backbone = self.config.vision_backbone.build(self.config.llm, device)
        self.special_ids = tokenizer.get_special_token_ids(self.config.build_tokenizer())
        if self.config.bi_directional_attn:
            self.__cache["image_tokens"] = torch.as_tensor([self.special_ids[x] for x in [
                tokenizer.IMAGE_PATCH_TOKEN,
                tokenizer.IM_COL_TOKEN,
                tokenizer.IM_START_TOKEN,
                tokenizer.IM_END_TOKEN,
                tokenizer.IMAGE_LOW_RES_TOKEN,
            ]], dtype=torch.long, device=get_default_device())
        self._image_end_token_id = self.special_ids[tokenizer.IM_END_TOKEN]
        self._image_start_token_id = self.special_ids[tokenizer.IM_START_TOKEN]
        self._image_low_res_id = self.special_ids[tokenizer.IMAGE_LOW_RES_TOKEN]
        self._image_high_res_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]

        # load the HF model. Initially, don't even shard
        # Hard Coded for now. Needs to be passed in via config. Has to be aligned with the image encoder to re-use pre-processor
        hf_source = "google/siglip2-so400m-patch14-384"
        cache_dir = os.path.join(VIDEO_DATA_HOME, "hf_init_encoders")
        self.frame_selection_model = AutoModel.from_pretrained(
            hf_source,
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
        )
        for param in self.frame_selection_model.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        """Re-initialize the weights from scratch"""
        self.transformer.reset_parameters()
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()

    def reset_with_pretrained_weights(self):
        """Re-initialize the weights, possibly loading pretrained weights for the LLM and ViT"""
        self.transformer.reset_with_pretrained_weights()
        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights()

    def apply_activation_checkpointing(self):
        """Enable activation checkpointing"""
        self.transformer.apply_activation_checkpointing()
        if self.vision_backbone is not None:
            self.vision_backbone.apply_activation_checkpointing()

    def apply_compile(self, **compile_kwargs):
        """Compile the model with `torch.compile`"""
        self.transformer.apply_compile(**compile_kwargs)
        self.vision_backbone.apply_compile(**compile_kwargs)

    def warmup_cache(self, device):
        """Pre-fill the buffer-cache"""
        if self.transformer.blocks[0].rotary_emb is not None:
            self.transformer.blocks[0].rotary_emb.warmup_cache(device)

    def apply_fsdp2(self, **fully_shard_kwargs):
        """Fully shard this model using `fully_shard`"""
        if self.vision_backbone is not None:
            self.vision_backbone.apply_fsdp2(**fully_shard_kwargs)
        self.transformer.apply_fsdp2(**fully_shard_kwargs)
        fully_shard(self, **fully_shard_kwargs)

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        """Get a FSDP1 wrap policy for this model."""
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

        wrap_layer_names = (OLMoBlock, ResidualAttentionBlock, MolmoVisionBackbone, VisionTransformer)

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
        return parameters

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

    @staticmethod
    def siglip_vision_forward(model, pixel_values):
        hidden_states = model.embeddings(pixel_values)

        encoder_outputs = model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
            return_dict=model.config.use_return_dict
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = model.post_layernorm(last_hidden_state)

        return model.head(last_hidden_state)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,

        # Image data
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        low_res_pooled_idx: Optional[torch.Tensor] = None,
        high_res_pooled_idx: Optional[torch.Tensor] = None,

        high_res_indices: Optional[torch.Tensor] = None,
        low_res_pooled_idx_no_offset: Optional[torch.Tensor] = None,
        high_res_pooled_idx_no_offset: Optional[torch.Tensor] = None,
        low_res_token_place_holders: Optional[torch.Tensor] = None,
        high_res_token_place_holders: Optional[torch.Tensor] = None,

        siglip_text_token_ids: Optional[torch.Tensor] = None,
        frame_list: Optional[torch.Tensor] = None,

        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.llm.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if self.config.llm.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = subsegment_mask & attention_mask[:, :, None] & attention_mask[:, None, :]
            attention_mask = attention_mask | torch.eye(seq_len, device=attention_mask.device, dtype=torch.bool)[None, :, :]
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.llm.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

        if images is not None:
            siglip_position_ids = torch.arange(siglip_text_token_ids.shape[1])
            siglip_position_ids = siglip_position_ids.repeat([siglip_text_token_ids.shape[0], 1]).to(siglip_text_token_ids.device)
            embedded_text = self.frame_selection_model.text_model(
                input_ids=siglip_text_token_ids,
                position_ids=siglip_position_ids
            ).pooler_output

            if self.frame_selection_model.vision_model.embeddings.position_ids.sum() == 0:
                self.frame_selection_model.vision_model.embeddings.position_ids = (torch.arange(729).unsqueeze(0).
                                                                                   to(self.frame_selection_model.vision_model.embeddings.position_ids.device))

            num_frames = images.shape[1]
            # break patches
            image_reshaped = images.reshape(batch_size, num_frames, 27, 27, 14, 14, 3)
            # rearrange to original order
            image_reshaped = image_reshaped.permute(0, 1, 2, 4, 3, 5, 6)
            # reshape to image arrangement
            image_reshaped = image_reshaped.reshape(batch_size, num_frames, 378, 378, 3)
            # pi = (image_reshaped[0,0].cpu().numpy() + 1)/2 * 255
            # Image.fromarray(pi.astype(np.uint8)).save("patches.png")
            # Image.fromarray(frame_list[0, 0].cpu().numpy().astype(np.uint8)).save("model_orig.png")

            # reorder to B, C, H, W
            image_reshaped = image_reshaped.permute(0, 1, 4, 2, 3)
            # merge batch_size and num_frames
            image_reshaped = image_reshaped.reshape(-1, 3, 378, 378)

            embedded_image = self.siglip_vision_forward(self.frame_selection_model.vision_model, image_reshaped)
            embedded_image = embedded_image.reshape(-1, num_frames, embedded_image.shape[-1])

            embedded_image = embedded_image / (embedded_image.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            embedded_text = embedded_text / (embedded_text.norm(p=2, dim=-1, keepdim=True) + 1e-6)

            bmm = torch.bmm(embedded_image, embedded_text.unsqueeze(2))
            bmm = bmm.squeeze(2)

            # import pdb; pdb.set_trace()
            #
            # frame_list_processed = (frame_list / 255) * 2 - 1
            # frame_list_processed = frame_list_processed[0, :1, :, :378, :378]
            #
            # embedded_frame_list = self.siglip_vision_forward(self.frame_selection_model.vision_model, frame_list_processed)
            # embedded_frame_list = embedded_frame_list / (embedded_frame_list.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            #
            # hf_source = "google/siglip2-so400m-patch14-384"
            # cache_dir = os.path.join(VIDEO_DATA_HOME, "hf_init_encoders")
            # local_frame_selection_model = AutoModel.from_pretrained(hf_source, cache_dir=cache_dir)
            #
            # local_embedded_image = self.siglip_vision_forward(local_frame_selection_model.vision_model, frame_list_processed.cpu())
            # local_embedded_image = local_embedded_image / (local_embedded_image.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            #
            # print(embedded_frame_list.cpu() @ local_embedded_image.cpu().t())
            # # embedded_frame_list.cpu() @ local_embedded_image.cpu().t() ==> tensor([[0.3756]], grad_fn=<MmBackward0>). Confirms this is the gap

            # cpu_frame_selection_model = self.frame_selection_model.cpu()
            # copied_to_cpu_embedded_frame_list = self.siglip_vision_forward(cpu_frame_selection_model.vision_model, frame_list_processed.cpu())
            # copied_to_cpu_embedded_frame_list = copied_to_cpu_embedded_frame_list / (copied_to_cpu_embedded_frame_list.norm(p=2, dim=-1, keepdim=True) + 1e-6)

            # local_embedded_image.cpu() @ embedded_text.cpu().t() <- confirmed that using local image and cuda text with frame list works. so will work for patches too
            # embedded_frame_list[0, :1].cpu() @ embedded_image[0, :1].cpu().t()  <- confirmed that the embedding is matching for patches and frame list

            # local_embedded_text = local_frame_selection_model.text_model(siglip_text_token_ids.cpu(), position_ids=siglip_position_ids.cpu()).pooler_output
            # local_embedded_text = local_embedded_text / (local_embedded_text.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            # embedded_text.cpu() @ local_embedded_text.cpu().t()

            # each batch has the same values
            low_res_image_col_token_place_holder = low_res_token_place_holders[0]
            high_res_image_col_token_place_holder = high_res_token_place_holders[0]

            num_image_tokens_in_low_res = torch.sum(low_res_image_col_token_place_holder == self.special_ids[tokenizer.IMAGE_LOW_RES_TOKEN])
            num_image_tokens_in_high_res = torch.sum(high_res_image_col_token_place_holder == self.special_ids[tokenizer.IMAGE_PATCH_TOKEN])

            final_low_res_pooled_idx = -1 * torch.ones(low_res_pooled_idx_no_offset.shape, device=low_res_pooled_idx_no_offset.device,
                                                       dtype=low_res_pooled_idx_no_offset.dtype)
            final_high_res_pooled_idx = -1 * torch.ones(high_res_pooled_idx_no_offset.shape, device=high_res_pooled_idx_no_offset.device,
                                                        dtype=high_res_pooled_idx_no_offset.dtype)

            final_tensors = torch.zeros_like(input_ids, device=input_ids.device, dtype=input_ids.dtype)
            final_tensors[:, 0] = input_ids[:, 0]  # bos_token_id at the start needs to be kept in place

            for instance_idx in range(batch_size):
                num_candidate_frames = (high_res_indices[instance_idx] != -1).sum().to(bmm.device)
                num_frames_to_select = (high_res_indices[instance_idx] == 1).sum().to(bmm.device)
                generated_indices = torch.topk(bmm[instance_idx, :num_candidate_frames], k=num_frames_to_select, dim=0).indices.detach()

                # Using set() may slow everything down. Will monitor performance.
                siglip_based_high_res_instance = set(generated_indices.cpu().numpy().tolist())

                single_instance_high_res_instance = high_res_indices[instance_idx]
                input_id_start = 1  # bos_token_id at the start needs to be kept in place
                token_position_offset = 0

                low_res_pooling_start = 0
                high_res_pooling_start = 0

                for flag_idx in range(num_frames):
                    # TODO: This is a hacky way to know when the relevant frames have ended. Use a better way
                    high_res_flag = single_instance_high_res_instance[flag_idx]
                    if high_res_flag == -1:
                        continue

                    # high_res = high_res_flag == 1
                    high_res = flag_idx in siglip_based_high_res_instance

                    if high_res:
                        final_tensors[instance_idx][input_id_start:input_id_start + len(high_res_image_col_token_place_holder)] = high_res_image_col_token_place_holder
                        input_id_start += len(high_res_image_col_token_place_holder)

                        ids_w_o_offset = high_res_pooled_idx_no_offset[instance_idx, high_res_pooling_start: high_res_pooling_start + num_image_tokens_in_high_res]
                        final_high_res_pooled_idx[instance_idx, high_res_pooling_start: high_res_pooling_start + num_image_tokens_in_high_res] = ids_w_o_offset + token_position_offset
                        token_position_offset += ids_w_o_offset.shape[0] * ids_w_o_offset.shape[1]

                        high_res_pooling_start += num_image_tokens_in_high_res
                    else:
                        final_tensors[instance_idx][input_id_start:input_id_start + len(low_res_image_col_token_place_holder)] = low_res_image_col_token_place_holder
                        input_id_start += len(low_res_image_col_token_place_holder)

                        ids_w_o_offset = low_res_pooled_idx_no_offset[instance_idx, low_res_pooling_start: low_res_pooling_start + num_image_tokens_in_low_res]
                        final_low_res_pooled_idx[instance_idx, low_res_pooling_start: low_res_pooling_start + num_image_tokens_in_low_res] = ids_w_o_offset + token_position_offset
                        token_position_offset += ids_w_o_offset.shape[0] * ids_w_o_offset.shape[1]

                        low_res_pooling_start += num_image_tokens_in_low_res

                final_tensors[instance_idx, input_id_start:] = input_ids[instance_idx, input_id_start:]

            input_ids = final_tensors
            low_res_pooled_idx = final_low_res_pooled_idx
            high_res_pooled_idx = final_high_res_pooled_idx

        if input_embeddings is not None:
            x = input_embeddings
        elif self.config.shared_low_high_embedding:
            x = self.transformer.wte(torch.where(input_ids == self._image_low_res_id, self._image_high_res_id, input_ids))
        else:
            x = self.transformer.wte(input_ids)

        if images is not None:
            if high_res_pooled_idx is None:
                image_features = self.vision_backbone(images, image_masks, low_res_pooled_idx)
                is_image_patch = input_ids.view(-1) == self._image_low_res_id
                x.view(-1, x.shape[-1])[is_image_patch] += image_features
            else:
                all_image_features = self.vision_backbone(images, image_masks, [low_res_pooled_idx, high_res_pooled_idx])
                for (image_features, mask), input_id in zip(all_image_features, [self._image_low_res_id, self._image_high_res_id]):
                    is_image_patch = input_ids.view(-1) == input_id
                    x = x.clone()
                    x.view(-1, x.shape[-1])[is_image_patch] += image_features.view(-1, image_features.shape[-1])[mask.view(-1)]

        if not self.config.llm.rope:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.llm.normalize_input_embeds:
            x = x * (self.config.llm.d_model ** 0.5)

        # Transform the attention mask into the 4D tensor blocks expect.
        attention_mask_len = past_length + seq_len  # mask should include the K/V cache
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, :attention_mask_len]
            attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = attention_mask.unsqueeze(1)
        assert attention_mask.shape[-1] == attention_mask_len

        # Combined with attention with the casual mask
        if "casual_mask" not in self.__cache or self.__cache["casual_mask"].shape[-1] < attention_mask_len:
            self.__cache["casual_mask"] = torch.tril(torch.ones(
                attention_mask_len, attention_mask_len,
                device=x.device, dtype=torch.bool))[None, None, :, :]
        casual_mask = self.__cache["casual_mask"].to(x.device)[:, :, :attention_mask_len, :attention_mask_len]

        if self.config.bi_directional_attn == "image_tokens":
            image_tokens = self.__cache["image_tokens"].to(input_ids.device)
            c = torch.any(input_ids[:, :, None] == image_tokens[None, None, :], -1)
            casual_mask = casual_mask | (c[:, :, None] & c[:, None, :])[:, None, :, :]
        elif self.config.bi_directional_attn == "image_to_question":
            if images is not None:
                # image tokens can attend to all non-response tokens
                image_tokens = self.__cache["image_tokens"].to(input_ids.device)
                is_image_token = torch.any(input_ids[:, :, None] == image_tokens[None, None, :], -1)
                if use_cache:
                    casual_mask = casual_mask | is_image_token[:, None, :, None]
                else:
                    casual_mask = casual_mask | (is_image_token[:, :, None] & (~response_mask[:, None, :]))[:, None, :, :]
        elif self.config.bi_directional_attn is not None:
            raise NotImplementedError(self.config.bi_directional_attn)

        attention_mask = attention_mask & casual_mask

        # Convert mask to a float mask, and possibly combine with `attention_bias`
        if attention_bias is not None:
            attention_bias = torch.where(attention_mask, attention_bias, torch.finfo(x.dtype).min)
        else:
            attention_bias = torch.where(attention_mask, 0, torch.finfo(x.dtype).min)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_hidden_states = []
        for block_idx, block in enumerate(self.transformer.blocks):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)

            layer_past = None if past_key_values is None else past_key_values[block_idx]
            x, cache = block(
                x, attention_bias=attention_bias, position_ids=position_ids,
                drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache
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

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.llm.weight_tying:
            logits = self.transformer.wte(x, logits=True)
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.llm.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.llm.d_model))

        if not last_logits_only and append_last_valid_logits is not None:
            last_valid_logit = logits[
                torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
            logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        return OLMoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]

    def generate(
        self,
        batch,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
        is_distributed: bool=False
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        input_ids: torch.LongTensor = batch["input_ids"]
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask")
        image_args = dict(
            images=batch.get("images"),
            image_masks=batch.get("image_masks"),
            low_res_pooled_idx=batch.get("low_res_pooled_idx"),
            high_res_pooled_idx=batch.get("high_res_pooled_idx"),
            high_res_indices=batch.get("high_res_indices"),
            low_res_token_place_holders=batch.get("low_res_token_place_holders"),
            high_res_token_place_holders=batch.get("high_res_token_place_holders"),
            low_res_pooled_idx_no_offset=batch.get("low_res_pooled_idx_no_offset"),
            high_res_pooled_idx_no_offset=batch.get("high_res_pooled_idx_no_offset"),
            siglip_text_token_ids=batch.get("siglip_text_token_ids")
        )

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

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        mask_len = seq_len + max_steps if llm_cfg.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if llm_cfg.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_steps))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= llm_cfg.max_sequence_length
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
            for i in range(self.config.llm.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            nonlocal tokens_generated
            nonlocal position_ids
            nonlocal image_args
            nonlocal append_last_valid_logits
            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if not llm_cfg.use_position_ids and attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
                if llm_cfg.use_position_ids:
                    position_ids = position_ids[:, -1:] + 1
                    _, *last_dims = position_ids.size()
                    _position_ids = (
                        position_ids.unsqueeze(1)
                        .expand(batch_size, beam_size, *last_dims)
                        .reshape(batch_size * beam_size, *last_dims)
                    )
                else:
                    _position_ids = None

                _image_args = {}
                _append_last_valid_logits = None
            else:
                past_key_values = None
                input_ids = state["input_ids"]
                _image_args = image_args
                _position_ids = position_ids
                _append_last_valid_logits = append_last_valid_logits

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            # We allow the pre-fill stage to compile, but generation is not compiled
            # since it would require recompiling for each step as the KV cache grows
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
                append_last_valid_logits=_append_last_valid_logits,
                **_image_args
            )
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
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

