import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional

import einops
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from olmo.config import BaseConfig, D, StrEnum
from olmo.nn.image_vit import VitConfig, VisionTransformer
from olmo.nn.llm import Activation
from olmo.torch_util import freeze_module
from olmo.util import resource_path
from torch.nn import functional as F
from torch.distributed.fsdp import fully_shard


class ImagePaddingEmbed(StrEnum):
    """How to embed image padding information"""
    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    """How to pool patch features"""
    attention = "attention"
    attention_meanq = "attention_meanq"
    attention_2wide = "attention_2wide"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    """How to project the pooled features into the LLM embedding space"""
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


@dataclass
class VisionBackboneConfig(BaseConfig):
    """Vision ViT and the Image/Language Connector"""

    vit: VitConfig = field(default_factory=VitConfig)
    """The vision ViT"""

    image_pooling_2d: ImagePooling2DType = ImagePooling2DType.attention
    """Layer to pool image features"""

    image_projector: ImageProjectType = ImageProjectType.mlp
    """Layer to project pooled image features to the LLM embedding space"""

    image_padding_embed: Optional[ImagePaddingEmbed] = None
    """
    Image padding mode to use to tell the model what parts of the image are padding
    """

    fix_image_padding: bool = True
    """
    Use a version of the image padding mask that fixes the an off-by-one error how the embeddings
    are computed, should only be false for legacy models 
    """

    vit_layers: Tuple = (-1,)
    """What layers to use from the VIT"""

    skip_unused_layers: bool = True
    """Don't load layers we don't need from the ViT"""

    image_pooling_h: int = 2
    """Pooling patch features height"""

    image_pooling_w: int = 2
    """Pooling patch features width"""

    image_feature_dropout: float = 0.0
    """Dropout for image patch features"""

    activation_checkpointing: bool = True
    """Allow activation checkpoint to the connector components"""

    compile_vit: Optional[str] = "blocks"
    """How to compile the ViT"""

    def __post_init__(self):
        self.vit_layers = tuple(self.vit_layers)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        return self.vit.image_num_patch

    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w

    def build(self, llm_config, device):

        return MolmoVisionBackbone(self, llm_config, device)


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(self, config, input_dim: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.hidden_size = config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=device,
            )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        # Activation function.
        self.act = Activation.build(config.activation_type, split_inputs=True)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def reset_parameters(self):
        self.submodule.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: VisionBackboneConfig, llm_config, device=None):
        super().__init__()
        self.config = config
        input_dim: int = None
        vit_cfg = config.vit
        pool_dim = vit_cfg.image_emb_dim * len(config.vit_layers)

        from olmo.nn.image_vit import ViTMultiHeadDotProductAttention

        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(config.vit, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_2wide:
            cfg = deepcopy(config.vit)
            vit_cfg.image_emb_dim *= 2
            vit_cfg.image_head_dim *= 2
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(cfg, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack]:
            self.image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * vit_cfg.image_emb_dim
            if config.image_pooling_2d == ImagePooling2DType.stack:
                input_dim *= 4
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")

        self.input_dim = input_dim

        if config.image_projector == ImageProjectType.mlp:
            self.image_projector = ImageProjectorMLP(llm_config, input_dim, device=device)
        elif config.image_projector == ImageProjectType.linear:
            self.image_projector = nn.Linear(input_dim, llm_config.d_model, bias=False, device=device)
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)

        self.vit_layers = []
        for layer in config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(config.vit.image_num_layers + layer)
        last_layer_needed = (max(self.vit_layers)+1)

        vit_cfg = self.config.vit
        if last_layer_needed < config.vit.image_num_layers:
            if self.config.skip_unused_layers:
                vit_cfg = replace(vit_cfg, image_num_layers=last_layer_needed)
                self.image_vit: VisionTransformer = vit_cfg.build(device)
            else:
                # We might need to keep the layers for checkpoint compatibility, but we
                # freeze them since unfrozen layers with no gradient confuses torch's distributed
                # optimizer checkpointer
                self.image_vit: VisionTransformer = vit_cfg.build(device)
                for block in self.image_vit.transformer.resblocks[last_layer_needed-1:]:
                    freeze_module(block)
        else:
            self.image_vit: VisionTransformer = vit_cfg.build(device)

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = vit_cfg.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=device))
            else:
                raise ValueError(config.image_padding_embed)

    @classmethod
    def build(cls, config: VisionBackboneConfig, outut_dim, device=None) -> 'MolmoVisionBackbone':
        return MolmoVisionBackbone(config, outut_dim, device)

    def reset_connector_parameters(self):
        if self.image_pooling_2d is not None:
            self.image_pooling_2d.reset_parameters()
        if self.image_projector == "2mlp":
            for module in self.image_projector:
                module.reset_parameters()
        elif self.image_projector == "linear":
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector.reset_parameters()
        if self.pad_embed is not None:
            nn.init.zeros_(self.pad_embed)

    def reset_parameters(self):
        self.reset_connector_parameters()
        self.image_vit.reset_parameters()

    def reset_with_pretrained_weights(self):
        self.reset_connector_parameters()  # resets the connector
        self.image_vit.reset_with_pretrained_weights()

    def apply_fsdp2(self, **kwargs):
        self.image_vit.apply_fsdp2(**kwargs)
        fully_shard(self.image_pooling_2d, **kwargs)
        fully_shard(self.image_projector, **kwargs)
        # For any remaining parameters in `self`, like the pad embed
        fully_shard(self, **kwargs)

    def apply_activation_checkpointing(self):
        self.image_vit.apply_activation_checkpointing()
        if self.config.activation_checkpointing:
            self.image_projector = checkpoint_wrapper(self.image_projector)
            self.image_pooling_2d = checkpoint_wrapper(self.image_pooling_2d)

    def apply_compile(self, **kwargs):
        if self.config.compile_vit == "blocks":
            for block in self.image_vit.transformer.resblocks:
                block.compile(**kwargs)
        elif self.config.compile_vit is not None:
            raise NotImplementedError(self.config.compile_vit)

    def get_connector_parameters(self):
        vit_params = set(self.image_vit.parameters())
        return (p for p in self.parameters() if p not in vit_params)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
        image_features = image_features.view(B, T, N, -1)
        return image_features

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=torch.float32)
                all_pad = all_pad.to(dtype=torch.float32)
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = self.image_feature_dropout(image_features)

        image_features = image_features.reshape(
            (batch_size, num_image) + cfg.image_num_patch + (-1,),
            )

        if cfg.image_num_patch[0] % cfg.image_pooling_h != 0 or cfg.image_num_patch[1] % cfg.image_pooling_w != 0:
            pad_h = cfg.image_num_patch[0] % cfg.image_pooling_h
            pad_w = cfg.image_num_patch[1] % cfg.image_pooling_w
            # Pad so we can still pool mxn patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, pad_w, 0, pad_h, 0, 0, 0, 0),
            )

        # image pooling
        image_features = einops.rearrange(
            image_features,
            'b n (h dh) (w dw) c -> (b n h w) (dh dw) c',
            dh=cfg.image_pooling_h,
            dw=cfg.image_pooling_w,
        )

        if cfg.image_pooling_2d == ImagePooling2DType.attention_meanq:
            query = image_features.mean(-2, keepdim=True)
            image_features = self.image_pooling_2d(query, image_features)
        elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
            image_features = self.image_pooling_2d(image_features[:, :1, :], image_features)

        h, w = cfg.llm_patches_per_crop()
        image_features = image_features.reshape(batch_size, num_image, h * w, -1)

        # MLP layer to map the feature.
        if cfg.image_projector == ImageProjectType.mlpx2:
            for module in self.image_projector:
                image_features = module(image_features)
        else:
            image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        return image_features

