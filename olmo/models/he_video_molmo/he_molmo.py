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
from olmo.models.he_molmo.he_molmo import HeMolmo, TokenScorerConfig
from olmo.models.he_video_molmo.he_video_preprocessor import HeVideoPreprocessorConfig
from olmo.models.molmo.collator import MMCollator
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.model_preprocessor import Preprocessor
from olmo.models.he_molmo.he_collator import HeMMCollator
from olmo.models.he_molmo.he_preprocessor import HePreprocessorConfig
from olmo.models.he_molmo.token_selector import TokenSelectionConfig, SelectionOutput
from olmo.models.video_olmo.video_preprocessor import VideoTextPreprocessor, VideoPreprocessor
from olmo.nn.beam_search import FinalSequenceScorer, Constraint, Sampler, BeamSearch
from olmo.nn.image_vit import ResidualAttentionBlock, VisionTransformer
from olmo.nn.legacy_config import convert_legacy_config
from olmo.nn.llm import LlmConfig, Llm, OLMoBlock, llm_activation_checkpoint_function
from olmo.models.model import FSDPWrapStrategy, OLMoOutput, OLMoGenerateOutput, ModelBase
from olmo.models.model_config import BaseModelConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig, MolmoVisionBackbone
from olmo.tokenizer import get_special_token_ids
from olmo.torch_util import BufferCache, get_default_device
from olmo.util import flatten_list


log = logging.getLogger(__name__)


@dataclasses.dataclass
class HeVideoMolmoConfig(BaseModelConfig):
    """Molmo model configuration"""

    _model_name: ClassVar[str] = "he_molmo"

    llm: LlmConfig = field(default_factory=LlmConfig)
    """LLM to use for generation"""

    vision_backbone: Optional[MolmoVisionBackboneConfig] = field(default_factory=MolmoVisionBackboneConfig)
    """Vision embedding module to get image features"""

    data_formatter: DataFormatter = field(default_factory=DataFormatter)
    """How to prompt the model for different tasks"""

    token_scorer: TokenScorerConfig = field(default_factory=TokenScorerConfig)
    """"How to get token scores"""

    token_selector: TokenSelectionConfig = field(default_factory=TokenSelectionConfig)
    """"How to select tokens using the scores"""

    mm_preprocessor: HeVideoPreprocessorConfig = field(default_factory=HeVideoPreprocessorConfig)
    """How to crop images and encoding jointly with text"""

    shared_low_high_embedding: bool = True
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
        max_seq_len: Optional[int] = None,
    ) -> Preprocessor:
        """
        Build a preprocessor that converts 'raw' image/text data from various tasks into tensors
        inputs/targets that can be passed to the model's forward/generate methods
        """
        return VideoPreprocessor(
            self.data_formatter,
            self.mm_preprocessor.build(self.build_tokenizer(), self.vision_backbone, max_seq_len),
            max_frames=self.mm_preprocessor.max_frames,
            for_inference=for_inference,
            is_training=is_training,
        )

    def build_collator(self, sequence_length, pad_mode: str, include_metadata=True) -> MMCollator:
        """Collators for tensors from the preprocessor produces"""
        padding_lens = self.mm_preprocessor.get_padding_lens(self.vision_backbone)
        if pad_mode:
            assert sequence_length <= self.max_sequence_length
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


