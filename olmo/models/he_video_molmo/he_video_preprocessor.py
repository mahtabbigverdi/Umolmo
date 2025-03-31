import dataclasses
from typing import Optional

from olmo.config import BaseConfig
from olmo.models.he_molmo.he_preprocessor import HeMultiModalPreprocessor
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig


@dataclasses.dataclass
class HeVideoPreprocessorConfig(BaseConfig):
    max_frames: int = 6
    """Max number of frames to use per an image"""

    use_col_tokens: bool = True
    """Use column tokens in the image tokens"""

    max_text_tokens: Optional[int] = None
    """Max query length"""

    max_query_tokens: Optional[int] = None
    """Max query length"""

    low_res_pooling: Optional[int] = 9
    """Pooling for low-res features"""

    high_res_pooling: Optional[int] = 3
    """Pooling for high-res features"""

    num_high_res_features: Optional[int] = None
    """Number of high-res features to select"""

    time_mode: Optional[str] = None
    """How to encode video times"""

    loss_token_weighting: Optional[str] = None

    def get_max_mm_tokens(self, vision_backbone_config):
        lens = self.get_padding_lens(vision_backbone_config)
        seq_len = lens["low_res_tokens_idx"] + self.num_high_res_features
        extra_per_frame = 1

        assert not self.use_col_tokens
        seq_len += self.max_frames * extra_per_frame + 8  # img start, img end,
        if self.time_mode == "fps-prefix":
            seq_len += 8
        elif self.time_mode is not None:
            raise NotImplementedError(self.max_frames)
        return seq_len

    def get_padding_lens(self, vision_backbone_config: MolmoVisionBackboneConfig):
        return self.build(None, vision_backbone_config).get_video_padding_lens(self.max_frames)

    def build(self, tokenizer, vision_backbone_config: MolmoVisionBackboneConfig):
        vit = vision_backbone_config.vit
        return HeMultiModalPreprocessor(
            loss_token_weighting=self.loss_token_weighting,
            time_mode=self.time_mode,
            tokenizer=tokenizer,
            resize=vit.resize_mode,
            normalize=vit.normalize,
            image_patch_size=vit.image_patch_size,
            base_image_input_size=vit.image_default_input_size,
            crop_mode="resize",
            use_col_tokens=self.use_col_tokens,
            use_high_res_col_tokens=self.use_col_tokens,
            max_text_tokens=self.max_text_tokens,
            max_query_tokens=self.max_query_tokens,
            video_low_res=self.low_res_pooling,
            video_high_res=self.high_res_pooling,
            num_high_res_features=self.num_high_res_features
        )
