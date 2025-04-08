import dataclasses
import logging
import math
from typing import Tuple, Union, List, Optional, Any
import numpy as np
import torch
import torchvision
from einops import einops
from torchvision.transforms import InterpolationMode
from transformers.image_utils import ImageInput

from olmo import tokenizer
from olmo.config import BaseConfig, D
from olmo.data.image_preprocessor import ImagePreprocessor
from olmo.data.interleaved_text_preprocessor import InterleavedTextPreprocessor
from olmo.models.molmo.model_preprocessor import arange_for_pooling, batch_pixels_to_patches
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.tokenizer import get_special_token_ids


def build_pos_ids(subsegment_ids):
    position_ids = np.zeros_like(subsegment_ids)
    for subsegment_id in np.unique(subsegment_ids):
        segment_position_ids = np.cumsum((subsegment_ids == subsegment_id) | (subsegment_ids == 10000)) - 1
        position_ids = np.where(subsegment_ids == subsegment_id, segment_position_ids, position_ids)
    return position_ids


@dataclasses.dataclass
class HePreprocessorConfig(BaseConfig):
    crop_mode: str = "resize"
    """How to divide the images into crops"""

    max_crops: int = 6
    """Max number of crops to produce per an image"""

    overlap_margins: Tuple[int, int] = (4, 4)
    """Overlap margins for overlapping crops modes"""

    use_col_tokens: bool = True
    """Use column tokens in the image tokens"""

    loss_token_weighting: Optional[str] = None
    """Automatically weight multi-message per image input"""

    indicate_k: Optional[str] = None
    """Indicate the amount of high-res tokens to use in the query"""

    max_text_len: Optional[int] = None
    """Max query length"""

    num_high_res_features: Optional[int] = 512
    """How many high-res features to use"""

    image_pooling_w: Optional[int] = 2
    image_pooling_h: Optional[int] = 2
    low_res_from_high: Optional[int] = None
    low_res_from_low: Optional[int] = 2

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "max_query_len" in config:
            config.max_text_len = config.pop("max_query_len")
        return config

    def get_max_crops(self) -> int:
        """Max numbers of that can be built for one image"""
        if self.crop_mode == "resize":
            return 1
        elif "resize" in self.crop_mode:
            # low_res crop + the high-res crops
            if not self.low_res_from_low:
                return self.max_crops
            else:
                return 1 + self.max_crops
        else:
            return self.max_crops

    def get_padding_lens(self, vision_backbone_config: MolmoVisionBackboneConfig):
        """Max numbers of image tokens can be built for one image"""
        return self.build(None, vision_backbone_config).get_image_padding_lens()

    def build(
        self,
        tokenizer,
        vision_backbone_config: MolmoVisionBackboneConfig,
        max_seq_len=None,
        low_to_high_interpolation_factor=2
    ):
        vit = vision_backbone_config.vit
        assert self.image_pooling_w == self.image_pooling_h

        return HeMultiModalPreprocessor(
            tokenizer=tokenizer,
            max_text_tokens=self.max_text_len,
            low_to_high_interpolation_factor=low_to_high_interpolation_factor,
            num_high_res_features=self.num_high_res_features,
            loss_token_weighting=self.loss_token_weighting,
            normalize=vit.normalize,
            crop_mode=self.crop_mode,
            max_crops=self.max_crops,
            overlap_margins=self.overlap_margins,
            resize=vit.resize_mode,
            use_col_tokens=self.use_col_tokens,
            use_high_res_col_tokens=self.use_col_tokens,
            base_image_input_size=vit.image_default_input_size,
            image_patch_size=vit.image_patch_size,
            image_padding_mask=vision_backbone_config.image_padding_embed is not None,
            pad_value=vit.pad_value,
            max_sequence_length=max_seq_len,
            image_high_res_from_crops=self.image_pooling_w,
            image_low_res_from_crops=self.low_res_from_high,
            image_low_res_from_resized=self.low_res_from_low,
        )


def arange_frames_for_pooling(idx_arr, pool_f, pool_h, pool_w):
    patches = np.prod(idx_arr.shape)
    idx = arange_for_pooling(idx_arr, pool_h, pool_w)
    idx = np.tile(np.expand_dims(idx, axis=0), [pool_f, 1, 1, 1])
    return np.where(
        idx < 0,
        idx,
        idx + (patches * np.arange(pool_f))[:, None, None, None]
    )


def arange_video_for_pooling(idx_arr, pool_h, pool_w):
    h_pad = pool_h * ((idx_arr.shape[1] + pool_h - 1) // pool_h) - idx_arr.shape[1]
    w_pad = pool_w * ((idx_arr.shape[2] + pool_w - 1) // pool_w) - idx_arr.shape[2]
    idx_arr = np.pad(idx_arr, [[0, 0], [h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant', constant_values=-1)
    return einops.rearrange(
        idx_arr, "f (h dh) (w dw) -> f h w (dh dw)", dh=pool_h, dw=pool_w)


@dataclasses.dataclass
class HeMultiModalPreprocessor(InterleavedTextPreprocessor, ImagePreprocessor):
    num_high_res_features: Optional[int] = None
    multi_res_min: Optional[int] = None
    multi_res_selection: Optional[int] = None

    # How to crops/resize images
    crop_mode: str = "resize"
    use_col_tokens: bool = True

    # For video
    video_high_res: int = 3
    video_low_res: int = 9
    video_low_res_collage: int = None
    time_mode: str = "fps-prefix"
    low_to_high_interpolation_factor: int = 2

    # For multi-crop image
    image_low_res_from_crops: Optional[int] = None
    image_low_res_from_resized: Optional[int] = 2
    image_high_res_from_crops: int = 2

    # Other settings
    image_padding_mask: Union[bool, int] = False
    indicate_k: bool = False
    use_high_res_col_tokens: bool = True

    def get_video_padding_lens(self, max_frames=None):
        crop_h = self.base_image_input_size[0]//self.image_patch_size
        crop_w = self.base_image_input_size[1]//self.image_patch_size
        tmp = np.zeros([crop_h, crop_w])
        high_res_idx = arange_for_pooling(tmp, self.video_high_res, self.video_high_res)
        if self.video_low_res_collage:
            n_in_collage = self.video_low_res_collage**2
            n_low_res_frames = (max_frames + n_in_collage  - 1) // n_in_collage
            n_frames = max_frames + n_low_res_frames
            n_high = np.prod(high_res_idx.shape[:2])*max_frames
            low_res_idx = arange_for_pooling(np.zeros([crop_h//self.video_low_res_collage, crop_w//self.video_low_res_collage]), self.video_low_res, self.video_low_res)
            n_low = np.prod(low_res_idx.shape[:2])*max_frames
        else:
            low_res_idx = arange_for_pooling(tmp, self.video_low_res, self.video_low_res)
            n_frames = max_frames
            n_high = np.prod(high_res_idx.shape[:2])*max_frames
            n_low = np.prod(low_res_idx.shape[:2])*max_frames
        return dict(
            images=n_frames,
            low_res_tokens_idx=n_low,
            high_res_tokens_idx=n_high,
            high_res_pos_ids=n_high
        )

    def get_image_padding_lens(self):
        base_h, base_w = self.base_image_input_size
        high, low = -1, -1
        for h, w in [
            [base_h, base_w*self.max_crops],
            [base_h*self.max_crops, base_w*self.max_crops],
            [base_h*2, base_w*self.max_crops//2 + 1],
            [base_h*self.max_crops, base_w],
        ]:
            new_high, new_low = self._compute_num_tokens(h, w)
            high = max(high, new_high)
            low = max(low, new_low)
        return dict(
            images=1 if self.crop_mode == "resize" else (1+self.max_crops),
            high_res_pos_ids=high,
            low_res_tokens_idx=low,
            high_res_tokens_idx=high,
        )

    def _compute_num_tokens(self, image_h, image_w) -> int:
        image_patch_size = self.image_patch_size
        crop_patch_w = self.base_image_input_size[1] // image_patch_size
        crop_patch_h = self.base_image_input_size[0] // image_patch_size

        if self.crop_mode == "resize":
            raise NotImplementedError()
        h, w = self.compute_overlapping_crops_size(image_h, image_w)
        idx_arr = arange_for_pooling(
            torch.zeros([h, w]), self.image_high_res_from_crops, self.image_high_res_from_crops)
        high_res_tokens = idx_arr.shape[0] * idx_arr.shape[1]

        low_res_tokens = 0
        if self.image_low_res_from_resized:
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            idx_arr = arange_for_pooling(resize_idx, self.image_low_res_from_resized, self.image_low_res_from_resized)
            low_res_tokens += idx_arr.shape[0] * idx_arr.shape[1]
        if self.image_low_res_from_crops:
            # (46, 84)
            idx_arr = arange_for_pooling(torch.zeros([h, w]), self.image_low_res_from_crops, self.image_low_res_from_crops)
            low_res_tokens += idx_arr.shape[0] * idx_arr.shape[1]
        return high_res_tokens, low_res_tokens

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        n_high_res,
        is_training=False,
        rng=None,
    ):
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode in ["overlap-and-resize-c2"]:
            crop_arr, mask_arr, patch_idx_arr = self.build_overlapping_crops(image, is_training=is_training, rng=rng)

            # Now arrange `patch_idx_arr` so it ready for pooling, possibly padding it
            high_res_idx = arange_for_pooling(patch_idx_arr, self.image_high_res_from_crops, self.image_high_res_from_crops)
            h, w = high_res_idx.shape[:2]
            high_res_idx = high_res_idx.reshape([-1, self.image_high_res_from_crops * self.image_high_res_from_crops])

            # Now build the output tokens
            image_k = n_high_res
            if self.use_high_res_col_tokens:
                col_token_positions = np.arange(1, h + 1) * (w + 1)
                joint = [
                    [self.tokenizer.image_start_token_id],
                    np.full(image_k, self.tokenizer.image_patch_token_id, dtype=np.int32),
                    np.full(h, self.tokenizer.image_col_token_id, dtype=np.int32),
                    [self.tokenizer.image_end_token_id]
                ]
                high_res_pos_ids = [
                    np.zeros([1], dtype=np.int32),  # IMG start
                    np.ones(image_k, dtype=np.int32),  # IMG patches
                    col_token_positions,  # IMG COL
                    [h*w + w+1]  # IMG End
                ]
                rows = np.arange(w, dtype=np.int32)
                cols = np.arange(h, dtype=np.int32) * (w + 1)
                high_res_patch_pos_ids = rows[None, :] + cols[:, None]
                high_res_patch_pos_ids = high_res_patch_pos_ids.reshape([-1])
            else:
                joint = [
                    [self.tokenizer.image_start_token_id],
                    np.full(image_k, self.tokenizer.image_patch_token_id, dtype=np.int32),
                    [self.tokenizer.image_end_token_id]
                ]
                high_res_pos_ids = [
                    np.zeros([1], dtype=np.int32),  # IMG start
                    np.ones(image_k, dtype=np.int32),  # IMG start/patches
                    [image_k+1]  # IMG End
                ]
                high_res_patch_pos_ids = np.arange(h*w)

            if self.image_low_res_from_crops:
                low_idx = arange_for_pooling(patch_idx_arr, self.image_low_res_from_crops, self.image_low_res_from_crops)
                low_h, low_w = low_idx.shape[:2]
                low_res_pooling_idx = low_idx.reshape([-1, self.image_low_res_from_crops ** 2])
            else:
                assert not self.image_low_res_from_crops
                # Finally do the same for the global image
                resized, resized_mask, resize_idx = self.build_resized_image(image, is_training=is_training, rng=rng)
                crop_arr = np.concatenate([resized, crop_arr], 0)
                mask_arr = np.concatenate([resized_mask, mask_arr], 0)

                low_idx = np.arange(crop_patch_h*crop_patch_w).reshape([crop_patch_h, crop_patch_w])
                low_idx = arange_for_pooling(low_idx, self.image_low_res_from_resized, self.image_low_res_from_resized)
                low_h, low_w = low_idx.shape[:2]
                low_res_pooling_idx = low_idx.reshape([-1, self.image_low_res_from_resized * self.image_low_res_from_resized])

                # Global image goes first, so the order of patches in previous crops gets increased
                high_res_idx = np.where(
                    high_res_idx >= 0,
                    high_res_idx + crop_patch_h*crop_patch_w,
                    -1
                )

            low_res_f = self.low_to_high_interpolation_factor
            low_to_high = np.eye((low_h*low_res_f*low_w*low_res_f), dtype=np.float32)
            low_to_high = low_to_high.reshape([low_h*low_w*low_res_f*low_res_f, low_h*low_res_f, low_w*low_res_f])
            low_to_high = torchvision.transforms.Resize(
                [h, w], InterpolationMode.BILINEAR, antialias=False)(
                torch.from_numpy(low_to_high)).numpy()
            # Re-arrange to match how the importance scores are predicted (four per a patch)
            # This save us having to transpose it in model
            low_to_high = einops.rearrange(
                # low_to_high, "(lh dh lw dw) h w -> (lw lh dw dh) (h w)",
                low_to_high, "(lh dh lw dw) h w -> (lh lw dh dw) (h w)",
                dh=low_res_f, dw=low_res_f, lh=low_h, lw=low_w)

            per_row = np.full(
                (low_w,),
                self.tokenizer.image_low_res_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [low_h])
            low_res_tokens = np.concatenate([
                [self.tokenizer.image_start_token_id],
                extra_tokens,
                [self.tokenizer.image_end_token_id],
            ])
            all_pos_ids = np.concatenate([
                np.arange(len(low_res_tokens)),
                np.concatenate(high_res_pos_ids) + len(low_res_tokens)
            ])
            joint = [low_res_tokens] + joint

            mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
            return (
                np.concatenate(joint, 0),
                all_pos_ids,
                batch_pixels_to_patches(crop_arr, image_patch_size),
                mask_arr, low_res_pooling_idx, high_res_idx,
                (low_to_high, high_res_patch_pos_ids)
            )
        else:
            raise NotImplementedError(self.crop_mode)

    def video_to_patches_and_tokens(
        self,
        image: ImageInput,
        max_high_res_tokens,
        is_training=False,
        rng=None,
    ):
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        n_frames = len(image)

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode == "resize":
            resized, _, patch_idx = self.build_resized_image(image, is_training=is_training, rng=rng)

            if self.video_low_res_collage:
                n = self.video_low_res_collage**2
                # Resize first so the image don't get blurred into one-another
                collage_resized, _, _ = self.build_resized_image(
                    image, is_training=is_training, rng=rng,
                    image_size=[
                        base_image_input_size[0]//self.video_low_res_collage,
                        base_image_input_size[1]//self.video_low_res_collage
                    ])

                # Pad and resize into a collage
                r = len(image) % n
                frame_pad = n - r
                if r != 0:
                    collage_resized = np.pad(collage_resized, [[0, n - r], [0, 0], [0, 0], [0, 0]])
                collage_resized = einops.rearrange(
                    collage_resized, "(b dh dw) h w c -> b (dh h) (dw w) c",
                    dh=self.video_low_res_collage, dw=self.video_low_res_collage
                )

                # Build the idx inverting the re-arangement, and the building the pooling
                # targets
                collage_idx = np.arange(len(collage_resized)*crop_patch_h*crop_patch_w).reshape(
                    len(collage_resized), crop_patch_h, crop_patch_w)
                # import pdb; pdb.set_trace()
                # collage_idx = collage_idx.transpose(0, 2, 1)
                collage_idx = einops.rearrange(
                    collage_idx, "b (dh h) (dw w) -> (b dh dw) h w",
                    dh=self.video_low_res_collage, dw=self.video_low_res_collage
                )[:len(image)]
                low_res_idx = arange_video_for_pooling(collage_idx, self.video_low_res, self.video_low_res)

                # Add the new frames, and offset the patch_ids for the original frames
                resized = np.concatenate([collage_resized, resized], 0)
                high_res_offset = np.prod(collage_resized.shape[:3]) // (image_patch_size**2)
            else:
                high_res_offset = 0
                low_res_idx = arange_frames_for_pooling(
                    patch_idx, len(image), self.video_low_res, self.video_low_res)
            low_h, low_w = low_res_idx.shape[1:3]

            per_row = np.full(
                (low_w,),
                self.tokenizer.image_low_res_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [low_h])
            low_res_joint = [
                [self.tokenizer.image_start_token_id],
                extra_tokens,
            ] * len(image) + [[self.tokenizer.image_end_token_id]]

            high_res_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            high_res_idx += high_res_offset
            high_res_idx = arange_frames_for_pooling(high_res_idx, len(image), self.video_high_res, self.video_high_res)
            high_h, high_w = high_res_idx.shape[1:3]
            max_high_res_per_crop = high_w * high_h
            max_high_res_tokens = min(max_high_res_tokens, max_high_res_per_crop*len(image))

            low_res_f = self.low_to_high_interpolation_factor
            low_to_high = np.eye((low_h*low_res_f*low_w*low_res_f), dtype=np.float32)
            low_to_high = low_to_high.reshape([low_h*low_w*low_res_f*low_res_f, low_h*low_res_f, low_w*low_res_f])
            low_to_high = torchvision.transforms.Resize(
                [high_h, high_w], InterpolationMode.BILINEAR, antialias=False)(
                torch.from_numpy(low_to_high)).numpy().astype(np.float32)

            low_to_high_frames = np.zeros([n_frames, n_frames] + list(low_to_high.shape), dtype=np.float32)
            low_to_high_frames[np.arange(n_frames), np.arange(n_frames)] = np.tile(low_to_high[None, :, :, :], [n_frames, 1, 1, 1])

            # Re-arrange to match how the importance scores are predicted (four per a patch)
            # This save us having to transpose it in the model
            low_to_high_frames = einops.rearrange(
                low_to_high_frames,
                # "fi fo (lh dh lw dw) h w -> (fi lw lh dw dh) (fo h w) ",
                "fi fo (lh dh lw dw) h w -> (fi lh lw dh dw) (fo h w) ",
                dh=low_res_f, dw=low_res_f, lh=low_h, lw=low_w)

            n_low = sum(len(x) for x in low_res_joint)
            high_res_pos_ids = np.ones(np.prod(high_res_idx.shape[:-1]), dtype=np.int32)
            high_res_per_crop = high_res_idx.shape[1] * high_res_idx.shape[2]
            high_res_pos_ids[::high_res_per_crop] = 2
            high_res_pos_ids[0] -= 1
            high_res_pos_ids = np.cumsum(high_res_pos_ids)

            joint_pos_ids = [
                np.arange(n_low),  # low res tokens
                n_low + np.arange(len(image)) * (max_high_res_per_crop + 1),  # frame starts
                np.full([max_high_res_tokens], n_low),  # high res
                [n_low + high_res_pos_ids.max() + 1]  # frame end
            ]
            joint = low_res_joint + [
                [self.tokenizer.image_start_token_id] * len(image),
                np.full([max_high_res_tokens], self.tokenizer.image_patch_token_id),
                [self.tokenizer.image_end_token_id],
            ]
            return (
                batch_pixels_to_patches(resized, image_patch_size),
                np.concatenate(joint, 0),
                np.concatenate(joint_pos_ids, 0),
                low_res_idx.reshape(-1, low_res_idx.shape[-1]),
                high_res_idx.reshape(-1, high_res_idx.shape[-1]),
                (
                    low_to_high_frames,
                    high_res_pos_ids
                )
            )
        else:
            raise ValueError()

    def _sample(self, rng: np.random, min_k, max_k, num_k=None, sample_k=None):
        if min_k is None:
            if num_k is None:
                return max_k
            else:
                return [max_k] * num_k
        if sample_k is None:
            return rng.randint(min_k, max_k, size=num_k if num_k else ())
        else:
            raise NotImplementedError(sample_k)

    def __call__(
        self,
        images,
        messages: Union[List[str], List[List[str]]],
        frame_times=None,
        weight=None,
        is_training=False,
        rng=None,
        require_image_features=False,
        max_high_res: Optional[int] = None,
        min_high_res: Optional[int] = None,
        sample_high_res: Optional[str] = None,
    ):
        """Interleave images and text tokens into multi-modal features for the model"""
        max_high_res = max_high_res or self.num_high_res_features
        min_high_res = min_high_res or self.multi_res_min

        if not is_training:
            n_high_res = max_high_res
        else:
            n_high_res = self._sample(rng, min_high_res, max_high_res, None, sample_high_res)
        if isinstance(images, (list, tuple)):
            if len(images) > 1:
                raise NotImplementedError("Multi-image input")
            images = images[0]

        if len(images.shape) == 4:
            (
                crops,
                image_tokens,
                image_pos_ids,
                low_pooled_idx,
                pooled_idx,
                _high_res_data
            ) = self.video_to_patches_and_tokens(images, n_high_res, is_training, rng)
            prefix = []
            if self.time_mode == "fps-prefix":
                tmp = np.array(frame_times)
                deltas = tmp[1:] - tmp[:-1]
                fps = np.mean(deltas)
                prefix.append(f"FPS {fps:0.2f}")
            if self.indicate_k:
                prefix.append(f"K={n_high_res}")
            if prefix:
                prefix = self.tokenizer.encode(" ".join(prefix))
                image_tokens = np.concatenate([prefix, image_tokens])
                image_pos_ids = np.concatenate([np.arange(len(prefix)), len(prefix) + image_pos_ids])
            elif self.time_mode is not None:
                raise NotImplementedError(self.time_mode)
        else:
            (
                image_tokens,
                image_pos_ids,
                crops,
                img_mask,
                low_pooled_idx,
                pooled_idx,
                _high_res_data
            ) = self.image_to_patches_and_tokens(images, n_high_res, is_training, rng)

        out = self.tokenize_and_interleave(
            messages,
            [image_tokens],
            [image_pos_ids],
        )
        n_high_res_tokens = (image_tokens == self.tokenizer.image_patch_token_id).sum()
        out.update({
            "images": crops,
            "low_res_tokens_idx": low_pooled_idx,
            "high_res_tokens_idx": pooled_idx,
            "low_to_high": _high_res_data[0],
            "high_res_pos_ids": _high_res_data[1],
            "high_res_features_weights": np.ones(n_high_res_tokens, dtype=np.float32),
        })
        return out


