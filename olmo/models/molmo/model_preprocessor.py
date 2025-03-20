import dataclasses
import math
from typing import List, Optional, Union, Any, Tuple

import PIL
from PIL import ImageFile
from einops import einops

from olmo import tokenizer
from olmo.config import BaseConfig
from olmo.data.image_preprocessor import load_image, ImagePreprocessor
from olmo.tokenizer import get_special_token_ids
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


import numpy as np
import torch

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
)

from olmo.models.molmo.data_formatter import DataFormatter


def batch_pixels_to_patches(array, patch_size):
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, w, h = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, w, h, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def arange_for_pooling(idx_arr, pool_h, pool_w):
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)



@dataclasses.dataclass
class MolmoPreprocessorConfig(BaseConfig):
    crop_mode: str = "resize"
    """How to divide the images into crops"""

    max_crops: int = 6
    """Max number of crops to produce per an image"""

    pooling_w: int = 2
    """Patch pooling w stride"""

    pooling_h: int = 2
    """Patch pooling h stride"""

    overlap_margins: Tuple[int, int] = (4, 4)
    """Overlap margins for overlapping crops modes"""

    use_col_tokens: bool = True
    """Use column tokens in the image tokens"""

    loss_token_weighting: Optional[str] = None
    """Automatically weight multi-message per image input"""

    legacy_image_mask: bool = False
    """Use legacy off-by-one mask, should only be true for old models"""

    def get_max_crops(self) -> int:
        """Max numbers of that can be built for one image"""
        if self.crop_mode == "resize":
            return 1
        elif "resize" in self.crop_mode:
            # low_res crop + the high-res crops
            return 1 + self.max_crops
        else:
            return self.max_crops

    def get_image_padding_lens(self, vision_backbone_config: MolmoVisionBackboneConfig):
        """Max numbers of image tokens can be built for one image"""
        padding_lens = dict(
            images=self.get_max_crops()
        )
        if vision_backbone_config.image_padding_embed:
            padding_lens["image_masks"] = self.get_max_crops()
        preprocessor = self.build(None, vision_backbone_config)
        padding_lens["pooled_patches_idx"] = preprocessor.max_image_tokens()
        return padding_lens

    def build(self, tokenizer, vision_backbone_config: MolmoVisionBackboneConfig):
        vit = vision_backbone_config.vit
        return MolmoPreprocessor(
            tokenizer=tokenizer,
            loss_token_weighting=self.loss_token_weighting,
            legacy_image_mask=self.legacy_image_mask,
            normalize=vit.normalize,
            crop_mode=self.crop_mode,
            max_crops=self.max_crops,
            overlap_margins=self.overlap_margins,
            resize=vit.resize_mode,
            use_col_tokens=self.use_col_tokens,

            base_image_input_size=vit.image_default_input_size,
            image_pooling_w=self.pooling_w,
            image_pooling_h=self.pooling_h,
            image_patch_size=vit.image_patch_size,
            image_padding_mask=vision_backbone_config.image_padding_embed is not None,
            pad_value=vit.pad_value,
        )


@dataclasses.dataclass
class MolmoPreprocessor(ImagePreprocessor):
    """
    Converts text/images inputs into tensors that can be used in the forward method
    for the a model
    """
    tokenizer: Any = None
    loss_token_weighting: Optional[str] = None
    legacy_image_mask: bool = False

    # How to crops/resize images
    crop_mode: str = "default"
    use_col_tokens: bool = True

    # Data about the ViT and connector we need when deciding the crops
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    image_padding_mask: Union[bool, int] = False

    image_patch_token_id: int = dataclasses.field(init=False)
    image_col_token_id: int = dataclasses.field(init=False)
    image_start_token_id: int = dataclasses.field(init=False)
    image_end_token_id: int = dataclasses.field(init=False)

    def __post_init__(self):
        if self.tokenizer is not None:
            special_tokens = get_special_token_ids(self.tokenizer)
            self.image_end_token_id = special_tokens[tokenizer.IM_END_TOKEN]
            self.image_start_token_id = special_tokens[tokenizer.IM_START_TOKEN]
            self.image_col_token_id = special_tokens[tokenizer.IM_COL_TOKEN]
            self.image_patch_token_id = special_tokens[tokenizer.IMAGE_PATCH_TOKEN]
            self.image_low_res_token_id = special_tokens[tokenizer.IMAGE_LOW_RES_TOKEN]
            self.image_prompt_token_id = special_tokens[tokenizer.IMAGE_PROMPT]

    def max_image_tokens(self) -> int:
        """Return the max number of pooled image tokens this could produce for any image"""
        base_h, base_w = self.base_image_input_size
        # Doing the math on what the max is not-trivial, just brute force it assuming long
        # skinny images are the wors due the having the least overlap
        max_tokens = -1
        for h, w in [
            [base_h, base_w*self.max_crops],
            [base_h*self.max_crops, base_w]
        ]:
            max_tokens = max(max_tokens, self.compute_num_tokens(
                h, w, self.image_pooling_h, self.image_pooling_w))
        return max_tokens

    def compute_num_tokens(self, image_h, image_w, pool_h, pool_w) -> int:
        """Return the number of pooled image tokens produced for an image of size image_w, image_h"""
        image_patch_size = self.image_patch_size
        crop_patch_w = self.base_image_input_size[1] // image_patch_size
        crop_patch_h = self.base_image_input_size[0] // image_patch_size

        resize_idx = np.zeros([crop_patch_h, crop_patch_w])
        idx_arr = arange_for_pooling(resize_idx, pool_h, pool_w)
        resize_tokens = idx_arr.shape[0] * idx_arr.shape[1]

        if self.crop_mode in ["resize"]:
            return resize_tokens

        h, w = self.compute_overlapping_crops_size(image_h, image_w)
        idx_arr = arange_for_pooling(torch.zeros([h, w]), pool_h, pool_w)
        overlap_tokens = idx_arr.shape[0] * idx_arr.shape[1]
        if self.crop_mode in ["overlap-and-resize-c2"]:
            return overlap_tokens + resize_tokens
        else:
            return overlap_tokens

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        pooling_h: int,
        pooling_w: int,
        patch_id: int,
        is_training=False,
        rng=None,
    ):
        """
        :return image_tokens, the token IDS for this image, including special tokens
        :return crops, the image crops to processes with the ViT
        :return mask, the padding mask for each crop
        :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                  patches in `crops` to pool for that token, masked with -1
        """
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

        if self.crop_mode == "resize":
            resized, resized_mask, resize_idx = self.build_resized_image(image, is_training=is_training, rng=rng)
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
            per_row = np.full(
                (w,),
                patch_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                        [self.image_start_token_id],
                        extra_tokens,
                        [self.image_end_token_id],
            ]
            return (np.concatenate(joint, 0), batch_pixels_to_patches(resized, image_patch_size),
                    batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1), pooling_idx)

        if self.crop_mode in ["overlap-and-resize-c2", "overlap-and-resize"]:
            crop_arr, mask_arr, patch_idx_arr = self.build_overlapping_crops(image, is_training=is_training, rng=rng)
            pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])

            # Now build the output tokens
            per_row = np.full(w, self.image_patch_token_id, dtype=np.int32)
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            joint = np.tile(per_row, [h])
            joint = [
                [self.image_start_token_id],
                joint,
                [self.image_end_token_id]
            ]

            if self.crop_mode == "overlap-and-resize":
                crop_arr = batch_pixels_to_patches(crop_arr, image_patch_size)
                mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
                return np.concatenate(joint, 0), crop_arr, mask_arr, pooling_idx

            # Finally do the same for the global image
            resized, resized_mask, resize_idx = self.build_resized_image(image, is_training=is_training, rng=rng)
            crop_arr = np.concatenate([resized, crop_arr], 0)

            if self.legacy_image_mask:
                mask_arr = np.pad(mask_arr.astype(np.float32), [[0, 1], [0, 0], [0, 0]], constant_values=-1)
            else:
                mask_arr = np.concatenate([resized, mask_arr], 0)

            resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = resize_idx.shape[:2]
            resize_idx = resize_idx.reshape([-1, pooling_h*pooling_w])

            # Global image goes first, so the order of patches in previous crops gets increased
            pooling_idx = np.where(
                pooling_idx >= 0,
                pooling_idx + crop_patch_h*crop_patch_w,
                -1
            )
            pooling_idx = np.concatenate([resize_idx, pooling_idx])

            per_row = np.full(
                (w,),
                patch_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                        [self.image_start_token_id],
                        extra_tokens,
                        [self.image_end_token_id],
                    ] + joint
            mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
            return (np.concatenate(joint, 0), batch_pixels_to_patches(crop_arr, image_patch_size),
                    mask_arr, pooling_idx)
        else:
            raise NotImplementedError(self.crop_mode)

    def __call__(
        self,
        images,
        messages: Union[List[str], List[List[str]]],
        weight=None,
        is_training=False,
        rng=None,
        require_image_features=False
    ):
        """Interleave images and text tokens into multi-modal features for the model"""
        if len(messages) == 0:
            raise ValueError("Given empty messages")
        if not isinstance(messages[0], str) and len(messages) == 1:
            messages = messages[0]
        if isinstance(messages[0], str):
            # List of user/system/user/system ect. prompts
            loss_masks = []
            token_ids = []
            for msg_ix, message in enumerate(messages):
                has_loss = msg_ix % 2 == 1
                message_ids = self.tokenizer.encode(message)
                if has_loss:
                    message_ids.append(self.tokenizer.eos_token_id)
                token_ids += message_ids
                if weight is None:
                    loss_masks += [has_loss]*len(message_ids)
                else:
                    loss_masks += [weight if has_loss else 0]*len(message_ids)
            tokens = np.array(token_ids, dtype=np.int32)
            loss_masks = np.array(loss_masks, dtype=np.float32)
            subsegments = None
        else:
            if weight is not None:
                raise NotImplementedError("Multi-messages with weights")
            # List of lists of user/system/user/system ect. prompts
            subsegments = []
            loss_masks = []
            token_ids = []
            for message_set_ix, message_set in enumerate(messages):
                n = 0
                for msg_ix, message in enumerate(message_set):
                    has_loss = msg_ix % 2 == 1
                    message_ids = self.tokenizer.encode(message)
                    if has_loss:
                        message_ids.append(self.tokenizer.eos_token_id)
                    token_ids += message_ids
                    loss_masks.append(np.full(len(message_ids), has_loss, dtype=np.bool_))
                    n += len(message_ids)
                subsegments.append(np.full(n, message_set_ix+1, dtype=np.int32))
            tokens = np.array(token_ids, dtype=np.int32)
            loss_masks = np.concatenate(loss_masks, dtype=np.float32)
            subsegments = np.concatenate(subsegments, dtype=np.int32)
            if weight is not None:
                loss_masks *= weight
            if self.loss_token_weighting == "root_subsegments":
                loss_masks *= math.sqrt(1/len(messages))
            elif self.loss_token_weighting is not None:
                raise NotImplementedError(self.loss_token_weighting)

        if images is None or (
            isinstance(images, (list, tuple)) and len(images) == 0
        ):
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            decoder_input_tokens = np.pad(tokens, [[1, 0]], constant_values=bos)[:-1]
            data = {"input_tokens": tokens, "loss_masks": loss_masks, "target_tokens": tokens}
            if subsegments is not None:
                subsegments = np.pad(subsegments, [[1, 0]], constant_values=subsegments[0])[:-1]
                data["subsegments"] = subsegments
            if require_image_features:
                raise NotImplementedError("")
            return data

        if not isinstance(images, (list, tuple)):
            images = [images]
        image_idx = np.argwhere(tokens == self.image_prompt_token_id)
        if len(image_idx) == 0:
            image_idx = [-1] * len(images)
        else:
            image_idx = image_idx[:, 0]
            assert len(image_idx) == len(images)

        n = len(images)
        all_crops = []
        out_tokens = []
        all_crop_masks = []
        all_subsegments = []
        all_loss_masks = []
        pooled_patches_idx = []

        for ix in range(n):
            token_ix = image_idx[ix]
            image_tokens, crops, img_mask, pooled_idx = self.image_to_patches_and_tokens(
                images[ix], self.image_pooling_h, self.image_pooling_w,  self.image_patch_token_id, is_training, rng)
            if token_ix == -1:  # -1 is an image inserted at the very start
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix-1] + 1
                end = token_ix + 1

            pooled_patches_idx.append(pooled_idx + sum(np.prod(x.shape[:2]) for x in all_crops))
            all_crops.append(crops)
            out_tokens.append(tokens[start:token_ix])
            all_loss_masks.append(loss_masks[start:token_ix])

            out_tokens.append(image_tokens)
            all_loss_masks.append(np.zeros(image_tokens.shape[0], dtype=np.float32))
            if subsegments is not None:
                all_subsegments.append(subsegments[start:token_ix])
                image_subsegment = 10000 if image_idx[ix] == -1 else token_ix
                all_subsegments.append(np.full(len(image_tokens), image_subsegment, np.int32))
            if self.image_padding_mask:
                all_crop_masks.append(img_mask)

        end = image_idx[-1] + 1
        out_tokens.append(tokens[end:])
        all_loss_masks.append(loss_masks[end:])
        if subsegments is not None:
            all_subsegments.append(subsegments[end:])

        input_ids = np.concatenate(out_tokens, 0)
        images = np.concatenate(all_crops, 0)
        pooled_patches_idx = np.concatenate(pooled_patches_idx, 0)
        all_loss_masks = np.concatenate(all_loss_masks, 0)

        target_tokens = input_ids
        ends_with_eos = input_ids[-1] == self.tokenizer.eos_token_id
        if not ends_with_eos and loss_masks[-1]:
            raise RuntimeError("EOS should not be masked")

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids = np.pad(input_ids, [[1, 0]], constant_values=bos)
        if ends_with_eos:
            input_ids = input_ids[:-1]
        else:
            # We are presumably doing inference since the messages end with user response instead
            # of a target response, so these fields should not be used, but pad them anyway
            # just so everything is a consistent length
            all_loss_masks = np.pad(all_loss_masks, [[0, 1]], constant_values=-1)
            target_tokens = np.pad(target_tokens, [[0, 1]], constant_values=-1)

        out = {
            "input_tokens": input_ids,
            "images": images,
            "pooled_patches_idx": pooled_patches_idx,
            "loss_masks": all_loss_masks,
            "target_tokens": target_tokens,
        }
        if self.image_padding_mask:
            out["image_masks"] = np.concatenate(all_crop_masks, 0)
        if subsegments is not None:
            all_subsegments = np.concatenate(all_subsegments, 0)
            all_subsegments = np.pad(all_subsegments, [[1, 0]], constant_values=all_subsegments[0])
            if ends_with_eos:
                all_subsegments = all_subsegments[:-1]
            out["subsegment_ids"] = all_subsegments
            position_ids = np.zeros_like(all_subsegments)
            for subsegment_id in np.unique(all_subsegments):
                segment_position_ids = np.cumsum(all_subsegments >= subsegment_id) - 1
                position_ids = np.where(all_subsegments == subsegment_id, segment_position_ids, position_ids)
            out["position_ids"] = position_ids
        else:
            out["position_ids"] = np.arange(len(input_ids), dtype=np.int64)
        return out


@dataclasses.dataclass
class Preprocessor:
    formater: DataFormatter
    mm_preprocessor: MolmoPreprocessor
    for_inference: bool = False
    is_training: bool = False
    include_image: bool = False
    require_image_features: bool = False

    def __call__(self, example, rng=np.random):
        example = dict(example)
        if "image" in example:
            try:
                image = load_image(example["image"])
            except Exception as e:
                raise ValueError(f"Could not load image: {example['image']}")
            else:
                example["image"] = image
        else:
            image = None

        messages, formatter_metadata = self.formater(example, self.is_training, self.for_inference, rng)
        if isinstance(messages[0], list):
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(messages)
        batch = self.mm_preprocessor(
            image,
            messages,
            weight=example.get("weight"),
            rng=rng,
            is_training=self.is_training,
            require_image_features=self.require_image_features
        )
        if formatter_metadata is None:
            formatter_metadata = {}
        if self.include_image and image is not None:
            formatter_metadata["image"] = image
        if image is not None:
            h, w = image.shape[:2]
            formatter_metadata["image_size"] = (w, h)
        if "metadata" in example or formatter_metadata:
            metadata = example.get("metadata", {})
            if formatter_metadata:
                metadata.update(formatter_metadata)
            batch["metadata"] = metadata
        return batch

    @property
    def tokenizer(self):
        return self.mm_preprocessor.tokenizer