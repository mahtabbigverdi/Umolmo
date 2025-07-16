import dataclasses
import math
from typing import List, Optional, Union, Any, Tuple

import PIL
from PIL import ImageFile
from einops import einops

from olmo import tokenizer
from olmo.config import BaseConfig
from olmo.data.image_preprocessor import load_image, ImagePreprocessor, get_image_collage
from olmo.data.interleaved_text_preprocessor import InterleavedTextPreprocessor
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
        n_crops, h, w = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
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

    max_images: Optional[int] = None
    """Max number of images per example, used for multi-image examples"""

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
        max_crops = self.get_max_crops()
        preprocessor = self.build(None, vision_backbone_config, None)
        max_image_tokens = preprocessor.max_image_tokens()
        if self.max_images is not None:
            max_crops = self.max_images * max_crops
            max_image_tokens = self.max_images * max_image_tokens
        """Max numbers of image tokens can be built for one image"""
        padding_lens = dict(
            images=max_crops
        )
        if vision_backbone_config.image_padding_embed:
            padding_lens["image_masks"] = max_crops
        padding_lens["pooled_patches_idx"] = max_image_tokens
        return padding_lens

    def build(self, tokenizer, vision_backbone_config: MolmoVisionBackboneConfig, max_seq_len):
        vit = vision_backbone_config.vit
        return MolmoPreprocessor(
            tokenizer=tokenizer,
            loss_token_weighting=self.loss_token_weighting,
            legacy_image_mask=self.legacy_image_mask,
            normalize=vit.normalize,
            crop_mode=self.crop_mode,
            max_crops=self.max_crops,
            max_images=self.max_images,
            overlap_margins=self.overlap_margins,
            resize=vit.resize_mode,
            use_col_tokens=self.use_col_tokens,
            max_sequence_length=max_seq_len,

            base_image_input_size=vit.image_default_input_size,
            image_pooling_w=self.pooling_w,
            image_pooling_h=self.pooling_h,
            image_patch_size=vit.image_patch_size,
            image_padding_mask=vision_backbone_config.image_padding_embed is not None,
            pad_value=vit.pad_value,
        )


@dataclasses.dataclass
class MolmoPreprocessor(InterleavedTextPreprocessor, ImagePreprocessor):
    """
    Converts text/images inputs into tensors that can be used in the forward method
    for the a model
    """
    legacy_image_mask: bool = False

    # How to crops/resize images
    crop_mode: str = "resize"
    use_col_tokens: bool = True

    # Data about the ViT and connector we need when deciding the crops
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    image_padding_mask: Union[bool, int] = False
    max_images: Optional[int] = None

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
        resize_h = idx_arr.shape[0]
        resize_w = idx_arr.shape[1] + int(self.use_col_tokens)
        resize_tokens = resize_h * resize_w + 2 # start and end tokens

        if self.crop_mode in ["resize"]:
            return resize_tokens

        h, w = self.compute_overlapping_crops_size(image_h, image_w)
        idx_arr = arange_for_pooling(torch.zeros([h, w]), pool_h, pool_w)
        overlap_h = idx_arr.shape[0]
        overlap_w = idx_arr.shape[1] + int(self.use_col_tokens)
        overlap_tokens = overlap_h * overlap_w + 2 # start and end tokens
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
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                [self.tokenizer.image_start_token_id],
                extra_tokens,
                [self.tokenizer.image_end_token_id],
            ]
            return (np.concatenate(joint, 0), batch_pixels_to_patches(resized, image_patch_size),
                    batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1), pooling_idx)

        if self.crop_mode in ["overlap-and-resize-c2", "overlap-and-resize"]:
            crop_arr, mask_arr, patch_idx_arr = self.build_overlapping_crops(image, is_training=is_training, rng=rng)
            pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])

            # Now build the output tokens
            per_row = np.full(w, self.tokenizer.image_patch_token_id, dtype=np.int32)
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            joint = np.tile(per_row, [h])
            joint = [
                [self.tokenizer.image_start_token_id],
                joint,
                [self.tokenizer.image_end_token_id]
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
                mask_arr = np.concatenate([resized_mask, mask_arr], 0)

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
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                        [self.tokenizer.image_start_token_id],
                        extra_tokens,
                        [self.tokenizer.image_end_token_id],
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
        image_outputs,
        weight=None,
        is_training=False,
        rng=None,
        require_image_features=False
    ):
        """Interleave images and text tokens into multi-modal features for the model"""
        if images is None or (
            isinstance(images, (list, tuple)) and len(images) == 0
        ):
            out = self.tokenize_and_interleave(messages, [])
            if require_image_features:
                raise NotImplementedError("")
            return out

        if not isinstance(images, (list, tuple)):
            images = [images]
        all_image_tokens = []
        all_crop_masks = []
        all_crops = []
        pooled_patches_idx = []
        
        for idx, image in enumerate(images):
            image_tokens, crops, img_mask, pooled_idx = self.image_to_patches_and_tokens(
                image, self.image_pooling_h, self.image_pooling_w,  self.tokenizer.image_patch_token_id, is_training, rng)
            pooled_patches_idx.append(pooled_idx + sum(np.prod(x.shape[:2]) for x in all_crops))
            all_crops.append(crops)
            if len(images) > 1:
                # Add prefix to the image tokens if there are multiple images
                prefix = f"Image {idx + 1}"
                image_tokens = np.concatenate([self.tokenizer.encode(prefix), image_tokens], 0)
            all_image_tokens.append(image_tokens)
            all_crop_masks.append(img_mask)
            if self.max_images is not None and idx == self.max_images - 1:
                break
        
        out = self.tokenize_and_interleave(messages, all_image_tokens, image_outputs, weight=weight)
        out["images"] = np.concatenate(all_crops)
        out["pooled_patches_idx"] = np.concatenate(pooled_patches_idx)
        if self.image_padding_mask:
            out["image_masks"] = np.concatenate(all_crop_masks)
        out['image_outputs'] = np.concatenate(image_outputs) if len(image_outputs) > 0 else np.empty((0,0))
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
                if isinstance(example["image"], (list, tuple)):
                    image = [load_image(x) for x in example["image"]]
                else:
                    image = load_image(example["image"])
            except Exception as e:
                raise ValueError(f"Could not load image: {example['image']}")
            else:
                example["image"] = image
        else:
            image = None
        
        if "image_outputs" in example and len(example["image_outputs"]) > 0:
            image_outputs = [np.load(example["image_outputs"][i]) for i in range(len(example["image_outputs"]))]
        else:
            image_outputs = []
        # if self.is_training == False:
        #     import pdb; pdb.set_trace()
        messages, formatter_metadata = self.formater(example, self.is_training, self.for_inference, rng)
        if isinstance(messages[0], list):
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(messages)
        batch = self.mm_preprocessor(
            image,
            messages,
            image_outputs,
            weight=example.get("weight"),
            rng=rng,
            is_training=self.is_training,
            require_image_features=self.require_image_features
        )

        

        if image is not None and isinstance(image, (list, tuple)):
            image = get_image_collage(image)
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