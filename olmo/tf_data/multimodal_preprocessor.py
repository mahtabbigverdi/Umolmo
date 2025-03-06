import dataclasses
import logging
import re
from collections import defaultdict
from typing import Tuple, Optional, Any, Dict, Union, Mapping

import einops
import seqio
import numpy as np
import tensorflow as tf

from olmo import tokenizer
from olmo.config import BaseConfig
from olmo.tf_data.data_utils import pad_to_bounding_box, \
    get_3d_subsegments, _append_to_innermost_axis, resize_and_pad, \
    apply_with_random_selector, make_autoregressive_inputs, \
    trim_and_pad_dataset, assert_not_truncated, normalize_image
from olmo.tf_data.prompts import apply_keyword_prompt, STYLE_TO_GENERAL_PROMPT, GENERAL_PROMPTS_V1
from olmo.tokenizer import get_special_token_ids
from olmo.vision_backbone import VisionBackboneConfig


def is_floating(image: Union[tf.Tensor, np.ndarray]) -> bool:
    if isinstance(image, tf.Tensor):
        return image.dtype.is_floating
    else:
        return np.issubdtype(image.dtype, np.floating)


def siglip_resize(src, imgsize, truncate):
    """Resize and preprocess for SigLIP ViT in the offical jax implementation"""
    # SigCLIP removes aspect ratio by default
    dtype = src.dtype
    if is_floating(src):
        in_min = 0
        in_max = 1.0
        resized = tf.image.resize(src, imgsize, method=tf.image.ResizeMethod.BILINEAR, antialias=False)
        resized = tf.cast(tf.clip_by_value(resized, 0.0, 1.0), dtype)
    else:
        assert src.dtype == tf.uint8, "SigLIP expects float images or uint8 images, but got {}".format(src.dtype)
        in_min = 0
        in_max = 255.0
        resized = tf.image.resize(src, imgsize, method=tf.image.ResizeMethod.BILINEAR, antialias=False)
        tf_dtype = tf.type_spec_from_value(src).dtype
        resized = tf.cast(tf.clip_by_value(resized, tf_dtype.min, tf_dtype.max), dtype)

    # Normalize between -1 and 1 without using imagenet standard mean/std
    vmin=-1; vmax=1
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(resized, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if truncate:
        image = image[:truncate, :truncate]
    return image


def dino_resize(image, size, truncate):
    if is_floating(image):
        image = tf.image.resize(image, size, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        image = tf.clip_by_value(image, 0.0, 1.0)
    else:
        assert image.dtype == tf.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
        image = tf.image.resize(image, size, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = image / 255.0
    image = normalize_image(image, offset=(0.485, 0.456, 0.406), scale=(0.229, 0.224, 0.225))
    if truncate:
        image = image[:truncate, :truncate]
    return image


def metaclip_resize(image, size, truncate):
    if is_floating(image):
        image = tf.image.resize(image, size, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        image = tf.clip_by_value(image, 0.0, 1.0)
    else:
        assert image.dtype == tf.uint8, "MetaCLIP expects float images or uint8 images, but got {}".format(image.dtype)
        image = tf.image.resize(image, size, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = image / 255.0
    image = normalize_image(image)
    if truncate:
        image = image[:truncate, :truncate]
    return image


def extract_annotated_points(caption, image_w, image_h):
    points = []
    for match in re.finditer("<point x=\"([0-9\\.]*)\" y=\"([0-9\\.]*)\" alt=\"([^\"]*)\">", caption):
        x = float(match.group(1))
        y = float(match.group(2))
        points.append(([[x, y]], match.group(3)))
    for match in re.finditer("<points ([^<]*) alt=\"([^\"]*)\">", caption):
        loc_str = match.group(1)
        locations = defaultdict(dict)
        if loc_str.startswith("points="):
            point_grp = []
            for point_match in re.finditer(r"([0-9]+\.[0-9]),? ([0-9]+\.[0-9])", loc_str):
                try:
                    point = [float(point_match.group(i)) for i in range(1, 3)]
                    point_grp.append(point)
                except ValueError:
                    pass
        else:
            for val in loc_str.split():
                try:
                    key, val = val.split("=")
                    locations[key[1:]][key[:1]] = float(val.strip("\""))
                except ValueError:
                    import pdb; pdb.set_trace()
                    logging.warning(f"Failed to parse {val} from {match.group(0)}")
            point_grp = []
            for key, coords in locations.items():
                if sorted(coords) == ["x", "y"]:
                    point_grp.append([coords["x"], coords["y"]])
        if point_grp:
            points.append((point_grp, match.group(2)))

    normalized = []
    for point_grp, point_text in points:
        normalized.append((
            np.array(point_grp) / 100.0 * np.array([image_w, image_h]),
            point_text,
        ))
    return normalized


def select_tiling(h, w, patch_size, max_num_patches):
    """Decide how best to divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = tf.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_patches+1):
        for j in range(1, max_num_patches+1):
            if i*j <= max_num_patches:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = tf.constant(tilings, dtype=tf.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    required_scale_d = tf.cast(candidate_resolutions, tf.float32) / tf.cast(original_size[None, :], tf.float32)
    required_scale = tf.reduce_min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if tf.reduce_all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = tf.argmax(required_scale)[0]
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = tf.where(required_scale < 1.0, 10e9, required_scale)
        ix = tf.argmin(required_scale)[0]
    return candidate_tilings[ix]


DEMO_STYLES = [
    "point_count",
    "pointing",
    "user_qa",
    "scifi_charts_exp",
    "scifi_charts_exp",
    "scifi_charts_exp",
    "scifi_charts_exp",
    "long_caption",
    "named_entity"
]


@dataclasses.dataclass
class MultiModalPreprocessor:
    """Turns text/image inputs into tensors that can be input to the model"""
    tokenizer: Any

    # How to prompt the model
    prompt_templates: str = "none"  # How to template prompts for examples
    message_format: str = "none"  # How to format messages
    system_prompt: Optional[str] = None  # How to generate system prompts
    prompt_override: Optional[str] = None  # Used for setting prompt manually
    always_start_with_space: bool = False  # Always include a leading space for the first bit of text
    default_inference_len: int = 65  # Inference len for length-conditioned prompting

    # How to crops/resize images
    crop_mode: str = "resize"
    max_crops: int = 6
    overlap_margins: Tuple[int, int] = (4, 4)
    do_random_scale: Optional[bool] = False
    resize: str = "default"
    use_col_tokens: bool = True

    # Data about the ViT and connector we need when deciding the crops
    base_image_input_size: Tuple[int, int] = (336, 336)
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    image_token_length_w: int = 12
    image_token_length_h: int = 12
    image_patch_size: int = 14
    image_padding_mask: Union[bool, int] = False

    # Other settings
    loss_token_weighting: Optional[str] = None
    unconditioned: Union[bool, float] = False  # Ignore images
    pad_value: Optional[float] = 0

    _special_tokens: Dict[str, int] = None
    include_raw_image: Optional[bool] = False

    def get_max_total_crops(self):
        if self.crop_mode == "resize":
            return 1
        elif "resize" in self.crop_mode:
            return 1 + self.max_crops
        else:
            return self.max_crops

    @property
    def image_num_patch(self):
        h, w = self.base_image_input_size
        return h//self.image_patch_size, w//self.image_patch_size

    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens

    def image_to_patches_and_tokens(self, image, is_training):
        """Preprocesses an image

        Args:
            image: [h, w, 3] image to preprocessing
        Returns:
            crops: (n_crops, n_patches, patch_dim) individual crops, `n_crops` might
                   change between images but the other dimension are fixed
            tokens: (n_tokens,) tf.int32 tokens, pad tokens indicate where to insert the
                                patch features, might include other special tokens as well
            patch_ordering: (n_crops, n_tokens_per_crop) order image features should be inserted
                            into the `tokens`, negative values indicates patches features to exclude
            padding_mask: (n_crops, h, w) mask of what pixels are padding, can be None
        """
        do_random_scale = self.do_random_scale
        if do_random_scale:
            do_random_scale = is_training

        base_image_input_size = self.base_image_input_size
        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        image_token_length_w, image_token_length_h = self.image_token_length_w, self.image_token_length_h
        base_image_input_d = self.image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d
        extra_image = False
        patch_ordering = None

        if self.resize == "default":
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            def _resize(_image, sz):
                return resize_and_pad(
                    _image, sz,
                    do_random_scale=do_random_scale,
                    random_scale_max=self.random_scale_max,
                    random_scale_min=self.random_scale_min,
                    random_scale_ratio=self.random_scale_ratio,
                    return_outputs=False,
                    pad_value=self.pad_value,
                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)
        elif self.resize == "stretch":
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            assert not do_random_scale

            def _resize(_image, sz):
                if not is_training:
                    img = tf.image.resize(_image, sz, antialias=True, method=tf.image.ResizeMethod.BILINEAR)
                else:
                    resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
                    img = apply_with_random_selector(
                        _image,
                        lambda x, method_idx: tf.image.resize(x, sz,
                                                              tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                                              antialias=True),
                        num_cases=len(resize_methods))
                return img, tf.ones(tf.shape(img)[:2], dtype=tf.bool)
        elif self.resize == "siglip":
            assert not do_random_scale

            def _resize(_image, sz):
                img = siglip_resize(_image, sz, truncate=None)
                return img, tf.ones(tf.shape(img)[:2], dtype=tf.bool)
        elif self.resize == "dino":
            assert not do_random_scale

            def _resize(_image, sz):
                img = dino_resize(_image, sz, truncate=None)
                return img, tf.ones(tf.shape(img)[:2], dtype=tf.bool)
        elif self.resize == "metaclip":
            assert not do_random_scale

            def _resize(_image, sz):
                img = metaclip_resize(_image, sz, truncate=None)
                return img, tf.ones(tf.shape(img)[:2], dtype=tf.bool)
        else:
            raise NotImplementedError(self.resize)

        def _img_to_patches(_img, _img_mask, dy=1, dx=1):
            _img = einops.rearrange(
                _img, '(dy h dh) (dx w dw) c -> (dy dx) (h w) (dh dw c)',
                dh=base_image_input_d,
                dw=base_image_input_d,
                dy=dy,
                dx=dx,
                h=image_base_patch_h,
                w=image_base_patch_w
            )
            _img_mask = einops.rearrange(
                _img_mask, '(dy h dh) (dx w dw) -> (dy dx) (h w) (dh dw)',
                dh=base_image_input_d,
                dw=base_image_input_d,
                dy=dy,
                dx=dx,
                h=image_base_patch_h,
                w=image_base_patch_w
            )
            return _img, tf.reduce_mean(tf.cast(_img_mask, tf.float32), -1)

        mode = self.crop_mode
        if mode == "resize":
            patches, img_mask = _resize(image, base_image_input_size)
            patches, img_mask = _img_to_patches(patches, img_mask)
            image_layout_impatch_w = 1
            image_layout_impatch_h = 1
            patch_ordering = tf.range(tokens_per_image)[None, :]

        elif mode in ["overlap", "overlap-and-resize-c2"]:
            original_image_h = tf.shape(image, out_type=tf.int32)[0]
            original_image_w = tf.shape(image, out_type=tf.int32)[1]
            crop_size = base_image_input_size[0]

            # Discard this many patches from the (left/top, right/bottom) of crops
            left_margin, right_margin = self.overlap_margins
            # Required for compatibility with image pooling
            assert left_margin % self.image_pooling_w == 0 and right_margin % self.image_pooling_w == 0
            assert left_margin % self.image_pooling_h == 0 and right_margin % self.image_pooling_h == 0
            total_margin_pixels = base_image_input_d*(right_margin + left_margin)  # pixels removed per dim
            crop_patches = base_image_input_size[0] // base_image_input_d  # patches per crop dim
            crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
            crop_window_size = crop_window_patches * base_image_input_d
            tiling = select_tiling(original_image_h - total_margin_pixels, original_image_w - total_margin_pixels,
                                   crop_window_size, self.max_crops)
            src, img_mask = _resize(
                image, [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels])

            n_crops = tiling[0]*tiling[1]
            patches_arr = tf.TensorArray(
                tf.float32, n_crops, element_shape=[crop_size, crop_size, 3])
            mask_arr = tf.TensorArray(
                tf.bool, n_crops, element_shape=[crop_size, crop_size])
            # We assume hxw pooling, but can allow padding the right/bottom with extra
            # patches if the number of patches per side is not even
            assert (crop_patches + self.image_pooling_h - 1) // self.image_pooling_h == image_token_length_h
            assert (crop_patches + self.image_pooling_w - 1) // self.image_pooling_w == image_token_length_w
            patch_ordering_arr = tf.TensorArray(
                tf.int32, n_crops, element_shape=[image_token_length_h, image_token_length_w])
            on = 0
            on_patch = 0
            for i in range(tiling[0]):
                y0 = i*crop_window_size
                if i == 0:
                    crop_y0 = 0
                else:
                    crop_y0 = left_margin // self.image_pooling_h

                crop_h = image_base_patch_h - (right_margin + left_margin)
                if i == 0:
                    crop_h += left_margin
                if i == (tiling[0]-1):
                    crop_h += right_margin
                for j in range(tiling[1]):
                    x0 = j*crop_window_size
                    if j == 0:
                        crop_x0 = 0
                    else:
                        crop_x0 = left_margin // self.image_pooling_w

                    crop_w = image_base_patch_w - (right_margin + left_margin)
                    if j == 0:
                        crop_w += left_margin
                    if j == (tiling[1]-1):
                        crop_w += right_margin

                    pooled_w = (crop_w + self.image_pooling_w - 1) // self.image_pooling_w
                    pooled_h = (crop_h + self.image_pooling_h - 1) // self.image_pooling_h
                    patch_ordering_arr = patch_ordering_arr.write(
                        on_patch,
                        pad_to_bounding_box(
                            tf.reshape(tf.range(on, on+pooled_h*pooled_w, dtype=tf.int32), (pooled_h, pooled_w, 1)),
                            crop_y0, crop_x0, image_token_length_h, image_token_length_w, value=-1
                        )[:, :, 0]
                    )
                    patches_arr = patches_arr.write(on_patch, src[y0:y0+crop_size, x0:x0+crop_size])
                    mask_arr = mask_arr.write(on_patch, img_mask[y0:y0+crop_size, x0:x0+crop_size])

                    on += pooled_h*pooled_w
                    on_patch += 1
            patches = patches_arr.stack()
            patch_ordering = patch_ordering_arr.stack()
            img_mask = mask_arr.stack()

            image_layout_impatch_w, image_layout_impatch_h = tiling[0], tiling[1]
            patches = einops.rearrange(
                patches, 'p (h dh) (w dw) c -> p (h w) (dh dw c)',
                dh=base_image_input_d,
                dw=base_image_input_d,
                h=image_base_patch_h,
                w=image_base_patch_w
            )
            img_mask = einops.rearrange(
                img_mask, 'p (h dh) (w dw) -> p (h w) (dh dw)',
                dh=base_image_input_d,
                dw=base_image_input_d,
                h=image_base_patch_h,
                w=image_base_patch_w
            )
            img_mask = tf.reduce_mean(tf.cast(img_mask, tf.float32), -1)
            patch_ordering = tf.reshape(patch_ordering, [-1])
            valid = patch_ordering >= 0

            # Transpose, to get left-to-right order
            patch_ordering_rh = tf.reshape(patch_ordering,
                                           [tiling[0], tiling[1], image_token_length_h, image_token_length_w])
            patch_ordering_rh = tf.transpose(patch_ordering_rh, [0, 2, 1, 3])
            patch_ordering_rh = tf.reshape(patch_ordering_rh, [-1])

            # The tranpose will screw up which patches are masked, project the
            # new order into sparse structure of `patch_ordering` to fix this
            patch_ordering = tf.tensor_scatter_nd_update(
                patch_ordering,
                tf.where(valid),
                tf.boolean_mask(patch_ordering_rh, patch_ordering_rh >= 0),
                name="patch_order_transpose_Scatter"
            )

            def get_num_patches(num_tiles: int, pooling_size: int) -> int:
                if num_tiles > 1:
                    left_crop_window_patches = (crop_window_patches + left_margin + pooling_size - 1) // pooling_size * pooling_size
                    middle_crop_window_patches = (crop_window_patches + pooling_size - 1) // pooling_size * pooling_size
                    right_crop_window_patches = (crop_window_patches + right_margin + pooling_size - 1) // pooling_size * pooling_size
                    return left_crop_window_patches + (num_tiles - 2) * middle_crop_window_patches + right_crop_window_patches
                else:
                    single_crop_window_patches = (crop_patches + pooling_size - 1) // pooling_size * pooling_size
                    return single_crop_window_patches

            special_token_ids = self.special_token_ids
            h = get_num_patches(tiling[0], self.image_pooling_h)
            w = get_num_patches(tiling[1], self.image_pooling_w)
            per_row = tf.fill((w // self.image_pooling_w,),
                              special_token_ids[tokenizer.DEFAULT_IMAGE_PATCH_TOKEN],)
            if self.use_col_tokens:
                per_row = tf.concat([per_row, [special_token_ids[tokenizer.DEFAULT_IM_COL_TOKEN]]], 0)

            joint = tf.tile(per_row, [h // self.image_pooling_h])
            joint = [
                [special_token_ids[tokenizer.DEFAULT_IM_START_TOKEN]],
                joint,
                [special_token_ids[tokenizer.DEFAULT_IM_END_TOKEN]]
            ]

            if "resize" in mode:
                resized, resized_mask = _resize(image, base_image_input_size)
                resized, resized_mask = _img_to_patches(resized, resized_mask)
                if 'c2' in mode:
                    patches = tf.concat([resized, patches], 0)
                    image_mask = tf.concat([resized_mask, img_mask], 0)
                else:
                    patches = tf.concat([patches, resized], 0)
                    image_mask = tf.concat([img_mask, resized_mask], 0)
                if self.image_padding_mask == 2:
                    img_mask = image_mask

                if patch_ordering is not None:
                    if 'c2' in mode:
                        patch_ordering = tf.where(
                            patch_ordering >= 0,
                            patch_ordering + tokens_per_image,
                            -1
                        )
                        patch_ordering = tf.concat([tf.range(0, tokens_per_image), patch_ordering], 0)
                    else:
                        raise ValueError()
                per_row = tf.fill((image_token_length_w,), special_token_ids[tokenizer.DEFAULT_IMAGE_PATCH_TOKEN],)
                if self.use_col_tokens:
                    per_row = tf.concat([per_row, [special_token_ids[tokenizer.DEFAULT_IM_COL_TOKEN]]], 0)
                extra_tokens = tf.tile(per_row, [image_token_length_h])
                joint = [
                            [special_token_ids[tokenizer.DEFAULT_IM_START_TOKEN]],
                            extra_tokens,
                            [special_token_ids[tokenizer.DEFAULT_IM_END_TOKEN]],
                        ] + joint

            joint = tf.concat(joint, 0)
            return patches, joint, patch_ordering, img_mask

        elif mode in ["patchify", "patchify-and-resize", "patchify-v2", "patchify-v2-and-resize", "patchify-v2-and-resize-c2"]:
            original_image_w = tf.shape(image, out_type=tf.int32)[0]
            original_image_h = tf.shape(image, out_type=tf.int32)[1]
            assert base_image_input_size[0] == base_image_input_size[1]
            base_patch_size = base_image_input_size[0]
            tiling = select_tiling(original_image_w, original_image_h, base_patch_size, self.max_crops)

            patches, img_mask = _resize(
                image, [tiling[0]*base_patch_size, tiling[1]*base_patch_size])
            patches, img_mask = _img_to_patches(patches, img_mask, tiling[0], tiling[1])
            if 'v2' in mode:
                # Order patches left-to-right not crop-by-crop
                patch_ordering = tf.reshape(
                    tf.range(tokens_per_image*tiling[0]*tiling[1]),
                    [tiling[0], tiling[1], image_token_length_w, image_token_length_h])
                patch_ordering = tf.transpose(patch_ordering, [0, 2, 1, 3])
                patch_ordering = tf.reshape(patch_ordering, (-1, tokens_per_image))
            else:
                patch_ordering = None

            # given image size, determine the number of patch size.
            image_layout_impatch_w = tiling[0]
            image_layout_impatch_h = tiling[1]

            if "resize" in mode:
                extra_image = True
                resized, resized_mask = _resize(image, base_image_input_size)
                resized, resized_mask = _img_to_patches(resized, resized_mask)
                if 'c2' in mode:
                    patches = tf.concat([resized, patches], 0)
                    image_mask = tf.concat([resized_mask, img_mask], 0)
                else:
                    patches = tf.concat([patches, resized], 0)
                    image_mask = tf.concat([img_mask, resized_mask], 0)

                if patch_ordering is not None:
                    if 'c2' in mode:
                        patch_ordering = tf.concat(
                            [tf.range(0, tokens_per_image)[None, :], patch_ordering+tokens_per_image], 0)
                    else:
                        n = tf.shape(patch_ordering)[0]
                        patch_ordering = tf.concat(patch_ordering, [tf.range(n, n+tokens_per_image)[None, :]], 0)
        else:
            raise NotImplementedError(mode)

        special_token_ids = self.special_token_ids

        per_row = tf.fill((image_token_length_w*image_layout_impatch_w,),
                          special_token_ids[tokenizer.DEFAULT_IMAGE_PATCH_TOKEN],)
        if self.use_col_tokens:
            per_row = tf.concat([per_row, [special_token_ids[tokenizer.DEFAULT_IM_COL_TOKEN]]], 0)

        joint = tf.tile(per_row, [image_token_length_h * image_layout_impatch_h])
        joint = [
            [special_token_ids[tokenizer.DEFAULT_IM_START_TOKEN]],
            joint,
            [special_token_ids[tokenizer.DEFAULT_IM_END_TOKEN]]
        ]
        if extra_image:
            assert self.image_padding_mask != 2
            per_row = tf.fill((image_token_length_w,), special_token_ids[tokenizer.DEFAULT_IMAGE_PATCH_TOKEN],)
            if self.use_col_tokens:
                per_row = tf.concat([per_row, [special_token_ids[tokenizer.DEFAULT_IM_COL_TOKEN]]], 0)
            extra_tokens = tf.tile(per_row, [image_token_length_h])
            if 'c2' in mode:
                joint = [
                            [special_token_ids[tokenizer.DEFAULT_IM_START_TOKEN]],
                            extra_tokens,
                            [special_token_ids[tokenizer.DEFAULT_IM_END_TOKEN]],
                        ] + joint
            else:
                joint += [
                    [special_token_ids[tokenizer.DEFAULT_IM_START_TOKEN]],
                    extra_tokens,
                    [special_token_ids[tokenizer.DEFAULT_IM_END_TOKEN]]
                ]
        if self.pad_to is not None:
            n = [tf.shape(x)[0] for x in joint]
            assert len(joint[-1]) == 1
            to_pad = self.pad_to - tf.reduce_sum(tf.stack(n))
            joint = tf.concat(joint[:-1] + [
                tf.zeros(to_pad, dtype=tf.int32) - 1,
                joint[-1]
            ], axis=0)
        else:
            joint = tf.concat(joint, 0)
        return patches, tf.concat(joint, 0), patch_ordering, img_mask

    def build_image_input_idx(self, input_tokens, patch_order, no_image=None):
        """Builds the index used to insert patch features into `input_tokens`"""
        tokens_per_image = self.image_token_length_w * self.image_token_length_h
        if no_image is not None and no_image:
            return tf.zeros((0, tokens_per_image), tf.int32)

        image_input_idx = input_tokens == self.special_token_ids[tokenizer.DEFAULT_IMAGE_PATCH_TOKEN]
        image_input_idx = tf.experimental.numpy.nonzero(image_input_idx)[0]
        image_input_idx = tf.cast(image_input_idx, tf.int32)

        if patch_order is not None:
            n_tokens = tf.shape(image_input_idx)[0]
            # Item N should have the value of image_input_index[where(patch_order == n)] if >= 0 else -1
            patch_order = tf.reshape(patch_order, [-1])
            n_patches = tf.shape(patch_order)[0]
            if n_tokens != n_patches:
                # Most complex case where some patches are dropped
                # First invert the valid tokens
                valid = patch_order >= 0
                sorted_patch_ixs = tf.scatter_nd(
                    tf.boolean_mask(patch_order, valid)[:, None],
                    tf.range(tf.reduce_sum(tf.cast(valid, tf.int32)), dtype=tf.int32),
                    [n_tokens],
                    name="valid_order_scatter"
                )

                # Project the inverted mapping into same sparse structure
                tmp = tf.fill(tf.shape(patch_order), -1)
                sorted_patch_ixs_ex = tf.tensor_scatter_nd_update(
                    tmp,
                    tf.where(valid),
                    sorted_patch_ixs,
                    name="order_with_padding_scatter"
                )

                # Do the gather and then re-masked outputs that were masked in `sorted_patch_ixs`
                valid = tf.cast(sorted_patch_ixs_ex >= 0, tf.int32)
                image_input_idx = tf.gather(image_input_idx, sorted_patch_ixs_ex*valid)
                image_input_idx = image_input_idx*valid - 100*(1 - valid)
            else:
                sorted_patch_ixs = tf.scatter_nd(patch_order[:, None], tf.range(n_patches), [n_patches])
                image_input_idx = tf.gather(tf.reshape(image_input_idx, [-1]), sorted_patch_ixs)
            image_input_idx = tf.reshape(image_input_idx, [-1, tokens_per_image])
        return image_input_idx

    def build_multimodel_features(self, tokens, mask, subsegments, images, is_training):
        """Builds input features by pre-processing `images` and modifying `tokens`
        to include image col/pad/start/end tokens instead image placeholder tokens
        """
        image_token_id = self.special_token_ids[tokenizer.IMAGE_PROMPT]
        image_idx = tf.experimental.numpy.nonzero(tokens == image_token_id)[0]
        if images is None or tf.shape(images)[0] == 0:
            tf.debugging.assert_equal(image_idx, tf.cast(0, tf.int64),
                                      "Image placeholders in input, but no images given!")
            tokens_per_image = self.image_token_length_w * self.image_token_length_h
            n_pixels = self.image_patch_size ** 2 * 3
            image_num_patch = np.prod(self.image_num_patch)
            crops = tf.zeros((0, image_num_patch, n_pixels), dtype=tf.float32)
            image_idx = tf.zeros((0, tokens_per_image), tf.int32)
            out = dict(
                target_tokens=tokens,
                images=crops,
                image_input_idx=image_idx,
                loss_masks=mask
            )
            if self.image_padding_mask:
                out["image_masks"] = tf.zeros((0, image_num_patch), dtype=tf.float32)
            if subsegments is not None:
                out["subsegment_ids"] = subsegments
            return out
        elif tf.shape(image_idx)[0] == 0 and tf.shape(images)[0] > 0:
            # As a special case, no image prompt means the images are all at the start
            image_idx = tf.zeros([tf.shape(images)[0]], tf.int64) - 1
        else:
            tf.debugging.assert_equal(
                tf.shape(images)[0], tf.shape(image_idx)[0],
                message="Different number of images and image placeholders")

        # Each image will produce a variable number of crops/tokens, so we aggregate things
        # the results tensor arrays and the concat them
        tokens_per_image = self.image_token_length_w * self.image_token_length_h
        n_pixels = self.image_patch_size*self.image_patch_size*3
        n_patches = self.image_num_patch[0]*self.image_num_patch[1]

        n = tf.shape(images)[0]
        all_crops = tf.TensorArray(dtype=tf.float32, size=n, infer_shape=False,
                                   element_shape=[None, n_patches, n_pixels])
        all_image_idx = tf.TensorArray(dtype=tf.int32, size=n, infer_shape=False,
                                       element_shape=[None, tokens_per_image])
        out_tokens = tf.TensorArray(dtype=tf.int32, size=n, infer_shape=False,
                                    element_shape=[None])
        out_masks = tf.TensorArray(dtype=tf.float32, size=n, infer_shape=False,
                                   element_shape=[None])
        if self.image_padding_mask:
            all_crop_masks = tf.TensorArray(dtype=tf.float32, size=n, infer_shape=False,
                                            element_shape=[None, None])
        else:
            # Dummy array to keep tensorflow's control analysis happy
            all_crop_masks = tf.TensorArray(dtype=tf.float32, size=0, infer_shape=False,
                                            element_shape=[None, None])
        if subsegments is not None:
            out_subsegments = tf.TensorArray(dtype=tf.int32, size=n, element_shape=[None])
        else:
            out_subsegments = tf.TensorArray(dtype=tf.int32, size=0, element_shape=[None])

        image_idx = tf.cast(image_idx, tf.int32)
        for ix in range(tf.shape(image_idx)[0]):
            token_ix = image_idx[ix]
            crops, image_tokens, patch_ordering, img_mask = self.image_to_patches_and_tokens(images[ix], is_training)
            patch_idx = self.build_image_input_idx(image_tokens, patch_ordering)

            if token_ix == -1:  # -1 is an image inserted at the very start
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix-1] + 1
                end = token_ix + 1

            all_image_idx = all_image_idx.write(ix, patch_idx + token_ix)
            all_crops = all_crops.write(ix, crops)
            image_token_mask = tf.zeros_like(image_tokens, dtype=tf.float32)

            if ix == (tf.shape(images)[0] - 1):
                tokens_part = tf.concat([tokens[start:token_ix], image_tokens, tokens[end:]], 0)
                mask_part = tf.concat([mask[start:token_ix], image_token_mask, mask[end:]], 0)
            else:
                tokens_part = tf.concat([tokens[start:token_ix], image_tokens], 0)
                mask_part = tf.concat([mask[start:token_ix], image_token_mask], 0)

            out_tokens = out_tokens.write(ix, tokens_part)
            out_masks = out_masks.write(ix, mask_part)
            if self.image_padding_mask:
                all_crop_masks = all_crop_masks.write(ix, img_mask)
            if subsegments is not None:
                parts = tf.fill([tf.shape(image_tokens)[0]], subsegments[token_ix])
                if ix == (tf.shape(images)[0] - 1):
                    seg = tf.concat([subsegments[start:token_ix], parts, subsegments[end:]], 0)
                else:
                    seg = tf.concat([subsegments[start:token_ix], parts], 0)
                out_subsegments = out_subsegments.write(ix, seg)

        out = dict(
            target_tokens=out_tokens.concat(),
            images=all_crops.concat(),
            image_input_idx=all_image_idx.concat(),
            loss_masks=out_masks.concat()
        )
        if self.image_padding_mask:
            out["image_masks"] = all_crop_masks.concat()
        if subsegments is not None:
            out["subsegment_ids"] = out_subsegments.concat()
        return out

    def _format_message(self, args):
        message, ix = args
        return self.format_message(message, ix)

    def format_message(self, message, ix):
        """Applies system formatting to ith message from a sequence of messages"""
        # If the image placeholder text is not preceded by space it will not get tokenized
        # correctly by some tokenizers, so double check it here
        assert tokenizer.IMAGE_PROMPT == "<|image|>"
        tf.debugging.assert_equal(
            tf.strings.regex_full_match(message, r".*[^ ]<\|image\|>.*"),
            False,
            message="Image token must always be preceded by a space"
        )
        is_user = ix % 2 == 0
        if self.message_format == "none" or self.message_format is None:
            pass
        elif self.message_format == "role":
            if is_user:
                # We put the "System:" prefix here since it doesn't need a loss
                message = tf.strings.join(["User: ", message, " Assistant:"])
        elif self.message_format == "cleanup":
            if is_user:
                # We put the "System:" prefix here since it doesn't need a loss
                message = tf.strings.join(
                    [
                        "[[User]]: Correct the spelling and punctuation mistakes on the following transcript based on what appears in the image.\n\n{before} ",
                        message,
                        "\n[[Assistant]]: {after}"
                    ]
                )
        elif self.message_format == "mistral":
            if is_user:
                message = tf.strings.join(["[INST] ", message, " [/INST]"])
        else:
            raise NotImplementedError(self.message_format)

        # For now assume a space will be used to separate the messages
        if not self.tokenizer.adds_space:
            if ix != 0 or self.always_start_with_space:
                message = tf.strings.join([" ", message])
        # Else space added automatically by the tokenizer

        return message

    def get_multi_message_token_input(self, conversations, text_weights=None):
        """Build inputs for a ragged tensor of conversations, where each row of the tensor,
        is a different conversation"""
        tf.debugging.assert_equal(tf.reduce_any(tf.strings.regex_full_match(
            conversations.values, re.escape(tokenizer.IMAGE_PROMPT))), False, "Segmented prompts must start with the image")

        n_conversation = tf.shape(conversations)[0]
        ar = tf.TensorArray(dtype=tf.int32, infer_shape=False, element_shape=[None],
                            size=n_conversation)
        n_messages_per_conversation = conversations.row_lengths()
        for ix in range(n_conversation):
            ar = ar.write(ix, tf.range(n_messages_per_conversation[ix], dtype=tf.int32))
        message_ix = ar.concat()
        messages = tf.map_fn(
            self._format_message, elems=(conversations.values, message_ix), fn_output_signature=tf.string)
        messages = self.tokenizer.encode_tf(messages)

        # Append EOS
        is_response = message_ix % 2 == 1
        is_response_int = tf.cast(is_response, tf.int32)
        eos = tf.RaggedTensor.from_row_lengths(
            tf.fill([tf.reduce_sum(is_response_int)], self.tokenizer.eos_token_id),
            tf.cast(is_response_int, messages.row_splits.dtype)
        )
        messages = tf.concat([messages, eos], axis=1)

        # Build mask over system responses
        mask = tf.ones_like(messages) * tf.cast(tf.expand_dims(is_response, axis=1), tf.int32)
        decoder_loss_weights = tf.cast(mask.values, tf.float32)

        # Build subsegment ids for each conversation
        tokens_per_message = tf.RaggedTensor.from_row_splits(
            row_splits=conversations.row_splits,
            values=messages.row_lengths()
        )
        token_per_conversation = tf.reduce_sum(tokens_per_message, axis=1)
        subsegment_ids = tf.repeat(tf.range(n_conversation, dtype=tf.int32)+1, token_per_conversation)

        image_ix = self.special_token_ids[tokenizer.IMAGE_PROMPT]
        messages = tf.concat([[image_ix], messages.values], axis=0)
        decoder_loss_weights = tf.concat([[0], decoder_loss_weights], axis=0)
        subsegment_ids = tf.concat([[10000], subsegment_ids], axis=0)
        return messages, decoder_loss_weights, subsegment_ids

    def get_multi_response_token_input(self, user_prompt, text, text_weights=None):
        """Build tokens for a multi-response-per-image example"""
        # FIXME this could be relaxed to just having the same prefix
        tf.debugging.assert_equal(tf.reduce_any(tf.strings.regex_full_match(
            user_prompt, re.escape(tokenizer.IMAGE_PROMPT))), False, "Segmented prompts must start with the image")
        user_prompt = self.format_message(user_prompt, 0)
        vocab = self.tokenizer
        prompts = vocab.encode_tf(user_prompt)
        response = self.format_message(text, 1)
        responses = vocab.encode_tf(response)
        responses = _append_to_innermost_axis(responses, vocab.eos_token_id)
        response_mask = tf.ones_like(responses, dtype=tf.float32)
        if text_weights is not None:
            response_mask *= text_weights
        image_tokens = tf.constant([self.special_token_ids[tokenizer.IMAGE_PROMPT]])

        if len(responses.shape) == 3:
            # Tricky case where we have multiple questions, each of which has multiple answers
            assert len(prompts.shape) == 2

            # Also shift the last tokens to the response segment since that tokens will
            # have multiple possible target tokens to predict
            last_prompt_tokens = prompts[:, -1:]
            last_prompt_tokens = tf.repeat(last_prompt_tokens, responses.row_lengths())
            last_prompt_tokens = tf.RaggedTensor.from_row_splits(
                values=tf.RaggedTensor.from_row_lengths(
                    values=last_prompt_tokens,
                    row_lengths=tf.ones_like(last_prompt_tokens, dtype=responses.row_splits.dtype)
                ),
                row_splits=responses.row_splits
            )
            responses = tf.concat([last_prompt_tokens,  responses], 2)
            prompts = prompts[:, :-1]

            shared_prefix = image_tokens
            segmented_suffix = tf.concat([tf.expand_dims(prompts, 1), responses], 1)
            targets = tf.concat([shared_prefix, segmented_suffix.values.values], 0)

            segmented_mask = tf.concat([
                tf.zeros_like(tf.expand_dims(prompts, 1), dtype=tf.float32),
                tf.concat([
                    tf.zeros_like(last_prompt_tokens, dtype=tf.float32),
                    response_mask
                ], 2)
            ], 1).values.values
            decoder_loss_weights = tf.concat(
                [tf.zeros_like(shared_prefix, dtype=tf.float32), segmented_mask], 0)

            text_segment_ids = get_3d_subsegments(segmented_suffix)
            subsegment_ids = tf.concat([
                tf.zeros_like(shared_prefix) + tf.reduce_max(text_segment_ids)+1,
                text_segment_ids], 0)
            subsegment_ids = tf.cast(subsegment_ids, tf.int32)
        else:
            if len(prompts.shape) == 1:
                # One prompt for all responses, we use the last token of the prompt as the
                # first token of each response segment since there will be multiple targets
                # for that token, the remaining targets are part of the prefix
                shared_prefix = tf.concat([image_tokens, prompts[:-1]], 0)
                prompts = prompts[-1:]
                prompts = tf.tile(tf.expand_dims(prompts, axis=0), [tf.shape(text)[0], 1])
            else:
                shared_prefix = image_tokens

            # Separate prompt for each response
            segmented_suffix = tf.concat([prompts, responses], 1)
            segmented_mask = tf.concat([tf.zeros_like(prompts, dtype=tf.float32), response_mask], 1).values

            targets = tf.concat([shared_prefix, segmented_suffix.values], 0)
            decoder_loss_weights = tf.concat(
                [tf.zeros_like(shared_prefix, dtype=tf.float32), segmented_mask], 0)
            subsegments = tf.ragged.row_splits_to_segment_ids(segmented_suffix.row_splits) + 1
            subsegment_ids = tf.concat([tf.zeros_like(shared_prefix)+10000,
                                        tf.cast(subsegments, tf.int32)], 0)
        return targets, decoder_loss_weights, subsegment_ids

    def get_tokens_input(self, messages, for_inference=False, text_weights=None):
        """Gets the token input for an example, using image placeholder tokens to
        indicate where images features should be inserted

        inputs
        messages: List or tensor users/system text messages, can have image placeholder tokens
        for_inference: bool, if true truncate the messages if it is a system message
        text_weights: Weights per a system message

        returns
        tokens: [n_tokens] tf.int32 token inputs with image placeholder tokens
        loss_mask: [n_tokens] tf.float32 token weights for loss
        subsegment: [n_tokens] tf.int32 or None, subsegment ids used to build more complex
                               attention masks if needed
        """
        if isinstance(messages, tf.RaggedTensor):
            assert not for_inference, "Cannot have multiple target messages for inference"
            return self.get_multi_message_token_input(messages, text_weights)
        elif len(tf.shape(messages[-1])) > 0:
            assert not for_inference, "Cannot have multiple target messages for inference"
            assert len(messages) == 2
            prompt = messages[0]
            response = messages[1]
            return self.get_multi_response_token_input(prompt, response, text_weights)
        else:
            messages = tf.convert_to_tensor(messages)
            if for_inference:
                if tf.shape(messages) % 2 == 0:
                    # Remove the last message since the model should predict it
                    messages = messages[:-1]

        # Apply system formatting
        ix = tf.range(tf.shape(messages)[0])
        is_response = ix % 2 == 1
        messages = tf.map_fn(
            self._format_message, elems=(messages, ix), fn_output_signature=tf.string)

        # Tokenize
        messages = self.tokenizer.encode_tf(messages)

        # Add EOS to system messages
        is_response_int = tf.cast(is_response, tf.int32)
        eos = tf.RaggedTensor.from_row_lengths(
            tf.fill([tf.reduce_sum(is_response_int)], self.tokenizer.eos_token_id),
            tf.cast(is_response_int, messages.row_splits.dtype)
        )
        messages = tf.concat([messages, eos], axis=1)
        targets = messages.values

        # Build mask over system responses
        mask = tf.ones_like(messages) * tf.cast(tf.expand_dims(is_response, axis=1), tf.int32)
        decoder_loss_weights = tf.cast(mask.values, tf.float32)
        if text_weights is not None:
            decoder_loss_weights = decoder_loss_weights * text_weights
        return messages.values, decoder_loss_weights, None

    def preprocess(self, image, input_text, is_training=False,
                   seq_len=None, pad_images=1, style=None, for_inference=True):
        """Get input tensors for the given image/text data

        image: [h, w, 3] numpy uint8 array of image pixels
        input_text: string input text, a list of text for a multi-turn conversation or dictionary
                    of inputs to use to build the prompt from a template
        is_training: allow training-time preprocessing (e.g., image augmentation)
        seq_len: pad input tokens to `seq_len`
        pad_images: pad input images to `self.get_max_total_crops()`
        style: Style to use for prompt templating
        """
        if image is not None and len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, axis=0)

        messages = self.get_messages(input_text, style, is_training, for_inference=for_inference, user_prompt_seed=None, system_prompt_seed=None)
        targets, loss_masks, subsegments = self.get_tokens_input(messages, for_inference=for_inference)
        batch = self.build_multimodel_features(
            targets, loss_masks, subsegments, image, is_training)

        # Optionally padding to get constant sized arrays
        if pad_images:
            max_crops = self.get_max_total_crops() * pad_images
            image = batch["images"]
            n = max_crops - tf.shape(batch["images"])[0]
            batch["images"] = tf.pad(image, [[0, n], [0, 0], [0, 0]], constant_values=-1)
            if self.image_padding_mask:
                m = max_crops - tf.shape(batch["image_masks"])[0]
                batch["image_masks"] = tf.pad(batch["image_masks"], [[0, m], [0, 0]], constant_values=-1)
            batch["image_input_idx"] = tf.pad(batch["image_input_idx"], [[0, n], [0, 0]], constant_values=-1)

        if seq_len is not None:
            targets = batch["target_tokens"]
            if seq_len < len(targets):
                raise ValueError("Sequence length too short")
            n = seq_len - len(targets)
            batch["target_tokens"] = tf.pad(targets, [[0, n]], constant_values=-1)
            batch["loss_masks"] = tf.pad(batch["loss_masks"], [[0, n]], constant_values=-1)

        batch = self.get_post_mixing_preprocessor(pack=False)._convert_example(batch)
        return batch

    def get_user_prompt(self, style, example, is_training=True, for_inference=False, seed=None):
        """Build a list of strings of what a user might type in to the model for the given example,
        and its responses, by applying a prompt template to the fields in `example`

        Can return multiple strings for one message for multi-response examples
        """
        if "style" in example:
            style = example["style"]

        if "prompt" in example:
            # Examples have a complete user prompt pre-specified, usually for eval sets
            prompt = example["prompt"]

        elif self.prompt_templates == "none":
            # Bare-bone prompt with not templating of instructions
            if "prompt" in example:
                prompt = example["prompt"]
            elif "refexp" in example:
                prompt = example["refexp"]
            elif "question" in example and "options" in example:
                prompt = tf.strings.join([example["question"], "\n", example["options"], "\n"])
            elif "question" in example:
                prompt = example["question"]
            else:
                prompt = ""

        elif self.prompt_templates == "uber_model":
            if not isinstance(style, str):
                tf.debugging.assert_equal(tf.logical_or(
                    style == "ai2_diagram_no_letter",
                    style == "ai2_diagram",
                ), True)
                prompt = tf.strings.join([example["question"], "\n", example["options"], "\n"])
            else:
                # We template long captions and pointing since they are "demo" tasks, and use
                # plain text for everything else
                if style == "long_caption":
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["long_caption"], example, seed)
                elif style == "pointing":
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["pointing"], example, seed)
                elif style == "point_count":
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["point_count"], example, seed)
                elif "prompt" in example:
                    prompt = example["prompt"]
                elif "refexp" in example:
                    prompt = example["refexp"]
                elif "question" in example and "options" in example:
                    prompt = tf.strings.join([example["question"], "\n", example["options"], "\n"])
                elif "question" in example:
                    prompt = example["question"]
                else:
                    prompt = ""

        elif self.prompt_templates == "uber_model_pointing":
            if style == "long_caption":
                long_captions = GENERAL_PROMPTS_V1["long_caption_no_pointing"]
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["long_caption"], example, seed)
            elif style == "pointing":
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["pointing"], example, seed)
            elif style in [
                "scifi_charts_explanation",
                "scifi_table_explanation",
                "scifi_document_explanation",
                "scifi_diagram_explanation",
                "user_qa",
                "long_caption",
            ]:
                raise NotImplementedError()
                if style == "long_caption":
                    prompts = GENERAL_PROMPTS_V1["long_caption"]
                elif "prompt" in example:
                    prompts = tf.expand_dims(example["prompt"], axis=0)
                else:
                    prompts = tf.expand_dims(example["question"], axis=0)
                suffixes = []
                for suffix in GENERAL_PROMPTS_V1["no_pointing_suffix"]:
                    if not suffix[0].isspace():
                        suffix = " " + suffix
                    suffixes.append(suffix)
                no_point_prompts = tf.reshape(tf.strings.join([
                    tf.tile(tf.expand_dims(suffixes, 1), [1, tf.shape(prompts)[1]]),
                    tf.tile(prompts, [len(suffixes), 1]),
                ]), [-1])
                # prefixes = []
                # for prefix in GENERAL_PROMPTS_V1["no_pointing_prefix"]:
                #     if not prefix[0].isspace():
                #         prefix = prefix + " "
                #     prefixes.append(prompts + prefix)
                prompt = apply_keyword_prompt(no_point_prompts, example, seed, keywords=[])
            elif "prompt" in example:
                prompt = example["prompt"]
            elif "refexp" in example:
                prompt = example["refexp"]
            elif "question" in example and "options" in example:
                prompt = tf.strings.join([example["question"], "\n", example["options"], "\n"])
            elif "question" in example:
                prompt = example["question"]
            else:
                prompt = ""

        elif self.prompt_templates == "general_instructions_v1":
            if isinstance(style, str):
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[STYLE_TO_GENERAL_PROMPT[style]], example, seed)
            elif isinstance(style, list):
                # This ia bit of hack to allow apply prompts to joint caption/transcript data
                # FIXME ideally we can apply the templating to multiple styles more generally
                def _apply(_style, ix):
                    tmp = dict(example)
                    # prevent apply_keyword_prompt for generating multiple templates
                    tmp["text"] = tmp["text"][0]
                    if _style == "long_caption":
                        return apply_keyword_prompt(GENERAL_PROMPTS_V1["long_caption"], tmp, seed)
                    elif _style == "transcript":
                        return apply_keyword_prompt(GENERAL_PROMPTS_V1["transcript"], tmp, seed)
                    else:
                        raise NotImplementedError(_style)
                prompt = [_apply(x, ix) for ix, x in enumerate(style)]
            else:
                raise NotImplementedError()

        elif self.prompt_templates == "zero_shot_v1":
            assert style is not None
            if not isinstance(style, str):
                # FIXME can we handle tensor style's in a better way?
                if style == "ai2_diagram":
                    prompt = "Question: {question}\nAnswer with correct answer option letter only\nOptions: {options}\nAnswer:"
                    prompt = apply_keyword_prompt([prompt], example, seed)
                elif style == "ai2_diagram_no_letter":
                    prompt = "Question: {question}\nAnswer with correct answer option only\nOptions: {options}\nAnswer:"
                    prompt = apply_keyword_prompt([prompt], example, seed)
                else:
                    prompt = ""
                tf.debugging.assert_equal(prompt != "", True)
            else:
                general_style = STYLE_TO_GENERAL_PROMPT[style]
                if general_style == "short_answer":
                    prompt = apply_keyword_prompt(["Question: {question} Answer with as few words as possible. Answer:"], example, seed)
                elif general_style == "multiple_choice":
                    prompt = apply_keyword_prompt(["Question: {question}\nAnswer with correct answer option letter only\nOptions: {options}\nAnswer:"], example, seed)
                elif general_style == "count_bench":
                    prompt = apply_keyword_prompt(["Question: How many {object} are there?\nRespond with only a number.\nAnswer:"], example, seed)
                else:
                    raise NotImplementedError(general_style)

        elif self.prompt_templates == "zero_shot_v2":
            assert style is not None

            if self.prompt_override:
                prompt = apply_keyword_prompt([self.prompt_override], example, seed)
            elif not isinstance(style, str):
                if style == "ai2_diagram":
                    prompt = "{question} Answer with correct answer option letter only. Options: {options}"
                    prompt = apply_keyword_prompt([prompt], example, seed)
                elif style == "ai2_diagram_no_letter":
                    prompt = "{question} Answer with correct answer option only. Options: {options}"
                    prompt = apply_keyword_prompt([prompt], example, seed)
                else:
                    prompt = ""
                tf.debugging.assert_equal(prompt != "", True)
            else:
                if style in ["vqa2", "gqa", "tally_qa", "okvqa", "a_okvqa_da"]:
                    prompt = "Answer with a single word. {question}"
                elif style in ["text_vqa", "doc_qa", "info_qa", "chart_qa", "st_qa", "ocr_vqa", "dv_qa", "tabwmp_da", "figure_qa", "figure_qa_zero_shot", "plot_qa"]:
                    prompt = "{question}\nRespond as concisely as possible, do not output anything other than the answer."
                elif STYLE_TO_GENERAL_PROMPT[style] == "multiple_choice":
                    prompt = "{question} Answer with correct answer option letter only. Options: {options}"
                elif STYLE_TO_GENERAL_PROMPT[style] == "short_answer":
                    prompt = "{question} Answer with as few words as possible."
                elif style == "vtabfact":
                    prompt = "{question}"
                elif style == "count_bench":
                    prompt = "How many {object} are there?\nRespond with only a number."
                else:
                    raise NotImplementedError(style)
                prompt = apply_keyword_prompt([prompt], example, seed)
        else:
            raise NotImplementedError(self.prompt_templates)

        if for_inference:
            return [prompt]
        else:
            return [prompt, example["text"]]

    def get_system_prompt(self, style, example, for_inference,
                          messages, seed=None):
        if isinstance(style, str) and style == "count_bench":
            style = "ok_vqa"

        if self.system_prompt == "style":
            if isinstance(style, str):
                prefix = style + ":"
            else:
                prefix = tf.strings.join([style, ":"])

        elif self.system_prompt == "demo_or_style":
            if isinstance(style, str):
                if style == "android_control" or style == "demo":
                    # android is a special case since I hacked in prefix in the preprocessor
                    prefix = ""
                elif style in ["scifi_demo", "synthetic_qa"] or style in DEMO_STYLES:
                    if getattr(self, "debug", False):
                        p_no_prompt = 0.0
                    elif style == "scifi_demo":
                        p_no_prompt = 0.2
                    elif style == "synthetic_qa":
                        p_no_prompt = 0.25
                    else:
                        p_no_prompt = 0.9
                    if len(tf.shape(messages)) > 1:
                        n_messages = tf.shape(messages)[1]
                        style = tf.tile(tf.expand_dims(style, axis=0), [n_messages])
                        r = tf.random.stateless_uniform([n_messages], seed, 0, 1)
                    else:
                        r = tf.random.stateless_uniform((), seed, 0, 1)
                    prefix = tf.where(r < p_no_prompt, "", tf.strings.join([style + ":"]))
                else:
                    prefix = style + ":"
            else:
                if tf.reduce_any(style == tf.constant(DEMO_STYLES + ["scifi_demo", "android_control", "demo"])):
                    prefix = ""
                else:
                    prefix = tf.strings.join([style, ":"])

        elif self.system_prompt in ["long_caption_length_hint", "style_long_caption_length_hint"]:
            if seed is not None:
                raise NotImplementedError("Determinism")
            std = 25
            use_hint = tf.logical_or(
                tf.equal(style, "long_caption"), tf.equal(style, "transcript"))
            if self.system_prompt == "style_long_caption_length_hint":
                default = tf.strings.join([style, ": "])
            else:
                default = ""
            if for_inference:
                assert len(tf.shape(use_hint)) == 0
                if self.default_inference_len and use_hint:
                    prefix = tf.strings.join([style, " ", str(self.default_inference_len), ": "])
                else:
                    prefix = default
            else:
                std = 25
                n = tf.strings.length(messages[-1])
                n += tf.cast(tf.random.normal(n.shape)*std, tf.int32)
                hint = tf.strings.join([style, " ", tf.strings.as_string(n//15), ": "])
                use_hint = tf.logical_and(use_hint, tf.random.uniform(tf.shape(hint)) > 0.1)
                prefix = tf.where(use_hint, hint, default)

        elif for_inference and self.system_prompt in ["style_and_length", "style_and_length_v2"]:
            v2 = self.system_prompt == "style_and_length_v2"
            if example.get("length_cond") is not None:
                # Examples have individual length conditioning
                n = tf.strings.as_string(example["length_cond"])
            else:
                inference_len = self.default_inference_len
                n = None if inference_len is None else str(inference_len)
                logging.warning(f"eval len: {n}")
            if n is not None and tf.strings.length(n) > 0:  # allow empty string to signal unconditioned
                prefix = tf.strings.join([style, " ", n, ":"])
            else:
                prefix = tf.strings.join([style, ":" if v2 else " :"])
        elif self.system_prompt in ["style_and_length", "style_and_length_v2"]:
            v2 = self.system_prompt == "style_and_length_v2"
            std = 25
            logging.info(f"style prompt std={std}, percent=10")
            if seed is not None:
                seeds = tf.random.split(seed)
                p = tf.random.stateless_uniform((), seed=seeds[0])
            else:
                p = tf.random.uniform(())
            if p > 0.10:
                n = tf.strings.length(messages[-1])
                if seed is not None:
                    n += tf.cast(tf.random.stateless_normal(n.shape, seed=seeds[1])*std, tf.int32)
                else:
                    n += tf.cast(tf.random.normal(n.shape)*std, tf.int32)
                n = tf.strings.as_string(n//15)
                prefix = tf.strings.join([style, " ", n, ":"])
            else:
                prefix = tf.strings.join([style, ":" if v2 else " :"])
        else:
            raise NotImplementedError(self.system_prompt)

        return prefix

    def preprend_system_prompt(self, style, example, for_inference, messages, seed=None):
        prefix = self.get_system_prompt(style, example, for_inference, messages, seed=seed)
        separator = tf.where(tf.logical_and(
            tf.strings.length(prefix) > 0, tf.strings.length(messages[0]) > 0), " ", "")
        with_system_prompt = tf.strings.join([prefix, separator, messages[0]])
        if isinstance(messages, list):
            messages = [with_system_prompt] + messages[1:]
        else:
            messages = tf.concat([tf.expand_dims(with_system_prompt, 0), messages[1:]], axis=0)
        return messages

    def get_messages(self, ex, style, is_training, for_inference, user_prompt_seed, system_prompt_seed):
        if isinstance(ex, list):
            messages = ex
        elif isinstance(ex, str):
            messages = [ex]
        elif "messages" in ex:
            messages = ex["messages"]
        else:
            # Apply a prompt template
            messages = self.get_user_prompt(style, ex, is_training, for_inference=for_inference, seed=user_prompt_seed)

        # Maybe add a system prompt. The system prompt gets concatenated with the first user input
        if self.system_prompt and self.system_prompt != "none":
            if isinstance(ex, dict):
                style = ex.get("style", style)

            if isinstance(messages, tf.RaggedTensor):
                n = tf.shape(messages)[0]
                message_arr = tf.TensorArray(dtype=tf.string, size=n, element_shape=(None,))
                seeds = tf.random.split(system_prompt_seed, n)
                for i in range(n):
                    message_arr = message_arr.write(i, self.preprend_system_prompt(style, None, for_inference, messages[i], seed=seeds[i]))
                messages = tf.RaggedTensor.from_row_splits(
                    values=message_arr.concat(), row_splits=messages.row_splits)
            else:
                messages = self.preprend_system_prompt(style, ex, for_inference, messages, seed=system_prompt_seed)

        return messages

    def get_preprocessor(self, is_training, for_inference, style=None, include_metadata=None):
        """Build a preprocessing function that can be applied ot a tf.data.Dataset"""
        vocab = self.tokenizer
        include_response = not for_inference
        if include_metadata is None:
            include_metadata = for_inference

        @seqio.map_over_dataset(num_seeds=2)
        def to_inputs_and_targets(ex, seeds):
            if "unconditioned" in ex:
                raise NotImplementedError()
            if "image" not in ex:
                image = None
            elif ex['image'].dtype == tf.string:
                image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
            else:
                image = ex['image']
            raw_image = image
            if image is not None and len(tf.shape(image)) == 3:
                image = tf.expand_dims(image, axis=0)

            unconditioned = self.unconditioned
            if unconditioned and isinstance(unconditioned, float):
                assert image is not None
                if is_training and tf.random.uniform((), 0, 1, dtype=tf.float32) < unconditioned:
                    image = image[:0]
            elif unconditioned:
                image = None

            messages = self.get_messages(ex, style, is_training, for_inference, seeds[0], seeds[1])
            targets, loss_masks, subsegments = self.get_tokens_input(
                messages, for_inference, ex.get("text_weights"))
            # if "scifi" in style and style.endswith("_explanation"):
            #     logging.warning(f"No loss on EOS for {style}")
            #     loss_masks = tf.where(targets == self.tokenizer.eos_token_id, tf.zeros_like(loss_masks), loss_masks)
            out = self.build_multimodel_features(targets, loss_masks, subsegments, image, is_training)

            if include_metadata:
                # FIXME remove these special cases
                if "text" in ex:
                    if len(ex["text"].shape) > 0:
                        # FIXME can this be variable lengths after all?
                        out["metadata/captions"] = tf.strings.reduce_join(
                            tf.strings.regex_replace(ex['text'], "\\s+", " "),
                            separator="\n"
                        )
                    else:
                        out["metadata/captions"] = ex["text"]

                if "image_url" in ex:
                    out["metadata/image_url"] = ex["image_url"]
                elif "url" in ex:
                    out["metadata/image_url"] = ex["url"]
                if "image_id" in ex:
                    out["metadata/image_id"] = ex["image_id"]
                for k, v in ex.items():
                    if k.startswith("metadata"):
                        out[k] = v
                if (raw_image is not None and "metadata/image_size" not in out):
                    img_h = tf.shape(raw_image)[0]
                    img_w = tf.shape(raw_image)[1]
                    out["metadata/image_size"] = [img_w, img_h]
                if self.include_raw_image:
                    out["metadata/image"] = ex["image"]
                elif ("metadata/image_url" not in out) and raw_image is not None :
                    if len(ex["image"].shape) < 4:
                        # For visualizations FIXME can we make this variable length
                        out["metadata/image"] = tf.io.encode_jpeg(
                            tf.image.convert_image_dtype(raw_image, tf.uint8))
            return out
        return to_inputs_and_targets

    def get_post_mixing_preprocessor(self, pack=False):
        """Build a feature conversion function that can be applied ot a tf.data.Dataset

        This function applies a second stage of pre-processing, but unlike `self.get_preprocessor`
        this stage can be applied after mixing tf.data.Datasets into a mixture
        """
        return MultiModalLMFeatureConverter(
            loss_token_weighting=self.loss_token_weighting,
            bos_id=self.tokenizer.bos_token_id,
            pack=pack,
            special_tokens=list(self.special_token_ids.values()),
        )


class MultiModalLMFeatureConverter:

    def __init__(
        self, pack: bool = False, loss_token_weighting: str=None, bos_id: int = 1,
        special_tokens=None,
    ):
        self.pack = pack
        self.bos_id = bos_id
        self.special_tokens = tf.constant(special_tokens) if special_tokens else None
        self.loss_token_weighting = loss_token_weighting

    def _convert_example(
        self, features: Mapping[str, tf.Tensor]
    ) -> Mapping[str, tf.Tensor]:
        """Convert an LM example into an example with model features."""
        # targets_segment_id is present only for a packed dataset.
        decoder_input_tokens = make_autoregressive_inputs(
            features["target_tokens"],
            sequence_id=features.get("targets_segment_ids", None),
            bos_id=self.bos_id,
        )

        tf.assert_equal(
            True,
            tf.reduce_all(decoder_input_tokens[-1] != self.special_tokens),
            message="An input ends with an image special token",
        )

        image_input_idx = features["image_input_idx"]
        # plus one sine we have added BOS to the inputs
        image_input_idx = tf.where(image_input_idx < 0,  image_input_idx, image_input_idx + 1)

        d = {
            "target_tokens": features["target_tokens"],
            "input_tokens": decoder_input_tokens,
            "loss_masks": features["loss_masks"],
            "images": features["images"],
            "image_input_idx": image_input_idx
        }
        if "image_masks" in features:
            d["image_masks"] = features["image_masks"]

        has_custom_text_weight = features.get("has_custom_loss_weight", False)

        if "subsegment_ids" in features:
            subsegment_ids = make_autoregressive_inputs(
                features["subsegment_ids"],
                sequence_id=features.get("targets_segment_ids", None),
                bos_id=features["subsegment_ids"][0],
            )

            # Subsegment have a position based on the sum of previous positions they can attend to
            position_ids = tf.zeros_like(subsegment_ids)
            unique_segments = tf.unique(subsegment_ids)[0]
            for i in unique_segments:
                segment_position_ids = tf.cumsum(tf.cast(subsegment_ids >= i, tf.int32)) - 1
                position_ids = tf.where(subsegment_ids == i, segment_position_ids, position_ids)

            # Apply loss weighting, this is done here so it occurs after truncation
            if has_custom_text_weight:
                pass
            elif self.loss_token_weighting in ["subsegments", "root_subsegments"]:
                n_loss_segments = tf.shape(tf.unique(tf.boolean_mask(subsegment_ids, d["loss_masks"] > 0))[0])[0]
                n_loss_segments = tf.maximum(tf.cast(n_loss_segments, tf.float32), 1)
                weight = 1/n_loss_segments if self.loss_token_weighting == "subsegments" else tf.math.rsqrt(n_loss_segments)
                d["loss_masks"] = tf.where(d["loss_masks"] > 0, d["loss_masks"]*weight, d["loss_masks"])
            elif self.loss_token_weighting is not None:
                raise NotImplementedError(self.loss_token_weighting)

            d["subsegment_ids"] = subsegment_ids
            d["position_ids"] = position_ids
        else:
            if self.loss_token_weighting not in [None, "subsegments", "root_subsegments"] and not has_custom_text_weight:
                raise NotImplementedError(self.loss_token_weighting)
        if self.pack:
            d["decoder_segment_ids"] = features["targets_segment_ids"]
            d["decoder_positions"] = features["targets_positions"]

        for k in features:
            if k.startswith("metadata/"):
                d[k] = features[k]
        return d

    def _pack_or_pad(self, ds, task_feature_lengths):
        if self.pack:
            raise NotImplementedError()
        else:
            return trim_and_pad_dataset(ds, task_feature_lengths)

    def __call__(self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
        """Convert the dataset to be fed to a language model."""
        task_feature_lengths = dict(task_feature_lengths)

        if "images" in ds.element_spec and "images" in task_feature_lengths:
            # Images should never be truncated
            ds = assert_not_truncated(ds, ["images", "image_input_idx"], task_feature_lengths["images"])

        if any(x.startswith("metadata/") for x in ds.element_spec) and "target_tokens" in task_feature_lengths:
            # Metadata indicates the dataset is being used for inference, inference datasets
            # should not be truncated
            ds = assert_not_truncated(ds, ["target_tokens"], task_feature_lengths["target_tokens"])

        if "image_masks" in ds.element_spec and "images" in task_feature_lengths:
            task_feature_lengths["image_masks"] = task_feature_lengths["images"]
        if "subsegment_ids" in ds.element_spec and "target_tokens" in task_feature_lengths:
            task_feature_lengths["subsegment_ids"] = task_feature_lengths["target_tokens"]
        if "loss_masks" not in task_feature_lengths and "target_tokens" in task_feature_lengths:
            task_feature_lengths["loss_masks"] = task_feature_lengths["target_tokens"]
        ds = self._pack_or_pad(ds, task_feature_lengths)

        return ds.map(
            self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
