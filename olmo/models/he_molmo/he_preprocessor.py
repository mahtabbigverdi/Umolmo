import dataclasses
from typing import Tuple, Union, List, Optional
import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from transformers.image_utils import ImageInput

from olmo.config import BaseConfig
from olmo.models.molmo.model_preprocessor import select_tiling, batch_pixels_to_patches, pixels_to_patches, \
    MultiModalPreprocessor
from olmo.nn.vision_backbone import VisionBackboneConfig


def build_pos_ids(subsegment_ids):
    position_ids = np.zeros_like(subsegment_ids)
    for subsegment_id in np.unique(subsegment_ids):
        segment_position_ids = np.cumsum(subsegment_ids <= subsegment_id) - 1
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

    max_query_len: Optional[int] = None
    """Max query length"""

    num_high_res_features: Optional[int] = 512
    """How many high-res features to use"""

    def get_max_crops(self) -> int:
        """Max numbers of that can be built for one image"""
        if self.crop_mode == "resize":
            return 1
        elif "resize" in self.crop_mode:
            # low_res crop + the high-res crops
            return 1 + self.max_crops
        else:
            return self.max_crops

    def build(
        self, tokenizer, vision_backbone_config: VisionBackboneConfig):
        h, w = vision_backbone_config.llm_patches_per_crop()
        vit = vision_backbone_config.vit

        return HeMultiModalPreprocessor(
            tokenizer,
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
            image_pooling_w=vision_backbone_config.image_pooling_w,
            image_pooling_h=vision_backbone_config.image_pooling_h,
            image_token_length_w=w,
            image_token_length_h=h,
            image_patch_size=vit.image_patch_size,
            image_padding_mask=vision_backbone_config.image_padding_embed is not None,
            pad_value=vit.pad_value
        )


@dataclasses.dataclass
class HeMultiModalPreprocessor(MultiModalPreprocessor):
    num_high_res_features: int=512
    max_query_len: Optional[int] = None
    multi_res_selection: Optional[int] = None
    multi_res_min: Optional[int] = None
    indicate_k: Optional[str] = None
    use_high_res_col_tokens: bool = True
    vector_query: bool = False
    debug: Optional[str] = None

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        is_training=False,
        rng=None,
        num_high_res_features=None
    ):
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_token_length_w = self.image_token_length_w
        image_token_length_h = self.image_token_length_h
        image_patch_size = self.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode in ["overlap-and-resize-c2",]:
            # Discard this many patches from the (left/top, right/bottom) of crops
            left_margin, right_margin = overlap_margins
            # Required for compatibility with image pooling
            assert left_margin % self.image_pooling_w == 0 and right_margin % self.image_pooling_w == 0
            assert left_margin % self.image_pooling_h == 0 and right_margin % self.image_pooling_h == 0
            total_margin_pixels = base_image_input_d*(right_margin + left_margin)  # pixels removed per dim
            crop_patches = base_image_input_size[0] // base_image_input_d  # patches per crop dim
            crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
            crop_window_size = crop_window_patches * base_image_input_d

            # Decide how to tile the image, to account for the overlap margins we compute the tiling
            # as if we had an image without the margins and were using a crop size without the margins
            tiling = select_tiling(
                original_image_h - total_margin_pixels,
                original_image_w - total_margin_pixels,
                crop_window_size,
                max_crops
            )
            src, img_mask = self.resize_image(
                image,
                [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
                is_training,
                rng
            )
            src = self._normalize(src)

            # Now we have to split the image into crops, while keeping track of how each patch in the
            # each crop should be ordered in the global image, this require a lot of tricky booking
            n_crops = tiling[0] * tiling[1]
            patches_arr = []
            mask_arr = []
            patch_ordering_arr = []

            # We assume hxw pooling, but can allow padding the right/bottom with extra
            # patches if the number of patches per side is not divisible by h/w
            assert (crop_patches + self.image_pooling_h - 1) // self.image_pooling_h == image_token_length_h
            assert (crop_patches + self.image_pooling_w - 1) // self.image_pooling_w == image_token_length_w
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
                    after_padding_width = image_token_length_w - pooled_w - crop_x0
                    after_padding_height = image_token_length_h - pooled_h - crop_y0
                    patch_ordering_arr.append(
                        np.pad(
                            np.reshape(
                                np.arange(on, on+pooled_h*pooled_w, dtype=np.int32),
                                (pooled_h, pooled_w)),
                            [[crop_y0, after_padding_height], [crop_x0, after_padding_width]],
                            constant_values=-1, mode='constant'
                        )
                    )
                    patches_arr.append(src[y0:y0+crop_size, x0:x0+crop_size])
                    mask_arr.append(img_mask[y0:y0+crop_size, x0:x0+crop_size])

                    on += pooled_h*pooled_w
                    on_patch += 1
            patches = np.stack(patches_arr)
            patch_ordering = np.stack(patch_ordering_arr)
            img_mask = np.stack(mask_arr)

            patch_ordering = np.reshape(patch_ordering, [-1])

            # Path order numbers the patches crop-by-crop, here we transpose
            # it to get left-to-right order
            valid = patch_ordering >= 0
            patch_ordering_rh = np.reshape(
                patch_ordering,
                [tiling[0], tiling[1], image_token_length_h, image_token_length_w]
            )
            patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
            patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])

            # The transpose will screw up which patches are masked, project the
            # new order into sparse structure of `patch_ordering` to fix it
            patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

            # Build a map of low-res -> high-res patches
            low_res_scaling = 2
            n_low_res_scores = 144*low_res_scaling*low_res_scaling
            patch_mapping = np.eye(n_low_res_scores, dtype=np.float32)
            patch_mapping = patch_mapping.reshape([n_low_res_scores, 12*low_res_scaling, 12*low_res_scaling])
            if self.resize != "metaclip":
                raise NotImplementedError(f"Resizing was {self.resize} but padding not implemented")
            high_res_pooled_h = src.shape[0]//(2*image_patch_size)
            high_res_pooled_w = src.shape[1]//(2*image_patch_size)
            patch_mapping = torchvision.transforms.Resize(
                [high_res_pooled_h, high_res_pooled_w], InterpolationMode.BILINEAR, antialias=False)(
                torch.from_numpy(patch_mapping)).numpy().transpose(1, 2, 0)

            if self.use_high_res_col_tokens:
                rows = np.arange(high_res_pooled_w, dtype=np.int32)
                # +1 for the col tokens
                cols = np.arange(high_res_pooled_h, dtype=np.int32) * (high_res_pooled_w + 1)
                high_res_pos_ids = rows[None, :] + cols[:, None]
                high_res_pos_ids = high_res_pos_ids[:, :, None]
            else:
                high_res_pos_ids = np.arange(high_res_pooled_h*high_res_pooled_w).reshape([high_res_pooled_h, high_res_pooled_w, 1])

            # Add in position ids and x/y coordinates
            patch_mapping = np.concatenate([
                patch_mapping,
                high_res_pos_ids,
                np.tile((np.arange(high_res_pooled_h)/high_res_pooled_h)[:, None, None], [1, high_res_pooled_w, 1]),
                np.tile((np.arange(high_res_pooled_w)/high_res_pooled_w)[None, :, None], [high_res_pooled_h, 1, 1])
            ], axis=2)

            # Re-order the patch data so it matches the order of the image patches
            patch_data_size = n_low_res_scores + 3
            ix = patch_ordering.ravel()
            ix = ix[ix >= 0]
            patch_data_ = np.full([len(ix), patch_data_size], -1, dtype=np.float32)
            patch_data_[ix] = patch_mapping.reshape(-1, patch_data_size)
            patch_data = np.full([tiling[0]*tiling[1]*12*12, patch_data_size], -1, dtype=np.float32)
            patch_data[patch_ordering.ravel() >= 0] = patch_data_
            patch_data = patch_data.reshape(tiling[0]*tiling[1], 12*12, -1)

            # Switch to [n_crops, n_patches, pixels_per_patch] format
            image_layout_impatch_w, image_layout_impatch_h = tiling[0], tiling[1]

            patches = batch_pixels_to_patches(patches, image_patch_size)
            img_mask = batch_pixels_to_patches(img_mask, image_patch_size)
            img_mask = img_mask.astype(np.float32).mean(axis=-1)

            def get_num_patches(num_tiles: int, pooling_size: int) -> int:
                if num_tiles > 1:
                    left_crop_window_patches = (crop_window_patches + left_margin + pooling_size - 1) // pooling_size * pooling_size
                    middle_crop_window_patches = (crop_window_patches + pooling_size - 1) // pooling_size * pooling_size
                    right_crop_window_patches = (crop_window_patches + right_margin + pooling_size - 1) // pooling_size * pooling_size
                    return left_crop_window_patches + (num_tiles - 2) * middle_crop_window_patches + right_crop_window_patches
                else:
                    single_crop_window_patches = (crop_patches + pooling_size - 1) // pooling_size * pooling_size
                    return single_crop_window_patches

            h = get_num_patches(tiling[0], self.image_pooling_h)
            w = get_num_patches(tiling[1], self.image_pooling_w)
            num_high_res_tokens = (h * w) // self.image_pooling_h // self.image_pooling_w

            # A low res image might have less than `self.num_high_res_features`
            # patches total
            image_k = num_high_res_features if num_high_res_features else self.num_high_res_features
            image_k = min(image_k, num_high_res_tokens)

            if self.use_high_res_col_tokens:
                n_col_tokens = 1+h // self.image_pooling_h
                col_token_positions = np.arange(1, h // self.image_pooling_h + 1) * (w // self.image_pooling_w + 1)
                patches_per_row = (tiling[0]*(12-4)+4) * 28
                joint = [
                    [self.image_start_token_id],
                    np.full(image_k, self.image_patch_token_id, dtype=np.int32),
                    np.full(h // self.image_pooling_h, self.image_col_token_id, dtype=np.int32),
                    [self.image_end_token_id]
                ]
                high_res_pos_ids = [
                    np.zeros([1], dtype=np.int32),  # IMG start
                    np.ones(image_k, dtype=np.int32),  # IMG patches
                    col_token_positions,  # IMG COL
                    [n_col_tokens + num_high_res_tokens + 1]  # IMG End
                ]
            else:
                joint = [
                    [self.image_start_token_id],
                    np.full(image_k, self.image_patch_token_id, dtype=np.int32),
                    [self.image_end_token_id]
                ]
                high_res_pos_ids = [
                    np.zeros([1], dtype=np.int32),  # IMG start
                    np.ones(image_k, dtype=np.int32),  # IMG start/patches
                    [num_high_res_tokens+1]  # IMG End
                ]

            # Finally do the same for the global image
            resized, _ = self.resize_image(image, base_image_input_size, is_training, rng)
            resized = self._normalize(resized)
            resized = pixels_to_patches(resized, image_patch_size)
            patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

            per_row = np.full(
                (image_token_length_w,),
                self.image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [image_token_length_h])
            low_res_tokens = np.concatenate([
                [self.image_start_token_id],
                extra_tokens,
                [self.image_end_token_id],
            ])
            all_pos_ids = np.concatenate([
                np.arange(len(low_res_tokens)),
                np.concatenate(high_res_pos_ids) + len(low_res_tokens)
            ])

            return low_res_tokens, np.concatenate(joint), patch_ordering, patches, img_mask, patch_data, all_pos_ids
            # return low_res_tokens, np.concatenate(joint), patch_ordering, patches, img_mask, patch_data, col_token_positions
            # joint = [
            #             [self.image_start_token_id],
            #             extra_tokens,
            #             [self.image_end_token_id],
            #         ] + joint
            #
            # joint = np.concatenate(joint, 0)
            # img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)
            # return patches, joint, patch_ordering, img_mask
        else:
            raise NotImplementedError(self.crop_mode)

    def preprocess(self, image, is_training: bool, rng=None, num_high_res_features=None):
        # [n_crops, n_patches, pixels]
        # [n_crops, n_patch, 3] -> [is_padding]
        # [n_crops, n_super_patches, 146] -> [is_padding]
        # [n_crops, 144, 3] -> [is_padding, x, y, low_res_mapping]

        # [4*patch_size*patch_size+2]  # [pixels, is_padding, x, y, low_res_mapping]
        low_res_tokens, high_res_tokens, high_res_position, crops, img_mask, patch_data, all_pos_ids = (
            self.image_to_patches_and_tokens(image, is_training, rng, num_high_res_features))
        patch_idx = self.build_image_input_idx(low_res_tokens, None)
        patch_idx = patch_idx.reshape([-1])
        high_res_idx = np.where(high_res_tokens == self.image_patch_token_id)[0]
        patch_idx = np.concatenate([patch_idx, len(low_res_tokens) + high_res_idx])
        if self.indicate_k == "before-low-res":
            n_high_res = len(high_res_idx)
            indicator = self.tokenizer.encode("K=" + str(n_high_res))
            joint_tokens = np.concatenate([indicator, low_res_tokens, high_res_tokens])
            all_pos_ids = np.concatenate([np.arange(len(indicator), dtype=np.int32), all_pos_ids + len(indicator)])
        else:
            assert self.indicate_k is None
            joint_tokens = np.concatenate([low_res_tokens, high_res_tokens])
        return crops, joint_tokens, patch_idx, img_mask, patch_data, all_pos_ids

    def _sample(self, rng: np.random, min_k, max_k, num_k=None, sample_k=None):
        if min_k is None:
            if num_k is None:
                return max_k
            else:
                return [max_k] * num_k
        if num_k is None:
            num_k = 1
        if sample_k is None:
            return rng.randint(min_k, max_k, size=num_k)
        elif sample_k == "lin0.2":
            probs = np.linspace(start=1, stop=0.2, num=max_k-min_k)
            probs = probs / probs.sum()
            return np.random.choice(max_k-min_k, [num_k], p=probs, replace=True) + min_k
        else:
            raise NotImplementedError(sample_k)

    def __call__(
        self,
        images,
        messages: Union[List[str], List[List[str]]],
        weight=None,
        is_training=False,
        rng=None,
        require_image_features=False,
        max_k: Optional[int] = None,
        min_k: Optional[int] = None,
        num_k: Optional[int] = None,
        sample_k: Optional[str] = None,
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
                if self.max_query_len and msg_ix == 0:
                    message_ids = message_ids[:self.max_query_len]
                if has_loss:
                    message_ids.append(self.tokenizer.eos_token_id)
                token_ids += message_ids
                if weight is None:
                    loss_masks += [has_loss]*len(message_ids)
                else:
                    loss_masks += [weight if has_loss else 0]*len(message_ids)
            tokens = np.array(token_ids, dtype=np.int32)
            loss_masks = np.array(loss_masks, dtype=np.float32)
            subsegment_ids = None
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
            subsegment_ids = np.concatenate(subsegments, dtype=np.int32)
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
            if subsegment_ids is not None:
                subsegment_ids = np.pad(subsegment_ids, [[1, 0]], constant_values=subsegment_ids[0])[:-1]
                data["subsegments"] = subsegment_ids
            if require_image_features:
                # Add size-zero image features, this can be useful to make sure all devices
                # get an image input when the image ViT is FSDP wrapped
                tokens_per_image = self.image_token_length_w * self.image_token_length_h
                n_pixels = self.image_patch_size ** 2 * 3
                h, w = self.base_image_input_size
                image_num_patch = (h//self.image_patch_size * w//self.image_patch_size)
                crops = np.zeros((0, image_num_patch, n_pixels), dtype=np.float32)
                image_idx = np.zeros((0, tokens_per_image), np.int32)
                data.update(dict(
                    images=crops,
                    image_input_idx=image_idx,
                ))
                if self.image_padding_mask:
                    data["image_masks"] = np.zeros((0, image_num_patch), dtype=np.float32)
            return data

        if not isinstance(images, (list, tuple)):
            images = [images]
        image_idx = np.argwhere(tokens == self.image_prompt_token_id)
        if len(image_idx) == 0:
            image_idx = [-1] * len(images)
        else:
            image_idx = image_idx[:, 0]
            assert len(image_idx) == len(images)

        max_k = max_k or self.num_high_res_features
        min_k = min_k or self.multi_res_min
        num_k = num_k or self.multi_res_selection

        if num_k is not None and num_k != 1:
            assert min_k is not None
            # Everything after the image get repeated `self.multi_res_selection` times
            # Add EOS now since the EOS should also get repeated
            assert len(image_idx) == 1
            image_start = image_idx[0]
            before_img_tokens = tokens[:image_start+1]
            after_img_tokens = tokens[image_start+1:]
            before_img_loss = loss_masks[:image_start+1]
            after_img_loss = loss_masks[image_start+1:]

            # This token can get the EOS token as a label, so it should not be trained on
            # We could handle this case by repeating the image end token but that makes
            # life even more complex so for now we don't support it
            assert after_img_loss[0] == 0
            tokens = np.concatenate([before_img_tokens, np.tile(after_img_tokens, [num_k])])
            loss_masks = np.concatenate([before_img_loss, np.tile(after_img_loss, [num_k])])
            
            # Now build the subsegment ids, before image tokens are subsegment 0 and
            # tokens after the image are a random id reflect how many high res tokens to attend to
            subsegment_ixs = self._sample(rng, min_k, max_k, num_k, sample_k)
            subsegment_ixs.sort()
            ks = subsegment_ixs[::-1]
            ks = np.concatenate([[0], ks])
            subsegment_ids = np.repeat(ks, [len(before_img_tokens)] + [len(after_img_tokens)]*num_k)
            image_token_weights = np.zeros([max_k], dtype=np.float32)
            image_token_weights[subsegment_ixs] = 1
            image_token_weights = np.cumsum(image_token_weights[::-1])[::-1]
            n_high_res = subsegment_ixs.max() + 1
        else:
            n_high_res = self._sample(rng, min_k, max_k, num_k, sample_k)

        max_total_crops = self.max_crops
        image_token_length_w = self.image_token_length_w
        image_token_length_h = self.image_token_length_h
        image_patch_size = self.image_patch_size
        base_image_input_size = self.base_image_input_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        image_padding_mask = self.image_padding_mask

        tokens_per_image = image_token_length_w * image_token_length_h
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]

        n = len(images)
        all_crops = []
        all_image_idx = []
        out_tokens = []
        all_crop_masks = []
        all_loss_masks = []
        all_patch_data = []
        all_subsegment_ids = []
        all_position_ids = []
        current_position = 0
        for ix in range(n):
            token_ix = image_idx[ix]
            crops, image_tokens, patch_idx, img_mask, patch_data, image_pos_ids = self.preprocess(
                images[ix], is_training, rng, n_high_res)

            # In case there were fewer than `n_high_res` high-res tokens for this image
            # (e.g., it was a low-resolution image)
            n_high_res_tokens = len(patch_idx) - 144
            assert n_high_res_tokens <= n_high_res
            assert n == 1

            if token_ix == -1:  # -1 is an image inserted at the very start
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix-1] + 1
                end = token_ix + 1

            all_image_idx.append(patch_idx + token_ix)
            all_crops.append(crops)
            all_patch_data.append(patch_data)

            if subsegment_ids is not None:
                all_subsegment_ids.append(subsegment_ids[start:token_ix])
            out_tokens.append(tokens[start:token_ix])
            all_loss_masks.append(loss_masks[start:token_ix])
            n = len(out_tokens[-1])
            all_position_ids.append(np.arange(current_position, current_position+n))
            current_position += n

            if subsegment_ids is not None:
                n_los_res = np.argmax(image_tokens == self.image_end_token_id) + 1
                # low_res tokens are 0
                all_subsegment_ids.append(np.zeros([n_los_res], dtype=np.int32))

                # image patch tokens start at 1 and go to max tokens
                is_patch = image_tokens[n_los_res:] == self.image_patch_token_id
                image_subsegment_ids = np.cumsum(is_patch)

                # End of image of is 0 so all tokens attend to it
                # import pdb; pdb.set_trace()
                image_subsegment_ids[image_tokens[n_los_res:] == self.image_end_token_id] = 0
                all_subsegment_ids.append(image_subsegment_ids)

            out_tokens.append(image_tokens)
            if subsegment_ids is not None:
                assert subsegment_ids[0] == 0
            if self.debug and "v1_explicit" in self.debug:
                pos_ids = np.arange(current_position, current_position+len(image_tokens))
                high_res_start = 5 + np.argmax(image_tokens[5:] == self.image_start_token_id)
                is_high_res = np.arange(len(image_tokens)) > high_res_start
                pos_ids[(image_tokens == self.image_patch_token_id) & is_high_res] = current_position + 1 + high_res_start
                all_position_ids.append(pos_ids)
                current_position += len(image_tokens)
            elif self.debug and "repeat_image_pos_ids" in self.debug:
                should_increment = np.cumsum((image_tokens != self.image_patch_token_id) & (image_tokens != self.image_col_token_id))
                all_position_ids.append(current_position + should_increment)
                current_position += should_increment.max() + 1
            elif self.debug and "compressed_pos_ids_32" in self.debug:
                # import pdb; pdb.set_trace()
                # image_pos_ids = np.cumsum(np.where(image_tokens == self.image_patch_token_id, 1.0/32, 1))
                patch_data[:, :, 576] /= 32.0
                image_pos_ids = image_pos_ids / 32.0
                image_pos_ids[-1] = np.ceil(image_pos_ids[-1])
                all_position_ids.append(current_position + image_pos_ids)
                current_position += (image_pos_ids.max() + 1)
            else:
                all_position_ids.append(image_pos_ids + current_position)
                if self.debug and "start_pos_ids_at_half" in self.debug:
                    current_position += image_pos_ids.max() // 2 + 1
                elif self.debug and "start_pos_ids_len" in self.debug:
                    current_position += len(image_pos_ids)
                else:
                    current_position += image_pos_ids.max() + 1
            all_loss_masks.append(np.zeros(image_tokens.shape[0], dtype=np.float32))
            if image_padding_mask:
                all_crop_masks.append(img_mask)

        end = image_idx[-1] + 1
        out_tokens.append(tokens[end:])
        all_loss_masks.append(loss_masks[end:])
        n = len(out_tokens[-1])
        if subsegment_ids is not None:
            all_position_ids.append(current_position + build_pos_ids(subsegment_ids[end:]))
        else:
            all_position_ids.append(np.arange(current_position, current_position+n))
        if subsegment_ids is not None:
            all_subsegment_ids.append(subsegment_ids[end:])

        input_ids = np.concatenate(out_tokens, 0)
        images = np.concatenate(all_crops, 0)
        image_input_idx = np.concatenate(all_image_idx, 0)
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

        image_input_idx = np.where(image_input_idx < 0, image_input_idx, image_input_idx + 1)
        out = {
            # [n_crops, n_patches, n_pixels]
            "images": images,

            # [n_super_patches + n_high_res_tokens], maps image features -> token_ids
            "image_input_idx": image_input_idx,

            # [n_crops-1, n_super_patches, n_super_patches+3], the low->high mapping, the
            # x/y positions, and the position id of each high-res super-patch
            "high_res_patch_data": np.concatenate(all_patch_data),

            # [n_high_res_tokens], weight of each high res token
            "high_res_features_weights": np.ones(n_high_res_tokens, dtype=np.float32),

            "input_tokens": input_ids,
            "loss_masks": all_loss_masks,
            "target_tokens": target_tokens,
        }

        if image_padding_mask:
            out["image_masks"] = np.concatenate(all_crop_masks, 0)
        if subsegment_ids is not None:
            pos_ids = np.concatenate(all_position_ids, -1)
            # Add BOS as the new position 0
            pos_ids += 1
            pos_ids = np.pad(pos_ids, [[1, 0]], constant_values=0)
            if ends_with_eos:
                pos_ids = pos_ids[:-1]

            all_subsegments = np.concatenate(all_subsegment_ids, 0)
            all_subsegments = np.pad(all_subsegments, [[1, 0]], constant_values=all_subsegments[0])
            if ends_with_eos:
                all_subsegments = all_subsegments[:-1]
            out["subsegment_ids"] = all_subsegments
            out["position_ids"] = pos_ids

        pos_ids = np.concatenate(all_position_ids, -1)
        # Add BOS as the new position 0
        pos_ids += 1
        pos_ids = np.pad(pos_ids, [[1, 0]], constant_values=0)
        if ends_with_eos:
            pos_ids = pos_ids[:-1]
        assert pos_ids.shape[0] == input_ids.shape[0]
        out["position_ids"] = pos_ids
        return out

