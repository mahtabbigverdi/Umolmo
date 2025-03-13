import dataclasses
import math
from typing import Tuple, Union, List, Optional
import numpy as np
import torch
import torchvision
from einops import einops
from torchvision.transforms import InterpolationMode
from transformers.image_utils import ImageInput

from olmo.config import BaseConfig
from olmo.models.molmo.model_preprocessor import select_tiling, batch_pixels_to_patches, \
    MolmoPreprocessor, arange_for_pooling
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

    low_res_from_high: Optional[int] = None

    low_res_from_low: Optional[int] = 2

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

    def get_max_tokens(self, vision_backbone_config: VisionBackboneConfig):
        """Max numbers of image tokens can be built for one image"""
        preprocessor = self.build(None, vision_backbone_config)
        return preprocessor.max_image_tokens()

    def build(
        self, tokenizer, vision_backbone_config: VisionBackboneConfig):
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
            image_patch_size=vit.image_patch_size,
            image_padding_mask=vision_backbone_config.image_padding_embed is not None,
            pad_value=vit.pad_value,
            low_res_from_high=self.low_res_from_high,
            low_res_from_low=self.low_res_from_low
        )


@dataclasses.dataclass
class HeMultiModalPreprocessor(MolmoPreprocessor):
    num_high_res_features: int=512
    max_query_len: Optional[int] = None
    multi_res_selection: Optional[int] = None
    multi_res_min: Optional[int] = None
    indicate_k: Optional[str] = None
    use_high_res_col_tokens: bool = True
    vector_query: bool = False
    debug: Optional[str] = None
    low_res_from_high: Optional[int] = None
    low_res_from_low: Optional[int] = 2

    def max_image_tokens(self) -> int:
        """Return the max number of pooled image tokens this could produce for any image"""
        base_h, base_w = self.base_image_input_size
        high, low = -1, -1
        for h, w in [
            [base_h, base_w*self.max_crops],
            [base_h*self.max_crops, base_w]
        ]:
            new_high, new_low = self.compute_num_tokens(h, w)
            high = max(high, new_high)
            low = max(low, new_low)
        return high, low

    def compute_num_tokens(self, image_h, image_w) -> int:
        """Return the number of pooled image tokens produced for an image of size image_w, image_h"""
        image_patch_size = self.image_patch_size
        crop_patch_w = self.base_image_input_size[1] // image_patch_size
        crop_patch_h = self.base_image_input_size[0] // image_patch_size

        margin_patches = sum(self.overlap_margins)
        margin_pixels = image_patch_size*margin_patches  # pixels removed per dim
        assert crop_patch_w == crop_patch_h
        crop_window_patches = crop_patch_w - margin_patches
        crop_window_size = crop_window_patches * image_patch_size
        tiling = select_tiling(
            image_h - margin_pixels,
            image_w - margin_pixels,
            crop_window_size,
            self.max_crops
        )
        h, w = [tiling[0]*crop_window_size+margin_pixels, tiling[1]*crop_window_size+margin_pixels]
        h, w = h//image_patch_size, w//image_patch_size
        idx_arr = arange_for_pooling(
            torch.zeros([h, w]), self.image_pooling_h, self.image_pooling_w)
        overlap_tokens = idx_arr.shape[0] * idx_arr.shape[1]

        low_res_tokens = 0
        if self.low_res_from_low:
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            idx_arr = arange_for_pooling(resize_idx, self.low_res_from_low, self.low_res_from_low)
            low_res_tokens += idx_arr.shape[0] * idx_arr.shape[1]
        if self.low_res_from_high:
            idx_arr = arange_for_pooling(torch.zeros([h, w]), self.low_res_from_high, self.low_res_from_high)
            low_res_tokens += idx_arr.shape[0] * idx_arr.shape[1]
        return overlap_tokens, low_res_tokens

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        n_high_res,
        is_training=False,
        rng=None,
    ):
        """
        :return image_tokens, the token IDS for this image
        :return crops, the image crops to processes with the ViT
        :return mask, the padding mask for each crop
        :return pooled_patch_idx, the padding mask for each crop
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
            resized, resized_mask = self.resize_image(image, base_image_input_size, is_training, rng)
            resized = np.expand_dims(resized, 0)
            resized_mask = np.expand_dims(resized_mask, 0)
            resized = self._normalize(resized)
            low_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            pooling_idx = arange_for_pooling(low_idx, self.image_pooling_w, self.image_pooling_h)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, self.image_pooling_w*self.image_pooling_h])

            per_row = np.full(
                (w,),
                self.image_patch_token_id,
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
                    batch_pixels_to_patches(resized_mask, image_patch_size), pooling_idx)

        if self.crop_mode in ["overlap-and-resize-c2", "overlap-and-resize"]:
            # Discard this many patches from the (left/top, right/bottom) of crops
            left_margin, right_margin = overlap_margins
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
            crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
            mask_arr = np.zeros([n_crops, crop_size, crop_size], dtype=img_mask.dtype)
            patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
            on = 0
            on_crop = 0
            for i in range(tiling[0]):
                # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
                # which results in overlapping crop windows
                y0 = i*crop_window_size
                for j in range(tiling[1]):
                    x0 = j*crop_window_size
                    crop_arr[on_crop] = src[y0:y0+crop_size, x0:x0+crop_size]
                    mask_arr[on_crop] = img_mask[y0:y0+crop_size, x0:x0+crop_size]
                    patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
                    patch_idx += on_crop * crop_patch_h * crop_patch_w

                    # Mask out idx that are in the overlap region
                    if i != 0:
                        patch_idx[:left_margin, :] = -1
                    if j != 0:
                        patch_idx[:, :left_margin] = -1
                    if i != tiling[0]-1:
                        patch_idx[-right_margin:, :] = -1
                    if j != tiling[1]-1:
                        patch_idx[:, -right_margin:] = -1
                    patch_idx_arr[on_crop] = patch_idx
                    on_crop += 1

            # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
            # so it is ordered left-to-right order
            patch_idx_arr = np.reshape(
                patch_idx_arr,
                [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
            )
            patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
            patch_idx_arr = np.reshape(patch_idx_arr, [-1])

            # Now get the non-masked parts, now it should map each patch in `src` to the
            # correct patch it should come from in `crop_arr`
            patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
                src.shape[0]//image_patch_size,
                src.shape[1]//image_patch_size,
            )

            # Now arrange `patch_idx_arr` so it ready for pooling, possibly padding it
            pooling_idx = arange_for_pooling(patch_idx_arr, self.image_pooling_w, self.image_pooling_h)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, self.image_pooling_w*self.image_pooling_h])

            # Now build the output tokens
            image_k = n_high_res
            if self.indicate_k:
                raise NotImplementedError()
            if self.use_high_res_col_tokens:
                col_token_positions = np.arange(1, h + 1) * (w + 1)
                joint = [
                    [self.image_start_token_id],
                    np.full(image_k, self.image_patch_token_id, dtype=np.int32),
                    np.full(h, self.image_col_token_id, dtype=np.int32),
                    [self.image_end_token_id]
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
                    [self.image_start_token_id],
                    np.full(image_k, self.image_patch_token_id, dtype=np.int32),
                    [self.image_end_token_id]
                ]
                high_res_pos_ids = [
                    np.zeros([1], dtype=np.int32),  # IMG start
                    np.ones(image_k, dtype=np.int32),  # IMG start/patches
                    [image_k+1]  # IMG End
                ]
                high_res_patch_pos_ids = np.arange(h*w)

            if self.low_res_from_high:
                low_idx = arange_for_pooling(patch_idx_arr, self.low_res_from_high, self.low_res_from_high)
                low_h, low_w = low_idx.shape[:2]
                low_res_pooling_idx = low_idx.reshape([-1, self.low_res_from_high**2])
            else:
                assert not self.low_res_from_high
                # Finally do the same for the global image
                resized, resized_mask = self.resize_image(image, base_image_input_size, is_training, rng)
                resized = self._normalize(resized)
                crop_arr = np.concatenate([np.expand_dims(resized, 0), crop_arr], 0)

                if self.legacy_image_mask:
                    raise NotImplementedError()
                else:
                    mask_arr = np.concatenate([np.expand_dims(resized_mask, 0), mask_arr], 0)

                low_idx = np.arange(crop_patch_h*crop_patch_w).reshape([crop_patch_h, crop_patch_w])
                low_idx = arange_for_pooling(low_idx, self.low_res_from_low, self.low_res_from_low)
                low_h, low_w = low_idx.shape[:2]
                low_res_pooling_idx = low_idx.reshape([-1, self.low_res_from_low*self.low_res_from_low])

                # Global image goes first, so the order of patches in previous crops gets increased
                pooling_idx = np.where(
                    pooling_idx >= 0,
                    pooling_idx + crop_patch_h*crop_patch_w,
                    -1
                )

            low_to_high = np.eye((low_h*2*low_w*2), dtype=np.float32)
            low_to_high = low_to_high.reshape([low_h*low_w*4, low_h*2, low_w*2])
            low_to_high = torchvision.transforms.Resize(
                [h, w], InterpolationMode.BILINEAR, antialias=False)(
                torch.from_numpy(low_to_high)).numpy()
            # Re-arrange to match how the importance scores are predicted (four per a patch)
            # This save us having to transpose it in model
            low_to_high = einops.rearrange(
                low_to_high, "(lh dh lw dw) h w -> (lw lh dw dh) (h w)",
                dh=2, dw=2, lh=low_h, lw=low_w)

            per_row = np.full(
                (low_w,),
                self.image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [low_h])
            low_res_tokens = np.concatenate([
                [self.image_start_token_id],
                extra_tokens,
                [self.image_end_token_id],
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
                mask_arr, low_res_pooling_idx, pooling_idx,
                (low_to_high, high_res_patch_pos_ids)
            )
        else:
            raise NotImplementedError(self.crop_mode)

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
                raise NotImplementedError()

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
        image_patch_size = self.image_patch_size
        base_image_input_size = self.base_image_input_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]

        n = len(images)
        all_crops = []
        out_tokens = []
        all_loss_masks = []
        high_res_features_weights = []
        low_pooled_patches_idx = []
        pooled_patches_idx = []
        all_subsegment_ids = []
        all_position_ids = []
        _high_res_data = None
        current_position = 0
        for ix in range(n):
            token_ix = image_idx[ix]
            image_tokens, image_pos_ids, crops, img_mask, low_pooled_idx, pooled_idx, _high_res_data = self.image_to_patches_and_tokens(images[ix], n_high_res, is_training, rng)

            if token_ix == -1:  # -1 is an image inserted at the very start
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix-1] + 1
                end = token_ix + 1

            n_high_res_tokens = (image_tokens == self.image_patch_token_id).sum() - len(low_pooled_idx)

            offset = sum(np.prod(x.shape[:2]) for x in all_crops)
            low_pooled_patches_idx.append(low_pooled_idx + offset)
            pooled_patches_idx.append(pooled_idx + offset)
            all_crops.append(crops)
            out_tokens.append(tokens[start:token_ix])
            all_loss_masks.append(loss_masks[start:token_ix])

            n = len(out_tokens[-1])
            all_position_ids.append(np.arange(current_position, current_position+n))
            current_position += n

            out_tokens.append(image_tokens)
            all_position_ids.append(image_pos_ids + current_position)
            current_position += image_pos_ids.max() + 1

            all_loss_masks.append(np.zeros(image_tokens.shape[0], dtype=np.float32))

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
            "images": images,
            "low_pooled_patches_idx": np.concatenate(low_pooled_patches_idx, 0),
            "pooled_patches_idx": np.concatenate(pooled_patches_idx, 0),
            "input_tokens": input_ids,
            "loss_masks": all_loss_masks,
            "target_tokens": target_tokens,
            "high_res_features_weights": np.ones(n_high_res_tokens, dtype=np.float32),
            "low_to_high": _high_res_data[0],
            "high_res_pos_ids": _high_res_data[1]
        }

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

