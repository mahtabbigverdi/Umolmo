from typing import Dict, List, Any
import numpy as np
import torch

from olmo.models.molmo.collator import _collate


class HeMMCollator:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks"]

    def __init__(self, max_sequence_length=None, include_metadata=True, pad=None,
                 max_crops=None, max_low_res_tokens=None, max_high_res_tokens=None):
        """
        :param max_sequence_length: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert max_sequence_length is not None and max_crops is not None
        self.max_sequence_length = max_sequence_length
        self.max_crops = max_crops
        self.include_metadata = include_metadata
        self.pad = pad
        self.max_low_res_tokens = max_low_res_tokens
        self.max_high_res_tokens = max_high_res_tokens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        out = {}

        for key in self.TEXT_KEYS:
            # If one examples has subsegment_ids, all examples need it so with ones
            # matching the input tokens
            if any(key in ex for ex in batch):
                if key == "subsegment_ids":
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])

                if key == "position_ids" and key in batch[0]:
                    dtype = batch[0][key].dtype
                else:
                    dtype = np.float32 if key == "loss_masks" else np.int64
                out[key] = _collate(
                    [ex.get(key) for ex in batch], self.max_sequence_length, dtype, pad=self.pad)

        for key in self.IMAGE_KEYS:
            if any(key in ex for ex in batch):
                out[key] = _collate([ex.get(key) for ex in batch], self.max_crops, pad=self.pad, allow_truncate=False)

        out["low_pooled_patches_idx"] = _collate(
            [ex["low_pooled_patches_idx"] for ex in batch],
            self.max_low_res_tokens, pad=self.pad, pad_value=-1)
        n_low_res = out["low_pooled_patches_idx"].shape[1]

        l2h = [ex["low_to_high"] for ex in batch]
        n_high, n_low = max(x.shape[1] for x in l2h), max(x.shape[0] for x in l2h)
        if self.pad == "to_max":
            assert n_low <= self.max_low_res_tokens*4
            assert n_high <= self.max_high_res_tokens
            n_low, n_high = self.max_low_res_tokens*4, self.max_high_res_tokens
        elif self.pad is not None:
            raise NotImplementedError(self.pad)
        out["low_to_high"] = torch.from_numpy(np.stack([
            np.pad(x, [[0, n_low-x.shape[0]], [0, n_high-x.shape[1]]])
            for x in l2h
        ]))
        out["high_res_patch_pos_ids"] = _collate(
            [ex["high_res_pos_ids"] for ex in batch],
            self.max_high_res_tokens, pad=self.pad, pad_value=-1, allow_truncate=False)
        out["pooled_patches_idx"] = _collate(
            [ex["pooled_patches_idx"] for ex in batch],
            self.max_high_res_tokens, pad=self.pad, pad_value=-1, allow_truncate=False)

        # Special case, we don't pad this to a fixed length since its shape does not affect
        # any parameterized module, just how we do top-k and then shuffle around the image features
        out["high_res_features_weights"] = _collate([ex["high_res_features_weights"] for ex in batch], pad_value=0)

        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
