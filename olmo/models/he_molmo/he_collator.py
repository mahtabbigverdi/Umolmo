from typing import Dict, List, Any
import numpy as np

from olmo.models.molmo.collator import _collate


class HeMMCollator:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks"]

    def __init__(self, max_sequence_length=None, include_metadata=True, pad=None, max_crops=None):
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

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        out = {}

        for ex in batch:
            assert ex["image_input_idx"].max() < len(ex["input_tokens"])
            if self.max_sequence_length:
                if ex["image_input_idx"].max() >= self.max_sequence_length:
                    import pdb; pdb.set_trace()
                assert ex["image_input_idx"].max() < self.max_sequence_length, f"Image features would get truncated: {batch[0]['image']}"

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

        out["high_res_patch_data"] = _collate(
            [ex["high_res_patch_data"] for ex in batch], self.max_crops-1, pad=self.pad,
            allow_truncate=False)

        # Note these are NOT padded to a max length
        out["high_res_features_weights"] = _collate([ex["high_res_features_weights"] for ex in batch], None, pad_value=0)
        out["image_input_idx"] = _collate([ex["image_input_idx"] for ex in batch], allow_truncate=False)
        assert out["image_input_idx"].shape[1] == out["high_res_features_weights"].shape[1] + 144

        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
