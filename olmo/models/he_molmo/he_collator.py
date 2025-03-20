from typing import Dict, List, Any
import numpy as np
import torch

from olmo import tokenizer
from olmo.models.molmo.collator import _collate
from olmo.tokenizer import get_special_token_ids


class HeMMCollator:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]

    def __init__(self, tok, max_sequence_length, padding_lens, include_metadata=True, pad=None):
        """
        :param max_sequence_length: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        self.max_sequence_length = max_sequence_length
        self.include_metadata = include_metadata
        self.padding_lens = padding_lens
        self.pad = pad
        special_tokens = get_special_token_ids(tok)
        self._image_tokens = np.array([
            special_tokens[tokenizer.IMAGE_LOW_RES_TOKEN],
            special_tokens[tokenizer.IMAGE_PATCH_TOKEN],
        ])[None, :]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        out = {}

        if self.pad:
            if any(np.any(self._image_tokens == ex["input_tokens"][self.max_sequence_length:][:, None]) for ex in batch):
                raise ValueError("An image would have gotten truncated!")

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

        for key, max_len in self.padding_lens.items():
            if any(key in ex for ex in batch):
                out[key] = _collate([ex.get(key) for ex in batch], max_len, pad=self.pad, allow_truncate=False)

        l2h = [ex["low_to_high"] for ex in batch]
        n_low = out["low_res_tokens_idx"].shape[1]*4
        n_high = out["high_res_tokens_idx"].shape[1]
        out["low_to_high"] = torch.from_numpy(np.stack([
            np.pad(x, [[0, n_low-x.shape[0]], [0, n_high-x.shape[1]]])
            for x in l2h
        ]))

        # Special case, we don't pad this to a fixed length since its shape does not affect
        # any parameterized moduyle, just how we do top-k and then shuffle around the image features
        out["high_res_features_weights"] = _collate([ex["high_res_features_weights"] for ex in batch], pad_value=0)

        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
