import logging
from typing import Dict, Any, List

import numpy as np
import torch

from olmo import tokenizer
from olmo.tokenizer import get_special_token_ids

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def _collate(tensors, max_sequence_length=None, dtype=None, pad=None, pad_value=-1, allow_truncate=True):
    tensor = [x for x in tensors if x is not None and x.shape[0]!=0][0]
    max_tensor_len = max((0 if x is None else x.shape[0]) for x in tensors)
    if pad == "to_max":
        max_len = max_sequence_length
        if not allow_truncate:
            assert max_tensor_len <= max_len
    elif pad == "truncate":
        max_len = min(max_tensor_len, max_sequence_length)
        if not allow_truncate:
            assert max_tensor_len <= max_len
    elif pad is None:
        max_len = max_tensor_len
    else:
        raise NotImplementedError(pad)

    arr = np.full([len(tensors), max_len] + list(tensor.shape[1:]), pad_value,
                  dtype=dtype or tensor.dtype)
    for ix, tensor in enumerate(tensors):
        if tensor is not None and tensor.shape[0] != 0:
            arr[ix, :len(tensor)] = tensor[:max_len]
    return torch.from_numpy(arr)


class MMCollator:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks"]

    def __init__(self, special_tokens, max_sequence_length=None, image_padding_lens=None, include_metadata=True, pad=None):
        """
        :param max_sequence_length: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert max_sequence_length is not None and image_padding_lens is not None
        self.max_sequence_length = max_sequence_length
        self.image_padding_lens = image_padding_lens
        self.include_metadata = include_metadata
        self.pad = pad
        self._special_tokens = np.array([
            special_tokens[tokenizer.IM_END_TOKEN],
            special_tokens[tokenizer.IM_START_TOKEN],
            special_tokens[tokenizer.IM_COL_TOKEN],
            special_tokens[tokenizer.IMAGE_LOW_RES_TOKEN],
            special_tokens[tokenizer.IMAGE_PATCH_TOKEN],
        ])[None, :]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()

        # Sanity checks
        
        for ex in batch:
            if self.pad:
                if np.any(self._special_tokens == ex["input_tokens"][self.max_sequence_length:][:, None]):
                    raise ValueError("An image would have gotten truncated!")
                if np.any(ex["loss_masks"] != 0) and np.all(ex["loss_masks"][:self.max_sequence_length] == 0):
                    raise ValueError("All loss tokens truncated!")

        out = {}
        for key in self.TEXT_KEYS:
            # If one example has subsegment_ids, all examples need it as well
            if key == "subsegment_ids":
                if any(key in ex for ex in batch):
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])
                else:
                    continue
            
            dtype = np.float32 if key == "loss_masks" else np.int64
            out[key] = _collate(
                [ex.get(key) for ex in batch], self.max_sequence_length, dtype, pad=self.pad)
        
        for key, max_len in self.image_padding_lens.items():
            out[key] = _collate([ex.get(key) for ex in batch], max_len, pad=self.pad, allow_truncate=False,)
        

        ## check if there exists at least one image output in the batch
        max_len = max(ex.get('image_outputs').shape[0] for ex in batch)
        if max_len > 0:
            out['image_outputs'] = _collate(
                [ex.get('image_outputs') for ex in batch],
                max_len,
                pad=self.pad,
                allow_truncate=False,
            )
        else:
            out['image_outputs'] = torch.zeros((len(batch), 0), dtype=torch.float32)
        
        # if any(ex.get('image_outputs').shape[0]!=0 for ex in batch):
        #     MAX_NUMBER_OUTPUT_IMAGE_TOKENS = 200
        #     out['image_outputs'] = _collate([ex.get('image_outputs') for ex in batch], MAX_NUMBER_OUTPUT_IMAGE_TOKENS, pad=self.pad, allow_truncate=False,)
        # else:
        #     out['image_outputs'] = torch.zeros((len(batch), 0), dtype=torch.float32)
        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
