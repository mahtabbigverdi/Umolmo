import dataclasses
from typing import Any, Dict, List, Tuple
import numpy as np

from olmo.config import BaseConfig


def pack(*examples: Dict) -> Dict:
    keys = set(x for x in examples[0].keys() if x != "metadata")
    if "subsegment_ids" not in keys:
        keys.add("subsegment_ids")
    patch_keys = [k for k in ["low_res_pooled_idx", "high_res_pooled_idx", "pooled_patches_idx"]
                  if k in keys]
    image_offset = 0
    for example_ix, example in enumerate(examples):
        # Patch indices need to be offset by total number of images
        for k in patch_keys:
            example[k] += image_offset
        image_offset += np.prod(example["images"].shape[:2])
        assert "position_ids" in example
        n_tokens = len(example["position_ids"])

        # Modify or add subsegment ids to prevent intra-example attention
        if "subsegment_ids" not in example:
            example["subsegment_ids"] = np.full([n_tokens], example_ix*100000)
        else:
            example["subsegment_ids"] += example_ix*100000
    return {k: np.concatenate([ex[k] for ex in examples]) for k in keys if k != "metadata"}


def packed_iterator(it, packer):
    for i, ex in enumerate(it):
        out = packer.add_example(i, ex)
        if out is not None:
            yield out


class PackingConfig(BaseConfig):
    buffer_size: int = 10
    max_frames: Tuple[int] = (4, -1)
    max_text_tokens: int = 512


@dataclasses.dataclass
class Packable:
    id: Any
    lens: Dict[str, int]
    size: int
    example: Dict[str, Any]


class PackPairs:
    def __init__(self):
        self._example = None

    def add_example(self, example_id, example) -> List:
        if self._example is None:
            self._example = example
            return None
        tmp = self._example
        self._example = None
        return pack(example, tmp)


class GreedyDynamicPacker:

    def __init__(self, buffer_size: int, max_lens: Dict[str, int]):
        self.max_lens = max_lens
        self.buffer_size = buffer_size
        self._buffer: List[Packable] = []

    def _can_pack_with(self, ex1: Dict, ex2: Dict):
        return all((ex1[k] + ex2[k]) < max_len for k, max_len in self.max_lens.items())

    def add_example(self, example_id, example) -> List:
        pack_lens = {k: len(v) for k, v in example.items()}
        obj = Packable(example_id, pack_lens, sum(pack_lens.values()), example)
        if len(self._buffer) < len(self.buffer_size):
            self._buffer.append(obj)
            return None
        best_can_pack_with = None
        best_ix = None
        for ix, example in enumerate(self._buffer):
            if (best_can_pack_with is None or best_can_pack_with.size > example.size) and self._can_pack_with(obj.lens, example.lens):
                best_can_pack_with = example
                best_ix = ix
        if best_can_pack_with is None:
            # Return the largest example in the buffer, or the input example
            ix = max(enumerate(self._buffer), key=lambda x: x.size)
            out = self._buffer[ix]
            if out.size > obj.size:
                return pack(obj.example)
            self._buffer[ix] = obj
            return pack(out.example)
        return pack(obj, self._buffer.pop(best_ix))







