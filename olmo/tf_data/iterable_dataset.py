import logging
import math
import multiprocessing
import os
import pickle
import queue
import socket
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.managers import BaseManager
from multiprocessing.shared_memory import SharedMemory
from os.path import exists
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import psutil
import tensorflow as tf
import numpy as np
import torch
import torch.utils.data
import clu
from clu.data.dataset_iterator import Element


from ..aliases import PathOrStr
from ..torch_util import barrier, get_fs_local_rank, get_global_rank, get_world_size, get_node_rank, \
    get_local_world_size, get_local_rank, move_to_device
from ..util import roundrobin, threaded_generator
from .data_factory import SeqioDataset
from ..multimodal_preprocessor import MultiModalPreprocessor
from .preprocesssors import rename
import torch.distributed as dist
from . import tasks

__all__ = ["MMIterableDataset"]

log = logging.getLogger(__name__)


def batch_fn(batch, for_inference):
    if for_inference:
        out = {}
        for k, v in batch.items():
            if k.startswith("metadata/"):
                out[k] = v
            else:
                out[k] = torch.from_numpy(v)
        return out
    else:
        out = {k: torch.from_numpy(v) for k, v in batch.items() if not k.startswith("metadata/")}
        out["metadata"] = [{} for _ in out["input_ids"]]
        return out


class PyTorchDatasetIterator(clu.data.dataset_iterator.TfDatasetIterator):
    def __init__(self, dataset, *, checkpoint: bool, for_inference: bool):
        self.for_inference = for_inference
        super().__init__(dataset, checkpoint=checkpoint)

    def __next__(self) -> Element:
        batch = {k: v.numpy() for k, v in next(self.iterator).items()}
        return batch_fn(batch, self.for_inference)

    def __len__(self) -> int:
        return len(self._dataset)


class MMIterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    def __init__(
        self,
        dataset: SeqioDataset,
        preprocessor: MultiModalPreprocessor,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        self.rank = rank if rank is not None else get_global_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.dataset_config = dataset

        data_iter = dataset.build(
            self.preprocessor,
            self.rank,
            self.world_size,
        )

        data_iter: tf.data.Dataset = rename(input_ids="input_tokens", labels="target_tokens")(data_iter)
        self.dataset = data_iter
        self.data_iter = PyTorchDatasetIterator(
            data_iter, checkpoint=True, for_inference=dataset.for_inference)

    def reset(self):
        self.data_iter.reset()

    def save(self, filename: PathOrStr):
        self.data_iter.save(filename)

    def restore(self, filename: PathOrStr):
        self.data_iter.restore(filename)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.data_iter
