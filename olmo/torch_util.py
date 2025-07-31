from __future__ import annotations

import gc
import os
import logging
from datetime import timedelta
from typing import Optional, TypeVar, List, Tuple, MutableMapping

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

T = TypeVar("T")


log = logging.getLogger(__name__)


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def init_process_group() -> bool:
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=int(os.environ.get("NCCL_TIMEOUT_MINUTES", 10))),
        device_id=torch.device(f"cuda:{get_local_rank()}")
    )

    # The math backend is very slow, make sure we don't use it accidentally
    torch.backends.cuda.enable_math_sdp(False)

    log.info("Process group initialized, math SDP disabled")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK") or (get_global_rank() - get_local_rank()) // get_local_world_size())


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_global_rank() -> int:
    if is_distributed():
        return int(os.environ.get("RANK") or dist.get_rank())
    else:
        return 0

    
def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def get_fs_local_rank() -> int:
    """Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_global_rank()`,
    but if nodes do not share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_local_rank()`.
    """
    if os.environ.get("OLMO_SHARED_FS"):
        return int(os.environ.get("FS_LOCAL_RANK") or get_global_rank())
    else:
        return int(os.environ.get("FS_LOCAL_RANK") or get_local_rank())


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def peak_gpu_memory(reset: bool = False) -> Optional[float]:
    """
    Get the peak GPU memory usage in MB across all ranks.
    Only rank 0 will get the final result.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if is_distributed():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


V = TypeVar("V", bool, int, float)


def synchronize_value(value: V, device: torch.device) -> V:
    if dist.is_available() and dist.is_initialized():
        value_tensor = torch.tensor(value, device=device)
        dist.broadcast(value_tensor, 0)
        return value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(flag: bool, device: torch.device) -> bool:
    return synchronize_value(flag, device)


def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def listinstr(lst, s, delimiter=None):
    assert isinstance(lst, list)
    for item in lst:
        if delimiter:
            if all(x in s for x in item.split(delimiter)):
                return True
        else:
            if item in s:
                return True
    return False


def freeze_module(module: torch.nn.Module, exclude_params: Optional[List[str]] = None):
    for name, param in module.named_parameters():
        if exclude_params is not None and listinstr(exclude_params, name):
            continue
        param.requires_grad = False


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def get_element_size(dtype: torch.dtype) -> int:
    """
    Get the size in bytes of element of the given PyTorch dtype.
    """
    return torch._utils._element_size(dtype)  # type: ignore


def clip_grad_norm(parameters, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None):
    # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348
    grads = [p.grad for p in parameters if p.grad is not None]
    # for p in grads:
    #     print(f"[Rank {get_global_rank()}] param norm={p.norm()}")
    total_norm = nn.utils.get_total_norm(
        grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
    )
    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
    return total_norm