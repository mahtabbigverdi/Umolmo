"""
From https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/distributed/checkpoint/__init__.py

A high-level distributed checkpointing module with a unified API for saving and
loading both local and remote checkpoints.

Features
--------

- Save with one distributed topology, seamlessly load with a different one. For example,
  with FSDP/FSDP2 you can save/load checkpoints with different world sizes or sharding strategies.
- Save/load directly to/from a remote object store like S3 or GCS. When loading from a remote object
  store each rank only downloads the fraction of the data it needs for its local
  (potentially sharded) tensors.

Overview
--------

Use :func:`save_model_and_optim_state()` to write a checkpoint with your model and optimizer's state,
then use :func:`load_model_and_optim_state()` to load the checkpoint in-place.

You can unshard a checkpoint saved this way with :func:`unshard_checkpoint()`.

API Reference
-------------
"""

from __future__ import annotations
from olmo.torch_util import get_global_rank
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from rich.progress import track
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo.config import StrEnum
from olmo.io import PathOrStr, normalize_path, clear_directory, dir_is_empty, is_url
from olmo.torch_util import barrier, get_fs_local_rank, is_distributed, gc_cuda, get_world_size

from torch.distributed._tensor import DTensor, DeviceMesh
from torch.distributed._tensor.placement_types import Replicate

from olmo.train.remote_filesystem import RemoteFileSystemWriter, RemoteFileSystemReader
from olmo.util import wait_for

log = logging.getLogger(__name__)


@torch.no_grad()
def save_state_dict(
    dir: PathOrStr,
    state_dict: Dict[str, Any],
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    thread_count: Optional[int] = None,
    throttle_uploads: bool = False,
):
    """
    Save an arbitrary state dictionary to a distributed format that can loaded again with
    a different distributed topology.

    .. important::
        Please use :func:`save_model_and_optim_state` to save model/optimizer state dicts instead
        unless you know what you're doing.

    :param dir: Path/URL to save to.
    :param state_dict: The state dict to save.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files.
    :param thread_count: Set this to override the number of threads used while writing data.
    :param throttle_uploads: If this is set to ``True`` and ``dir`` is a URL then only one
        rank from each node will upload data at a time.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    dist_cp.state_dict_saver.save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
    )


@torch.no_grad()
def save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    thread_count: Optional[int] = None,
    throttle_uploads: bool = False,
) -> None:
    """
    Save model and optimizer state dictionaries. The model state can be a sharded model, in which
    case this method will correctly handle the optimizer state to ensure it can be loaded again with
    a different distributed topology through :func:`load_model_and_optim_state()`.

    .. seealso::
        - :func:`load_model_and_optim_state()`
        - :func:`async_save_model_and_optim_state()`
        - :func:`unshard_checkpoint()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`load_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    :param dir: Path/URL to save to.
    :param model: The model to save state from.
    :param optim: The optimizer to save state from.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files.
    :param thread_count: Set this to override the number of threads used while writing data.
    :param throttle_uploads: If this is set to ``True`` and ``dir`` is a URL then only one
        rank from each node will upload data at a time.

    :raises FileExistsError: If the checkpoint dir exists and is non-empty unless ``save_overwrite=True``.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    state_dict = _prepare_state_dict(model, optim=optim, process_group=process_group)
    planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)
    dist_cp.state_dict_saver.save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def async_save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    thread_count: Optional[int] = None,
    throttle_uploads: bool = False,
) -> Future[None]:
    """
    An async version of :func:`save_model_and_optim_state()`.

    This code first de-stages the state dict on the CPU, then writes it in a separate thread.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    state_dict = _prepare_state_dict(model, optim=optim, process_group=process_group)
    planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)
    return dist_cp.state_dict_saver.async_save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def load_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    key_mapping: Optional[Dict[str, str]] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    thread_count: Optional[int] = None,
):
    """
    Load model and optimizer state in-place from a checkpoint saved via :func:`save_model_and_optim_state()`.
    This method is agnostic to the distributed topology in that it can load checkpoints saved with a different
    distributed topology (e.g. FSDP/FSDP2, DDP).

    .. seealso::
        - :func:`save_model_and_optim_state()`
        - :func:`unshard_checkpoint()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`save_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    .. warning::
        Due to the way :mod:`torch.distributed.checkpoint` works, if you have keys in the checkpoint
        dict that are not present in the current state of the model or optimizer, those keys won't
        be loaded.

        For example, if you added a custom field to one of your optimizer's param groups
        before saving the checkpoint, but don't have that field in the param group of the optimizer
        you're loading into, it won't be added.

        This can cause unexpected behavior if you're not careful. In this case the best thing to do
        is to ensure all keys are in present param groups when you initialize the optimizer, before saving
        or loading a checkpoint.

    :param dir: Path/URL to the checkpoint saved via :func:`save_model_and_optim_state()`.
    :param model: The model to load the state into.
    :param optim: The optimizer to load the state into.
    :param process_group: The process group to use for distributed collectives.
    :param key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.
    :param thread_count: Set the number of threads used for certain operations.
    """
    dir = normalize_path(dir)
    state_dict = _prepare_state_dict(model, optim, process_group=process_group)
    reader = RemoteFileSystemReader(
        dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
    )
    metadata = reader.read_metadata()

    if key_mapping is not None:
        for current_key, original_key in key_mapping.items():
            if f"model.{original_key}" not in metadata.state_dict_metadata:
                continue

            log.info(f"Mapping current param '{current_key}' to '{original_key}' in checkpoint")
            state_dict["model"][original_key] = state_dict["model"].pop(current_key)

            if optim is None:
                continue

            state_dict["optim"]["state"][original_key] = state_dict["optim"]["state"].pop(
                current_key
            )
            for group in state_dict["optim"]["param_groups"]:
                if current_key in group["params"]:
                    idx = group["params"].index(current_key)
                    group["params"][idx] = original_key
                    break

    model_key_list = list(state_dict['model'].keys())
    skipped_key_list = []
    for model_key in model_key_list:
        if f"model.{model_key}" not in metadata.state_dict_metadata:
            state_dict['model'].pop(model_key)
            skipped_key_list.append(model_key)
   
    ## todo remove
    # missing_optimizer_keys = []
    # if optim is not None:
    #     for key in list(state_dict["optim"]["state"].keys()):
    #         # Check if the checkpoint actually contains this optimizer key
    #         if f"optim.state.{key}.step" not in metadata.state_dict_metadata:
    #             missing_optimizer_keys.append(key)
    #             state_dict["optim"]["state"].pop(key, None)
    #             # Remove from param_groups to avoid invalid references
    #             for group in state_dict["optim"]["param_groups"]:
    #                 group["params"] = [p for p in group["params"]
    #                                 if p not in missing_optimizer_keys]
    #     if missing_optimizer_keys:
    #         log.warning(f"Missing optimizer state for params: {missing_optimizer_keys}")
    ## todo remove

    dist_cp.load(
        state_dict,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
    )
    ## todo remove
    # for key in ["vision_decoder_head.weight"]:
    #     if key not in state_dict["optim"]["state"]:
    #         t = model.vision_decoder_head.weight
    #         state_dict["optim"]["state"][key] = {
    #             "step": torch.tensor(0, dtype=torch.int64),
    #             "exp_avg": torch.zeros_like(t),
    #             "exp_avg_sq": torch.zeros_like(t),
    #         }
    ## todo remove

    
    if len(skipped_key_list) > 0:
        import os
        from transformers import AutoModel
        from olmo.data.dataset import VIDEO_DATA_HOME

        hf_source = "google/siglip2-so400m-patch14-384"
        cache_dir = os.path.join(VIDEO_DATA_HOME, "hf_init_encoders")
        frame_selection_model = AutoModel.from_pretrained(hf_source, cache_dir=cache_dir)

        class WrapperModule(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.frame_selection_model = module

        wrapped_model = WrapperModule(frame_selection_model)
        frame_sel_sd = wrapped_model.state_dict()

        for model_key in skipped_key_list:
            assert model_key in frame_sel_sd, f"Missing key {model_key} not from frame selection model"

        # Create a device mesh, e.g., for available CUDA devices
        mesh = DeviceMesh("cuda", torch.arange(get_world_size()))

        # Define placements; for instance, Replicate means the tensor is replicated across devices
        placements = [Replicate()]

        def convert_state_dict_to_dtensor(state_dict, mesh, placements):
            converted_state_dict = {}
            for key, tensor in state_dict.items():
                if not isinstance(tensor, DTensor):
                    converted_state_dict[key] = DTensor.from_local(tensor, mesh, placements)
                else:
                    converted_state_dict[key] = tensor
            return converted_state_dict

        frame_sel_sd_dtensor = convert_state_dict_to_dtensor(frame_sel_sd, mesh, placements)

        state_dict['model'].update(frame_sel_sd_dtensor)

    if key_mapping is not None:
        for current_key, original_key in key_mapping.items():
            if f"model.{original_key}" not in metadata.state_dict_metadata:
                continue

            state_dict["model"][current_key] = state_dict["model"].pop(original_key)

            if optim is None:
                continue

            state_dict["optim"]["state"][current_key] = state_dict["optim"]["state"].pop(
                original_key
            )
            for group in state_dict["optim"]["param_groups"]:
                if original_key in group["params"]:
                    idx = group["params"].index(original_key)
                    group["params"][idx] = current_key
                    break

    dist_cp_sd.set_model_state_dict(
        model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=True)
    )
    gc_cuda()

    if optim is not None:
        dist_cp_sd.set_optimizer_state_dict(
            model, optim, state_dict["optim"], options=dist_cp_sd.StateDictOptions(strict=True)
        )
        ## todo remove
        # for name in missing_optimizer_keys:
        #     for group in optim.param_groups:
        #         for p in group["params"]:
        #             if hasattr(p, "name") and p.name == name or getattr(p, "_name", None) == name:
        #                 optim.state[p]["step"] = torch.tensor(0, dtype=torch.int64)
        #                 optim.state[p]["exp_avg"] = torch.zeros_like(p)
        #                 optim.state[p]["exp_avg_sq"] = torch.zeros_like(p)
        ## todo remove
        gc_cuda()


class UnshardStrategyType(StrEnum):
    """
    An enumeration of the unsharding strategies that can be used with :func:`unshard_checkpoint`.
    """

    one_file = "one_file"
    """
    Save the unsharded model state into a one file, and optionally the optimizer state into
    another file. The bigger the model, the more memory this requires. For very big models,
    :data:`one_file_per_tensor` will scale better.
    """

    one_file_per_tensor = "one_file_per_tensor"
    """
    Save each unsharded tensor to its own file. Currently this is not compatible with optimizer
    state.
    """

    chunks = "chunks"
    """
    Like :data:`one_file_per_tensor` but multiple tensors and objects may be grouped into the same file
    up to the limit defined by :data:`UnshardStrategy.chunk_size_bytes`.
    """


@dataclass
class UnshardStrategy:
    """
    Unsharding strategy config for :func:`unshard_checkpoint`.
    """

    name: UnshardStrategyType = UnshardStrategyType.one_file
    """
    The strategy type.
    """

    chunk_size_bytes: Optional[int] = None
    """
    The approximate max chunk size (per file size), in bytes, for the :data:`UnshardStrategyType.chunks` strategy.
    """

    def __post_init__(self):
        if self.name == UnshardStrategyType.chunks and self.chunk_size_bytes is None:
            raise ValueError("'chunk_size_bytes' is required for the 'chunks' strategy")
        if self.chunk_size_bytes is not None and self.name != UnshardStrategyType.chunks:
            raise ValueError("'chunk_size_bytes' is only valid for the 'chunks' strategy")

    @classmethod
    def one_file(cls) -> "UnshardStrategy":
        """
        Use the :data:`UnshardStrategy.one_file` strategy.
        """
        return cls(name=UnshardStrategyType.one_file)

    @classmethod
    def one_file_per_tensor(cls) -> "UnshardStrategy":
        """
        Use the :data:`UnshardStrategy.one_file_per_tensor` strategy.
        """
        return cls(name=UnshardStrategyType.one_file_per_tensor)

    @classmethod
    def chunks(cls, chunk_size_in_bytes: int) -> "UnshardStrategy":
        """
        Use the :data:`UnshardStrategy.chunks` strategy.
        """
        return cls(name=UnshardStrategyType.chunks, chunk_size_bytes=chunk_size_in_bytes)


def unshard_checkpoint(
    dir: PathOrStr,
    target_dir: PathOrStr,
    *,
    optim: Optional[bool] = None,
    save_overwrite: bool = False,
    use_safetensors: bool = False,
    unshard_strategy: Optional[UnshardStrategy] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    quiet: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    Convert a checkpoint saved via :func:`save_model_and_optim_state()` into unsharded
    model and optimizer checkpoint files that can be loaded directly with :func:`torch.load()`
    or `safetensors <https://github.com/huggingface/safetensors>`_ if ``use_safetensors=True``.

    .. warning::
        The safetensors format cannot be used to save optimizer state, since optimizer state
        can contain arbitrary Python objects that need to be pickled.
        Therefore ``optim=True`` and ``use_safetensors=True`` is incompatible.

    .. warning::
        This should only be called in a non-distributed context. Otherwise a :class:`RuntimeError` is raised.

    .. seealso::
        :func:`load_keys()` if you only need to load and unshard certain keys in the checkpoint.

    :param dir: The path/URL to the original checkpoint created via :func:`save_model_and_optim_state()`.
    :param target_dir: The directory to save the unsharded model/optimizer checkpoint files to.
        This must be a local directory. URLs are not supported.
    :param optim: Whether to unshard the optimizer state. This defaults to ``True`` as long as
        ``use_safetensors=False``.
    :param save_overwrite: Overwrite any existing files in ``target_dir``.
    :param use_safetensors: Save the unsharded files with :func:`safetensors.torch.save_file()` instead
        of :func:`torch.save()`.
    :param unshard_strategy: The strategy to use. Defaults to :meth:`UnshardStrategy.one_file`.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.
    :param quiet: Do not show progress messages.

    :return: The path to the unsharded model checkpoint and the path to the unsharded
        optimizer checkpoint if ``optim=True``. These paths may represent files or directories
        depending on the ``unshard_strategy``.

    :raises FileExistsError: If the ``target_dir`` is non-empty and ``save_overwrite=False``.
    """
    # Adapted from `torch.distributed.checkpoint.format_utils.dcp_to_torch_save()`.

    if unshard_strategy is None:
        unshard_strategy = UnshardStrategy.one_file()

    if optim is None:
        optim = (not use_safetensors) and (unshard_strategy.name == UnshardStrategyType.one_file)
    elif optim and use_safetensors:
        raise NotImplementedError("`optim=True` is incompatible with `use_safetensors=True`")
    elif optim and unshard_strategy.name != UnshardStrategyType.one_file:
        raise NotImplementedError(
            f"`optim=True` is incompatible with `unshard_strategy={unshard_strategy}`"
        )

    if is_distributed():
        raise RuntimeError("'unshard_checkpoint' cannot be called in a distributed context")

    dir = normalize_path(dir)

    if is_url(target_dir):
        raise ValueError("'target_dir' must be a local directory")
    target_dir = Path(normalize_path(target_dir))
    target_dir.mkdir(exist_ok=True, parents=True)

    ext = "pt" if not use_safetensors else "safetensors"
    metadata = get_checkpoint_metadata(dir)

    def save(state_dict: Dict[str, Any], path: Path):
        if path.is_file() and not save_overwrite:
            raise FileExistsError(
                f"'{path}' already exists, use `save_overwrite=True` to overwrite it"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        if use_safetensors:
            from safetensors.torch import save_file

            save_file(state_dict, path)
        else:
            torch.save(state_dict, path)

    def get_chunks(prefix: str) -> Tuple[Path, List[Tuple[Path, List[str]]]]:
        assert unshard_strategy is not None
        assert isinstance(target_dir, Path)

        if unshard_strategy.name == UnshardStrategyType.one_file:
            path = target_dir / f"{prefix}.{ext}"
            return path, [(path, [prefix])]
        elif unshard_strategy.name == UnshardStrategyType.one_file_per_tensor:
            path = target_dir / prefix
            chunks = []
            for key in metadata.state_dict_metadata.keys():
                if key.startswith(f"{prefix}."):
                    chunks.append((path / f"{key.replace('.', '-')}.{ext}", [key]))
            return path, chunks
        elif unshard_strategy.name == UnshardStrategyType.chunks:
            assert unshard_strategy.chunk_size_bytes is not None
            path = target_dir / prefix
            chunks = []
            current_size = 0
            current_keys: List[str] = []
            for key, meta in metadata.state_dict_metadata.items():
                if key.startswith(f"{prefix}."):
                    if isinstance(meta, TensorStorageMetadata):
                        size = meta.size.numel() * get_element_size(meta.properties.dtype)
                        if current_keys and current_size + size > unshard_strategy.chunk_size_bytes:
                            chunks.append((path / f"chunk-{len(chunks):05d}.{ext}", current_keys))
                            current_size = 0
                            current_keys = []
                        current_size += size
                        current_keys.append(key)
                    else:
                        # This is a pickled Python object, which is probably pretty small,
                        # so we don't worry about recording the size.
                        current_keys.append(key)
            if current_keys:
                chunks.append((path / f"chunk-{len(chunks):05d}.{ext}", current_keys))
            return path, chunks
        else:
            raise NotImplementedError(unshard_strategy.name)

    def unshard_chunk(prefix: str, path: Path, keys: List[str]):
        state_dict: Dict[str, Any] = _load_unsharded_keys(
            dir, keys, pre_download=pre_download, work_dir=work_dir
        )
        if not state_dict:
            raise RuntimeError(f"missing keys '{keys}' in checkpoint")

        save(state_dict[prefix], path)
        del state_dict
        gc_cuda()

    model_path, model_chunks = get_chunks("model")
    for chunk_path, chunk_keys in track(
        model_chunks, description="Unsharding model chunks...", disable=quiet
    ):
        unshard_chunk("model", chunk_path, chunk_keys)

    optim_path: Optional[Path] = None
    if optim:
        optim_path, optim_chunks = get_chunks("optim")
        for chunk_path, chunk_keys in track(
            optim_chunks, description="Unsharding optim chunks...", disable=quiet
        ):
            unshard_chunk("optim", chunk_path, chunk_keys)

    return model_path, optim_path


def load_keys(
    dir: PathOrStr,
    keys: Iterable[str],
    *,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
) -> Generator[Any, None, None]:
    """
    Load specific keys from a checkpoint.

    .. warning::
        This should only be called in a non-distributed context. Otherwise a :class:`RuntimeError` is raised.

    :param dir: The path/URL to the original checkpoint created via :func:`save_model_and_optim_state()`,
        :func:`save_state_dict`, or one of the other functions in this module.
    :param keys: The keys to load.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.

    :returns: The (unsharded) objects from the checkpoint corresponding to the given keys, in the
        same order as the keys.
    """
    if is_distributed():
        raise RuntimeError("'load_keys' cannot be called in a distributed context")

    dir = normalize_path(dir)
    # validate checkpoint.
    get_checkpoint_metadata(dir)

    keys = list(keys)
    state_dict = _load_unsharded_keys(dir, keys, pre_download=pre_download, work_dir=work_dir)
    for key in keys:
        yield _get_key(state_dict, key, pop=True)


def get_checkpoint_metadata(dir: PathOrStr) -> Metadata:
    """
    Load the metadata from a checkpoint.

    :param dir: The path/URL to the checkpoint.
    """
    dir = normalize_path(dir)
    storage_reader = RemoteFileSystemReader(dir)
    return storage_reader.read_metadata()


def _prepare_env_for_save(
    dir: PathOrStr,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
) -> str:
    dir = normalize_path(dir)

    # Prepare checkpoint folder.
    if save_overwrite:
        if get_fs_local_rank() == 0:
            clear_directory(dir)
    elif not dir_is_empty(dir):
        raise FileExistsError(dir)

    barrier()

    if not is_url(dir):
        if get_fs_local_rank() == 0:
            Path(dir).mkdir(exist_ok=True, parents=True)
        # Ensure the dir exists for all ranks before continuing. This might take a second if we're
        # saving to an NFS drive or something like that.
        wait_for(Path(dir).exists, description=f"Waiting on '{dir}' to be created...")
        barrier()

    return dir


def _prepare_state_dict(
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    del process_group  # I feel like these torch functions should take a process group argument.
    sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=False)
    state_dict: Dict[str, Any] = {
        "model": dist_cp_sd.get_model_state_dict(model, options=sd_options)
    }
    if optim is not None:
        state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(model, optim, options=sd_options)

    return state_dict


def _load_unsharded_keys(
    dir: PathOrStr,
    keys: List[str],
    *,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
) -> Dict[str, Any]:
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    state_dict: Dict[str, Any] = {}
    _load_state_dict(
        state_dict,
        storage_reader=RemoteFileSystemReader(dir, pre_download=pre_download, work_dir=work_dir),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )
    return state_dict


def _get_key(state_dict: Dict[str, Any], key: str, pop: bool = False) -> Any:
    if key in state_dict:
        if pop:
            return state_dict.pop(key)
        else:
            return state_dict[key]

    if "." not in key:
        raise KeyError(key)

    root, key = key.split(".", 1)
    if root not in state_dict:
        raise KeyError(root)

    return _get_key(state_dict[root], key, pop=pop)


def _set_key(state_dict: Dict[str, Any], key: str, value: Any):
    if "." not in key or all(["." in k for k in state_dict.keys()]):
        state_dict[key] = value
        return

    root, key = key.split(".", 1)
    if root not in state_dict:
        raise KeyError(root)

    return _set_key(state_dict[root], key, value=value)