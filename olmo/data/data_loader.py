from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import omegaconf
from torch.utils.data import DataLoader, DistributedSampler

from olmo.config import BaseConfig
from olmo.data.dataset import DeterministicDataset
from olmo.data.get_dataset import get_dataset_by_name
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture
from olmo.models.molmo.molmo import MolmoConfig
from olmo.torch_util import get_global_rank, get_world_size

log = logging.getLogger(__name__)


@dataclass
class RootSizeMixture(BaseConfig):
    rate: float
    mixture: Dict[str, Optional[float]]


@dataclass
class DataConfig(BaseConfig):
    """Configuration for a torch `DataLoader`"""

    dataset: Optional[str] = None
    """Dataset name, will be used for `get_dataset_by_name`"""

    mixture: Optional[Dict[str, float]] = None
    """Mixture of dataset names and sampling rates"""

    root_size_mixture: Optional[List[RootSizeMixture]] = None
    """Mixture-of-mixtures where sub-mixtures rates are determined by the root dataset size"""

    split: str = omegaconf.MISSING
    """Dataset split to load"""

    seed: int = omegaconf.MISSING
    """Dataset seed for shuffling and augmentation"""

    pad: Optional[str] = "to_max"
    """How to pad in the collator"""

    sequence_length: Optional[int] = None
    """Max sequence length to truncate examples to in the Collator"""

    shuffle: Optional[bool] = True
    """Should the data be shuffled"""

    # DataLoader args
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0

    def build_eval_dataloader(
        self,
        model_config: MolmoConfig,
        batch_size: int,
        for_inference: bool,
        include_metadata: bool = None,
        pad_batches: bool = False,
        max_steps_for_padding=None
    ) -> DataLoader:
        assert self.mixture is None and self.root_size_mixture is None
        log.info(f"Loading eval dataset: {self.dataset}/{self.split}")
        if include_metadata is None:
            include_metadata = for_inference

        dataset = get_dataset_by_name(self.dataset, self.split)
        n_pad = 0
        if pad_batches:
            assert not self.drop_last
            global_batch_size = batch_size*get_world_size()
            n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
            if max_steps_for_padding:
                n_steps = min(n_steps, max_steps_for_padding)
            if n_steps*global_batch_size > len(dataset):
                # Pad the dataset so that it can produce enough batches of `global_batch_size` size
                # to cover the entire dataset without dropping any examples
                # We need this if evaluating FSDP models since they will need all devices to get
                # exactly the same number of batches
                n_pad = (n_steps*global_batch_size) - len(dataset)

        preprocessor = model_config.build_preprocessor(
            for_inference=for_inference, is_training=False)
        dataset = DeterministicDataset(
            dataset=dataset,
            seed=self.seed,
            preprocessor=preprocessor,
            n_pad=n_pad
        )
        sampler = DistributedSampler(
            dataset,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=model_config.build_collator(
                self.sequence_length, self.pad, include_metadata=include_metadata),
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=self.pin_memory,
            prefetch_factor=None if self.num_workers == 0 else self.prefetch_factor,
            persistent_workers=False if self.num_workers == 0 else self.persistent_workers,
            timeout=self.timeout,
        )

    def build_train_dataloader(
        self, 
        model_config: MolmoConfig,
        global_batch_size: int, 
        device=None
    ) -> DataLoader:
        if device is None:
            device = "cpu"
        if self.dataset:
            datasets = [get_dataset_by_name(
                self.dataset, self.split)]
            rates = [1]
        else:
            if self.mixture:
                mixture = {}
                for name, rate in self.mixture.items():
                    log.info(f"Loading train dataset {name}/{self.split}")
                    mixture[name] = (get_dataset_by_name(name, self.split), rate)
            else:
                mixture = {}
                for root_size_mixture in self.root_size_mixture:
                    group_datasets = {}
                    for name, as_size in root_size_mixture.mixture.items():
                        log.info(f"Loading train dataset {name}/{self.split}")
                        dataset = get_dataset_by_name(name, self.split)
                        if as_size is not None:
                            size = as_size
                        else:
                            size = len(dataset)
                        group_datasets[name] = (dataset, np.sqrt(size))
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*root_size_mixture.rate)
                                    for name, (ds, r) in group_datasets.items()})

            total_rate = sum(x[1] for x in mixture.values())
            mixture = sorted(mixture.items(), key=lambda x: x[0])
            rates = [rate/total_rate for (_, (_, rate)) in mixture]
            datasets = [ds for (_, (ds, _)) in mixture]
            log.info("Sampling rates:")
            names = list(x[0] for x in mixture)
            for ix in np.argsort(rates)[::-1]:
                log.info(f"{names[ix]}: {100*rates[ix]:0.2f}")

        preprocessor = model_config.build_preprocessor(
            for_inference=False, is_training=True)
        datasets = [DeterministicDataset(ds, preprocessor, self.seed) for ds in datasets]

        dataset = IterableDatasetMixture(
            datasets=datasets,
            mixture_rates=rates,
            global_batch_size=global_batch_size,
            seed=self.seed,
            shuffle=self.shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=dataset.device_batch_size,
            drop_last=self.drop_last,
            collate_fn=model_config.build_collator(
                self.sequence_length, self.pad, include_metadata=False),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=None if self.num_workers == 0 else self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            timeout=self.timeout,
        )

