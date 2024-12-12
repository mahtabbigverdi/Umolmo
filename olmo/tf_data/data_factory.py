'''
Dataset factory to load data from huggingface and others.
'''
import dataclasses
import logging
from typing import List, Optional

import numpy as np
import tensorflow as tf

from .data_utils import add_segment_ids
from .dataset_sizes import get_dataset_size
from .tasks import get_task
from ..multimodal_preprocessor import MultiModalPreprocessor
import seqio

from ..torch_util import get_global_rank

log = logging.getLogger(__name__)


TASK_MAPPING = {
    "pixmo_cap_with_transcripts": "cockatoo_and_transcript_712k_sept6",
    "pixmo_cap": "cockatoo_712k_sept6",

    "pixmo_cap_qa": "synthetic_qa_v3_as_user_qa",

    "pixmo_clocks": "clocks",

    "pixmo_ask_model_anything": "user_qa",

    "pixmo_pointing_high_freq": "pointing_high_freq",
    "pixmo_point_count_high_freq": "point_count_high_freq",
    "pixmo_pointing": "pointing",
    "pixmo_counting": "point_count",
    "pixmo_pointing_explanations": "point_qa",

    "pixmo_count_counting": "fast_flickr_count_qa_point_count",
    "pixmo_count_pointing": "fast_flickr_count_qa_pointing",

    "pixmo_docs_other": "scifi_document_qa",
    "pixmo_docs_tables": "scifi_table_qa",
    "pixmo_docs_diagrams": "scifi_table_qa",
    "pixmo_docs_charts": "scifi_table_qa",
}


@dataclasses.dataclass
class SeqioDataset:
    mixture_or_task_name: str
    seq_len: int
    global_batch_size: int
    max_crops: int = None
    is_training: bool = False
    for_inference: bool = False
    split: str = 'train'
    shuffle: bool = True
    num_epochs: int = None
    drop_remainder: bool = True
    seed: int = None
    pack: bool = False
    use_custom_packing_ops: bool = False
    use_memory_cache: bool = False
    shuffle_buffer_size: Optional[int] = None
    different_host_mixture_seeds: bool = True
    disable_autotune: bool = True
    trim_output_features: bool = True

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def get_task_feature_lengths_dict(self, max_crops):
        if self.max_crops is not None:
            assert self.max_crops >= max_crops
            max_crops = self.max_crops
        return dict(
            target_tokens=self.seq_len,
            loss_masks=self.seq_len,
            images=max_crops,
            image_positions=max_crops,
            image_input_idx=max_crops,
            is_training=self.is_training
        )

    def build(self, preprocessor: MultiModalPreprocessor, shard_id, num_shards):
        shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)
        task_feature_lengths_dict = self.get_task_feature_lengths_dict(
            preprocessor.get_max_total_crops())

        seed = self.seed
        assert seed is not None

        batch_size = self.global_batch_size // num_shards

        if isinstance(self.mixture_or_task_name, (dict, list, tuple)):
            if isinstance(self.mixture_or_task_name, dict):
                items = self.mixture_or_task_name.items()
            else:
                items = self.mixture_or_task_name
            items = [(TASK_MAPPING.get(k, k), v) for k, v in items]
            task_list = []
            for task, weight in items:
                task = get_task(preprocessor, task, self.is_training, self.for_inference)
                task_list.append((task, weight))
            mixture_or_task = task_list
        else:
            mixture_or_task = get_task(
                preprocessor, TASK_MAPPING.get(self.mixture_or_task_name, self.mixture_or_task_name),
                self.is_training, self.for_inference)

        in_memory_shuffle = self.shuffle
        if not self.drop_remainder:
            # Used if we want to evaluate on an eval dataset without dropping any examples.
            # To do this, we pad the dataset with dummy examples marked as invalid in their
            # metadata so we can still get fixed-sized batches.
            assert self.num_epochs is not None
            assert not self.pack
            assert not isinstance(mixture_or_task, list), "Inference datasets cannot be mixtures"
            logging.info(
                f"Initializing inf. dataset {mixture_or_task.name}: replica_batch_size={batch_size}"
                f' seed={seed}, sharding={shard_info.index}/{shard_info.num_shards}'
            )
            ds = mixture_or_task.get_dataset(
                sequence_length=task_feature_lengths_dict,
                split=self.split,
                shuffle=in_memory_shuffle,
                num_epochs=self.num_epochs,
                seed=seed,
                try_in_mem_cache=self.use_memory_cache,
                trim_output_features=self.trim_output_features
            )

            try:
                n = len(ds)
            except TypeError:
                dataset_len = get_dataset_size(self.mixture_or_task_name, self.split)
                logging.info(f"Setting dataset len to {dataset_len} based on DATASET_SIZES")
                n = dataset_len
                ds = tf.data.experimental.assert_cardinality(n)(ds)

            remainder = n % self.global_batch_size
            if remainder > 0:
                n_to_pad = self.global_batch_size - remainder
            else:
                n_to_pad = 0
            assert "metadata/valid" not in ds.element_spec
            def add_valid(x):
                x["metadata/valid"] = True
                return x
            def add_invalid(x):
                x["metadata/valid"] = False
                return x
            ds = ds.map(add_valid)
            if n_to_pad > 0:
                to_pad = ds.take(1).map(add_invalid).cache().repeat(n_to_pad)
                ds = ds.concatenate(to_pad)

            # Shard after padding to ensure shards are the same length
            ds = ds.shard(num_shards=num_shards, index=shard_id)

            ds = preprocessor.get_post_mixing_preprocessor()(
                ds, task_feature_lengths=task_feature_lengths_dict)
            data_iter = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Make it possible for client to get the size of the batched/sharded dataset with `len()`
            new_len = (n + n_to_pad) // self.global_batch_size
            data_iter = tf.data.experimental.assert_cardinality(new_len)(data_iter)
        else:
            if isinstance(mixture_or_task, list):
                total_rate = sum(x[1] for x in mixture_or_task)
                mixture_or_task = [(task, r/total_rate) for task, r in mixture_or_task]
                sorted_tasks: List[seqio.Task] = sorted(mixture_or_task, key=lambda x: -x[1])

                if self.different_host_mixture_seeds and shard_info:
                    # If each process has the same seed they will draw from the datasets in the same
                    # order, which can make the global batches very non-random if there are
                    # many processes each with a small batch size. To fix this, we give each host
                    # a different seed based on its rank to use when mixing
                    mix_seed = seed + shard_info.index*4397
                else:
                    mix_seed = seed

                logging.info(
                    f"Initializing mixture: replica_batch_size={batch_size} seed={seed}, "
                    f"mix_seed={mix_seed}, sharding={shard_info.index}/{shard_info.num_shards} rates:"
                )
                for task, rate in sorted_tasks:
                    logging.info(f"\t{task.name}: {rate:0.4f}")

                datasets = []
                rates = []
                for task, rate in sorted_tasks:
                    assert rate > 0
                    datasets.append(task.get_dataset(
                        task_feature_lengths_dict,
                        split=self.split,
                        shuffle=self.shuffle,
                        seed=seed,
                        shard_info=shard_info,
                        num_epochs=self.num_epochs,
                        try_in_mem_cache=self.use_memory_cache,
                        trim_output_features=self.trim_output_features
                    ))
                    rates.append(rate)

                # If any of the sub-tasks have subsegment_ids, we need to ensure all the tasks have
                # a subsegment_ids field so they can be mixed
                if any("subsegment_ids" in ds.element_spec for ds in datasets):
                    for ix, ds in enumerate(datasets):
                        if "subsegment_ids" not in ds.element_spec:
                            datasets[ix] = add_segment_ids(ds)

                ds = tf.data.Dataset.sample_from_datasets(
                    datasets, rates, seed=mix_seed, stop_on_empty_dataset=False)
            else:
                logging.info(
                    f"Initializing dataset {mixture_or_task.name}: replica_batch_size={batch_size}"
                    f' seed={seed}, sharding={shard_info.index}/{shard_info.num_shards}'
                )
                ds = mixture_or_task.get_dataset(
                    task_feature_lengths_dict,
                    split=self.split,
                    shuffle=self.shuffle,
                    seed=seed,
                    shard_info=shard_info,
                    num_epochs=self.num_epochs,
                    try_in_mem_cache=self.use_memory_cache,
                    trim_output_features=self.trim_output_features
                )
            data_iter = preprocessor.get_post_mixing_preprocessor()(
                ds, task_feature_lengths=task_feature_lengths_dict)
            data_iter = data_iter.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(2)

        # Following https://github.com/google-research/big_vision/blob/b8dab6e4de3436849415f37c591399c93b1eaf39/big_vision/input_pipeline.py#L228
        # These options try to stop tf datasets from eating all our RAM if we are using a
        # large mixture
        # This options are used by default in some google codebases
        # For example: (https://github.com/google-research/big_vision/blob/b8dab6e4de3436849415f37c591399c93b1eaf39/big_vision/input_pipeline.py#L228)
        # They don't seem to harm throughput and can save RAM so we use them as well
        options = tf.data.Options()
        options.experimental_optimization.inject_prefetch = False
        options.threading.max_intra_op_parallelism = 1
        if self.disable_autotune:
            # Following https://www.tensorflow.org/datasets/performances
            # This reduces RAM and checkpoint size by a lot
            options.autotune.enabled = False
        data_iter = data_iter.with_options(options)

        return data_iter
