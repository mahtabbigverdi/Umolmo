# Module that can be imported to register all tasks
import dataclasses
import functools
import logging
import os
from typing import List, Dict, Any

import tensorflow_datasets as tfds

from .data_utils import _strip_metadata
from .preprocesssors import *
from .preprocesssors import _preprocess_scifi
from ..tokenizer import build_tokenizer


@dataclasses.dataclass
class TaskSpec:
    name: str
    source: seqio.DataSourceInterface
    preprocessors: List
    style: str
    inference_preprocessors: List = None
    inference_only: bool = False
    decode_image: bool = False
    shuffle_after: Optional[int] = None
    ignore_errors: bool = False


MULTITASK_TFDS_DATA_DIR = "/weka/oe-training-default/mm-olmo/tensorflow_datasets"
if not os.path.exists(MULTITASK_TFDS_DATA_DIR):
    print("Using default gs based TFDS data dir")
    MULTITASK_TFDS_DATA_DIR = "gs://mm-olmo-data"

TASKS: Dict[str, TaskSpec] = {}


def add_task(
    name,
    source: seqio.DataSourceInterface,
    preprocessors: List,
    style: str,
    inf_preprocessor=None,
    inf_only=False,
    decode_image=False,
    shuffle_after=None,
    ignore_errors=False
):
    TASKS[name] = TaskSpec(
        name, source, preprocessors, style, inf_preprocessor, inf_only, decode_image,
        shuffle_after, ignore_errors)


@seqio.map_over_dataset
def add_image_size(ex):
    if ex["image"].dtype == tf.string:
        ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]
    ex["metadata/image_size"] = [img_w, img_h]


@dataclasses.dataclass
class TaskDatasetBuilder:
    """tf.data.Dataset builder for task after shuffling, sharding, and initial model pre-processing
    have been applied"""
    # This class is a simplified and customized version of seqio.Task
    #
    # The main differences are:
    #     1: Does not prefetch by default, which wastes a small amount of RAM if we are using the
    #        dataset in a mixture which can just have its own top-level prefetch
    #     2: Reduce threshold for memory caching which is way too high for image datasets by default
    #     3: Can customize when shuffling occurs to help minimizes RAM usage, in general shuffling
    #        should happen before building image crops and tokenization so the shuffle and
    #        dataset checkpoint take less memory
    #     4: Don't decoding images until after shuffling for the same reason
    #     5: Support splitting with tfds.map_split so we never have to fall back to example sharding
    #        not default at the moment since its not well tested
    #     6: Removes caching/output feature spec stuff from seqio that we don't need

    name: str
    source: Any
    preprocessors: List
    keep_metadata: bool
    shuffle_after: int
    sharding: str = "tfds_split"
    decode_image: bool = False
    ignore_errors: bool = False

    def get_dataset(
        self,  # pytype: disable=signature-mismatch  # overriding-default-value-checks
        sequence_length: Optional[Mapping[str, int]] = None,
        split: str = tfds.Split.TRAIN,
        shuffle: bool = True,
        shuffle_buffer_size: Optional[int] = 1000,
        seed: Optional[int] = None,
        shard_info: Optional[seqio.ShardInfo] = None,
        num_epochs: Optional[int] = 1,
        try_in_mem_cache: bool = True,
        trim_output_features: bool=True
    ) -> tf.data.Dataset:
        source = self.source

        if self.sharding == "seqio":
            if source.supports_arbitrary_sharding:
                shard_data_source = True
            elif shard_info:
                # Whether we should shard at source or on the examples from the source.
                shard_data_source = (
                    len(source.list_shards(split=split)) >= shard_info.num_shards
                )
                logging.info(
                    "Sharding at the %s: %d of %d",
                    "data source" if shard_data_source else "examples",
                    shard_info.index + 1,
                    shard_info.num_shards,
                    )
            else:
                # Call get_dataset on the source without a shard_info.
                shard_data_source = True
                shard_info = None

            if "image" in source.tfds_dataset.info.features:
                if not self.decode_image:
                    source.tfds_dataset._decoders = dict(image=tfds.decode.SkipDecoding())

            if shard_data_source:
                ds = source.get_dataset(
                    split=split, shuffle=shuffle, seed=seed, shard_info=shard_info)
            else:
                ds = source.get_dataset(split=split, shuffle=shuffle, seed=seed)
                ds = ds.shard(shard_info.num_shards, shard_info.index)
        elif self.sharding == "tfds_split":
            # Shard with `tfds.even_splits`, which is seems to be recommended for mult-host training
            # https://github.com/tensorflow/datasets/blob/master/docs/splits.md#tfdseven_splits--multi-host-training
            assert isinstance(self.source, seqio.TfdsDataSource)
            loader: seqio.LazyTfdsLoader = self.source.tfds_dataset
            dataset, data_dir = loader.get_split_params(split)
            shard_split = loader._map_split(split)
            if shard_info and shard_info.num_shards > 1:
                shard_split = tfds.even_splits(shard_split, n=shard_info.num_shards, drop_remainder=False)[shard_info.index]
            else:
                shard_split = shard_split
            read_config = loader.read_config
            read_config.shuffle_seed = seed
            read_config.skip_prefetch = True
            read_config.input_context = None
            # Don't decode images until after shuffling to save RAM
            if "image" in loader.info.features:
                decoders = dict(image=tfds.decode.SkipDecoding())
            else:
                decoders = None
            ds = tfds.load(
                dataset,
                split=shard_split,
                data_dir=data_dir,
                shuffle_files=shuffle,
                download=True,
                try_gcs=True,
                read_config=read_config,
                decoders=decoders
            )
        else:
            raise NotImplementedError(self.sharding)

        num_shards = shard_info.num_shards if shard_info else 1
        if try_in_mem_cache and (
            source.num_input_examples(split)
            and source.num_input_examples(split)
            < 10000 * num_shards
        ):
            logging.info(f"Automatically caching small dataset in memory: {self.name}:{split}")
            ds = ds.cache()

        # We repeat before calling any (potentially) stochastic
        # preprocessors in order to take new samples each epoch.
        if num_epochs != 1:
            ds = ds.repeat(num_epochs)

        preprocessors = [
            seqio.add_kwargs_to_transform(
                _fn,
                sequence_length=sequence_length,
                output_features=None,
            ) for _fn in self.preprocessors
        ]

        with seqio.utils.map_seed_manager(seed):
            for fn in preprocessors[:self.shuffle_after]:
                ds = fn(ds)

            # Strip metadata before shuffling if possible so its doesn't waste space
            if not self.keep_metadata:
                ds = _strip_metadata(ds)

            if shuffle:
                if shuffle_buffer_size is None:
                    raise ValueError("Shuffle is true, but shuffle_buffer_size is None")
                else:
                    ds = ds.shuffle(shuffle_buffer_size, seed=seed)

            for fn in preprocessors[self.shuffle_after:]:
                ds = fn(ds)

        if self.ignore_errors:
            ds = ds.ignore_errors(log_warning=True)

        if trim_output_features:
            ds = seqio.trim_dataset(ds, sequence_length, sequence_length)

        return ds


def get_task(preprocessor, name, is_training, for_inference,
             include_metadata=None, style_override=None) -> TaskDatasetBuilder:
    """Get a builder for task `name` that is pre-processed by `preprocessor`"""

    task_spec = TASKS[name]
    if for_inference is None:
        for_inference = task_spec.inference_only
    elif task_spec.inference_only and not for_inference:
        raise ValueError(f"Inference=only task {task_spec.name} can only be used in inference mode")

    if include_metadata is None:
        include_metadata = for_inference

    if preprocessor is not None:
        style = style_override if style_override else task_spec.style
        preprocessor = preprocessor.build_preprocessor(
            is_training, for_inference, style, include_metadata)
        preprocessor = [preprocessor]
    else:
        preprocessor = []
    task_preprocessors = task_spec.preprocessors
    if for_inference and task_spec.inference_preprocessors is not None:
        task_preprocessors = task_spec.inference_preprocessors
    if isinstance(task_spec.source, seqio.TfdsDataSource):
        from seqio.utils import _TFDS_DATA_DIR_OVERRIDE
        if _TFDS_DATA_DIR_OVERRIDE:
            # Stop annoying override warnings flooding the log
            task_spec.source.tfds_dataset._data_dir = None

    return TaskDatasetBuilder(
        task_spec.name,
        task_spec.source,
        task_preprocessors + preprocessor,
        keep_metadata=include_metadata,
        shuffle_after=(task_spec.shuffle_after if task_spec.shuffle_after
                       else len(task_spec.preprocessors)),
        sharding="seqio",
        decode_image=task_spec.decode_image,
        ignore_errors=task_spec.ignore_errors,
    )


add_task(
    "coco_caption_2017",
    source=seqio.TfdsDataSource(
        tfds_name="coco_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "image/filename": ["image/filename"],
            "image": ["image"],
            "text": ["captions", "text"]
        }),
        functools.partial(flatten_parts, parts=["text"]),
    ],
    inf_preprocessor=[
        functools.partial(rekey, key_map={
            "image/filename": ["image/filename"],
            "image": ["image"],
            "text": ["captions", "text"]
        })
    ],
    style="coco_captioning",
)


add_task(
    "coco_captioning_karpathy",
    source=seqio.TfdsDataSource(
        tfds_name="coco_captioning_karpathy:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(text="captions"),
        functools.partial(flatten_parts, parts=["text"]),
    ],
    inf_preprocessor=[add_coco_url],
    style="coco_captioning",
)


add_task(
    "synth_counting",
    source=seqio.TfdsDataSource(
        tfds_name="synth_counting:0.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[5120:]", "validation": "train[:5120]"}
    ),
    preprocessors=[synth_count_preprocessor],
    inf_preprocessor=[synth_count_inf_preprocessor],
    style="synth_counting",
)


add_task(
    "khan_academy",
    source=seqio.TfdsDataSource(
        tfds_name="khan_academy:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]"}
    ),
    preprocessors=[extract_khan_academy],
    style="khan_academy",
)

for name, src in [
    ("vaia_qa_latex_image_math_subset", seqio.TfdsDataSource(
        tfds_name=f"vaia_qa_latex_image_short_answer:0.1.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "validation"}
    )),
    ("vaia_qa_latex_image_all", seqio.TfdsDataSource(
        tfds_name=f"vaia_qa_latex_image_short_answer:0.1.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "validation"}
    )),
]:
  add_task(
    f"{name}_short_answer",
    source=src,
    preprocessors=[
      remove_is_long,
      remove_has_multiple_parts,
      functools.partial(extract_vaia_qa_latex_image, add_short_answer=True),
    ],
    style="vaia_qa",
  )
  add_task(
    f"{name}_short_answer_first",
    source=src,
    preprocessors=[
      remove_is_long,
      remove_has_multiple_parts,
      functools.partial(extract_vaia_qa_latex_image, add_short_answer=True, set_short_answer_first=True),
    ],
    style="vaia_qa_short_answer_first",
  )
  add_task(
    f"{name}_mc_only_short_answer",
    source=src,
    preprocessors=[
      remove_is_long,
      remove_has_multiple_parts,
      filter_mc,
      functools.partial(extract_vaia_qa_latex_image, add_short_answer=True),
    ],
    style="vaia_qa_short_answer",
  )
  add_task(
    f"{name}_mc_only_short_answer_first",
    source=src,
    preprocessors=[
      remove_is_long,
      remove_has_multiple_parts,
      filter_mc,
      functools.partial(extract_vaia_qa_latex_image, add_short_answer=True, set_short_answer_first=True),
    ],
    style="vaia_qa_short_answer_first",
  )
  add_task(
    f"{name}_image_only_short_answer",
    source=src,
    preprocessors=[
      image_only,
      remove_is_long,
      remove_has_multiple_parts,
      functools.partial(extract_vaia_qa_latex_image, add_short_answer=True),
    ],
    style="vaia_qa_short_answer",
  )
  add_task(
    f"{name}_image_only_short_answer_first",
    source=src,
    preprocessors=[
      image_only,
      remove_is_long,
      remove_has_multiple_parts,
      functools.partial(extract_vaia_qa_latex_image, add_short_answer=True, set_short_answer_first=True),
    ],
    style="vaia_qa_short_answer_first",
  )

add_task(
  "vqa_online",
  source=seqio.TfdsDataSource(
    tfds_name="vqa_online:1.0.1",
    tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"}
  ),
  preprocessors=[
    build_question_with_context,
    extract_vqa_online,
  ],
  style="vqa_online",
)

add_task(
  "vqa_online_gpt_longQ_longA",
  source=seqio.TfdsDataSource(
    tfds_name="vqa_online_gpt_parsed:1.1.0",
    tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"}
  ),
  preprocessors=[
    rename(question="question_long", answer="answer_long"),
    extract_vqa_online,
  ],
  style="vqa_online",
)


add_task(
    "famous_birthdays",
    source=seqio.TfdsDataSource(
        tfds_name="famous_birth_days:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[5120:]", "validation": "train[:5120]"}
    ),
    preprocessors=[
        famous_birthdays_preprocessor,
        functools.partial(name_entity_augmentation, p_high_color=0.0),
    ],
    style="famous_birthdays",
)


add_task(
    "wiki_art",
    source=seqio.TfdsDataSource(
        tfds_name="wiki_art:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[5120:]", "validation": "train[:5120]"}
    ),
    preprocessors=[name_entity_augmentation, wiki_art_preprocessor],
    style="wiki_art",
)

add_task(
    "wiki_art_no_aug",
    source=seqio.TfdsDataSource(
        tfds_name="wiki_art:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[5120:]", "validation": "train[:5120]"}
    ),
    preprocessors=[wiki_art_preprocessor],
    style="wiki_art",
)

add_task(
    "atlas_obscura",
    source=seqio.TfdsDataSource(
        tfds_name="atlas_obscura:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[5120:]", "validation": "train[:5120]"}
    ),
    preprocessors=[
        atlas_obscura_preprocessor,
        mild_color_aug_preprocessor
    ],
    style="atlas_obscura",
)


add_task(
    "clocks",
    source=seqio.TfdsDataSource(
        tfds_name="clocks:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        clocks_preprocessor,
        clock_augmentation
    ],
    style="clocks",
    shuffle_after=0
)


add_task(
    "count_bench",
    source=seqio.TfdsDataSource(
        tfds_name="count_bench:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        count_bench_preprocessor,
    ],
    style="count_bench",
)


add_task(
    "tulu_v2_sft",
    source=seqio.TfdsDataSource(
        tfds_name="allenai__tulu_v2_sft_mixture:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[tulu_preprocessor],
    style="tulu_v2",
)


# Pointing / Point+Count datasets
for task in ["pointing", "point_count", "only_count", "count_then_point", "point_count_random_points", "point_count_random_points_and_length", "point_count_random_order"]:
    style = task
    point_order = "xy"
    
    if task == "pointing":
        is_count = False
        just_count = False
        count_first = False
    elif task == "only_count":
        is_count = False
        just_count = True
        count_first = False
    elif task == "count_then_point":
        is_count = False
        just_count = False
        count_first = True
    elif task == "point_count":
        is_count = True
        just_count = False
        count_first=False
    elif task == "point_count_random_points":
        is_count = True
        just_count = False
        count_first = False
        point_order = "random_points"
        style = "point_count"
    elif task == "point_count_random_points_and_length":
        is_count = True
        just_count = False
        count_first = False
        point_order = "random_points_and_length"
        style = "point_count"
    elif task == "point_count_random_order":
        is_count = True
        just_count = False
        count_first = False
        point_order = "random"
        style = "point_count"
    else:
        raise ValueError(f"Invalid task: {task}")
    add_task(
        task,
        source=seqio.TfdsDataSource(
            tfds_name="pointing:1.0.1",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits={"train": "train", "validation": "validation"}
        ),
        preprocessors=[
            filter_points,
            functools.partial(pointing_preprocessor, with_count=is_count, just_count=just_count, count_first=count_first, point_order=point_order),
            split
        ],
        style=style,
    )
    add_task(
        task + "_eval",  # pointing validation set
        source=seqio.TfdsDataSource(
            tfds_name="pointing:1.0.2",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        ),
        preprocessors=[
            filter_points,
            functools.partial(pointing_preprocessor, with_count=is_count, just_count=just_count, count_first=count_first, point_order=point_order),
            split
        ],
        style=style,
    )
    add_task(
        task + "_high_freq",
        source=seqio.TfdsDataSource(
            tfds_name="count_qa:0.0.2",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits=dict(
                train="train[2048:]",
                validation="train[:2048]"
            )
        ),
        preprocessors=[
            filter_points,
            fix_count_qa,  # Fix a tfrecord bug TODO fix the underlying records
            functools.partial(pointing_preprocessor, with_count=is_count, just_count=just_count, count_first=count_first, point_order=point_order),
            split,
        ],
        style=style,
    )
    add_task(
        "fast_flickr_count_qa_" + task,
        source=seqio.TfdsDataSource(
            tfds_name="fast_flickr_count_qa:1.0.4",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        ),
        preprocessors=[
            functools.partial(count_qa_preprocessor, with_count=is_count, just_count=just_count, count_first=count_first, point_order=point_order),
        ],
        inf_preprocessor=[
            functools.partial(count_qa_preprocessor, with_count=is_count, for_inference=True, point_order=point_order),
        ],
        style=style,
    )
    add_task(
        "countbench_qa_" + task,
        source=seqio.TfdsDataSource(
            tfds_name="countbench_qa:1.2.0",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        ),
        inf_only=True,
        preprocessors=[
            count_qa_preprocessor_inf,
        ],
        style=style,
    )


add_task(
    "countbench_qa",
    source=seqio.TfdsDataSource(
        tfds_name="countbench_qa:1.2.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    inf_only=True,
    preprocessors=[
        count_qa_preprocessor_inf,
    ],
    style="point_count",
)

add_task(
    f"pointing_test",  # pointing set with segmentation ground truths
    source=seqio.TfdsDataSource(
        tfds_name="pointing:1.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        pointing_inf_preprocessor
    ],
    style="pointing",
    inf_only=True,
)


add_task(
    "point_qa",
    source=seqio.TfdsDataSource(
        tfds_name="point_qa:0.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[512:]",
            validation="train[:512]"
        )
    ),
    preprocessors=[extract_point_qa, split],
    style="point_qa",
)

add_task(
    "clocks_no_aug",
    source=seqio.TfdsDataSource(
        tfds_name="clocks:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        clocks_preprocessor
    ],
    style="clocks",
)


add_task(
    "clocks_dbg",
    source=seqio.TfdsDataSource(
        tfds_name="clocks:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        lambda ds: ds.filter(lambda ex: (ex["hour"] == 0)),
        clocks_preprocessor,
    ],
    style="clocks",
)


add_task(
    "wiki_data",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_wiki:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[10240:]", "validation": "train[:5120]", "test": "train[5120:10240]"}
    ),
    preprocessors=[extract_wiki_data],
    style="wiki_data",
)


add_task(
    "wiki_data_name",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_wiki:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[10240:]", "validation": "train[:5120]", "test": "train[5120:10240]"}
    ),
    preprocessors=[
        extract_wiki_data_name,
        mild_color_aug_preprocessor
    ],
    style="wiki_data",
)

add_task(
    "wiki_data_describe",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_wiki:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[10240:]", "validation": "train[:5120]", "test": "train[5120:10240]"}
    ),
    preprocessors=[extract_wiki_data_describe],
    inf_only=True,
    style="wiki_data",
)

add_task(
    "wiki_data_describe",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_wiki:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[10240:]", "validation": "train[:5120]", "test": "train[5120:10240]"}
    ),
    preprocessors=[extract_wiki_data_describe],
    inf_only=True,
    style="wiki_data",
)


for name, src in [
    ("scifi_charts", seqio.TfdsDataSource(
        tfds_name="sci_fi_charts:1.0.6",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]"}
    )),
    ("scifi_table", seqio.TfdsDataSource(
        tfds_name="sci_fi_table:1.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]"}
    )),
    ("scifi_document", seqio.TfdsDataSource(
        tfds_name="sci_fi_document:1.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]"}
    )),
    ("scifi_diagram", seqio.TfdsDataSource(
        tfds_name="sci_fi_diagram:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]"}
    )),
    ("scifi_natural", seqio.TfdsDataSource(
        tfds_name="sci_fi_natural:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[128:]", "validation": "train[:128]"}
    )),
    ("scifi_nutrition", seqio.TfdsDataSource(
        tfds_name="sci_fi_nutrition:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[128:]", "validation": "train[:128]"}
    ))
]:
    add_task(
        name + "_qa",
        source=src,
        preprocessors=[
            remove_no_qa,
            _preprocess_scifi,
            extract_individual_vqa,
        ],
        inf_preprocessor=[
            remove_no_qa, _preprocess_scifi,
            functools.partial(flatten_parts, parts=["question", "answer"]),
            extract_individual_vqa,
        ],
        style=name,
    )
    add_task(
        name + "_qa_split",
        source=src,
        preprocessors=[
            remove_no_qa,
            _preprocess_scifi,
            extract_individual_vqa,
            split
        ],
        inf_preprocessor=[
            remove_no_qa, _preprocess_scifi,
            functools.partial(flatten_parts, parts=["question", "answer"]),
            extract_individual_vqa,
        ],
        style=name,
        )
    add_task(
        name + "_qa_exp",
        source=src,
        preprocessors=[
            remove_no_qa,
            _preprocess_scifi,
            extract_scifi_qa_exp,
            extract_individual_vqa,
        ],
        inf_preprocessor=[
            remove_no_qa, _preprocess_scifi,
            extract_scifi_qa_exp,
            functools.partial(flatten_parts, parts=["question", "answer"]),
            extract_individual_vqa,
        ],
        style=name + "_qa_exp",
    )
    add_task(
        name + "_qa_exp_split",
        source=src,
        preprocessors=[
            remove_no_qa,
            _preprocess_scifi,
            extract_scifi_qa_exp,
            extract_individual_vqa,
            split,
        ],
        inf_preprocessor=[
            remove_no_qa, _preprocess_scifi,
            extract_scifi_qa_exp,
            functools.partial(flatten_parts, parts=["question", "answer"]),
            extract_individual_vqa,
        ],
        style=name + "_qa_exp",
    )
    add_task(
        name + "_exp",
        source=src,
        preprocessors=[
            remove_no_qa,
            _preprocess_scifi,
            scifi_explanation_only,
            extract_individual_vqa,
            split
        ],
        style=name + "_exp"
    )
    add_task(
        name + "_demo",
        source=src,
        preprocessors=[
            remove_no_qa,
            _preprocess_scifi,
            extract_scifi_qa_demo,
            extract_individual_vqa,
            split
        ],
        style="scifi_demo"
    )


add_task(
    "chart_qa_scifi",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label", **{"metadata/is_human": "is_human"}),
        extract_individual_vqa,
    ],
    style="scifi_charts_qa_exp",
)


add_task(
    "chart_qa_prompting",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label", **{"metadata/is_human": "is_human"}),
        chartqa_prompting,
        extract_individual_vqa,
    ],
    style="chart_qa",
)


add_task(
    "chart_qa_prompting_explanation",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label", **{"metadata/is_human": "is_human"}),
        chartqa_explanation,
        extract_individual_vqa,
    ],
    style="chart_qa",
)



add_task(
    "coco_captioning_karpathy_multi",
    source=seqio.TfdsDataSource(
        tfds_name="coco_captioning_karpathy:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[rename(text="captions")],
    inf_preprocessor=[add_coco_url],
    style="coco_captioning",
)


add_task(
    "coco_caption_2017_grouped",
    source=seqio.TfdsDataSource(
        tfds_name="coco_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(
            rekey, key_map={
                "image/filename": ["image/filename"],
                "image": ["image"],
                "text": ["captions", "text"]
            }),
        join_captions
    ],
    style="coco_captioning_multiple",
)


add_task(
    "llava_pretrain",
    source=seqio.TfdsDataSource(
        tfds_name="llava_pretrain:1.0.0",
        tfds_data_dir="gs://mm-olmo-datasets/",
        splits=dict(
            train="train[4096:]",
            validation="train[:4096]"
        )
    ),
    preprocessors=[extract_llava],
    style="web_caption"
)


add_task(
    "rohun_images",
    source=seqio.TfdsDataSource(
        tfds_name="rohun_images:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[],
    style="long_caption",
    inf_only=True
)


add_task(
    "dense_caption_eval",
    source=seqio.TfdsDataSource(
        tfds_name="dense_captioning_eval:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(validation="train")
    ),
    preprocessors=[],
    style="long_caption",
    inf_only=True
)


add_task(
    "dense_caption_eval_dbg",
    source=seqio.TfdsDataSource(
        tfds_name="dense_captioning_eval:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(validation="train")
    ),
    preprocessors=[
        lambda ds: ds.filter(lambda x: x["url"] == "https://explore-multimodal-datasets.s3.us-west-2.amazonaws.com/eval-set/v0/eval-set/a211be07e2c9c722ef75093026a608856bd07ad935ebdedea6f2944b1f2d2b0e.jpg")
    ],
    style="long_caption",
    inf_only=True
)


add_task(
    "dense_caption_sample",
    source=seqio.TfdsDataSource(
        tfds_name="dense_captioning_eval:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            validation="train"
        )
    ),
    preprocessors=[select_dense_caption_sample],
    style="long_caption",
)


add_task(
    "cockatoo_1per_caption_287k",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_1per_caption_287k:1.0.5",
        tfds_data_dir="gs://mm-olmo-data/",
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    ),
    preprocessors=[
        rename(text="caption"),
    ],
    style="long_caption"
)


def _filter_large_ratio(ds):
    return ds.filter(
        lambda x: tf.shape(x["image"])[0] > tf.shape(x["image"])[1]*2
    )


add_task(
    f"cockatoo_dbg",
    source=    seqio.TfdsDataSource(
        tfds_name="cockatoo_476k:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    )
    ,
    preprocessors=[
        _filter_large_ratio,
        extract_caption_and_transcript
    ],
    style=["long_caption", "transcript"]
)


for name, src in [
    ("712k_sept6", seqio.TfdsDataSource(
        tfds_name="cockatoo_712k_sept6:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    )),
    ("476k", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    )),
    ("476k_gpt_captions", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k_gpt_captions:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    )),
    ("100k_of_476k_gpt_captions", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k_gpt_captions:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:105120]",
            validation="train[:5120]"
        )
    )),
    ("200k_of_476k_gpt_captions", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k_gpt_captions:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:205120]",
            validation="train[:5120]"
        )
    )),
    ("300k_of_476k_gpt_captions", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k_gpt_captions:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:305120]",
            validation="train[:5120]"
        )
    )),
    ("400k_of_476k_gpt_captions", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k_gpt_captions:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:405120]",
            validation="train[:5120]"
        )
    )),
    ("400k_of_476k", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:405120]",
            validation="train[:5120]"
        )
    )),
    ("300k_of_476k", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:305120]",
            validation="train[:5120]"
        )
    )),
    ("200k_of_476k", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:205120]",
            validation="train[:5120]"
        )
    )),
    ("100k_of_476k", seqio.TfdsDataSource(
        tfds_name="cockatoo_476k:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:105120]",
            validation="train[:5120]"
        )
    )),
    ("276k", seqio.TfdsDataSource(
        tfds_name="cockatoo:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    )),
    ("180k", seqio.TfdsDataSource(
        tfds_name="cockatoo:1.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[4096:]",
            validation="train[:4096]"
        )
    )),
    ("84k_claude_captions", seqio.TfdsDataSource(
        tfds_name="cockatoo_84k_claude_captions:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[1000:]",
            validation="train[:1000]"
        )
    )),
]:
    add_task(
        f"cockatoo_{name}",
        source=src,
        preprocessors=[extract_caption],
        style="long_caption"
    )
    add_task(
        f"transcript_{name}",
        source=src,
        preprocessors=[extract_transcript],
        style="transcript"
    )
    add_task(
        f"cockatoo_{name}_transcript1",
        source=src,
        preprocessors=[extract_caption_and_transcript1],
        style=["long_caption", "transcript"]
    )

    add_task(
        f"cockatoo_and_transcript_{name}",
        source=src,
        preprocessors=[extract_caption_and_transcript],
        style=["long_caption", "transcript"]
    )

    add_task(
        f"cockatoo_and_all_transcripts_{name}",
        source=src,
        preprocessors=[extract_caption_and_all_transcripts],
        style=["long_caption", "transcript", "transcript", "transcript"]
    )

    add_task(
        f"cockatoo_all_transcripts_{name}",
        source=src,
        preprocessors=[extract_all_transcripts],
        style="transcript"
    )
    add_task(
        f"cockatoo_transcripts_{name}",
        source=src,
        preprocessors=[extract_transcript],
        style="transcript"
    )


TFRECORD_IMAGE_TEXT_FEATURES = {
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'text':tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}


add_task(
    "laion400m",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": os.path.join("gs://unified-io-2-us-east/", "pretrain-datasets", "laion400m", "1.0.0", "laion400m-train*"),
        },
        feature_description=TFRECORD_IMAGE_TEXT_FEATURES,
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "image": ["image"],
            "text": ["text"]
        }),
    ],
    style="laion",
)


add_task(
    "laion_2B",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": os.path.join(MULTITASK_TFDS_DATA_DIR, "laion2b_en", "1.0.0", "laion2b_en-train*"),
        },
        feature_description=TFRECORD_IMAGE_TEXT_FEATURES,
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "image": ["image"],
            "text": ["text"]
        }),
    ],
    style="laion",
)


add_task(
    "region_caption_vg",
    source=seqio.TfdsDataSource(
        tfds_name="vg:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[region_captions_to_dense],
    style="region_captions",
)


add_task(
    "pdfa_eng_wds",
    source=seqio.TfdsDataSource(
        tfds_name="pdfa_eng_wds:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(max_words, max_words=400),
        format_pdfa_eng_wds
    ],
    style="pdfa_eng_wds",
)


add_task(
    "idl_words",
    source=seqio.TfdsDataSource(
        tfds_name="idl_words:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[],
    style="idl_words",
)



open_image_v6_keys_to_features = {
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'image_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'detection/label':tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'detection/bbox':tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32,  allow_missing=True),
    'detection/num':tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'vrd/sub_label': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'vrd/obj_label': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'vrd/sub_bbox':tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32, allow_missing=True),
    'vrd/obj_bbox':tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32, allow_missing=True),
    'vrd/relation': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'vrd/num':tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'cap/cap_caption': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'cap/num':tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'seg/masks': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'seg/num':tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'seg/label': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True),
    'seg/bbox': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32, allow_missing=True),
}


add_task(
    "localized_narratives_v6",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": os.path.join(MULTITASK_TFDS_DATA_DIR, "open_image_v6", "1.0.0", "open_image_v6-train*"),
        },
        feature_description=open_image_v6_keys_to_features,
    ),
    preprocessors=[extract_localized_narrative],
    style="localized_narratives",
)


add_task(
    "lvis_objects",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="lvis:1.2.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        extract_lvis,
        region_captions_to_dense,
    ],
    style="lvis_objects",
)


add_task(
    "open_images_with_objects",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": os.path.join(MULTITASK_TFDS_DATA_DIR, "open_image_v6", "1.0.0", "open_image_v6-train*"),
        },
        feature_description=open_image_v6_keys_to_features,
    ),
    preprocessors=[
        extract_open_images_boxes,
        region_captions_to_dense,
    ],
    style="visual_narratives_with_objects",
)


add_task(
    "cockatoo_with_acc_476k_gpt_captions",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_with_acc_476k_gpt_captions:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    ),
    preprocessors=[accuracy_conditioned_joint],
    inf_preprocessor=[functools.partial(accuracy_conditioned_joint, is_eval=True)],
    style=None
)


add_task(
    "dense_caption_eval_with_acc",
    source=seqio.TfdsDataSource(
        tfds_name="dense_captioning_eval:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(validation="train")
    ),
    preprocessors=[functools.partial(accuracy_conditioned_joint, is_eval=True)],
    style="long_caption",
    inf_only=True
)

# ************************
# VQA Datasets
# ************************

add_task(
    "science_qa_img",
    source=seqio.TfdsDataSource(
        tfds_name="science_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        image_only,
        rename(answer_idx="answer"),
        build_question_with_hint,
        format_multiple_choice_qa
    ],
    style="science_qa",
)


add_task(
    "tabwmp_da",
    source=seqio.TfdsDataSource(
        tfds_name="tab_mwp:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "dev", "test": "test"}
    ),
    preprocessors=[
        rename(**{"text": "answer", "metadata/example_id": "example_id"})
    ],
    style="tabwmp_da",
)


add_task(
    "figure_qa",
    source=seqio.TfdsDataSource(
        tfds_name="figure_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train1", "validation": "validation1", "test": "no_annot_test1"}
    ),
    preprocessors=[extract_figureqa, extract_individual_vqa],
    style="figure_qa",
)

add_task(
    "figure_qa_zero_shot",
    source=seqio.TfdsDataSource(
        tfds_name="figure_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train1", "validation": "validation1", "test": "no_annot_test1"}
    ),
    preprocessors=[extract_figureqa, convert_figureqa_answer, extract_individual_vqa],
    style="figure_qa",
)


add_task(
    "plot_qa",
    source=seqio.TfdsDataSource(
        tfds_name="plot_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[extract_figureqa, extract_individual_vqa],
    inf_preprocessor=[
        extract_figureqa,
        functools.partial(flatten_parts, parts=["questions", "answer", "question_id"]),
        extract_individual_vqa
    ],
    style="plot_qa",
)


add_task(
    "ai2_diagram",
    source=seqio.TfdsDataSource(
        tfds_name="ai2_diagram:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]", "test": "test"}
    ),
    preprocessors=[
        rename(choices="answer_texts", answer_idx="correct_answer"),
        format_multiple_choice_qa
    ],
    style="ai2_diagram",
)


add_task(
    "ai2_diagram_v2",
    source=seqio.TfdsDataSource(
        tfds_name="ai2_diagram_v2:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        rename(choices="answer_texts", answer_idx="correct_answer"),
        format_ai2d
    ],
    style="ai2_diagram",
)


add_task(
    "ai2_diagram_v2_transparent",
    source=seqio.TfdsDataSource(
        tfds_name="ai2_diagram_v2_transparent:1.0.5",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        rename(choices="answer_texts", answer_idx="correct_answer"),
        format_ai2d
    ],
    style="ai2_diagram",
)

# ai2_diagram_v2 mixed with addiitonal abc label questions with transparent box.
# Shares the same image split as ai2_diagram_v2.
add_task(
    "ai2_diagram_v2_mix_transparent",
    source=seqio.TfdsDataSource(
        tfds_name="ai2_diagram_v2_mix_transparent:1.0.6",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={
            "train": "train_mix",
            "validation": "validation_mix",
            "test": "test_mix", # test should only use either transparent or opaque
            # "test": "test_opaque",
        }
    ),
    preprocessors=[
        rename(choices="answer_texts", answer_idx="correct_answer"),
        format_ai2d
    ],
    style="ai2_diagram",
)

add_task(
    "ai2_diagram_v2_mix_transparent_one_style",
    source=seqio.TfdsDataSource(
        tfds_name="ai2_diagram_v2_mix_transparent:1.0.6",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={
            "train": "train_mix",
            "validation": "validation_mix",
            "test": "test_mix", # test should only use either transparent or opaque
            # "test": "test_opaque",
        }
    ),
    preprocessors=[
        rename(choices="answer_texts", answer_idx="correct_answer"),
        functools.partial(format_ai2d, variable_style=False),
    ],
    style="ai2_diagram",
)


for src, test_sets in [
    ["refclef_unc", ["testA", "testB", "testC", "testAB", "testBC"]],
    ["refcoco_unc", ["testA", "testB"]],
    ["refcocoplus_unc", ["testA", "testB"]],
    ["refcocog_umd", ["test"]],
]:
    if "coco" in src:
        add_url = [add_coco_url]
    else:
        add_url = []
    splits = {x: x for x in test_sets}
    splits.update({"train": "train", "validation": "val"})
    add_task(
        src,
        source=seqio.TfdsDataSource(
            tfds_name=f"{src}:1.0.2",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits=splits
        ),
        preprocessors=[refexp],
        inf_preprocessor=add_url + [
            refexp_inf,
            # Flatten objects
            functools.partial(flatten_parts, parts=["refexp", "metadata/bbox"]),
            # Flatten expressions
            functools.partial(flatten_parts, parts=["refexp"])
        ],
        style="refexp",
        decode_image=True,
    )
    add_task(
        src + "_pointing",
        source=seqio.TfdsDataSource(
            tfds_name=f"{src}:1.0.2",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits=splits
        ),
        preprocessors=[refexp_pointing],
        inf_preprocessor=add_url + [
            refexp_pointing_inf,
            functools.partial(flatten_parts, parts=["refexp", "metadata/bbox", "metadata/mask", "metadata/answer"]),
            functools.partial(flatten_parts, parts=["refexp"])
        ],
        decode_image=True,
        style="refexp_pointing",
    )


# FIXME
add_task(
    "ai2_diagram_test",
    source=seqio.TfdsDataSource(
        tfds_name="ai2_diagram:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]", "test": "test"}
    ),
    preprocessors=[
        rename(choices="answer_texts", answer_idx="correct_answer"),
        format_multiple_choice_qa
    ],
    style="ai2_diagram",
)


add_task(
    "gqa",
    source=seqio.TfdsDataSource(
        tfds_name="gqa:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        functools.partial(format_gqa, is_balanced=True),
        extract_individual_vqa,
    ],
    inf_preprocessor=[
        functools.partial(format_gqa, is_balanced=True),
        extract_individual_vqa,
    ],
    style="gqa",
)


add_task(
    "gqa_multi",
    source=seqio.TfdsDataSource(
        tfds_name="gqa:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        functools.partial(format_gqa, is_balanced=True, flatten=False),
        extract_individual_vqa,
    ],
    inf_preprocessor=[
        functools.partial(format_gqa, is_balanced=True, flatten=False),
        extract_individual_vqa,
    ],
    style="gqa",
)


add_task(
    "text_vqa",
    source=seqio.TfdsDataSource(
        tfds_name="text_vqa:1.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(
            rekey, key_map={
                "image": ["image"],
                "questions": ["question"],
                "answers": ["answers"],
                "id": ["question_id"]
            }),
        extract_individual_vqa,
    ],
    style="text_vqa",
)


add_task(
    "okvqa",
    source=seqio.TfdsDataSource(
        tfds_name="ok_vqa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        rename(example_id="question_id"),
        add_coco_url,
        extract_individual_vqa,
    ],
    style="okvqa",
)

add_task(
    "a_okvqa_da",
    source=seqio.TfdsDataSource(
        tfds_name="a_ok_vqa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(**{
            "example_id": "question_id",
            "answers": "direct_answers",
            "metadata/difficult_direct_answer": "difficult_direct_answer"
        }),
        extract_individual_vqa,
    ],
    inf_preprocessor=[
        filter_difficult_direct_answer,
        rename(**{
            "example_id": "question_id",
            "answers": "direct_answers",
            "metadata/difficult_direct_answer": "difficult_direct_answer"
        }),
        add_coco_url,
        extract_individual_vqa,
    ],
    style="a_okvqa_da",
)


add_task(
    "a_okvqa_mc",
    source=seqio.TfdsDataSource(
        tfds_name="a_ok_vqa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(**{
            "example_id": "question_id",
            "metadata/difficult_direct_answer": "difficult_direct_answer",
            "answer_idx": "correct_choice_idx"
        }),
        add_coco_url,
        format_multiple_choice_qa,
    ],
    style="a_okvqa_mc",
)


add_task(
    "dv_qa",
    source=seqio.TfdsDataSource(
        tfds_name="dv_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val_easy"}
    ),
    preprocessors=[
        extract_figureqa,
        extract_individual_vqa,
    ],
    inf_preprocessor=[
        extract_figureqa,
        flatten_vqa,
        extract_individual_vqa
    ],
    style="dv_qa",
)


@seqio.map_over_dataset
def add_image_question_example_id(ex):
    key = tf.strings.join([ex["question"], "\n\n", ex["image"]])
    ex["metadata/example_id"] = tf.strings.to_hash_bucket(key, 2**30)
    return ex


add_task(
    "chart_qa",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label", **{"metadata/is_human": "is_human"}),
        add_image_question_example_id,
        extract_individual_vqa,
    ],
    style="chart_qa",
)


add_task(
    "chart_qa_ex",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label", **{"metadata/is_human": "is_human"}),
        extract_individual_vqa,
    ],
    style="scifi_charts_qa_exp",
)


add_task(
    "chart_qa_weighted",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label", **{"metadata/is_human": "is_human"}),
        extract_individual_vqa,
        functools.partial(reweight_chartqa, human=2*20901/(20901+7398), aug=2*7398/(20901+7398)),
    ],
    style="chart_qa",
)


add_task(
    "chart_qa_human",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label"),
        add_image_question_example_id,
        filter_human,
        extract_individual_vqa,
    ],
    style="chart_qa",
)


add_task(
    "chart_qa_aug",
    source=seqio.TfdsDataSource(
        tfds_name="chart_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[
        rename(question="query", answer="label"),
        filter_aug,
        extract_individual_vqa,
    ],
    style="chart_qa",
)


add_task(
    "doc_qa",
    source=seqio.TfdsDataSource(
        tfds_name="doc_qa:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[fix_doqa_url, extract_individual_vqa],
    style="doc_qa",
)


add_task(
    "ocr_qa",
    source=seqio.TfdsDataSource(
        tfds_name="ocr_vqa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[extract_individual_vqa],
    inf_preprocessor=[flatten_vqa, extract_individual_vqa],
    style="ocr_vqa",
)


add_task(
    "st_qa",
    source=seqio.TfdsDataSource(
        tfds_name="st_vqa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train[1024:]", "validation": "train[:1024]", "test": "test"}
    ),
    preprocessors=[extract_individual_vqa],
    inf_preprocessor=[extract_individual_vqa],
    style="st_qa",
)


add_task(
    "tally_qa",
    source=seqio.TfdsDataSource(
        tfds_name="tally_qa:1.0.2",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "test"}
    ),
    preprocessors=[
        extract_tally_qa,
        extract_individual_vqa
    ],
    inf_preprocessor=[
        extract_tally_qa,
        flatten_vqa,
        extract_individual_vqa
    ],
    style="tally_qa",
)


add_task(
    "info_qa",
    source=seqio.TfdsDataSource(
        tfds_name="info_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[extract_individual_vqa],
    style="info_qa",
)

add_task(
    "android_control",
    source=seqio.TfdsDataSource(
        tfds_name="android_control:2.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "train", "validation": "val", "test": "test"}
    ),
    preprocessors=[extract_android_control],
    style="android_control",
)

for mode in ["ll", "hl", "hl_ll", "hl_cot"]:
    add_task(
        f"android_control_{mode}",
        source=seqio.TfdsDataSource(
            tfds_name="android_control:2.0.0",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits={"train": "train", "validation": "val", "test": "test"}
        ),
        preprocessors=[functools.partial(extract_andriod_control_inf, mode=mode)],
        style="android_control",
    )


map_coco_vqa = functools.partial(rekey, key_map={
    "image": ["image"],
    "questions": ["vqa", "questions"],
    "answers": ["vqa", "answers"],
    "id": ["vqa", "id"],
    "metadata/image_url": ["metadata/image_url"],
    "metadata/image_id": ["image/filename"],
})


add_task(
    "coco_2017_vqa",
    source=seqio.TfdsDataSource(
        tfds_name="coco_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        add_coco_url,
        map_coco_vqa,
        flatten_vqa,
        extract_individual_vqa
    ],
    style="vqa2",
)


add_task(
    "cockatoo_qa",
    source=seqio.TfdsDataSource(
        tfds_name="cockatoo_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[5120:]",
            validation="train[:5120]"
        )
    ),
    preprocessors=[rename(text="answer")],
    style=None,
)


add_task(
    "synthetic_qa_v3",
    source=seqio.TfdsDataSource(
        tfds_name="synthetic_qa_v3:0.0.4",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[2048:]",
            validation="train[:2048]"
        )
    ),
    preprocessors=[extract_cockatoo_qa_v2, prefix_how_many_messages],
    style="synthetic_qa",
)


add_task(
    "synthetic_qa_v3_style_tag",
    source=seqio.TfdsDataSource(
        tfds_name="synthetic_qa_v3:0.0.4",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[2048:]",
            validation="train[:2048]"
        )
    ),
    preprocessors=[extract_cockatoo_qa_v2, prefix_how_many_messages],
    style="llm_qa",
)


add_task(
    "synthetic_qa_v3_as_user_qa",
    source=seqio.TfdsDataSource(
        tfds_name="synthetic_qa_v3:0.0.4",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[2048:]",
            validation="train[:2048]"
        )
    ),
    preprocessors=[extract_cockatoo_qa_v2, prefix_how_many_messages],
    style="user_qa",
)


add_task(
    "synthetic_qa_v3_multi_turn",
    source=seqio.TfdsDataSource(
        tfds_name="synthetic_qa_v3:0.0.4",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[2048:]",
            validation="train[:2048]"
        )
    ),
    preprocessors=[extract_cockatoo_qa_v2, filter_single_turn, prefix_how_many_messages],
    style="synthetic_qa",
)


NE_SHARDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

for i in NE_SHARDS:
    add_task(
        f"named_entity{i}",
        source=seqio.TfdsDataSource(
            tfds_name=f"named_entities_qa_{i}_of_18:1.0.0",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits=dict(
                train="train[1024:]",
                validation="train[:1024]"
            )
        ),
        preprocessors=[filter_named_entity, extract_named_entity, extract_individual_vqa],
        inf_preprocessor=[
            filter_named_entity,
            extract_named_entity,
            flatten_vqa,
            extract_individual_vqa
        ],
        style="named_entity",
        ignore_errors=True
    )


add_task(
    "user_qa",
    source=seqio.TfdsDataSource(
        tfds_name="user_qa:0.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits=dict(
            train="train[2048:]",
            validation="train[:2048]"
        )
    ),
    preprocessors=[extract_cockatoo_qa_v2, prefix_how_many_messages],
    style="user_qa",
)

add_task(
    "user_questions_for_elo",
    source=seqio.TfdsDataSource(
        tfds_name="user_questions_for_elo:0.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[functools.partial(extract_individual_vqa, test=True)],
    inf_only=True,
    style="demo",
)


def _filter_by_id(ds, prediction_file, max_seq_len):
    with open(prediction_file) as f:
        predictions = json.load(f)
    is_long = []
    lens = []
    tokenizer = build_tokenizer("hf-Qwen/Qwen2-7B")
    for pred in predictions:
        n_tokens = len(tokenizer.encode(pred["prediction"]))
        lens.append(n_tokens)
        if n_tokens >= max_seq_len:
            is_long.append(pred["example_id"])
    is_long = tf.constant(is_long)
    logging.info(f"Filtering for {len(is_long)} ids")
    return ds.filter(lambda ex: tf.reduce_any(ex["example_id"] == is_long))



add_task(
    "user_questions_for_elo",
    source=seqio.TfdsDataSource(
        tfds_name="user_questions_for_elo:0.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[functools.partial(extract_individual_vqa, test=True)],
    inf_only=True,
    style="demo",
)


add_task(
    "user_questions_for_elo_long",
    source=seqio.TfdsDataSource(
        tfds_name="user_questions_for_elo:0.0.3",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(_filter_by_id, prediction_file="/weka/oe-training-default/chrisc/cockatoo/models/uber-model-v11/70b-335-30k-3.2-resume8k-noopt/predictions-ck20000-user_questions_for_elo-test/predictions.json", max_seq_len=230),
        functools.partial(extract_individual_vqa, test=True)
    ],
    inf_only=True,
    style="demo",
)


add_task(
    "coco_2014_vqa",
    source=seqio.TfdsDataSource(
        tfds_name="coco_2014_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        add_coco_url,
        map_coco_vqa,
        flatten_vqa,
        extract_individual_vqa
    ],
    inf_preprocessor=[
        add_coco_url,
        map_coco_vqa,
        flatten_vqa,
        extract_individual_vqa
    ],
    style="vqa2",
)


add_task(
    "coco_2014_vqa_multi",
    source=seqio.TfdsDataSource(
        tfds_name="coco_2014_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        add_coco_url,
        map_coco_vqa,
        extract_individual_vqa
    ],
    inf_preprocessor=[
        add_coco_url,
        map_coco_vqa,
        flatten_vqa,
        extract_individual_vqa
    ],
    style="vqa2",
)


add_task(
    "coco_2017_vqa_multi",
    source=seqio.TfdsDataSource(
        tfds_name="coco_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        add_coco_url,
        map_coco_vqa,
        extract_individual_vqa
    ],
    inf_preprocessor=[
        add_coco_url,
        map_coco_vqa,
        flatten_vqa,
        extract_individual_vqa
    ],
    style="vqa2",
)


add_task(
    "vqa_v2_test",
    source=seqio.TfdsDataSource(
        tfds_name="coco_test_all:1.0.1",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "image": ["image"],
            "questions": ["vqa", "questions"],
            "answers": ["vqa", "answers"],
            "id": ["vqa", "id"],
        }),
        flatten_vqa,
        functools.partial(extract_individual_vqa, test=True)
    ],
    style="vqa2",
    inf_only=True
)

# ************************
# Eval-only Datasets
# ************************

add_task(
    "seed_bench_test",
    source=seqio.TfdsDataSource(
        tfds_name="seed_bench:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        format_multiple_choice_qa,
    ],
    style="a_okvqa_mc",
    inf_only=True
)


add_task(
    "pope_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="pope:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        add_coco_url,
        extract_individual_vqa
    ],
    style="vqa2",
    inf_only=True
)


MME_SOURCE = seqio.TfdsDataSource(
    tfds_name="mme:1.0.0",
    tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
)


add_task(
    "mme_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=MME_SOURCE,
    preprocessors=[
        functools.partial(flatten_parts, parts=["questions", "answers"]),
        rename(question="questions", answer="answers"),
        extract_individual_vqa,
    ],
    style="vqa2",
    inf_only=True
)

add_task(
    "real_world_qa_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="real_world_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(
            format_multiple_style_qa,
            types=['multiple_choice', 'short_answer'],
            styles=['a_okvqa_mc', 'vqa2'],
            default_style="a_okvqa_mc",
        ),
    ],
    style=None,
    inf_only=True
)

add_task(
    "real_world_qa_no_instruction",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="real_world_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(
            functools.partial(format_multiple_style_qa, strip_instruction=True),
            types=['multiple_choice', 'short_answer'],
            styles=['a_okvqa_mc', 'vqa2'],
            default_style="a_okvqa_mc",
        ),
    ],
    style=None,
    inf_only=True
)

add_task(
    "real_world_qa_dbg",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="real_world_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(
            format_multiple_style_qa,
            types=['multiple_choice', 'short_answer'],
            styles=['user_qa', 'user_qa'],
            default_style="user_qa",
        ),
    ],
    style=None,
    inf_only=True
)


add_task(
    "mmmu",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="mmmu:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"train": "dev"},
    ),
    preprocessors=[
        rename(img_type="metadata/img_type"),
        functools.partial(
            extract_mmmu,
            types=['multiple-choice', 'open'],
            styles=['a_okvqa_mc', 'vqa2'],
            default_style="a_okvqa_mc",
        ),
    ],
    style=None,
)


add_task(
    "mmmu_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="mmmu:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "validation", "test": "test"},
    ),
    preprocessors=[
        rename(img_type="metadata/img_type"),
        extract_mmmu,
    ],
    style=None,
    inf_only=True
)

for style in ["vaia_qa", "vaia_qa_short_answer_first", "vqa_online", ]:
  add_task(
    f"mmmu_test_{style}",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
      tfds_name="mmmu:1.0.0",
      # tfds_name="mmmu_khan_academy:1.0.1",
      tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
      splits={"validation": "validation", "test": "test", "dev": "dev"},
    ),
    preprocessors=[
      rename(img_type="metadata/img_type"),
      extract_mmmu_cot,
    ],
    style=style,
    inf_only=True
  )


add_task(
    "math_vista_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="math_vista:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "testmini", "test": "test"},
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "id": ["id"],
            "query": ["query"],
            "image": ["image"],
            "choices": ["choices"],
            "answer": ["answer"],
            "metadata/question_type": ["question_type"],
            "metadata/answer_type": ["answer_type"],
            "metadata/precision": ["precision"],
            "metadata/split": ["metadata/split"],
        }),
        functools.partial(extract_math_vista, styles=['a_okvqa_mc', 'vqa2']),
    ],
    style=None,
    inf_only=True
)


add_task(
    "math_vista_v2",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="math_vista:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "testmini", "test": "test"},
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "id": ["id"],
            "query": ["query"],
            "image": ["image"],
            "choices": ["choices"],
            "answer": ["answer"],
            "metadata/question_type": ["question_type"],
            "metadata/answer_type": ["answer_type"],
            "metadata/precision": ["precision"],
            "metadata/split": ["metadata/split"],
        }),
        reformat_math_vista,
        functools.partial(
            extract_math_vista,
            styles=['a_okvqa_mc', 'vqa2'],
        ),
    ],
    style=None,
    inf_only=True
)


MM_BENCH_SRC = seqio.TfdsDataSource(
    tfds_name="mmbench:1.0.0",
    tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    splits={"validation": "dev", "test": "test"},
)

add_task(
    "mmbench_test",
    source=MM_BENCH_SRC,
    preprocessors=[format_mmbench],
    style="a_okvqa_mc",
    inf_only=True
)


add_task(
    "sugar_crepe_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="sugar_crepe:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        add_coco_url,
        functools.partial(flatten_parts, parts=["choices", "answer_idx", "metadata/answer_type"]),
        format_multiple_choice_qa,
    ],
    style="a_okvqa_mc",
    inf_only=True
)


add_task(
    "blink_test",
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
        tfds_name="blink:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
        functools.partial(rekey, key_map={
            "id": ["id"],
            "question": ["prompt"],
            "image": ["image_concat"],
            "choices": ["choices"],
            "answer_idx": ["answer_idx"],
            "metadata/subtask": ["metadata/subtask"],
            "metadata/question": ["question"],
        }),
        format_multiple_choice_qa,
        output_options,
    ],
    style="a_okvqa_mc",
    inf_only=True
)

add_task(
    "oscarbench_qa",
    source=seqio.TfdsDataSource(
        tfds_name="oscarbench_qa:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "val"}
    ),
    preprocessors=[oscar_preprocessor],
    style="oscarbench_qa"

)

add_task(
    "charxiv",
    source=seqio.TfdsDataSource(
        tfds_name="charxiv:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "validation", "test": "test"}
    ),
    preprocessors=[charxiv_preprocessor, extract_individual_vqa],
    inf_preprocessor=[
        charxiv_preprocessor,
        functools.partial(flatten_parts, parts=["question", "answer"]),
        extract_individual_vqa,
    ],
    style="charxiv",
)

add_task(
    "charxiv_descriptive",
    source=seqio.TfdsDataSource(
        tfds_name="charxiv:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "validation", "test": "test"}
    ),
    preprocessors=[charxiv_descriptive_preprocessor, extract_individual_vqa],
    inf_preprocessor=[
        charxiv_descriptive_preprocessor,
        functools.partial(flatten_parts, parts=["question", "answer"]),
        extract_individual_vqa,
    ],
    style="charxiv_descriptive",
)

add_task(
    "charxiv_reasoning",
    source=seqio.TfdsDataSource(
        tfds_name="charxiv:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "validation", "test": "test"}
    ),
    preprocessors=[charxiv_reasoning_preprocessor, extract_individual_vqa],
    style="charxiv_reasoning",
)

for tablevqa_name in ["fintabnetqa", "vwtq", "vwtq_syn"]:
    add_task(
        tablevqa_name,
        source=seqio.TfdsDataSource(
            tfds_name=f"{tablevqa_name}:1.0.0",
            tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
            splits={"validation": "test[:125]", "test": "test"}
        ),
        preprocessors=[tablevqa_preprocessor, extract_individual_vqa],
        style=tablevqa_name,
    )

add_task(
    "vtabfact",
    source=seqio.TfdsDataSource(
        tfds_name="vtabfact:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "test[:125]", "test": "test"}
    ),
    preprocessors=[vtabfact_preprocessor, extract_individual_vqa],
    style="vtabfact",
)

add_task(
    "nutrition_fact",
    source=seqio.TfdsDataSource(
        tfds_name="nutrition_fact:1.0.0",
        tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
        splits={"validation": "test", "test": "test"}
    ),
    preprocessors=[nutrition_fact_preprocessor, extract_individual_vqa],
    inf_preprocessor=[
            nutrition_fact_preprocessor,
            functools.partial(flatten_parts, parts=["question", "answer"]),
            extract_individual_vqa,
        ],
    style="nutrition_fact",
    inf_only=True
)

for k in ["chart_qa", "info_qa", "doc_qa", "text_vqa", "coco_2014_vqa",
          "ai2_diagram_v2_mix_transparent", "chart_qa_human"]:
    TASKS[k + "_demo"] = dataclasses.replace(TASKS[k], style="demo")
