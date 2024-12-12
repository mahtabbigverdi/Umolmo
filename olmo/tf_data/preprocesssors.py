import hashlib
import json
import math
from functools import reduce
from typing import Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf
import seqio
import gin

from .data_utils import flatten_parts, stateless_permutation, stateless_shuffle
from .. import config


def get_from_dict(data, keys):
    """Iterate nested dictionary"""
    return reduce(dict.get, keys, data)

def get_blank_image():
    image = tf.zeros([224, 224, 3], dtype=tf.uint8)
    image = tf.expand_dims(image, 0)[:1]
    return image


@seqio.utils.map_over_dataset
def rekey(x, key_map=None):
    """Replace the feature keys according to the mapping in `key_map`.
    For example, if the dataset returns examples of the format:
    {'foo': 'something', 'bar': 'something else'}
    and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
    examples with the format
    {'boo': 'something', 'spar': 'something else'}
    If a mapping is to an empty key or None, set the new key to an empty string.
    Args:
        x: an example to process.
        key_map: dictionary mapping new keys to original keys
    Returns:
        A preprocessed example with the format listed above.
    """
    if key_map:
        out = {}
        for new_key, old_key in key_map.items():
            if isinstance(old_key, list):
                out[new_key] = get_from_dict(x, old_key)
            else:
                out[new_key] = x[old_key]
        return out
    return x


def rename(**kwargs):
    @seqio.map_over_dataset
    def _fn(x):
        updates = {}
        for new_key, old_key in kwargs.items():
            if isinstance(old_key, list):
                val = x[old_key[0]]
                for k in old_key[1:-1]:
                    val = val[k]
                updates[new_key] = val.pop(old_key[-1])
            else:
                updates[new_key] = x.pop(old_key)
        x.update(updates)
        return x
    return _fn


def extract_transcripts(ds):
    ds = flatten_parts(ds, ["transcripts"])
    def _map(ex):
        return dict(
            image=ex["image"],
            text=ex["transcripts"],
            url=ex["url"]
        )
    return ds.map(_map)


@seqio.map_over_dataset
def extract_caption_and_all_transcripts(ex):
    transcripts = tf.random.shuffle(ex["transcripts"])[:3]
    weight = 1.0 / tf.cast(tf.shape(transcripts)[0], tf.float32)
    return dict(
        image=ex["image"],
        text=tf.concat([tf.expand_dims(ex["caption"], 0), transcripts], 0),
        url=ex["url"],
        text_weights=tf.pad(
            tf.ones((1,), dtype=tf.float32), [[0, tf.shape(transcripts)[0]]],
            constant_values=weight),
    )


@seqio.map_over_dataset
def extract_all_transcripts(ex):
    transcripts = tf.random.shuffle(ex["transcripts"])[:3]
    weight = 3.0 / tf.cast(tf.shape(transcripts)[0], tf.float32)
    return dict(
        image=ex["image"],
        text=transcripts,
        url=ex["url"],
        text_weights=tf.fill((tf.shape(transcripts)[0],), weight),
    )


@seqio.map_over_dataset
def extract_transcript(ex):
    transcripts = tf.random.shuffle(ex["transcripts"])
    return dict(
        image=ex["image"],
        text=transcripts[0],
        url=ex["url"],
    )


@seqio.map_over_dataset
def extract_caption(ex):
    caption = ex["caption"]
    if len(caption.shape) > 0:
        ex["text"] = caption[0]
    else:
        ex["text"] = caption
    return ex


@seqio.map_over_dataset
def extract_joint_captions(ex):
    caption = ex["caption"]
    if len(caption.shape) > 0:
        caption = caption[0]
    _ix = tf.random.uniform((), 0, tf.shape(ex["transcripts"])[0], dtype=tf.int32)
    _ix = _ix % tf.shape(ex["transcripts"])[0]
    return dict(
        image=ex["image"],
        text=tf.stack([caption, ex["mistral_caption"], ex["transcripts"][_ix]], 0),
        url=ex["url"]
    )


@seqio.map_over_dataset()
def extract_caption_and_transcript1(ex):
    caption = ex["caption"]
    if len(caption.shape) > 0:
        caption = caption[0]
    return dict(
        image=ex["image"],
        text=tf.stack([caption, ex["transcripts"][0]], 0),
        url=ex["url"]
    )


@seqio.map_over_dataset(num_seeds=1)
def extract_caption_and_transcript(ex, seed):
    caption = ex["caption"]
    if len(caption.shape) > 0:
        caption = caption[0]
    _ix = tf.random.stateless_uniform((), seed, 0, tf.shape(ex["transcripts"])[0], dtype=tf.int32)
    return dict(
        image=ex["image"],
        text=tf.stack([caption, ex["transcripts"][_ix]], 0),
        url=ex["url"]
    )


@seqio.map_over_dataset
def caption_transcript_augmented(ex, sequence_length):
    caption = ex["caption"]
    if len(caption.shape) > 0:
        caption = caption[0]
    image = ex["image"]
    properties = []

    do_augmentation = sequence_length["is_training"]
    # do_augmentation = False

    # Keep this off, it screws up OCR
    # do_hflip = (tf.random.uniform(()) > 0.2 and do_augmentation)
    do_hflip = False
    if do_hflip:
        image = image[:, ::-1]

    # Mild color jitter
    do_color = (tf.random.uniform(()) > 0.5 and do_augmentation)
    if do_color:
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)

    # Mild affine transformation
    do_affine = (tf.random.uniform(()) > 0.5 and do_augmentation)
    if do_affine and do_augmentation:
        shift_x = tf.random.uniform((), -10, 10) * 0
        shift_y = tf.random.uniform((), -10, 10) * 0
        shear_x = tf.random.uniform((), -2, 2)
        shear_y = tf.random.uniform((), -2, 2)
        rotation = tf.random.uniform((), -6, 6)
        max_scale = 1.1
        scale = tf.random.uniform((), 0.8, max_scale)
        center = tf.cast(tf.shape(image), tf.float32)/2

        image = tf.keras.ops.image.affine_transform(
            image,
            tf.stack(get_affine_matrix(
                [center[0], center[1]],
                rotation,
                [shift_x, shift_y],
                1/scale,
                [shear_x, shear_y]
            ) + [0., 0.]),
            interpolation='bilinear',
            fill_mode='constant',
            fill_value=1.,
            data_format='channels_last'
        )

    properties = tf.stack([
        ("[hflip]" if do_hflip else ""),
        ("[color]" if do_color else ""),
        ("[affine]" if do_affine else "")
    ])
    properties = tf.boolean_mask(properties, tf.strings.length(properties) > 0)
    prompt = tf.strings.reduce_join(properties, separator=" ")
    ix = tf.random.uniform((), 0, tf.shape(ex["transcripts"])[0], dtype=tf.int32)
    out = dict(
        image=image,
        text=tf.stack([caption, ex["transcripts"][ix]], 0),
        url=ex["url"],
        prompt=prompt,
    )
    # out["metadata/unaugmented_image"] = image
    return out


def extract_caption_and_transcript_hflip(ds):

    # Just in case they are ordered somehow in Matt's data
    @seqio.map_over_dataset
    def _shuffle_transcripts(_ex):
        _ex["transcripts"] = tf.random.shuffle(_ex["transcripts"])
        _ex["hflip"] = tf.random.uniform((), 0, 3, dtype=tf.int32)
        return _ex

    ds = _shuffle_transcripts(ds)

    # Build a 3x long dataset with each individual transcript so we iterate through
    # each transcript
    @seqio.map_over_dataset
    def _with_transcript(ex, _ix):
        caption = ex["caption"]
        if len(caption.shape) > 0:
            caption = caption[0]
        hflip = ex["hflip"] == _ix
        if hflip:
            ex["image"] = ex["image"][:, ::-1]
            style = ["long_caption_flipped", "transcript_flipped"]
        else:
            style = ["long_caption", "transcript"]
        return dict(
            image=ex["image"],
            text=tf.stack([caption, ex["transcripts"][_ix]], 0),
            url=ex["url"],
            style=style
        )

    joint_ds = _with_transcript(ds, 0)
    for i in range(1, 3):
        joint_ds = joint_ds.concatenate(_with_transcript(ds, i))

    return joint_ds


@seqio.map_over_dataset
def extract_llava(ex, sequence_length, output_features):
    tf.assert_equal(tf.shape(ex['conversations']['value'])[0], 2)
    prompt = ex['conversations']['value'][0]
    text = ex['conversations']['value'][1]
    ex.pop('conversations')
    ex["text"] = text
    ex["prompt"] = prompt
    return ex


def extract_localized_narrative(ds):
    ds = ds.filter(lambda ex: tf.shape(ex["cap/cap_caption"])[0] > 0)
    def _map(ex):
        return dict(
            image=ex["image"],
            text=tf.strings.reduce_join(ex["cap/cap_caption"], separator="\n")
        )
    return ds.map(_map)


def float_to_text(val):
    return tf.strings.as_string(tf.cast(val * 100, tf.int32))


@seqio.map_over_dataset
def extract_vqa(ex):
    questions = ex["vqa"]["questions"]
    answers = ex["vqa"]["answers"]
    answers = tf.strings.reduce_join(answers, 1, separator="; ")
    qas = tf.strings.reduce_join(tf.stack([questions, answers], 1), separator=" ")
    return dict(
        image=ex["image"],
        text=tf.strings.reduce_join(qas, separator="\n")
    )


@seqio.map_over_dataset
def coco_image_id_from_path(ex):
    image_id = tf.strings.substr(ex["image/filename"], 0, tf.strings.length(ex["image/filename"])-4)
    ex["image_id"] = tf.strings.to_number(image_id)
    return ex


@seqio.map_over_dataset
def add_coco_url(ex):
    """Turns a COCO path into a URL, which can then be used in visualizations"""
    path = ex["image/filename"]
    if not tf.strings.regex_full_match(path, ".*/.*"):
        prefix = tf.strings.regex_replace(path, "COCO_", "")
        prefix = tf.strings.regex_replace(prefix, "_[0-9]+.jpg", "")
        path = tf.strings.join([prefix, path], separator="/")

    # images are hosted by the COCO website here
    url = tf.strings.join(["https://s3.us-east-1.amazonaws.com/images.cocodataset.org/", path])
    ex["metadata/image_url"] = url
    ex["metadata/image_id"] = path
    return ex


def flatten_vqa(ds):
    parts = ["questions", "answers"]
    for k in ["id", "question_id"]:
        if k in ds.element_spec:
            parts.append(k)
    return flatten_parts(ds, parts)


def format_gqa(ds, is_balanced=True, flatten=True):
    if is_balanced:
        ds = ds.filter(lambda x: tf.reduce_any(x["questions"]["is_balanced"]))
        def _filter_qs(ex):
            qs = ex["questions"]
            mask = qs["is_balanced"]
            qs = {k: tf.boolean_mask(v, mask) for k, v in qs.items()}
            ex["questions"] = qs
            return ex
        ds = ds.map(_filter_qs)

    if flatten:
        ds = flatten_parts(ds, ["questions"])

    def _rename(ex):
        out = ex["questions"]
        out["image"] = ex["image"]
        out["image_id"] = ex["image_id"]
        return out
    return ds.map(_rename)


@seqio.map_over_dataset
def fix_doqa_url(x):
    x["image_url"] = tf.strings.regex_replace(x["image_url"], "gs://", "")
    return x


def _add_metadata(ex):
    out = {}
    if "id" in ex:
        out["metadata/example_id"] = ex["id"]
    elif "example_id" in ex:
        out["metadata/example_id"] = ex["example_id"]
    elif "question_id" in ex:
        out["metadata/example_id"] = ex["question_id"]
    if "image_url" in ex:
        out["metadata/image_url"] = ex["image_url"]
    for k, v in ex.items():
        if k.startswith("metadata/"):
            out[k] = v
    return out


def image_only(ds):
    return ds.filter(lambda x: x["has_image"])


def filter_difficult_direct_answer(ds):
    return ds.filter(lambda x: not x["difficult_direct_answer"])


@seqio.map_over_dataset()
def format_ai2d(ex, variable_style=True):
    abc = tf.constant(list("abcdefg".upper()))
    out = dict(image=ex["image"])
    out.update(_add_metadata(ex))

    options = ex["choices"]
    # >= 3 in case of none of the above like answers
    n_options = tf.shape(ex["option_is_abc"])[0]
    if ex["abc_label"] and tf.reduce_sum(tf.cast(ex["option_is_abc"], tf.int32)) >= (n_options - 1):
        # The image labels are always upper, so use upper in the answer ptions
        options = tf.where(
            ex["option_is_abc"],
            tf.strings.upper(options),
            options
        )
        short_options = options
        style = "ai2_diagram_no_letter"
    else:
        short_options = abc[:tf.shape(options)[0]]
        options = tf.stack([short_options, options,], 1)
        options = tf.strings.reduce_join(options, axis=-1, separator=": ")
        style = "ai2_diagram"

    options = tf.strings.reduce_join(options, separator="\n")
    out["question"] = ex["question"]
    out["options"] = options
    if variable_style:
        out["style"] = style
    if ex["answer_idx"] < 0:
        out["text"] = "?"
    else:
        out["text"] = short_options[ex["answer_idx"]]
    out["metadata/answer_idx"] = ex["answer_idx"]
    tf.debugging.assert_equal(tf.reduce_any(tf.strings.regex_full_match(options, ".*\|\|\|.*")), False)
    out["metadata/option_names"] = tf.strings.reduce_join(short_options, separator="|||")
    out["metadata/has_transparent_box"] = ex.get("has_transparent_box", tf.constant(False))
    out["metadata/abc_label"] = ex["abc_label"]
    return out


@gin.configurable()
@seqio.map_over_dataset()
def format_multiple_choice_qa(ex, option_format="abc"):
    assert option_format == "abc"
    abc = tf.constant(list("abcdefg".upper()))
    out = dict(image=ex["image"])
    out.update(_add_metadata(ex))
    options = ex["choices"]
    short_options = abc[:tf.shape(options)[0]]
    options = tf.stack([short_options, options,], 1)
    options = tf.strings.reduce_join(options, axis=-1, separator=": ")
    options = tf.strings.reduce_join(options, separator="\n")
    out["question"] = ex["question"]
    out["options"] = options
    if ex["answer_idx"] < 0:
        out["text"] = "?"
    else:
        out["text"] = short_options[ex["answer_idx"]]
    out["metadata/answer_idx"] = ex["answer_idx"]
    tf.debugging.assert_equal(tf.reduce_any(tf.strings.regex_full_match(options, ".*\|\|\|.*")), False)
    out["metadata/option_names"] = tf.strings.reduce_join(short_options, separator="|||")
    if "problem_id" in ex:
        out["metadata/example_id"] = ex["problem_id"]
    # out["metadata/option_names"] = tf.RaggedTensor.from_row_lengths(short_options, tf.shape(short_options))
    # out["metadata/option_names"] = short_options
    return out


@seqio.map_over_dataset()
def output_options(ex):
    ex["metadata/options"] = ex["options"]
    return ex


@seqio.map_over_dataset()
def extract_tally_qa(ex):
    questions = ex.pop("questions")
    ex["questions"] = questions["question"]
    ex["answers"] = tf.strings.as_string(questions["answer"])
    ex["question_id"] = questions["question_id"]
    ex["metadata/image_id"] = ex["image_id"]
    return ex


@seqio.map_over_dataset()
def count_bench_preprocessor(ex):
    return {
        "image": ex["image"],
        "text": tf.strings.as_string(ex["number"]),
        "object": ex["noun"],
        "question": tf.strings.join([
            "How many ", ex["noun"], " are there?"
        ]),
        "metadata/count": ex["number"],
    }


def filter_human(ds):
    return ds.filter(lambda x: x["is_human"])


def filter_aug(ds):
    return ds.filter(lambda x: not x["is_human"])


@seqio.map_over_dataset()
def reweight_chartqa(ex, human, aug):
    is_human = ex["metadata/is_human"]
    ex["text_weights"] = human if is_human else aug
    return ex


@seqio.map_over_dataset()
def chartqa_prompting(ex):
    question = tf.strings.join([ex["question"], " Answer:"])
    return dict(
        image=ex["image"],
        question=question,
        answer=ex["answer"]
    )


@seqio.map_over_dataset()
def chartqa_explanation(ex):
    question = tf.strings.join([ex["question"], " Explanation:"])
    out = {
        "image": ex["image"],
        "question": question,
        "answer": ex["answer"],
    }
    out.update({k: v for k, v in ex.items() if k.startswith("metadata/")})
    return out


@gin.configurable()
@seqio.map_over_dataset(num_seeds=1)
def _preprocess_scifi(ex, seed, shuffle_questions=True):
    if "qa_pairs" in ex:
        q = ex["qa_pairs"]
    else:
        q = ex["qa"]
    if shuffle_questions:
        ix = stateless_permutation(tf.shape(q["question"])[0], seed)
    else:
        ix = tf.range(tf.shape(q["question"])[0])
    return dict(
        example_id=ex["image_path"],
        image=ex["image"],
        question=tf.gather(q["question"], ix),
        explanation=tf.gather(q["explanation"], ix),
        answer=tf.gather(q["answer"], ix),
    )

@seqio.map_over_dataset
def scifi_explanation_only(ex):
    return dict(
        image=ex["image"],
        question=ex["question"],
        answer=ex["explanation"],
    )


def filter_named_entity(ds):
    @seqio.map_over_dataset
    def _load_image(ex):
        ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
        return ex

    ds = _load_image(ds)
    return ds.filter(lambda x: tf.reduce_min(tf.shape(x["image"])[:2]) >= 32)


@seqio.map_over_dataset()
def extract_named_entity(ex):
    qs = ex["questions"]
    return {
        "image": ex["image"],
        "metadata/image_url": ex["url"],
        "metadata/entity": ex["entity"],
        "questions": qs["question"],
        "answers": qs["answer"],
    }

@gin.configurable()
def extract_individual_vqa(ds, test=False, answer_mode="best"):

    @seqio.map_over_dataset(num_seeds=1)
    def _extract(ex, seed):
        if "questions" in ex:
            question = ex["questions"]
        else:
            question = ex["question"]
        out = dict(
            image=ex["image"],
            question=question,
        )
        out.update(_add_metadata(ex))
        out["metadata/question"] = question
        if ex.get("answers") is not None:
            out["metadata/references"] = tf.strings.reduce_join(ex["answers"], separator="\n")
        elif ex.get("answer") is not None:
            out["metadata/references"] = ex["answer"]

        if not test:
            if "answer" in ex:
                answer = ex["answer"]
            else:
                answer = ex["answers"]
            if answer.dtype in [tf.int32, tf.int64]:
                answer = tf.strings.as_string(answer)
            if len(answer.shape) == 1 and tf.shape(answer)[0] == 0:
                answer = tf.expand_dims("", 0)
            if len(answer.shape) == len(question.shape):
                pass
            # Handle questions with multiple answers
            elif answer_mode == "random":
                assert len(answer.shape) == 1
                answer = answer[tf.random.stateless_uniform((), seed, 0, tf.shape(answer)[0], dtype=tf.int32)]
            elif answer_mode == "first":
                if len(answer.shape) == 1:
                    answer = tf.py_function(lambda x: np.min(x), Tout=tf.TensorSpec((), tf.string), inp=[answer])
                    answer = tf.ensure_shape(answer, ())
                else:
                    answer = tf.numpy_function(
                        lambda x: np.array([min(y) for y in x]),
                        inp=[answer],
                        Tout=tf.string
                    )
                    answer = tf.ensure_shape(answer, [None])
            elif answer_mode == "best":
                def _get_best(_answer):
                    vals, _, counts = tf.unique_with_counts(_answer)
                    count_thresh = tf.reduce_max(counts)
                    vals = tf.boolean_mask(vals, counts >= count_thresh)
                    return vals[tf.random.stateless_uniform((), seed, 0, tf.shape(vals)[0], dtype=tf.int32)]
                if len(answer.shape) == 1:
                    answer = _get_best(answer)
                elif isinstance(answer, tf.RaggedTensor):
                    n = tf.shape(answer)[0]
                    answer_arr = tf.TensorArray(dtype=tf.string, size=n, element_shape=())
                    for i in range(n):
                        answer_arr = answer_arr.write(i, _get_best(answer[i]))
                    answer = answer_arr.stack()
                else:
                    answer = tf.map_fn(_get_best, answer)
            elif answer_mode == "all_segments":
                out["text"] = answer
            elif answer_mode == "all_segments_weighted":
                out["text"] = answer
                out["text_weights"] = 1.0 / tf.cast(tf.shape(answer)[-1], tf.float32)
            elif answer_mode == "all":
                if len(answer.shape) == 1:
                    answer = stateless_shuffle(answer, seed)
                    answer = tf.strings.reduce_join(answer, separator="\n", axis=-1)
                elif isinstance(answer, tf.RaggedTensor):
                    n = tf.shape(answer)[0]
                    answer_arr = tf.TensorArray(dtype=tf.string, size=n, element_shape=())
                    for i in range(n):
                        answer_arr = answer_arr.write(i, tf.strings.reduce_join(tf.random.shuffle(answer[i]), separator="\n", axis=-1))
                    answer = answer_arr.stack()
                else:
                    answer = tf.map_fn(tf.random.shuffle, answer)
                    answer = tf.strings.reduce_join(answer, separator="\n", axis=-1)
            else:
                raise NotImplementedError()
            out["text"] = answer
        return out
    return _extract(ds)


@seqio.map_over_dataset()
def extract_khan_academy(ex):
    return dict(
        image=ex["image"],
        image_url=ex["image_url"],
        prompt="Answer this question",
        text=ex["gptResponse"]
    )

@seqio.map_over_dataset()
def extract_vaia_qa_latex_image(ex, add_short_answer=False, set_short_answer_first=False):
    if ex["has_image"]:
        image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
        image = tf.expand_dims(image, 0)[:1]
    else:
        # image = get_blank_image() # blank image
        image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
        image = tf.expand_dims(image, 0)[:0]
    img_h = tf.shape(image)[1]
    img_w = tf.shape(image)[2]

    if add_short_answer:
        if set_short_answer_first:
            answer = tf.strings.join(["Answer: ", ex["short_answer"], "\n\n", ex["answer"]])
        else:
            answer = tf.strings.join([ex["answer"], "\n\n", "Answer: ", ex["short_answer"]])
    else:
        answer = ex["answer"]
    out = dict(
        image=image, # 4-d tensor
        text=answer,
        prompt=tf.strings.join([ex["latex_question"], "\n"]),
    )
    out["metadata/images"] = image
    out.update(_add_metadata(ex))
    out["metadata/batch_id"] = ex["batch_id"]
    out["metadata/image_size"] = [img_w, img_h]
    return out

@seqio.map_over_dataset()
def extract_vqa_online(ex):
    out = dict(
        image=ex["image"],
        prompt=tf.strings.join([ex["question"], "\n"]),
        text=ex["answer"]
    )
    out.update(_add_metadata(ex))
    out["metadata/row_id"] = ex["row_id"]
    return out


@seqio.map_over_dataset()
def extract_scifi_joint(ex):
    if "qa_pairs" in ex:
        q = ex["qa_pairs"]
    else:
        q = ex["qa"]
    prompts = tf.concat([["Describe this image in detail."], q["question"]], 0)
    responses = tf.concat([ex["summary"][None], q["answer"]], 0)
    return dict(
        image=ex["image"],
        prompt=prompts,
        text=responses,
    )


def remove_no_qa(ds):
    def _filter(ex):
        if "qa_pairs" in ex:
            q = ex["qa_pairs"]
        else:
            q = ex["qa"]
        return tf.shape(q["question"])[0] > 0
    return ds.filter(_filter)


@seqio.map_over_dataset()
def extract_scifi_qa_exp(ex):
    return dict(
        image=ex["image"],
        question=ex["question"],  # Array of questions
        answer=tf.strings.join([ex["explanation"], " Answer: ", ex["answer"]]),
    )


@seqio.map_over_dataset(num_seeds=1)
def extract_scifi_qa_demo(ex, seed):
    # if tf.random.stateless_uniform((), 0, 1) > 0.5:
    answer = tf.strings.join([ex["explanation"], " Answer: ", ex["answer"]])
    # else:
    #     answer = ex["explanation"]
    return dict(
        image=ex["image"],
        question=ex["question"],  # Array of questions
        answer=answer,
    )


def deg2rad(x):
    return x*math.pi/180.0


def get_affine_matrix(center, angle, translate, scale, shear):
    # From https://github.com/pytorch/vision/blob/f96c42fca53230057b16941b078a0a9eee06e20f/torchvision/transforms/functional.py#L1006
    rot = deg2rad(angle)
    sx = deg2rad(shear[0])
    sy = deg2rad(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = tf.cos(rot - sy) / tf.cos(sy)
    b = -tf.cos(rot - sy) * tf.tan(sx) / tf.cos(sy) - tf.sin(rot)
    c = tf.sin(rot - sy) / tf.cos(sy)
    d = -tf.sin(rot - sy) * tf.tan(sx) / tf.cos(sy) + tf.cos(rot)

    matrix = [a, b, 0.0, c, d, 0.0]
    matrix = [x * scale for x in matrix]
    # Apply inverse of center translation: RSS * C^-1
    matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
    matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
    # Apply translation and center : T * C * RSS * C^-1
    matrix[2] += cx + tx
    matrix[5] += cy + ty
    return matrix


def quantize_point(coor, max_dim, mode="percent-precision-1"):
    max_dim = tf.cast(max_dim, tf.float32)
    coor = tf.cast(coor, tf.float32)
    x = (coor / max_dim)
    if mode == "percent-precision-1":
        return tf.strings.as_string(x*100, precision=1)
    elif mode == "zero_to_one":
        return tf.strings.as_string(x, precision=3)
    elif mode == "1k":
        return tf.strings.as_string(x*1000, precision=0)
    else:
        raise NotImplementedError(mode)


def construct_pointing_format(label_text, alt_text, x_str, y_str):
    if alt_text is None:
        alt_text = label_text
    np = tf.shape(x_str)[0]
    if np == 0:
        output = ""
    elif np == 1:
        output = tf.strings.join([
            '<point x="', x_str[0], '" y="', y_str[0], '" alt="',
            alt_text, '">', label_text, '</point>'
        ])
    else:
        ids = tf.strings.as_string(tf.range(1, np + 1, dtype=tf.int32))
        xs = tf.strings.join(["x", ids, '="', x_str, '"'])
        ys = tf.strings.join(["y", ids, '="', y_str, '"'])
        points = tf.strings.reduce_join(tf.reshape(tf.stack([xs, ys], 1), [-1]), separator=' ', axis=-1)
        output = tf.strings.join(
            ["<points ", points, ' alt="', alt_text, '">', label_text, "</points>"])
    return output


def order_points(x, y, seed, point_order):
    if point_order == "natural":
        return x, y
    elif point_order == "random_points":
        random_tensor = tf.random.uniform(shape=(tf.shape(x)[0], 2), minval=0, maxval=100)
        x = random_tensor[:, 0]
        y = random_tensor[:, 1]
        x_str = tf.strings.as_string(x, precision=1)
        y_str = tf.strings.as_string(y, precision=1)
        return x_str, y_str
    elif point_order == "random_points_and_length":
        random_length = tf.random.uniform(shape=[], minval=1, maxval=tf.shape(x)[0] * 2, dtype=tf.int32)
        random_tensor = tf.random.uniform(shape=(random_length, 2), minval=0, maxval=100)
        x = random_tensor[:, 0]
        y = random_tensor[:, 1]
        x_str = tf.strings.as_string(x, precision=1)
        y_str = tf.strings.as_string(y, precision=1)
        return x_str, y_str

    if point_order == "random":
        ix = stateless_permutation(tf.shape(x)[0], seed)
    elif point_order == "xy":
        x_float, y_float = tf.strings.to_number(x), tf.strings.to_number(y)
        ix = tf.argsort(x_float*100000 + y_float)
    elif point_order == "yx":
        x_float, y_float = tf.strings.to_number(x), tf.strings.to_number(y)
        ix = tf.argsort(y_float*100000 + x_float)
    else:
        raise NotImplementedError(point_order)
    return tf.gather(x, ix), tf.gather(y, ix)


@gin.configurable()
def points_to_text(x, y, w, h, seed, label=None, alt_text=None, point_mode="percent-precision-1",
                   point_order="xy", point_list_mode="tag"):
    """Returns a string encoding of a list of points"""
    x = quantize_point(x, w, point_mode)
    y = quantize_point(y, h, point_mode)
    # Order the quantized points to make the order matches what was generated, this can matter
    # when points have the same quantized value e.g, (10.001, 20) (10.002, 10) should be
    # represented (10, 10), (10, 20), but if we sort before quantization we get (10, 20), (10, 10)
    x, y = order_points(x, y, seed, point_order)
    if point_list_mode == "tag":
        return construct_pointing_format(label, alt_text, x, y)
    elif point_list_mode == "paren":
        n = tf.shape(x)[0]
        return tf.strings.reduce_join(tf.strings.join([
            "(", x, ", ", y, ")"
        ]), separator=", ")
    else:
        raise NotImplementedError(point_list_mode)


def points_to_answer(x, y, w, h, seed, label, is_counting, just_count=False, count_first=False, point_order="xy", alt_text=None):
    count = tf.shape(x)[0]
    if count == 0:
        return "There are none."
    
    if is_counting:
        point_text = points_to_text(x, y, w, h, seed, label, alt_text, point_order=point_order)
        return tf.strings.join([
            "Counting the ", point_text,
            " shows a total of ",
            tf.strings.as_string(count),
            "."
        ])
    elif just_count:
        return tf.strings.join(["Counting the ", label, " shows a total of ",
                                tf.strings.as_string(count), "."])
    elif count_first:
        point_text = points_to_text(x, y, w, h, seed, label, alt_text, point_order=point_order)
        return tf.strings.join([
            "Counting the ", label, " shows a total of ",
            tf.strings.as_string(count),". The points are ", point_text, ".",
        ])
    else:
        return points_to_text(x, y, w, h, seed, label, alt_text, point_order=point_order)


@seqio.map_over_dataset(num_seeds=2)
def extract_point_qa(ex, seeds, answer_type="y_major"):
    ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]

    questions = ex["questions"]
    question = questions["question"]
    n = tf.shape(question)[0]
    answers = tf.TensorArray(tf.string, size=n, element_shape=())
    point_text = questions["annotations"]["point_text"]
    point_seeds = tf.RaggedTensor.from_row_splits(
        row_splits=point_text.row_splits,
        values=tf.random.split(seeds[0], num=tf.shape(point_text.values)[0])
    )
    for question_ix in range(n):
        anno = questions["annotations"]
        answer = questions["answer_with_placeholders"][question_ix]
        n_anno = tf.shape(anno["point_text"][question_ix])[0]
        for anno_ix in range(n_anno):
            points = anno["points"][question_ix, anno_ix]
            point_text = points_to_answer(
                points[:, 0], points[:, 1], 100, 100,
                point_seeds[question_ix, anno_ix],
                anno["point_text"][question_ix, anno_ix],
                False,
                alt_text=anno["alt_text"][question_ix, anno_ix],
            )
            answer_split = tf.strings.split(answer, sep="<|POINT|>", maxsplit=1)
            answer = tf.strings.join([answer_split[0], point_text, answer_split[1]])
        # Make sure all placeholders where used
        tf.debugging.assert_equal(tf.shape(tf.strings.split(answer, sep="<|POINT|>"))[0], 1)
        answers = answers.write(question_ix, answer)

    messages = tf.stack([question, answers.stack()], axis=1)
    messages = tf.reshape(messages, [-1])
    conversation_ids = tf.range(tf.shape(messages)[0] // 2, dtype=tf.int32)
    conversation_ids = tf.repeat(conversation_ids, 2)
    out = dict(
        image=ex["image"],
        messages=tf.RaggedTensor.from_value_rowids(messages, conversation_ids)
    )
    ix = stateless_permutation(tf.shape(messages)[0], seeds[1])
    messages = tf.gather(messages, ix)
    out.update(_add_metadata(ex))
    out["metadata/image_size"] = [img_w, img_h]
    return out


def select_point(mask):
    bs = tf.shape(mask)[0]
    valid = tf.cast(mask, tf.float32)
    h, w = tf.shape(mask)[1], tf.shape(mask)[2]
    ys = tf.range(h, dtype=tf.int32)
    xs = tf.range(w, dtype=tf.int32)

    n = tf.reduce_sum(valid, [1, 2])
    cy = tf.reduce_sum(tf.cast(ys[None, :, None], tf.float32) * valid, [1, 2]) / n  # [bs]
    cx = tf.reduce_sum(tf.cast(xs[None, None, :], tf.float32) * valid, [1, 2]) / n  # [bs]

    dist_y = tf.square(tf.range(h, dtype=tf.float32)[None, :] - cy[:, None])  # [bs, h]
    dist_x = tf.square(tf.range(w, dtype=tf.float32)[None, :] - cx[:, None])  # [bs, w]
    dist = dist_y[:, :, None] + dist_x[:, None, :]  # [batch, h, w]
    dist = dist + (1 - valid) * 1e12
    min_dist = tf.argmin(tf.reshape(dist, [bs, -1]), axis=-1)  # [batch]
    w = tf.cast(w, min_dist.dtype)
    cy = tf.cast(min_dist // w, tf.float32)
    cx = tf.cast(min_dist % w, tf.float32)
    return cx, cy


@seqio.map_over_dataset
def refexp_pointing(ex):
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]
    objects = ex["objects"]

    # Shuffle objects so what object gets truncated if the sequence gets truncated is randomized
    refexps = objects['refexp']['raw']
    bbox = objects["bbox"]
    mask = tf.squeeze(objects["mask"], -1)

    ix = tf.range(0, tf.shape(refexps)[0], dtype=tf.int32)
    ix = tf.random.shuffle(ix)
    refexps = tf.gather(refexps, ix)
    bbox = tf.gather(bbox, ix)
    mask = tf.gather(mask, ix)

    cx, cy = select_point(mask)
    answers = points_to_text(img_h, img_w, cx, cy)

    out = {
        "image": ex["image"],
        "refexp": refexps.values,
        "metadata/image_size": tf.stack([img_w, img_h,]),
        "text": tf.repeat(answers, refexps.row_lengths()),
    }
    if "image_url" in ex:
        out["metadata/image_url"] = ex["image_url"]
    return out


@seqio.map_over_dataset
def refexp_pointing_inf(ex):
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]

    objects = ex["objects"]
    mask = tf.squeeze(objects["mask"], -1)
    cx, cy = select_point(mask)
    answers = points_to_text(img_h, img_w, cx, cy)

    refexps = objects["refexp"]["raw"]

    # We can't use `mask` directly since it is variable size, and thus it
    # will break batching. Here we serialize it instead
    serialized_masks = tf.map_fn(tf.io.serialize_tensor, mask, fn_output_signature=tf.string)
    out = {
        "image": ex["image"],
        "refexp": refexps,
        "metadata/bbox": objects["bbox"],
        "metadata/answer": answers,
        "metadata/mask": serialized_masks,
        "metadata/image_size": tf.stack([img_w, img_h]),
    }
    out.update({k: v for k, v in ex.items() if k.startswith("metadata/")})
    return out

@seqio.map_over_dataset
def extract_andriod_control_inf(ex, mode):
    if mode == "ll":
        prompt = tf.strings.join(["low_level: ", ex["metadata/ll_instruction"]])
    elif mode == "hl_ll":
        prompt = tf.strings.join([
            "high_level: ", ex["metadata/hl_instruction"],
            " low_level: ", ex["metadata/ll_instruction"]
        ])
    elif mode == "hl":
        prompt = tf.strings.join(["high_level: ", ex["metadata/hl_instruction"]])
    elif mode == "hl_cot":
        prompt = tf.strings.join(["high_level_cot: ", ex["metadata/hl_instruction"]])
    else:
        raise NotImplementedError()

    out = dict(
        image=ex["image"],
        prompt=prompt,
        text=ex["metadata/target_action"]
    )
    out.update(_add_metadata(ex))
    return out

@seqio.map_over_dataset
def extract_android_control(ex):
    # Each image has three tasks:
    # low level -> action
    # high+low level -> action
    # high level -> action
    # high level -> low level + action (CoT)
    out = dict(
        image=ex["image"],
        prompt=tf.stack([
            tf.strings.join(["low_level: ", ex["metadata/ll_instruction"]]),
            tf.strings.join([
                "high_level: ", ex["metadata/hl_instruction"],
                " low_level: ", ex["metadata/ll_instruction"]
            ]),
            tf.strings.join(["high_level: ", ex["metadata/hl_instruction"]]),
            tf.strings.join(["high_level_cot: ", ex["metadata/hl_instruction"]]),
        ]),
        text=tf.stack([
            ex["metadata/target_action"],
            ex["metadata/target_action"],
            ex["metadata/target_action"],
            tf.strings.join(["Plan: ", ex["metadata/ll_instruction"], " Action: ", ex["metadata/target_action"]]),
        ])
    )
    # Only needed if visualizing
    # ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    # img_h = tf.shape(ex["image"])[0]
    # img_w = tf.shape(ex["image"])[1]
    # out["metadata/image_size"] = tf.stack([img_w, img_h,])
    out.update(_add_metadata(ex))
    return out


@seqio.map_over_dataset(num_seeds=1)
def refexp(ex, seed):
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]
    objects = ex["objects"]

    # Shuffle objects so what object gets truncated if the sequence gets truncated is randomized
    refexps = objects['refexp']['raw']
    bbox = objects["bbox"]
    ix = stateless_permutation(tf.shape(refexps)[0], seed)
    refexps = tf.gather(refexps, ix)
    bbox = tf.gather(bbox, ix)

    x2 = bbox[:, 0] + bbox[:, 2]
    y2 = bbox[:, 1] + bbox[:, 3]
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.reduce_any(x2 <= tf.cast(img_w, tf.float32)), True),
        tf.debugging.assert_equal(tf.reduce_any(y2 <= tf.cast(img_h, tf.float32)), True)
    ]):
        answers = points_to_text(
            img_h, img_w,
            tf.reshape(tf.stack([bbox[:, 0], x2], 1), [-1]),
            tf.reshape(tf.stack([bbox[:, 1], y2], 1), [-1]))
        answers = tf.strings.reduce_join(tf.reshape(answers, [-1, 2]), separator=" ", axis=1)

    out = {
        "image": ex["image"],
        "refexp": refexps.values,
        "metadata/bbox": bbox,
        "metadata/image_size": tf.stack([img_w, img_h,]),
        "text": tf.repeat(answers, refexps.row_lengths()),
    }

    if "image_url" in ex:
        out["image_url"] = ex["image_url"]
    return out


@seqio.map_over_dataset
def refexp_inf(ex):
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]
    out = {
        "image": ex["image"],
        "refexp": ex["objects"]["refexp"]["raw"],
        "metadata/bbox": ex["objects"]["bbox"],
        "metadata/image_size": tf.stack([img_w, img_h,]),
    }
    out.update({k: v for k, v in ex.items() if k.startswith("metadata/")})
    return out


def point_text_interleaved(*args):
    raise NotImplementedError()


@seqio.map_over_dataset
def web_pointing_preprocessor(ex):
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]

    question = point_text_interleaved(
        img_h, img_w, ex["question"], ex["question_points"]["x"], ex["question_points"]["y"])
    answer = point_text_interleaved(
        img_h, img_w, ex["answer"], ex["answer_points"]["x"], ex["answer_points"]["y"])
    answer_points = tf.stack([ex["answer_points"]["x"], ex["answer_points"]["y"]], axis=1)
    return {
        "question": question,
        "answer": answer,
        "image": ex["image"],
        "metadata/image_size": [img_w, img_h],
        "metadata/question_type": ex["question_type"],
        "metadata/answer_points": tf.io.serialize_tensor(answer_points),
        "metadata/answer": answer,
    }


def filter_pointing(ds):
    return ds.filter(lambda ex: tf.shape(ex["answer_points"]["x"])[0] >= 1)


def filter_qa(ds):
    return ds.filter(lambda ex: tf.shape(ex["answer_points"]["x"])[0] == 0)

# vaia filtering
def filter_image_only(ds):
    return ds.filter(lambda ex: ex["has_image"])

def filter_mc(ds):
    return ds.filter(lambda ex: ex["is_mc"])

def remove_is_long(ds):
    return ds.filter(lambda ex: not ex["is_long"])

def remove_has_multiple_parts(ds):
    return ds.filter(lambda ex: not ex["has_multiple_parts"])


def _split(ds: tf.data.Dataset, keys, n_splits=2):
    def _map(ex):
        n = tf.shape(ex[keys[0]])[0]
        if n < n_splits:
            return tf.data.Dataset.from_tensors(ex)
        else:
            bs = n // n_splits
            remainder = n - bs*n_splits
            lens = tf.concat([
                tf.ones([remainder], dtype=tf.int32),
                tf.zeros([n_splits-remainder], dtype=tf.int32),
            ], axis=0) + bs
            tf.debugging.assert_equal(tf.reduce_sum(lens), n)
            ends = tf.cumsum(lens)

            parts = []
            for split_ix in range(n_splits):
                part_ex = dict(ex)
                e = ends[split_ix]
                s = e - lens[split_ix]
                for k in keys:
                    if isinstance(k, tuple):
                        assert len(k) == 2
                        part_ex[k[0]][k[1]] = ex[k[0]][k[1]][s:e]
                    else:
                        part_ex[k] = ex[k][s:e]
                parts.append(part_ex)

            ds = tf.data.Dataset.from_tensors(parts[0])
            for sub_ds in parts[1:]:
                sub_ds = tf.data.Dataset.from_tensors(sub_ds)
                ds = ds.concatenate(sub_ds)
            return ds

    return ds.flat_map(_map)



def split(ds, n=2):
    # return ds
    return _split(ds, [k for k in [
        "question",
        "label",
        "text",
        "entity",
        "messages"
    ] if k in ds.element_spec], n_splits=n)


def split_points(ds, max_points=50):
    label = "question" if "question" in ds.element_spec else "label"
    return _split(ds, [
        "question", label, "notInImage",
        ("answer_points", "x"),
        ("answer_points", "y"),
    ])


@seqio.map_over_dataset
def fix_count_qa(ex):
    ex["label"] = ex["label"][::2]
    tf.debugging.assert_equal(tf.shape(ex["answer_points"]["x"])[0], tf.shape(ex["label"])[0])
    return ex


def filter_points(ds, max_number=40):

    def _add_valid(ex):
        valid = (
            tf.reduce_all(ex["answer_points"]["x"] >= 0.0, axis=-1) &
            tf.reduce_all(ex["answer_points"]["x"] <= 100.0, axis=-1) &
            tf.reduce_all(ex["answer_points"]["y"] >= 0.0, axis=-1) &
            tf.reduce_all(ex["answer_points"]["y"] <= 100.0, axis=-1) &
            (ex["answer_points"]["y"].row_lengths() <= max_number)
        )
        ex["valid"] = valid
        return ex
    ds = ds.map(_add_valid)
    ds = ds.filter(lambda ex: tf.reduce_any(ex["valid"]))
    return ds


@gin.configurable()
@seqio.map_over_dataset(num_seeds=2)
def pointing_preprocessor(ex, sequence_length, seeds, with_count=False,
                          just_count=False, count_first=False, point_order="xy", shuffle=True):
    image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]

    ix = tf.where(ex["valid"])[:, 0]
    if shuffle:
        ix = stateless_shuffle(ix, seeds[0])
    if "label" in ex:
        question = tf.strings.lower(ex["label"])
    else:
        question = ex["question"]
    question = tf.gather(question, ix)  # [n_question]
    points_x = tf.gather(ex["answer_points"]["x"], ix)  # [n_question, n_points[ragged]]]
    points_y = tf.gather(ex["answer_points"]["y"], ix)
    not_in_image = tf.gather(ex["notInImage"], ix)  # [n_question]

    n = tf.shape(points_x)[0]
    point_text = tf.TensorArray(dtype=tf.string, size=n, element_shape=())  # [n_question]
    point_seeds = tf.random.split(seeds[1], n)
    for i in range(n):
        answer = points_to_answer(points_x[i], points_y[i], 100, 100, point_seeds[i], question[i], with_count, just_count, count_first, point_order=point_order)
        point_text = point_text.write(i, answer)
    return {
        "image": image,
        "metadata/image_size": [img_w, img_h],
        "metadata/image_url": ex["image_url"],
        "entity": question,
        "question": question,
        "text": point_text.stack(),
    }


@seqio.map_over_dataset
def pointing_inf_preprocessor(ex):
    ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    img_h = tf.shape(ex["image"])[0]
    img_w = tf.shape(ex["image"])[1]

    question = ex["question"]
    not_in_image = tf.shape(ex["answer_points"]["x"])[0] == 0

    # points are stored in normalized format, de-normalize here
    points_x = ex["answer_points"]["x"] * tf.cast(img_w, tf.float32) / 100.0
    points_y = ex["answer_points"]["y"] * tf.cast(img_h, tf.float32) / 100.0

    out = dict(
        image=ex["image"],
        question=question,
        entity=question,
    )
    out.update(_add_metadata(ex))
    out["metadata/not_in_image"] = not_in_image
    # We can't use `mask` directly since it is variable size, and thus it
    # will break batching. Here we serialize it instead
    serialized_masks = tf.map_fn(tf.io.serialize_tensor, ex["masks"], fn_output_signature=tf.string)
    serialized_masks = tf.strings.reduce_join(serialized_masks, separator="|||")
    out["metadata/mask"] = serialized_masks
    out["metadata/question"] = question
    out["metadata/answer_points"] = tf.io.serialize_tensor(tf.stack([points_x, points_y], 1))
    out["metadata/image_size"] = [img_w, img_h]

    return out


@seqio.map_over_dataset(num_seeds=1)
def count_qa_preprocessor_inf(ex, sequence_length, seed):
    image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]

    entity = tf.strings.substr(
        ex["question"], len("How many "), tf.strings.length(ex["question"]) - len("How many "))
    entity = tf.strings.split(entity, sep=" are ", maxsplit=1)[0]
    entity = tf.strings.lower(entity)
    tf.debugging.assert_equal(tf.strings.length(entity) != 0, True)
    out = {
        "image": image,
        "metadata/image_size": [img_w, img_h],
        "metadata/count": tf.strings.to_number(ex["answer"]),
        "question": ex["question"],
        "entity": entity,
    }
    if "image_url" in ex:
        out["metadata/image_url"] = ex["image_url"]
    return out

@seqio.map_over_dataset(num_seeds=1)
def count_qa_preprocessor(ex, sequence_length, seed, with_count=False, just_count=False, count_first=False, for_inference=False, point_order="xy"):
    point_answer = ex["point_answer"]
    numbers_str = tf.strings.regex_replace(point_answer, r'\.$', '')
    numbers_str = tf.strings.regex_replace(numbers_str, r'[^\d\.\s]+', '')
    numbers_str = tf.strings.strip(numbers_str)
    numbers = tf.strings.split(numbers_str)
    float_numbers = tf.strings.to_number(numbers, out_type=tf.float32)
    coordinates = tf.reshape(float_numbers, (-1, 3))
    points_x = coordinates[:, 1]
    points_y = coordinates[:, 2]

    image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]
    entity = tf.strings.substr(
        ex["question"], len("How many "), tf.strings.length(ex["question"]) - len("How many "))
    entity = tf.strings.split(entity, sep=" are ", maxsplit=1)[0]
    entity = tf.strings.lower(entity)
    tf.debugging.assert_equal(tf.strings.length(entity) != 0, True)
    count = tf.strings.to_number(ex["answer"], out_type=tf.int32)
    if for_inference:
        return {
            "image": image,
            "metadata/image_size": [img_w, img_h],
            "metadata/count": count,
            "metadata/image_url": ex["url"],
            "metadata/label": ex["question"],
            "question": ex["question"],
            "entity": entity,
        }
    else:
        tf.debugging.assert_equal(count, tf.shape(points_x)[0])
        # points are already normalized so use w=1, h=1
        answer = points_to_answer(points_x, points_y, 1, 1, seed, entity, with_count, just_count, count_first, point_order=point_order)
        return {
            "image": image,
            "metadata/image_size": [img_w, img_h],
            "metadata/count": count,
            "metadata/image_url": ex["url"],
            "metadata/label": ex["question"],
            "question": ex["question"],
            "entity": entity,
            "text": answer,
        }


@gin.configurable()
@seqio.map_over_dataset
def cleanup_preprocessor(ex, preprocess=False):
    if preprocess:
        ex["prompt"] = tf.strings.join(
            [
                "[[User]]: Correct the spelling and punctuation mistakes on the following transcript based on what appears in the image.\n\n{before} ",
                ex["prompt"],
                "\n[[Assistant]]: {after}"
            ]
        )
        return ex
    else:
        return ex


@gin.configurable()
@seqio.map_over_dataset
def random_text_preprocessor(ex, preprocess=False):
    ex["prompt"] = "What does the text say in this image?"
    if preprocess:
        ex["prompt"] = tf.strings.join(["[[User]]: ", ex["prompt"], "\n[[Assistant]]:"])
        return ex
    else:
        return ex


@seqio.map_over_dataset(num_seeds=25)
def clock_augmentation(ex, seeds):
    seeds = list(seeds)
    image = ex["image"]

    # Apply shear, rotation, and scale through one affine matrix
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

    _call_id = [0]

    def _rng(_minval=0, _maxval=1, shape=(), dtype=tf.float32):
        return tf.random.stateless_uniform(shape, seeds.pop(), _minval, _maxval, dtype=dtype)

    sel = _rng(0, 1)
    if sel < 0.1:
        # Straight on
        shear_x = 0.
        shear_y = 0.
        rotation = 0.
    elif sel < 0.5:
        # Normal looking
        shear_x = _rng(-10, 10)
        shear_y = _rng(-10, 10)
        rotation = _rng(-25, 25)
    else:
        # Allowed to be very wonky
        # if tf.random.stateless_uniform((), seeds.pop(), 0, 1) > 0.8:
        #     image = image[:, ::-1]

        if _rng() > 0.5:
            shear_x = _rng( -30, 30)
            shear_y = _rng( -30, 30)
        else:
            shear_x = _rng( -10, 10)
            shear_y = _rng( -10, 10)
        rng = _rng( 0, 1)
        if rng < 0.2:
            rotation = _rng( -25, 25)
        elif rng < 0.6:
            rotation = _rng( -80, 80)
        else:
            rotation = _rng( -180, 180)

    if _rng() > 0.5:
        scale = _rng( 0.3, 2)
    else:
        scale = _rng( 0.3, 1)
    # Pad so upscaling/rotation will not move the image out of bounds
    pad = tf.cast(tf.maximum(height, width)*0.5, tf.int32)
    image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], constant_values=1)
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

    image = tf.keras.ops.image.affine_transform(
        image,
        tf.stack(get_affine_matrix(
            [height/2, width/2],
            rotation,
            [0, 0],
            1/scale,
            [shear_x, shear_y]
        ) + [0., 0.]),
        interpolation='bilinear',
        fill_mode='constant',
        fill_value=1.,
        data_format='channels_last'
    )

    # Crop, otherwise it would be impossible to put the image at the corner of the image
    not_white = tf.logical_not(tf.reduce_all(image > 0.99, -1))
    no_white_ix = tf.where(not_white)
    top_left = tf.reduce_min(no_white_ix, axis=0)
    bottom_right = tf.reduce_max(no_white_ix, axis=0)
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=tf.cast(top_left[0], tf.int32),
        offset_width=tf.cast(top_left[1], tf.int32),
        target_height=tf.cast(bottom_right[0] - top_left[0] + 1, tf.int32),
        target_width=tf.cast(bottom_right[1] - top_left[1] + 1, tf.int32),
    )

    # Translate
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    translation_seed = _rng(0, 1)
    if translation_seed < 0.2:
        h_pad = _rng(0, height//2, (2,), dtype=tf.int32)
        w_pad = _rng(0, width//2, (2,), dtype=tf.int32)
    else:
        h_pad = _rng(0, height*2, (2,), dtype=tf.int32)
        w_pad = _rng(0, width*2, (2,), dtype=tf.int32)
    image = tf.pad(image, [[h_pad[0], w_pad[0]], [h_pad[1], w_pad[1]], [0, 0]],
                   constant_values=1)

    # Random background color
    # color_rng = tf.random.stateless_uniform((4,), seeds.pop(), 0, 1)
    # random_color = color_rng[:3]
    # valid = tf.reduce_all(tf.reduce_sum(tf.abs(random_color[None, None, :] - image), -1) > 0.03)
    # if color_rng[0] < 0.2 and valid:
    #     image = tf.where(tf.reduce_all(image < 0.99, axis=-1, keepdims=True),
    #                      image, image * 0 + random_color[None, None, :])

    # Mild color hitter
    image = tf.image.stateless_random_hue(image, max_delta=0.05, seed=seeds.pop())
    image = tf.image.stateless_random_brightness(image, max_delta=0.15, seed=seeds.pop())
    image = tf.image.stateless_random_saturation(image, 0.8, 1.2, seed=seeds.pop())
    image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed=seeds.pop())

    # ex["metadata/unaugmented_image"] = ex["image"]
    ex["image"] = image
    return ex


@seqio.map_over_dataset
def clocks_preprocessor(ex):
    time_format = ex["time_format"]
    shows_seconds = ex["shows_seconds"]
    hour, minute, second = [tf.cast(ex[k], tf.int32) for k in ["hour", "minute", "second"]]
    if hour == 0:
        am_pm = "AM"
        hour_str = 24  # Midnight of the previous day
    elif hour > 12:
        am_pm = "PM"
        hour_str = hour - 12
    else:
        hour_str = hour
        am_pm = "AM"
    hour_str = tf.strings.as_string(hour_str)
    minute_str = tf.strings.as_string(minute)
    if tf.strings.length(minute_str) == 1:
        minute_str = tf.strings.join(["0", minute_str])

    second_str = tf.strings.as_string(second)
    if tf.strings.length(second_str) == 1:
        second_str = tf.strings.join(["0", second_str])

    prefix = "The time shown is "

    if time_format == "The time is not shown":
        text = "The time is not shown in the image."
        hour, minute, second = -1, -1, -1
    else:
        if not shows_seconds:
            second = -1
        if time_format == "12 hour clock (without AM/PM)" and shows_seconds:
            if hour >= 12:
                hour = hour - 12
            time = tf.strings.join([hour_str, ":", minute_str, ":", second_str])
        elif time_format == "12 hour clock (with AM/PM)" and shows_seconds:
            time = tf.strings.join([hour_str, ":", minute_str, ":", second_str, " ", am_pm])
        elif time_format == "12 hour clock (with AM/PM)" and not shows_seconds:
            time = tf.strings.join([hour_str, ":", minute_str, " ", am_pm])
        elif time_format == "12 hour clock (without AM/PM)" and not shows_seconds:
            if hour >= 12:
                hour = hour - 12
            time = tf.strings.join([hour_str, ":", minute_str])
        else:
            time = ""  # Should never occur, but needed for tf analysis
        tf.debugging.assert_equal(tf.strings.length(time) > 0, True)
        text = tf.strings.join(["The time shown is ", time])
    image = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)[:-120]  # remove the black shadow at the bottom
    return {
        "image": image,
        "prompt": "What time is being shown?",
        "text": text,
        "metadata/time_format": time_format,
        "metadata/hour": hour,
        "metadata/minute": minute,
        "metadata/text": text,
        "metadata/second": second,
    }


@seqio.map_over_dataset()
def atlas_obscura_preprocessor(ex):
    out = dict(
        image=ex["image"],
        prompt="Where was this picture taken?",
        text=tf.strings.join([
            ex["place"],
            " in ",
            ex["city"]
        ])
    )
    out["metadata/image_url"] = ex["image_url"]
    out["metadata/references"] = out["text"]
    return out


@seqio.map_over_dataset()
def famous_birthdays_preprocessor(ex):
    out = dict(
        image=ex["image"],
        image_url=ex["image_url"],
        prompt="Who is this?",
        text=ex["name"]
    )
    out["metadata/references"] = out["text"]
    return out


@seqio.map_over_dataset()
def mild_color_aug_preprocessor(ex):
    if "image_url" in ex:  # URL won't show the augmentations
        del ex["image_url"]
    # ex["metadata/unaugmented_image"] = ex["image"]
    ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    ex["image"] = mild_color_aug(ex["image"])
    return ex


def build_text_with_points(text, points, img_h, img_w):
    points = points_to_text(img_h, img_w, points[:, 0], points[:, 1])
    parts = tf.strings.split(text, sep="<ANS>")
    with_points = tf.strings.reduce_join(tf.reshape(tf.stack([
        parts,
        tf.pad(points, [[0, 1]], constant_values=""),
    ], 1), [-1]), separator="")
    return tf.strings.split(with_points, "\n\n")


@seqio.map_over_dataset()
def synth_count_preprocessor(example):
    image_shape = tf.shape(example["image"])
    h, w = image_shape[0], image_shape[1]
    questions = build_text_with_points(example["questions"], example["question_points"], h, w)
    answers = build_text_with_points(example["answers"], example["answer_points"], h, w)
    keep_q = tf.strings.regex_full_match(questions, "How many.*")
    keep_ans = tf.strings.regex_full_match(answers, "There are [0-9]+.*")
    keep = tf.logical_and(keep_q, keep_ans)
    questions = tf.boolean_mask(questions, keep)
    answers = tf.boolean_mask(answers, keep)
    ix = tf.range(0, tf.shape(answers)[0], dtype=tf.int32)
    ix = tf.random.shuffle(ix)
    return dict(
        image=example["image"],
        prompt=tf.gather(questions, ix),
        text=tf.gather(answers, ix),
    )


def synth_count_inf_preprocessor(ds):

    @seqio.map_over_dataset(num_seeds=1)
    def get_two(example, seed):
        image_shape = tf.shape(example["image"])
        h, w = image_shape[0], image_shape[1]
        questions = build_text_with_points(example["questions"], example["question_points"], h, w)
        answers = build_text_with_points(example["answers"], example["answer_points"], h, w)
        keep_q = tf.strings.regex_full_match(questions, "How many.*")
        keep_ans = tf.strings.regex_full_match(answers, "There are [0-9]+.*")
        keep = tf.logical_and(keep_q, keep_ans)
        questions = tf.boolean_mask(questions, keep)
        answers = tf.boolean_mask(answers, keep)

        ix = stateless_permutation(tf.shape(answers)[0], seed)[:2]
        return {
            "image": example["image"],
            "prompt": tf.gather(questions, ix),
            "metadata/references": tf.gather(answers, ix),
        }

    ds = get_two(ds)
    return flatten_parts(ds, ["prompt", "metadata/references"])


def mild_color_aug(image):
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image


@seqio.map_over_dataset()
def name_entity_augmentation(ex, p_high_color=0.7):
    ex["image"] = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
    image = ex["image"]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Horizontal flip
    if tf.random.uniform((), 0, 1) > 0.85:
        image = image[:, ::-1]

    # Random crop
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    crop_rng = tf.random.uniform((), 0, 1)
    if crop_rng < 0.2:
        pass
    else:
        if crop_rng < 0.4:
            h_crop = height * 0.15
            w_crop = width * 0.15
        else:
            h_crop = height * 0.4
            w_crop = width * 0.4
        crop_h = tf.cast(tf.random.uniform((2,), 0, h_crop/2), tf.int32)
        crop_w = tf.cast(tf.random.uniform((2,), 0, w_crop/2), tf.int32)
        image = image[crop_h[0]:-crop_h[1]-1, crop_w[0]:-crop_w[1]-1]
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)

    if tf.random.uniform(()) > p_high_color:
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_contrast(image, 0.8, 1.2)
    else:
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_saturation(image, 0.0, 2.0)
        image = tf.image.random_contrast(image, 0.2, 1.5)

    # Apply shear, rotation, and scale through one affine matrix
    sel = tf.random.uniform((), 0, 1)
    if sel < 0.1:
        pass
    else:
        if sel < 0.15:  # Scale only
            shear_x = 0
            shear_y = 0
            rotation = 0
        if sel < 0.7:  # Mild
            shear_x = tf.random.uniform((), -2, 2)
            shear_y = tf.random.uniform((), -2, 2)
            rotation = tf.random.uniform((), -5, 5)
        else:  # Severe
            shear_x = tf.random.uniform((), -10, 10)
            shear_y = tf.random.uniform((), -10, 10)
            rotation = tf.random.uniform((), -20, 20)

        max_scale = 1.2
        scale = tf.random.uniform((), 0.4, max_scale)

        # Pad so upscaling/rotation will not move the image out of bounds
        pad = tf.cast(tf.maximum(height, width)*0.2, tf.int32)
        image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], constant_values=1)

        image = tf.keras.ops.image.affine_transform(
            image,
            tf.stack(get_affine_matrix(
                [height/2, width/2],
                rotation,
                [0, 0],
                1/scale,
                [shear_x, shear_y]
            ) + [0., 0.]),
            interpolation='bilinear',
            fill_mode='constant',
            fill_value=1.,
            data_format='channels_last'
        )

    # Crop, otherwise it would be impossible to put the image at the corner of the image
    not_white = tf.logical_not(tf.reduce_all(image > 0.99, -1))
    no_white_ix = tf.where(not_white)
    top_left = tf.reduce_min(no_white_ix, axis=0)
    bottom_right = tf.reduce_max(no_white_ix, axis=0)

    # Very low chance center crop will get nothing but white space, we just skip
    if (
        (bottom_right[0] - top_left[0]) > 1 and (bottom_right[1] - top_left[1]) > 1
    ):
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=tf.cast(top_left[0], tf.int32),
            offset_width=tf.cast(top_left[1], tf.int32),
            target_height=tf.cast(bottom_right[0] - top_left[0] + 1, tf.int32),
            target_width=tf.cast(bottom_right[1] - top_left[1] + 1, tf.int32),
        )

    # Translate
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    if tf.random.uniform((), 0, 1) < 0.1:
        h_pad = tf.zeros((2,), dtype=tf.int32)
        w_pad = tf.zeros((2,), dtype=tf.int32)
    elif tf.random.uniform((), 0, 1) < 0.8:
        h_pad = tf.random.uniform((2,), 0, 50, dtype=tf.int32)
        w_pad = tf.random.uniform((2,), 0, 50, dtype=tf.int32)
    else:
        pad = tf.cast(tf.maximum(height, width), tf.int32)
        h_pad = tf.random.uniform((2,), 0, pad, dtype=tf.int32)
        w_pad = tf.random.uniform((2,), 0, pad, dtype=tf.int32)
    image = tf.pad(image, [[h_pad[0], w_pad[0]], [h_pad[1], w_pad[1]], [0, 0]],
                   constant_values=1)

    if "image_url" in ex:  # URL won't show the augmentations
        del ex["image_url"]
    # ex["metadata/unaugmented_image"] = ex["image"]
    ex["image"] = image
    return ex


@seqio.map_over_dataset()
def wiki_art_preprocessor(ex):
    out = dict(
        image=ex["image"],
        prompt="What is this?",
        text=ex["question"]
    )
    out["metadata/title"] = ex["title"]
    out["metadata/gt"] = ex["question"]
    out["metadata/artist"] = ex["artist"]
    out["metadata/painting_url"] = ex["painting_url"]
    # if "metadata/unaugmented_image" in ex:
    #     out["metadata/unaugmented_image"] = ex["metadata/unaugmented_image"]
    return out

@seqio.map_over_dataset()
def oscar_preprocessor(ex):
    out = dict(
        image=ex["image"],
        prompt=ex["question"]
    )
    out.update(_add_metadata(ex))
    out["metadata/question"] = ex["question"]
    out["metadata/answer"] = ex["answer"]
    out["metadata/category"] = ex["category"]
    return out


@seqio.map_over_dataset()
def tulu_preprocessor(ex):
    return {
        "messages": ex["messages"]["content"],
    }
    # logging.info("Debugging tulue")
    # return {"messages": ex["messages"]["content"], "text_weights": 1e-6}


WIKI_DATA_QUESTION = "What is this? Respond with just a proper name."


@seqio.map_over_dataset()
def extract_wiki_data(ex):
    return dict(
        image=ex["image"],
        image_url=ex["image_url"],
        prompt=[
            WIKI_DATA_QUESTION,
            "What is this? Respond with the proper name of the main focus of the image and a few details about it."
        ],
        text=[
            tf.strings.strip(tf.strings.regex_replace(ex["question"], r"\(.*\)", "")),
            ex["gptResponse"],
        ]
    )


@seqio.map_over_dataset()
def extract_wiki_data_name(ex):
    target = tf.strings.strip(tf.strings.regex_replace(ex["question"], r"\(.*\)", ""))
    out = dict(
        image=ex["image"],
        image_url=ex["image_url"],
        prompt=WIKI_DATA_QUESTION,
        text=target,
    )
    out["metadata/references"] = target
    return out


@seqio.map_over_dataset()
def extract_wiki_data_describe(ex):
    out = dict(
        image=ex["image"],
        image_url=ex["image_url"],
        prompt="What is this? Respond with the proper name of the main focus of the image and a few details about it.",
    )
    out["metadata/references"] = ex["gptResponse"]
    return out


@gin.configurable()
def format_multiple_style_qa(ds, types=['multiple_choice', 'short_answer'], styles=['ai2_diagram', 'vqa2'], default_style='vqa2',
                             strip_instruction=False):
    def _extract(ex):
        prompt = ex["question"]
        out = dict(image=ex["image"])
        out.update(_add_metadata(ex))

        out["text"] = ex["answer"]
        out["metadata/references"] = ex["answer"]
        out["metadata/prompt"] = prompt

        if ex["metadata/question_type"] == 'multiple_choice':
            style = styles[0]
        else:
            style = styles[1]
        if strip_instruction:
            if ex["metadata/question_type"] == "multiple_choice":
                # parts = tf.strings.split(prompt, "\n")
                # parts 1 is blank and part -1 is the instruction
                # prompt = tf.strings.reduce_join(tf.concat([parts[:1], parts[2:-1]], 0), separator="\n")
                prompt = prompt
            else:
                prompt = tf.strings.split(prompt, "\n")[0]

        out["style"] = style
        out["prompt"] = prompt
        return out
    ds = ds.map(_extract, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


@gin.configurable()
def extract_mmmu(ds, types=['multiple-choice', 'open'], styles=['ai2_diagram', 'vqa2'], default_style='ai2_diagram', option_format="abc"):
    assert option_format == "abc"
    keys_tensor = tf.constant(types, dtype=tf.string)
    values_tensor = tf.constant(styles, dtype=tf.string)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
        default_value=tf.constant(default_style, dtype=tf.string),
    )
    def _extract(ex):
        out = dict(image=tf.expand_dims(ex["image_1"], 0))
        out.update(_add_metadata(ex))
        style = table.lookup(ex["metadata/question_type"])
        out["style"] = style
        out["text"] = ex["answer"]
        out["metadata/references"] = ex["answer"]

        if style == styles[0]:
            abc = tf.constant(list("abcdefghi".upper()))
            options = ex["options"]
            num_options = tf.shape(options)[0]
            dummy_options = tf.tile(tf.constant([""], dtype=tf.string), [9 - num_options])
            out["metadata/options"] = tf.concat([options, dummy_options], axis=0)
            out["metadata/options"] = tf.ensure_shape(out["metadata/options"], [9])

            short_options = abc[:num_options]
            options = tf.stack([short_options, options,], 1)
            options = tf.strings.reduce_join(options, axis=-1, separator=": ")
            options = tf.strings.reduce_join(options, separator="\n")
            out["prompt"] = tf.strings.join([ex["question"], "\n", options, "\n"])
            if tf.reduce_sum(tf.cast(tf.strings.regex_full_match(options, "<img='(.*?)'>"), tf.int32)) > 1:
                # Following LLaVa, don't use any images if there are multiple images paths
                # I think the rationale is that this means the image are answer-options
                out["image"] = out["image"][:0]
        else:
            out["metadata/options"] = tf.constant([""] * 9, dtype=tf.string)
            out["prompt"] = ex["question"]
            out["image"] = out["image"]
        return out
    ds = ds.map(_extract, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

@gin.configurable()
def extract_mmmu_cot(ds, types=['multiple-choice', 'open'], styles=['ai2_diagram', 'vqa2'], default_style='ai2_diagram', option_format="abc"):
    assert option_format == "abc"
    keys_tensor = tf.constant(types, dtype=tf.string)
    values_tensor = tf.constant(styles, dtype=tf.string)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
        default_value=tf.constant(default_style, dtype=tf.string),
    )
    def _extract(ex):
        # out = dict(image=tf.expand_dims(ex["image_with_question"], 0))
        out = dict(image=tf.expand_dims(ex["image_1"], 0))
        out.update(_add_metadata(ex))
        style = table.lookup(ex["metadata/question_type"])
        # out["style"] = style
        out["text"] = ex["answer"]
        out["metadata/question"] = ex["question"]
        out["metadata/references"] = ex["answer"]

        if style == styles[0]:
            abc = tf.constant(list("abcdefghi".upper()))
            options = ex["options"]
            num_options = tf.shape(options)[0]
            dummy_options = tf.tile(tf.constant([""], dtype=tf.string), [9 - num_options])
            out["metadata/options"] = tf.concat([options, dummy_options], axis=0)
            out["metadata/options"] = tf.ensure_shape(out["metadata/options"], [9])
            
            short_options = abc[:num_options]
            options = tf.stack([short_options, options,], 1)
            options = tf.strings.reduce_join(options, axis=-1, separator=": ")
            options = tf.strings.reduce_join(options, separator="\n")
            out["prompt"] = tf.strings.join([ex["question"], "\n", options, "\n"])
            # out["prompt"] = ex["question"]
            if tf.reduce_sum(tf.cast(tf.strings.regex_full_match(options, "<img='(.*?)'>"), tf.int32)) > 1:
                # Following LLaVa, don't use any images if there are multiple images paths
                # I think the rationale is that this means the image are answer-options
                out["image"] = out["image"][:0]
        else:
            out["metadata/options"] = tf.constant([""] * 9, dtype=tf.string)
            out["prompt"] = ex["question"]
            # out["image"] = out["image"][:0]
        return out
    ds = ds.map(_extract, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


@seqio.map_over_dataset
def reformat_math_vista(ex):
    query = ex["query"]
    query = tf.strings.split(query, sep="Question:")[-1]
    query = tf.strings.strip(tf.strings.split(query, sep="Hint:")[0])
    ex["query"] = query
    return ex


@seqio.map_over_dataset
def extract_math_vista(ex, styles=['ai2_diagram', 'vqa2']):
    out = dict(image=ex["image"])
    out.update(_add_metadata(ex))

    is_mc = ex["metadata/question_type"] == 'multi_choice'
    if is_mc:
        style = styles[0]
        abc = tf.constant(list("abcdefghi".upper()))
        options = ex["choices"]
        num_options = tf.shape(options)[0]
        dummy_options = tf.tile(tf.constant([""], dtype=tf.string), [9 - num_options])
        out["metadata/options"] = tf.concat([options, dummy_options], axis=0)
        out["metadata/options"] = tf.ensure_shape(out["metadata/options"], [9])

        if ex["metadata/split"] != "test":
            short_options = abc[:num_options]
            answer_short_option = tf.boolean_mask(short_options, options == ex["answer"])[0]
            out["text"] = answer_short_option
        else:
            out["text"] = ex["answer"]
    else:
        style = styles[1]
        out["metadata/options"] = tf.constant([""] * 9, dtype=tf.string)
        out["text"] = ex["answer"]
    out["style"] = style
    out["prompt"] = ex["query"]
    out["metadata/query"] = ex["query"]
    out["metadata/references"] = ex["answer"]
    return out


NO_POINT_PREFIX = [
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n"
]
assert all(x[-1].isspace() for x in NO_POINT_PREFIX)
NO_POINT_PREFIX_TF = tf.constant(NO_POINT_PREFIX)


def prefix_how_many(messages, seed):
    question = messages[0]
    if tf.strings.regex_full_match(tf.strings.lower(question), "how many.*"):
        ix = tf.random.stateless_uniform((), seed, 0, len(NO_POINT_PREFIX),  tf.int32)
        question = tf.strings.join([NO_POINT_PREFIX_TF[ix], question])
        return tf.concat([tf.expand_dims(question, 0), messages[1:]], axis=0)
    else:
        return messages


@seqio.map_over_dataset(num_seeds=1)
def prefix_how_many_messages(ex, seed):
    messages = ex["messages"]
    n = tf.shape(messages)[0]
    seeds = tf.random.split(seed, n)
    message_arr = tf.TensorArray(dtype=tf.string, size=n, element_shape=(None,))
    for i in range(n):
        message_arr = message_arr.write(i, prefix_how_many(messages[i], seeds[i]))
    ex["messages"] = tf.RaggedTensor.from_row_splits(
        values=message_arr.concat(), row_splits=messages.row_splits)
    return ex


def filter_single_turn(ds):
    @seqio.map_over_dataset
    def _filter(ex):
        multi_turn = ex["messages"].row_lengths() > 2
        ex["messages"] = tf.ragged.boolean_mask(ex["messages"], multi_turn)
        return ex

    ds = _filter(ds)
    ds = ds.filter(lambda x: tf.shape(x["messages"])[0] > 0)
    return ds


@gin.configurable()
@seqio.map_over_dataset(num_seeds=1)
def extract_cockatoo_qa_v2(ex, seed, shuffle_messages=True):
    messages = tf.RaggedTensor.from_value_rowids(ex["messages"], ex["conversation_ids"])
    if shuffle_messages:
        ix = stateless_permutation(tf.shape(messages)[0], seed)
        messages = tf.gather(messages, ix)
    out = dict(
        image=ex["image"],
        messages=messages
    )
    out.update(_add_metadata(ex))
    return out


def format_mmbench(ds):

    def _trim(ex):
        num_passes = tf.shape(ex["id"])[0]
        ex["choices"] = ex["choices"][:num_passes, :num_passes]
        ex["answer"] = ex["answer"][:num_passes]
        return ex

    ds = ds.map(_trim)
    ds = flatten_parts(ds, ["id", "query", "choices", "answer"])

    def _extract(ex):
        out = dict(image=ex["image"])
        out.update(_add_metadata(ex))
        out["prompt"] = ex["query"]
        out["text"] = ex["answer"]
        options = ex["choices"]
        tf.debugging.assert_equal(tf.reduce_any(tf.strings.regex_full_match(options, ".*\|\|\|.*")), False)
        out["metadata/options"] = tf.strings.reduce_join(options, separator="|||")
        out["metadata/question"] = ex["question"]
        out["metadata/references"] = ex["answer"]
        return out

    ds = ds.map(_extract, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


@seqio.map_over_dataset
def extract_lvis(ex, class_name_file="gs://oe-training-chrisc/cockatoo/data/lvis_class_names.json"):
    with tf.io.gfile.GFile(class_name_file) as f:
        class_names = json.load(f)
    class_names_arr = [None]*len(class_names)
    for k, v in class_names.items():
        class_names_arr[int(k)] = v
    assert all(x is not None for x in class_names_arr)
    class_names_arr = tf.constant(class_names_arr)

    return dict(
        image=ex["image"],
        bbox=ex["objects"]["bbox"],
        label=tf.gather(class_names_arr, ex["objects"]["label"]),
    )


def extract_open_images_boxes(ds):
    # ds = ds.filter(lambda ex: tf.logical_or(
    #     tf.shape(ex["cap/cap_caption"])[0] > 0,
    # tf.shape(ex["detection/bbox"])[0] > 0
    # ))
    ds = ds.filter(lambda ex: tf.shape(ex["cap/cap_caption"])[0] > 0)

    @seqio.map_over_dataset
    def _map(ex):
        bbox = tf.reshape(ex["detection/bbox"], (-1, 4))
        bbox = tf.stack([
            bbox[:, 2],
            bbox[:, 0],
            bbox[:, 3],
            bbox[:, 1]
        ], 1)
        return dict(
            image=tf.image.decode_jpeg(ex["image"]),
            bbox=bbox,
            label=ex["detection/label"],
            caption=tf.strings.reduce_join(ex["cap/cap_caption"], separator="\n")
        )

    return _map(ds)


@seqio.map_over_dataset
def region_captions_to_dense(ex):
    if "captions" in ex:
        captions = ex["captions"]["text"]
        boxes = ex["captions"]["bbox"]
    else:
        captions = ex["label"]
        boxes = ex["bbox"]


    sh = tf.cast(tf.shape(ex["image"])[:2], tf.float32)
    # image_h, image_w = sh[0], sh[1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    cx = tf.cast(boxes[:, 0] + w/2, tf.float32)
    cy = tf.cast(boxes[:, 1] + h/2, tf.float32)
    # w = w / image_w
    # h = h / image_h
    coor = tf.strings.reduce_join(
        float_to_text(tf.stack([cx, cy, w, h], 1)), separator=",", axis=1)

    area = w*h
    if tf.random.uniform(()) < 0.5:
        coor_text = "before"
        captions = tf.strings.join([coor, captions], separator=": ")
    else:
        coor_text = "after"
        captions = tf.strings.join([captions, coor], separator=": ")

    ix = tf.random.uniform((), 0, 6, tf.int32)
    center = boxes
    if ix == 0:
        order_text = "left"
        sort_by = boxes[:, 0]
    elif ix == 1:
        order_text = "right"
        sort_by = -boxes[:, 2]
    elif ix == 2:
        order_text = "top"
        sort_by = boxes[:, 1]
    elif ix == 3:
        order_text = "bottom"
        sort_by = -boxes[:, 3]
    elif ix == 4:
        order_text = "largest"
        sort_by = area
    else:
        order_text = "smallest"
        sort_by = -area
    ixs = tf.argsort(sort_by)
    captions = tf.gather(captions, ixs)
    text = tf.strings.join([
        order_text,
        coor_text,
        tf.strings.reduce_join(captions, separator="\n")
    ], separator="; ")

    if "caption" in ex:
        if tf.random.uniform(()) > 0.5:
            text = tf.strings.join([text, "\ncaption: ", ex["caption"]])
        else:
            text = tf.strings.join(["caption: ", ex["caption"], "\n", text])

    return dict(
        image=ex["image"],
        text=text
    )


@seqio.map_over_dataset()
def join_captions(ex):
    text = tf.random.shuffle(ex['text'])
    ex["text"] = tf.strings.reduce_join(text, separator="\n")
    return ex


@gin.configurable()
@seqio.map_over_dataset(num_seeds=1)
def extract_figureqa(ex, seed, shuffle_questions=True):
    questions = ex["questions"]
    if shuffle_questions:
        n = stateless_permutation(tf.shape(questions["question"])[0], seed)
    else:
        n = tf.range(tf.shape(questions["question"])[0])
    out = {
        "image": ex["image"],
        "questions": tf.gather(questions["question"], n),
        "question_id": tf.gather(questions["question_id"], n),
        "answer": tf.gather(tf.strings.as_string(questions["answer"]), n),
    }
    if "image_id" in ex:
        out["metadata/image_id"] = ex["image_id"]
    if "question_id" in questions:
        out["metadata/question_ids"] = questions["question_id"]
    return out


@seqio.map_over_dataset
def convert_figureqa_answer(ex):
    keys_tensor = tf.constant(["0", "1"])
    values_tensor = tf.constant(["no", "yes"])
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
        default_value=tf.constant("nan", dtype=tf.string),
    )
    answer = table.lookup(ex["answer"])
    ex["answer"] = answer
    return ex


@seqio.map_over_dataset()
def build_question_with_hint(ex):
    hint = ex["hint"]
    if tf.strings.length(hint) > 0:
        ex["question"] = tf.strings.join([hint, ex["question"]], separator="\n")
    # sh = tf.shape(tf.image.decode_image(ex['image'], channels=3, expand_animations=False))
    # ex["metadata/example_hash"] = (
    #     tf.strings.join([hint, ex["question"], ex["lecture"],
    #                      tf.strings.reduce_join(ex["choices"], separator="\t"),
    #                      tf.strings.as_string(sh[0]),
    #                      tf.strings.as_string(sh[1])
    #                      ], separator="\t")
    # )
    return ex

@seqio.map_over_dataset()
def build_question_with_context(ex):
    context = ex["context"]
    if tf.strings.length(context) > 0:
        ex["question"] = tf.strings.join([context, ex["question"]], separator="\n")
    return ex


def max_words(ds, max_words):
    return ds.filter(lambda x: x["n_words"] <= max_words)


@seqio.map_over_dataset
def format_pdfa_eng_wds(example):
    return dict(
        image=example["image"],
        text=tf.strings.reduce_join(example["lines"]["text"], separator="\n"),
    )


@gin.configurable()
def accuracy_conditioned_joint(ds, sequence_length, is_eval=False, eval_quality=17,
                               transcript_quality=None):
    # v2: Transcripts no longer get a quality score
    is_training = sequence_length.get('is_training', True)
    if not is_training:
        if is_eval:
            prompt = f"quality {eval_quality}:"
        else:
            prompt = f"quality 17:"

        @seqio.map_over_dataset
        def _with_prompt(ex):
            out = dict(
                image=ex["image"],
                url=ex["url"],
                prompt=prompt,
            )
            if "text" in ex:
                out["text"] = ex["text"]
            elif "caption" in ex:
                out["text"] = ex["caption"]
            return out
        return _with_prompt(ds)

    elif is_eval:
        raise ValueError("is_eval=True and is_training=False")

    # each transcript
    @seqio.map_over_dataset
    def _with_transcript(ex):
        if tf.shape(ex["edited_captions"]["caption"])[0] > 0:
            edited_caption = ex["edited_captions"]["caption"][0]
            n = ex["edited_captions"]["n_edits"][0]
        else:
            edited_caption = ""
            n = 0
        text = [
            ex["caption"],
            ex["transcripts"][tf.random.uniform((), 0, tf.shape(ex["transcripts"])[0], dtype=tf.int32)],
            edited_caption
        ]
        edit_quality = 17 - n
        prompt = [
            "quality 17:",
            "" if transcript_quality is None else f"quality: {edit_quality}:",
            tf.strings.join(["quality ", tf.strings.as_string(edit_quality), ":"])
        ]
        return dict(
            image=ex["image"],
            text=tf.stack(text, 0),
            url=ex["url"],
            prompt=tf.stack(prompt, 0),
            style=["long_caption", "transcript", "long_caption"]
        )
    return _with_transcript(ds)


def select_dense_caption_sample(ds, samples=200):
    def compute_hash(string: str) -> str:
        return hashlib.sha256(string.encode("utf-8")).hexdigest()

    with tf.io.gfile.GFile("gs://oe-training-chrisc/cockatoo/data/dense-caption-eval-v0-final-data.json") as f:
        data = json.load(f)
    for ex in data:
        ex["image_id"] = compute_hash(ex["image"])
    data.sort(key=lambda x: x["image_id"])
    np.random.RandomState(12312).shuffle(data)
    keep = tf.constant([x["image"] for x in data[:samples]])

    def _keep(ex):
        return tf.reduce_any(ex["url"] == keep)
    ds = ds.filter(_keep)
    ds = tf.data.experimental.assert_cardinality(samples)(ds)
    return ds

@seqio.map_over_dataset()
def charxiv_preprocessor(ex):
    question_names = ["descriptive_q1", "descriptive_q2", "descriptive_q3", "descriptive_q4", "reasoning_q"]
    answer_names = ["descriptive_a1", "descriptive_a2", "descriptive_a3", "descriptive_a4", "reasoning_a"]

    questions = [ex[name] for name in question_names]
    answers = [ex[name] for name in answer_names]

    return dict(
        image=ex["image"],
        question=tf.stack(questions, 0),
        answer=tf.stack(answers, 0)
    )

@seqio.map_over_dataset()
def charxiv_descriptive_preprocessor(ex):
    question_names = ["descriptive_q1", "descriptive_q2", "descriptive_q3", "descriptive_q4"]
    answer_names = ["descriptive_a1", "descriptive_a2", "descriptive_a3", "descriptive_a4"]

    questions = [ex[name] for name in question_names]
    answers = [ex[name] for name in answer_names]

    return dict(
        image=ex["image"],
        question=tf.stack(questions, 0),
        answer=tf.stack(answers, 0)
    )

@seqio.map_over_dataset()
def charxiv_reasoning_preprocessor(ex):
    return dict(
        image=ex["image"],
        question=ex["reasoning_q"],
        answer=ex["reasoning_a"]
    )

@seqio.map_over_dataset()
def tablevqa_preprocessor(ex):
    return dict(
        image=ex["image"],
        question=ex["question"],
        answer=ex["gt"]
    )

@seqio.map_over_dataset()
def vtabfact_preprocessor(ex):
    return dict(
        image=ex["image"],
        question=tf.strings.join([ex["question"], "Answer with yes or no."], separator="\n"),
        answer=ex["gt"]
    )

@seqio.map_over_dataset()
def nutrition_fact_preprocessor(ex):
    question_names = ["descriptive_q", "reasoning_q"]
    answer_names = ["descriptive_a", "reasoning_a"]

    questions = [ex[name] for name in question_names]
    answers = [ex[name] for name in answer_names]

    return dict(
        image=ex["image"],
        question=tf.stack(questions, 0),
        answer=tf.stack(answers, 0)
    )
