from collections import Counter

import numpy as np
import pytest

from olmo.models.he_molmo.he_preprocessor import HeMultiModalPreprocessor
from olmo.tokenizer import build_tokenizer, get_special_token_ids, IM_START_TOKEN, \
    IMAGE_PROMPT, IM_END_TOKEN, IM_COL_TOKEN, IMAGE_PATCH_TOKEN
from olmo.util import flatten_lists


def test_preprocessing(col_tokens: bool=False, max_crops=4, siglip=False, multi_message=False):
    tokenizer = build_tokenizer("Qwen/Qwen2-7B")
    n_high_res = 256

    special_token_ids = get_special_token_ids(tokenizer)
    start_token_id = special_token_ids[IM_START_TOKEN]
    end_token_id = special_token_ids[IM_END_TOKEN]
    col_token_id = special_token_ids[IM_COL_TOKEN]
    patch_token_id = special_token_ids[IMAGE_PATCH_TOKEN]

    preprocessor = HeMultiModalPreprocessor(
        tokenizer=tokenizer,
        crop_mode="overlap-and-resize-c2",
        max_crops=max_crops,
        resize="metaclip",
        num_high_res_features=n_high_res,
        multi_res_selection=None,
        multi_res_min=None,
        use_high_res_col_tokens=col_tokens,
    )
    if siglip:
        preprocessor.base_image_input_size = (378, 378)
        preprocessor.image_token_length_h = 14
        preprocessor.image_token_length_w = 14
        preprocessor.overlap_margins = (4, 3)

    if multi_message:
        messages = [
            ["Here is question1", " answer is 2"],
            ["And here is a longer question3", " 3"],
            ["Q3", " a longer answer"]
        ]
    else:
        messages = ["A simple question" + IMAGE_PROMPT + "The", " answer is 3"]
    batch = preprocessor(
        images=np.zeros((500, 500, 3), dtype=np.uint8),
        messages=messages,
        rng=np.random,
    )
    input_ids = batch["input_tokens"]
    position_ids = batch["position_ids"]
    first_image_start = np.argmax(input_ids == start_token_id)
    high_res_pos_ids = batch["high_res_pos_ids"]

    # low-to-high should make sense
    assert np.allclose(batch["low_to_high"].sum(0), 1)

    # Check the first set of tokens which should be the input query
    if not multi_message:
        assert input_ids[0] == tokenizer.bos_token_id
        assert (input_ids[1:] == tokenizer.eos_token_id).sum() == 0
        assert tokenizer.decode(input_ids[:first_image_start]) == "A simple question"
        assert np.all(position_ids[:first_image_start] == np.arange(first_image_start))

    # Now check the low-res image
    second_image_start = 1 + np.argmax(input_ids == end_token_id)
    low_res_counts = Counter(input_ids[first_image_start:second_image_start])
    assert low_res_counts[start_token_id] == 1
    assert low_res_counts[end_token_id] == 1
    assert low_res_counts[col_token_id] == (14 if siglip else 12)
    assert len(low_res_counts) == 4
    assert np.all(position_ids[:second_image_start] == np.arange(second_image_start))

    second_image_end = second_image_start + np.argmax(input_ids[second_image_start:] == end_token_id) + 1

    # Check the high-res token inputs ids
    high_ids = input_ids[second_image_start:second_image_end]
    high_pos = position_ids[second_image_start:second_image_end]
    assert high_ids[0] == start_token_id
    assert high_pos[0] == second_image_start
    assert high_pos[1] == second_image_start + 1
    assert high_ids[-1] == end_token_id
    assert np.all(high_pos[-1] > high_pos[:-1])
    assert np.unique(high_pos[high_ids != patch_token_id], return_counts=True)[1].max() == 1

    # Check the high-res position ids
    possible_high_res_positions = high_res_pos_ids.ravel()[high_res_pos_ids.ravel() >= 0]
    high_res_patch_base_pos_id = high_pos[1]
    assert np.all(high_pos[high_ids == patch_token_id] == high_res_patch_base_pos_id)
    assert high_res_patch_base_pos_id == high_pos[0] + 1

    # Check the high-res col tokens and patch ids
    if col_tokens:
        assert (high_ids == col_token_id).sum() > 1
        col_pos_ids = high_pos[high_ids == col_token_id]
        joint_pos_ids = np.concatenate([possible_high_res_positions+high_res_patch_base_pos_id, col_pos_ids])
        assert np.all(np.sort(joint_pos_ids) == np.arange(high_res_patch_base_pos_id, high_res_patch_base_pos_id+len(joint_pos_ids)))
    else:
        assert np.all(np.sort(possible_high_res_positions) == np.arange(len(possible_high_res_positions)))

    # Check text after the image:
    if not multi_message:
        assert tokenizer.decode(input_ids[second_image_end:]) == "The answer is 3"
        assert high_pos[-1] == position_ids[second_image_end:].min() - 1
        assert np.all(position_ids[second_image_end:] ==
                      (high_pos[-1] + 1 + np.arange(len(input_ids[second_image_end:]))))
    else:
        subsegment_ids = batch["subsegment_ids"]
        for i in range(len(messages)):
            mask = (subsegment_ids == i) | (subsegment_ids == 10000)
            assert np.all(mask[:second_image_end])
            text_mask = np.copy(mask)
            text_mask[:second_image_end] = 0
            text = tokenizer.decode(input_ids[text_mask])
            assert text == "".join(messages[i])
            # assert np.all(
            #     position_ids[text_mask] ==
            #     (np.arange(text_mask.sum()) + position_ids[second_image_end-1]+1)
            # )


@pytest.mark.parametrize("col_tokens", [True, False])
@pytest.mark.parametrize("max_crops", [1, 4])
@pytest.mark.parametrize("siglip", [True, False])
def test_preprocessor(col_tokens, max_crops, siglip):
    test_preprocessing(col_tokens, max_crops=max_crops, siglip=siglip)


def test_preprocessor_multi_message():
    test_preprocessing(True, max_crops=2, siglip=False, multi_message=True)


if __name__ == '__main__':
    test_preprocessing(False, max_crops=4, siglip=False, multi_message=False)