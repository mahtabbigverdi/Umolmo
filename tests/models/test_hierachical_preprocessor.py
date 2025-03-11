from collections import Counter

import numpy as np
import pytest

from olmo.models.he_molmo.he_preprocessor import HeMultiModalPreprocessor
from olmo.tokenizer import build_tokenizer, get_special_token_ids, DEFAULT_IM_START_TOKEN, \
    IMAGE_PROMPT, DEFAULT_IM_END_TOKEN, DEFAULT_IM_COL_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN


def test_preprocessing(
    col_tokens: bool=False, multi_res_selection=None, max_crops=4,
    siglip=False
):
    tokenizer = build_tokenizer("Qwen/Qwen2-7B")
    n_high_res = 256

    special_token_ids = get_special_token_ids(tokenizer)
    start_token_id = special_token_ids[DEFAULT_IM_START_TOKEN]
    end_token_id = special_token_ids[DEFAULT_IM_END_TOKEN]
    col_token_id = special_token_ids[DEFAULT_IM_COL_TOKEN]
    patch_token_id = special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]

    preprocessor = HeMultiModalPreprocessor(
        tokenizer=tokenizer,
        crop_mode="overlap-and-resize-c2",
        max_crops=max_crops,
        resize="metaclip",
        num_high_res_features=n_high_res,
        multi_res_selection=multi_res_selection,
        multi_res_min=32,
        use_high_res_col_tokens=col_tokens,
    )
    if siglip:
        preprocessor.base_image_input_size = (378, 378)
        preprocessor.image_token_length_h = 14
        preprocessor.image_token_length_w = 14
        preprocessor.overlap_margins = (4, 3)
    tokens_per_image = preprocessor.tokens_per_image

    batch = preprocessor(
        images=np.zeros((500, 500, 3), dtype=np.uint8),
        messages=["A simple question" + IMAGE_PROMPT + "The", " answer is 3"],
        rng=np.random,
    )
    input_ids = batch["input_tokens"]
    position_ids = batch["position_ids"]
    high_res_pos_ids = batch["high_res_patch_data"][:, :, tokens_per_image*4]
    first_image_start = np.argmax(input_ids == start_token_id)

    # Check the first set of tokens which should be the input query
    assert input_ids[0] == tokenizer.bos_token_id
    if multi_res_selection:
        assert (input_ids[1:] == tokenizer.eos_token_id).sum() == (multi_res_selection - 1)
    else:
        assert (input_ids[1:] == tokenizer.eos_token_id).sum() == 0
    assert tokenizer.decode(input_ids[:first_image_start]) == "A simple question"
    assert np.all(position_ids[:first_image_start] == np.arange(first_image_start))

    # Now check the low-res image
    second_image_start = 1 + np.argmax(input_ids == end_token_id)
    low_res_counts = Counter(input_ids[first_image_start:second_image_start])
    assert low_res_counts[start_token_id] == 1
    assert low_res_counts[end_token_id] == 1
    assert low_res_counts[col_token_id] == preprocessor.image_token_length_w
    assert low_res_counts[patch_token_id] == tokens_per_image
    assert len(low_res_counts) == 4
    assert np.all(position_ids[:second_image_start] == np.arange(second_image_start))

    second_image_end = second_image_start + np.argmax(input_ids[second_image_start:] == end_token_id) + 1

    # Check the high-res index, it should map to the high-res patch tokens
    image_input_idx = batch["image_input_idx"]
    high_res_idx = image_input_idx[tokens_per_image:]
    high_res_idx = high_res_idx[high_res_idx >= 0]
    assert np.unique(high_res_idx, return_counts=True)[1].max() == 1, \
        "high_res_idx should map to unique positions"
    assert np.all(input_ids[high_res_idx] == patch_token_id), \
        "high_res_idx should map to pos tokens"
    assert (input_ids[second_image_start:] == patch_token_id).sum() == len(high_res_idx), \
        "high_res_idx should cover allpos tokens"

    # Check the high-res token inputs ids
    high_ids = input_ids[second_image_start:second_image_end]
    high_pos = position_ids[second_image_start:second_image_end]
    assert high_ids[0] == start_token_id
    assert high_pos[0] == second_image_start
    assert high_ids[-1] == end_token_id
    assert np.all(high_pos[-1] > high_pos[:-1])
    assert high_pos[-1] > high_pos[0] + (300 if max_crops > 1 else tokens_per_image)
    assert np.unique(high_pos[high_ids != patch_token_id], return_counts=True)[1].max() == 1

    # Check the high-res patch position ids
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

    # Check the output(s)
    if multi_res_selection:
        res_start = second_image_end
        prev_subsegment_id = None
        for response_ix in range(multi_res_selection):
            if response_ix == (multi_res_selection - 1):
                res_end = len(input_ids)
            else:
                res_end = res_start + np.argmax(input_ids[res_start:] == tokenizer.eos_token_id)
            assert tokenizer.decode(input_ids[res_start:res_end]) == "The answer is 3"
            pos_start = position_ids[res_start]
            assert high_pos[-1] == pos_start - 1
            assert np.all(position_ids[res_start:res_end] == pos_start + np.arange(res_end-res_start))
            subsegment_ids = batch["subsegment_ids"]
            subsegment_id = subsegment_ids[res_start]
            assert np.all(subsegment_ids[res_start:res_end] == subsegment_id)

            # Cannot attend to the previous segment(s)
            if prev_subsegment_id is not None:
                assert prev_subsegment_id != subsegment_id

            # Can attend to all columns tokens
            if col_tokens:
                assert np.all(subsegment_ids[position_ids == col_token_id] <= subsegment_id)

            # Can attend to start/end tokens
            assert np.all(subsegment_ids[(input_ids == start_token_id) | (input_ids == end_token_id)] == 0)
            assert subsegment_id > subsegment_ids[second_image_start]

            # Can attent to the low res/query
            assert np.all(subsegment_id > subsegment_ids[:second_image_start])

            # Sanity check can attend to feasible number of image pathess
            can_attend = subsegment_ids <= subsegment_id
            n_patches_can_attend = ((input_ids == patch_token_id) & can_attend).sum()
            n_high_res_patches_can_attend = n_patches_can_attend - 144
            assert 1 < n_high_res_patches_can_attend < n_high_res
            res_start = res_end + 1
            subsegment_id = prev_subsegment_id
    else:
        assert tokenizer.decode(input_ids[second_image_end:]) == "The answer is 3"
        assert high_pos[-1] == position_ids[second_image_end:].min() - 1
        assert np.all(position_ids[second_image_end:] ==
                      (high_pos[-1] + 1 + np.arange(len(input_ids[second_image_end:]))))


@pytest.mark.parametrize("col_tokens", [True, False])
@pytest.mark.parametrize("max_crops", [1, 4])
@pytest.mark.parametrize("multi_res_selection", [None])  # No longer trying to use multi-res
@pytest.mark.parametrize("siglip", [True, False])
def test_preprocessor(col_tokens, multi_res_selection, max_crops, siglip):
    test_preprocessing(col_tokens, multi_res_selection, max_crops=max_crops, siglip=siglip)
