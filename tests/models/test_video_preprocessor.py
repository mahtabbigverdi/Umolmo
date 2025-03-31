import re
from collections import Counter

import numpy as np
from olmo import tokenizer

from olmo.models.video_olmo.video_preprocessor import VideoTextPreprocessor, VIDEO_SUBSEGMENT_ID
from olmo.tokenizer import build_tokenizer, get_special_token_ids


def _remove_video_text(text):
    return re.sub(fr"(time [0-9]|FPS)(.*{tokenizer.IM_END_TOKEN})+", tokenizer.IMAGE_PROMPT, text)


def _test_preprocessor(n_frames, message_list, periodic_high_res_frame=None):
    if isinstance(message_list[0], str):
        n_messages = 1
    else:
        n_messages = len(message_list)

    tok = build_tokenizer("Qwen/Qwen2-7B")
    pre = VideoTextPreprocessor(
        tokenizer=tok,
        periodic_high_res_frame=periodic_high_res_frame,
        crop_mode="resize",
        image_pooling_h=3, image_pooling_w=3,
        high_res_pooling_h=2, high_res_pooling_w=2,
        image_patch_size=14,
        base_image_input_size=(378, 378)
    )
    video = np.zeros((n_frames, 400, 400, 3), dtype=np.uint8)
    batch = pre(
        frames=video,
        frame_times=np.arange(len(video)),
        message_list=message_list,
        rng=np.random
    )
    input_tokens = batch["input_tokens"]

    # Check some basic invariants
    assert batch["images"].shape == (n_frames, (378//14)**2, 14*14*3)
    assert input_tokens[0] == tok.bos_token_id
    assert batch["target_tokens"][-1] == tok.eos_token_id
    assert batch["loss_masks"][-1] == 1.0

    # Sanity check the special tokens by checking the counts
    counts = Counter(input_tokens.tolist())
    assert counts[tok.image_low_res_token_id] == len(batch["low_res_pooled_idx"])
    if periodic_high_res_frame:
        assert counts[tok.image_patch_token_id] == len(batch["high_res_pooled_idx"])
    else:
        assert counts[tok.image_patch_token_id] == 0
    assert counts[tok.image_end_token_id] == len(video)
    assert counts[tok.image_start_token_id] == len(video)
    if pre.use_col_tokens:
        if not periodic_high_res_frame:
            assert counts[tok.image_col_token_id] == len(video) * 9

    # Check text and position ids
    if isinstance(message_list[0], str):
        actual = tok.decode(input_tokens, False)
        expected = "".join(message_list)
        if tokenizer.IMAGE_PROMPT not in expected:
            expected = tokenizer.IMAGE_PROMPT + expected
        assert _remove_video_text(actual) == expected
        assert np.all(batch["position_ids"] == np.arange(len(input_tokens)))
    else:
        subsegments = batch["subsegment_ids"]
        pos_ids = batch["position_ids"]
        for ix in range(len(message_list)):
            expected = "".join(message_list[ix])
            if tokenizer.IMAGE_PROMPT not in expected:
                expected = tokenizer.IMAGE_PROMPT + expected
            msg_mask = (subsegments == ix) | (subsegments == VIDEO_SUBSEGMENT_ID)
            assert np.all(pos_ids[msg_mask] - pos_ids[msg_mask][0] == np.arange(msg_mask.sum()))
            seg_tokens = input_tokens[msg_mask]
            seg_targets = batch["target_tokens"][msg_mask]
            seg_loss = batch["loss_masks"][msg_mask]
            assert seg_tokens[0] == tok.bos_token_id
            assert seg_targets[-1] == tok.eos_token_id


def test_basic():
    _test_preprocessor(
        n_frames=3,
        message_list=["What is this?", "A cat"],
    )


def test_high_res():
    _test_preprocessor(
        n_frames=13,
        periodic_high_res_frame=2,
        message_list=["What is this?", "A cat"],
    )


def test_video_in_middle():
    _test_preprocessor(
        n_frames=3,
        message_list=[f"What is {tokenizer.IMAGE_PROMPT} this?", "A cat"],
    )


def test_multi_message():
    _test_preprocessor(
        n_frames=2,
        message_list=[
            [f"What is this?", "A cat"],
            [f"He", "False"],
            [f"Another", "a few words"],
        ]
    )


def test_multi_message_video_middle():
    _test_preprocessor(
        n_frames=2,
        message_list=[
            [f"What is {tokenizer.IMAGE_PROMPT} this?", "A cat"],
            [f"A video {tokenizer.IMAGE_PROMPT}.", "another answer"],
            [f"Video first", "the answer"],
        ]
    )