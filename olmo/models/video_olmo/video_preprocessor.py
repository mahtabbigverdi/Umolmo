import dataclasses
from dataclasses import dataclass
from os.path import basename, dirname
from typing import List, Optional, Union, Tuple, Any

import decord
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from olmo import tokenizer
from olmo.data.image_preprocessor import ImagePreprocessor, load_image
from olmo.data.interleaved_text_preprocessor import InterleavedTextPreprocessor
from olmo.io import resource_path
from olmo.models.he_molmo.he_preprocessor import HeMultiModalPreprocessor
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.tokenizer import get_special_token_ids

decord.logging.set_level(2)
from decord import VideoReader, cpu

import concurrent.futures

from olmo.models.molmo.model_preprocessor import MolmoPreprocessorConfig, \
    batch_pixels_to_patches, arange_for_pooling

from olmo.nn.vision_backbone import MolmoVisionBackboneConfig


def get_sampling_fps(
    video_fps: float,
    max_frames: int,
    total_frames: int,
    frame_sample_mode: str,
    candidate_sampling_fps: Tuple[float],
) -> float:
    """
    Get the sampling fps that best spans the video and has the most frames sampled
    """
    if frame_sample_mode == "uniform":
        return None

    num_frames_sampled = 0
    selected_sampling_fps = None
    for sampling_fps in candidate_sampling_fps:
        step_size = max(int(video_fps / sampling_fps), 1)
        num_frames_sampled_at_fps = int(total_frames / step_size)
        if num_frames_sampled == 0:
            if "uniform" in frame_sample_mode:
                if num_frames_sampled_at_fps > max_frames:
                    break
            selected_sampling_fps = sampling_fps
            num_frames_sampled = num_frames_sampled_at_fps

        else:
            # the candidate sampling fps increases so frame count can't decrease
            assert num_frames_sampled <= num_frames_sampled_at_fps
            if num_frames_sampled_at_fps > max_frames:
                # choose the sampling fps that spans the video
                continue

            elif num_frames_sampled_at_fps > num_frames_sampled:
                # both are less than max_frames, choose the one with higher density of frames sampled
                selected_sampling_fps = sampling_fps
                num_frames_sampled = num_frames_sampled_at_fps
    return selected_sampling_fps


def load_decord_video(
    video_path: str,
    max_frames: int = 8,
    use_timeout: bool = False,
    frame_sample_mode: str = "fps",
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
) -> np.ndarray:
    """
    Load a video and returns frames as RGB numpy array.
    """
    video_path = resource_path(dirname(video_path), basename(video_path)).as_posix()
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

    # Get video properties
    video_fps = vr.get_avg_fps()  # Get FPS
    total_frames = len(vr)  # Total frame count
    selected_sampling_fps = get_sampling_fps(video_fps, max_frames, total_frames, frame_sample_mode, candidate_sampling_fps)

    if selected_sampling_fps is None:
        frame_indices = np.linspace(0, total_frames, max_frames, endpoint=False)
    else:
        step_size = max(int(video_fps / selected_sampling_fps), 1)
        frame_indices = np.arange(0, total_frames, step_size)
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]

    frame_times = [i / int(video_fps) for i in frame_indices]

    if use_timeout:
        # Fetch frames in batch (faster than looping)
        def fetch_frames():
            """Function to get frames in a separate thread."""
            return vr.get_batch(frame_indices).asnumpy()

        # Run the function with a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch_frames)
            try:
                # Enforce timeout
                return future.result(timeout=1), frame_times
            except concurrent.futures.TimeoutError:
                print(f"Timeout! Frame extraction took longer than 1 seconds.")
                return None, None  # Return None to indicate failure

    else:
        return vr.get_batch(frame_indices).asnumpy(), frame_times


def load_pyav_video(
    video_path: str,
    max_frames: int = 8,
    frame_sample_mode: str = "fps",
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
) -> np.ndarray:
    """
    Load a video and returns frames as RGB numpy array.
    """
    video_path = resource_path(dirname(video_path), basename(video_path)).as_posix()
    meta = iio.immeta(video_path)
    video_fps = meta["fps"]
    duration = meta["duration"]
    total_frames = int(np.floor(video_fps * duration))

    selected_sampling_fps = get_sampling_fps(video_fps, max_frames, total_frames, frame_sample_mode, candidate_sampling_fps)

    if selected_sampling_fps is None:
        frame_indices = np.linspace(0, total_frames, max_frames, endpoint=False)
    else:
        step_size = max(int(video_fps / selected_sampling_fps), 1)
        frame_indices = np.arange(0, total_frames, step_size)

    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    frame_times = [i / int(video_fps) for i in frame_indices]

    frame_idx, frames = 0, []
    for frame in iio.imiter(video_path, plugin="pyav"):
        if frame_idx in frame_indices:
            frames.append(frame)
        frame_idx += 1
        if len(frames) == max_frames:
            break
    return np.stack(frames), frame_times


def load_video_decord_or_pyav(
    video_path: str,
    max_frames: int = 24,
    frame_sample_mode: str = "fps",
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
) -> Tuple[np.ndarray, List[float]]:
    """
    Load a video and returns frames as RGB numpy array.
    """
    try:
        outputs = load_decord_video(video_path, max_frames, frame_sample_mode=frame_sample_mode, candidate_sampling_fps=candidate_sampling_fps)
    except Exception as e:
        outputs = load_pyav_video(video_path, max_frames, frame_sample_mode=frame_sample_mode, candidate_sampling_fps=candidate_sampling_fps)

    return outputs


def get_image_collage(frames: np.ndarray, frame_size: int = 128) -> np.ndarray:
    """
    Creates a collage of frames arranged in a Nx4 grid in reading order.
    Each frame is resized to 224x224 while maintaining aspect ratio.

    Args:
        frames: numpy array of shape (num_frames, height, width, channels)
        frame_size: size of each frame in the collage (default: 224)

    Returns:
        collage: numpy array of shape (N*224, 896, 3) where N is ceil(num_frames/4)
    """
    num_frames = len(frames)
    num_rows = (num_frames + 3) // 4  # Ceiling division for number of columns
    # Create black canvas of appropriate size
    canvas = np.zeros((num_rows * frame_size, 4 * frame_size, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // 4
        col = idx % 4

        largest_dim = max(frame.shape[0], frame.shape[1])
        square_frame = np.zeros((largest_dim, largest_dim, 3), dtype=np.uint8)

        # Center the frame in the square
        square_frame[
        (largest_dim - frame.shape[0]) // 2:(largest_dim - frame.shape[0]) // 2 + frame.shape[0],
        (largest_dim - frame.shape[1]) // 2:(largest_dim - frame.shape[1]) // 2 + frame.shape[1]
        ] = frame

        resized = Image.fromarray(square_frame).resize((frame_size, frame_size), Image.Resampling.BILINEAR)
        resized = np.array(resized)

        canvas[row * frame_size:(row + 1) * frame_size, col * frame_size:(col + 1) * frame_size] = resized

    return canvas


@dataclass
class MultiModalVideoPreprocessorConfig(MolmoPreprocessorConfig):
    time_mode: str = "per-frame"

    crop_mode: str = "frame_sampling"
    """How to divide the images into crops"""

    max_frames: Optional[int] = None
    """Max number of frames to sample from the video"""

    frame_sample_mode: str = "fps"
    """How to sample the frames from the video"""

    periodic_high_res_frame: Optional[int] = None
    """Periodic high resolution frame rate"""

    high_res_pooling_w: Optional[int] = None
    """High res pooling w stride"""

    high_res_pooling_h: Optional[int] = None
    """High res pooling h stride"""

    max_text_tokens: Optional[int] = None
    """For multi-message data, the drop message the if text token length is larger then this"""

    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    """Candidate sampling fps to sample the frames from the video"""

    def get_image_padding_lens(self, vision_backbone_config: MolmoVisionBackboneConfig):
        """Max numbers of image tokens can be built for one image"""
        padding_lens = dict(
            images=self.get_max_crops()*self.max_frames
        )
        if vision_backbone_config.image_padding_embed:
            padding_lens["image_masks"] = self.get_max_crops()*self.max_frames
        preprocessor = self.build(None, vision_backbone_config, None)
        vit = vision_backbone_config.vit
        h, w = vit.image_default_input_size
        low_res_tokens = preprocessor.compute_num_tokens(h, w, self.pooling_h, self.pooling_w)

        if self.periodic_high_res_frame:
            high_res_tokens = preprocessor.compute_num_tokens(h, w, self.high_res_pooling_h, self.high_res_pooling_w)
            n_high_res = 1 + (self.max_frames - 1) // self.periodic_high_res_frame
            n_low_res = self.max_frames - n_high_res
            padding_lens["low_res_pooled_idx"] = low_res_tokens*n_low_res
            padding_lens["high_res_pooled_idx"] = high_res_tokens*n_high_res
        else:
            padding_lens["low_res_pooled_idx"] = low_res_tokens*self.max_frames
        return padding_lens

    def get_max_image_tokens(self, vision_backbone_config: MolmoVisionBackboneConfig):
        lens = self.get_image_padding_lens(vision_backbone_config)
        seq_len = lens["low_res_pooled_idx"] + lens.get("high_res_pooled_idx", 0)
        extra_per_frame = 2  # start/end tokens
        if self.time_mode == "per-frame":
            extra_per_frame += 8  # time markers
        elif self.time_mode == "fps-prefix":
            seq_len += 10  # time prefix
        else:
            assert self.time_mode is None
        if self.use_col_tokens:
            sz = vision_backbone_config.vit.image_default_input_size
            patch_size = vision_backbone_config.vit.image_patch_size
            extra_per_frame += (sz[1] // patch_size // self.pooling_w)
        seq_len += self.max_frames * extra_per_frame
        return seq_len

    def build(self, tokenizer, vision_backbone_config: MolmoVisionBackboneConfig, max_sequence_length) -> 'VideoTextPreprocessor':
        vit = vision_backbone_config.vit
        return VideoTextPreprocessor(
            tokenizer=tokenizer,
            loss_token_weighting=self.loss_token_weighting,
            normalize=vit.normalize,
            crop_mode=self.crop_mode,
            max_crops=self.max_crops,
            overlap_margins=self.overlap_margins,
            resize=vit.resize_mode,
            use_col_tokens=self.use_col_tokens,
            base_image_input_size=vit.image_default_input_size,
            image_pooling_w=self.pooling_w,
            image_pooling_h=self.pooling_h,
            high_res_pooling_w=self.high_res_pooling_h,
            high_res_pooling_h=self.high_res_pooling_w,
            periodic_high_res_frame=self.periodic_high_res_frame,
            image_patch_size=vit.image_patch_size,
            max_text_tokens=self.max_text_tokens,
            max_sequence_length=max_sequence_length,
            image_padding_mask=vision_backbone_config.image_padding_embed is not None,
            pad_value=vit.pad_value,
            time_mode=self.time_mode
        )

    def __post_init__(self):
        if self.candidate_sampling_fps is not None:
            self.candidate_sampling_fps = tuple(self.candidate_sampling_fps)  # type: ignore[assignment]


VIDEO_SUBSEGMENT_ID = 10000


@dataclass
class VideoTextPreprocessor(InterleavedTextPreprocessor, ImagePreprocessor):
    crop_mode: str = "default"
    image_padding_mask: bool = False
    use_col_tokens: bool = True
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    high_res_pooling_w: Optional[int] = None
    high_res_pooling_h: Optional[int] = None
    time_mode: str = "per-frame"
    max_text_tokens: int = None
    periodic_high_res_frame: Optional[int] = None

    def compute_num_tokens(self, image_h, image_w, pool_h, pool_w) -> int:
        """Return the number of pooled image tokens produced for an image of size image_w, image_h"""
        image_patch_size = self.image_patch_size
        crop_patch_w = self.base_image_input_size[1] // image_patch_size
        crop_patch_h = self.base_image_input_size[0] // image_patch_size

        resize_idx = np.zeros([crop_patch_h, crop_patch_w])
        idx_arr = arange_for_pooling(resize_idx, pool_h, pool_w)
        resize_tokens = idx_arr.shape[0] * idx_arr.shape[1]

        if self.crop_mode in ["resize"]:
            return resize_tokens

        h, w = self.compute_overlapping_crops_size(image_h, image_w)
        idx_arr = arange_for_pooling(torch.zeros([h, w]), pool_h, pool_w)
        overlap_tokens = idx_arr.shape[0] * idx_arr.shape[1]
        if self.crop_mode in ["overlap-and-resize-c2"]:
            return overlap_tokens + resize_tokens
        else:
            return overlap_tokens

    def image_to_patches_and_tokens(
        self,
        image,
        pooling_h: int,
        pooling_w: int,
        patch_id: int,
        is_training=False,
        rng=None,
    ):
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode == "resize":
            resized, resized_mask, resize_idx = self.build_resized_image(image, is_training=is_training, rng=rng)
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
            per_row = np.full(
                (w,),
                patch_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                [self.tokenizer.image_start_token_id],
                extra_tokens,
                [self.tokenizer.image_end_token_id],
            ]
            return (np.concatenate(joint, 0), batch_pixels_to_patches(resized, image_patch_size),
                    batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1), pooling_idx)
        else:
            raise ValueError()

    def __call__(
        self,
        frames,
        frame_times,
        message_list: Union[List[str], List[List[str]]],
        weight=None,
        is_training=False,
        rng=None
    ):
        """
        Interleave video and text tokens into multi-modal features for the model
        """
        frame_id_token_ids = []
        for frame_idx, frame_time in enumerate(frame_times):
            if self.time_mode == "per-frame":
                prev_space = " " if frame_idx > 0 else ""
                frame_id = prev_space + f"time {frame_time:.2f} " # explicit whitespace before/after image tokens
                frame_id_token_ids.append(self.tokenizer.encode(frame_id))
            else:
                frame_id_token_ids.append(None)

        all_frame_patches = []
        low_res_pooled_idx = []
        high_res_pooled_idx = []
        video_masks = []
        video_tokens = []

        if self.time_mode == "fps-prefix":
            tmp = np.array(frame_times)
            deltas = tmp[1:] - tmp[:-1]
            fps = np.mean(deltas)
            prefix = self.tokenizer.encode(f"FPS {fps:0.2f}")
            video_tokens.append(prefix)

        for frame_idx, frame in enumerate(frames):
            frame_pooling_w = self.image_pooling_w
            frame_pooling_h = self.image_pooling_h
            patch_id = self.tokenizer.image_low_res_token_id
            is_high_res = self.periodic_high_res_frame is not None and frame_idx % self.periodic_high_res_frame == 0
            if is_high_res:
                # If the frame is a high res frame, use the high res token length
                frame_pooling_w = self.high_res_pooling_w
                frame_pooling_h = self.high_res_pooling_h
                patch_id = self.tokenizer.image_patch_token_id

            frame_tokens, frame_patches, frame_masks, pooled_idx = self.image_to_patches_and_tokens(
                frame, frame_pooling_w, frame_pooling_h, patch_id, is_training, rng)
            offset = sum(np.prod(x.shape[:2]) for x in all_frame_patches)
            pooled_idx = np.where(
                pooled_idx >= 0,
                pooled_idx + offset,
                pooled_idx
            )
            if is_high_res:
                high_res_pooled_idx.append(pooled_idx)
            else:
                low_res_pooled_idx.append(pooled_idx)
            all_frame_patches.append(frame_patches)
            video_masks.append(frame_masks)
            if frame_id_token_ids[frame_idx]:
                video_tokens.append(np.array(frame_id_token_ids[frame_idx], dtype=np.int32))
            video_tokens.append(frame_tokens)

        all_frame_patches = np.concatenate(all_frame_patches, 0)
        video_tokens = np.concatenate(video_tokens, 0)

        out = self.tokenize_and_interleave(
            message_list,
            [video_tokens],
            weight=weight
        )
        out["images"] = all_frame_patches
        if self.image_padding_mask:
            out["image_masks"] = np.concatenate(video_masks, 0)
        if low_res_pooled_idx:
            out["low_res_pooled_idx"] = np.concatenate(low_res_pooled_idx, 0)
        if high_res_pooled_idx:
            out["high_res_pooled_idx"] = np.concatenate(high_res_pooled_idx, 0)
        return out


@dataclass
class VideoPreprocessor:
    formater: DataFormatter
    mm_preprocessor: Union[VideoTextPreprocessor, HeMultiModalPreprocessor]
    for_inference: bool = False
    is_training: bool = False
    frame_sample_mode: str = "fps"
    include_image: bool = False
    max_frames: int = 24
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    image_to_video: Any = None

    def __post_init__(self):
        self.candidate_sampling_fps = tuple(self.candidate_sampling_fps)  # type: ignore[assignment]

    def __call__(self, example, rng=np.random):
        example = dict(example)

        if self.image_to_video and "image" in example:
            image = load_image(example["image"])
            frames, frame_times, patch_ids = self.image_to_video(image, rng)
        else:
            patch_ids = None
            assert "video" in example, "Video is required for video preprocessor"
            try:
                frames, frame_times = load_video_decord_or_pyav(example["video"], self.max_frames, self.frame_sample_mode, self.candidate_sampling_fps)
            except Exception as e:
                e.add_note(f"Could not load video: {example['video']}")
                raise e

        if "message_list" in example:
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(example["message_list"])

        message_list, formatter_metadata = self.formater(example, self.is_training, self.for_inference, rng)
        if isinstance(self.mm_preprocessor, VideoTextPreprocessor):
            processed_video = self.mm_preprocessor(
                frames,
                frame_times,
                message_list,
                weight=example.get("weight"),
                is_training=self.is_training,
                rng=rng,
            )
        else:
            processed_video = self.mm_preprocessor(
                frames,
                message_list,
                frame_times=frame_times,
                weight=example.get("weight"),
                is_training=self.is_training,
                rng=rng,
            )

        if formatter_metadata is None:
            formatter_metadata = {}
        if self.include_image:
            image_collage = get_image_collage(frames)
            processed_video["image_collage"] = image_collage
            formatter_metadata["image"] = image_collage
            h, w = image_collage.shape[:2]
        else:
            h, w = frames[0].shape[:2]
        formatter_metadata["image_size"] = (w, h)

        if "metadata" in example or formatter_metadata:
            metadata = example.get("metadata", {})
            if formatter_metadata:
                metadata.update(formatter_metadata)
            processed_video["metadata"] = metadata
        return processed_video

    @property
    def tokenizer(self):
        return self.mm_preprocessor.tokenizer
