# This file is used to load the video dataset for the MVBench task.
# Code is adapted from https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb
import logging
import os
import json
import tarfile
from collections import Counter

import subprocess
import random

import yaml
import pandas as pd
import numpy as np

import imageio.v3 as iio
from moviepy import VideoFileClip

import ffmpeg

import ast
import decord
from olmo import tokenizer
from tqdm import tqdm

from olmo.io import read_file, is_url, glob, file_exists, get_bytes_range
from olmo.util import flatten_lists, resource_path, split_into_groups

decord.logging.set_level(2)
from decord import VideoReader, cpu

from os.path import join, exists

from olmo.data.dataset import DatasetBase, VIDEO_DATA_HOME

from olmo.models.video_olmo.video_preprocessor import load_video_decord_or_pyav


def create_video_from_frames(frames_dir, start_frame, end_frame, fps=3):
    """
    Creates a video file from a sequence of frames in a directory.
    
    Args:
        frames_dir (str): Directory containing the frames
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
        fps (int): Frames per second for the output video
    
    Returns:
        str: Path to the created video file
    """
    # Generate output path
    output_path = os.path.join(frames_dir, f"video_{start_frame:05d}_{end_frame:05d}.mp4")
    
    if file_exists(output_path):
        return output_path

    # Get list of frame files within the range
    frame_files = []
    for i in range(start_frame, end_frame + 1):
        frame_path = os.path.join(frames_dir, f"{i:05d}.jpg")  # Using 5-digit frame numbers
        if os.path.exists(frame_path):
            frame_files.append(frame_path)
    
    if not frame_files:
        print(frames_dir)
        raise ValueError(f"No frames found in range {start_frame} to {end_frame}")

    # Read frames and write video
    frames = [iio.imread(f) for f in frame_files]
    iio.imwrite(output_path, frames, fps=fps, codec='libx264')
    
    return output_path


def save_bounded_video(video_path, start_time, end_time, task_type="default",
                       output_folder=None, clip_tag="bounded_decimal_2", use_video_file_clip=False):
    """
    Creates a new video file containing only the segment between start_time and end_time.
    
    Args:
        video_path (str): Path to the original video file or frames directory
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        task_type (str): Type of task, determines handling of frames vs video
        output_folder (str): Path to the output folder, if None, output folder will be created
        clip_tag (str): Tag for the new video file, if None, no tag will be added to the new video file

    Returns:
        str: Path to the new bounded video file
    """
    if task_type == "Episodic Reasoning":
        # For frame-based videos, convert frame numbers from time
        fps = 3  # As specified in the frames directory name
        start_frame = int(start_time * fps) + 1  # +1 because frames start at 1
        end_frame = int(end_time * fps)
        return create_video_from_frames(video_path, start_frame, end_frame, fps)

    # Original video handling code
    base_path, ext = os.path.splitext(video_path)
    
    # round start_time and end_time to 2 decimal places
    start_time = round(start_time, 2)
    end_time = round(end_time, 2)

    clip_id = f"{start_time}_{end_time}{ext}"
    if clip_tag:
        clip_id = f"{clip_tag}_{clip_id}"

    if output_folder is None:
        output_path = f"{base_path}_{clip_id}"
    else:
        file_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_folder, f"{file_name_no_ext}_{clip_id}")

    if file_exists(output_path):
        return output_path

    metadata = iio.immeta(video_path)
    duration = metadata['duration']

    if use_video_file_clip:
        video = VideoFileClip(video_path)
        clip = video.subclipped(start_time, min(end_time, duration))
        clip.write_videofile(output_path, codec='libx264')
        video.close()
        clip.close()
    else:
        input_file = ffmpeg.input(video_path).trim(start=start_time, end=end_time).setpts('PTS-STARTPTS')
        output_file = ffmpeg.output(input_file, output_path)
        ffmpeg.run(output_file)
    
    return output_path


def reformat_webm_to_mp4_if_needed(input_path):
    """
    Reformats a WebM file to MP4 format.

    Args:
        input_path (str): Path to the input WebM file.

    Returns:
        str: The path to the reformatted MP4 file.
    """

    if ".webm" not in input_path:
        return input_path

    output_path = input_path.replace(".webm", ".mp4")
    if os.path.exists(output_path):
        return output_path

    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video.close()
    return output_path


class PlmFGQAEval(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA")
    ego_exo_data_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA/egoexo4d")
    ego_4d_data_path = os.path.join(VIDEO_DATA_HOME, "Ego4d/ego4d_data/v2/full_scale")
    yt_dataset_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA/plm-eval-videos/")

    def __init__(self, split):
        assert split in ["validation"]
        super().__init__(split)

    @staticmethod
    def qa_template(option_candidates, correct_choice, question):
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(option_candidates))
        answer = f"{chr(ord('A') + correct_choice)}"
        question = "\n".join(
            [
                question,
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer

    def get_ego_exo_video_path(self, item, row, egoexo_segment_id_to_camera_id):
        segment_id = row['video'].split(".mp4")[0]
        selected_camera_id = egoexo_segment_id_to_camera_id[segment_id]
        camera, cam_source = selected_camera_id.split("_")
        camera_metadata = item['frame_aligned_videos'][camera]
        return os.path.join(self.ego_exo_data_path, item['root_dir'], camera_metadata[cam_source]["relative_path"])

    def load(self):
        ego_exo_metadata_path = os.path.join(self.ego_exo_data_path, "takes.json")
        with open(ego_exo_metadata_path) as f:
            ego_exo_metadata = json.load(f)
        ego_exo_uid_to_metadata = {}
        for item in ego_exo_metadata:
            ego_exo_uid_to_metadata[item['take_uid']] = item

        egoexo_segment_to_camera_map = pd.read_csv(os.path.join(self.data_path, "fgqa_test_egoexo4d_segment2cam.csv"))
        egoexo_segment_id_to_camera_id = {}
        for index, row in egoexo_segment_to_camera_map.iterrows():
            egoexo_segment_id_to_camera_id[row['segment_uid']] = row['camera_name']

        data = []
        # get the metadata column and get list of source
        fgqa_df = pd.read_parquet(os.path.join(self.data_path, "plm_fgqa_test.parquet"))
        for index, row in fgqa_df.iterrows():
            metadata = row['metadata']
            source_id = metadata['source_video_id']
            source = metadata['source_dataset']
            question_group_id = row['qa_uid']
            if source == "egoexo4d":
                video_path = self.get_ego_exo_video_path(ego_exo_uid_to_metadata[source_id], row, egoexo_segment_id_to_camera_id)
            elif source == "ego4d":
                video_path = os.path.join(self.ego_4d_data_path, source_id + ".mp4")
            else:
                if not os.path.exists(os.path.join(self.yt_dataset_path, source_id)):
                    continue  # Some videos in the PLM eval set and no longer found on the internet

                file_list = os.listdir(os.path.join(self.yt_dataset_path, source_id))
                assert len(file_list) >= 2, f"Number of files in {source_id} is {len(file_list)}, expected >= 2. {file_list}"
                video_path = None
                for file_name in file_list:
                    if ".json" not in file_name and "_bounded_" not in file_name:
                        video_path = os.path.join(self.yt_dataset_path, source_id, file_name)
                        break
                assert video_path is not None, f"No video file found for {source_id}"

                _, video_ext = os.path.splitext(video_path)
                if "webm" in video_ext:
                    video_path = reformat_webm_to_mp4_if_needed(video_path)

            start_time = metadata['source_start_time']
            end_time = metadata['source_end_time']
            video_path = save_bounded_video(video_path, start_time, end_time, task_type="default")

            option_tuple_list = [(option[0], option[1]) for option in row['options'].items()]
            sorted_options = sorted(option_tuple_list, key=lambda x: int(x[0].lstrip("option_")))
            option_list = [option[1] for option in sorted_options]

            question, answer = self.qa_template(option_list, int(row['answer_index']), row['question'])
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=row["uid"],
                    question_group_id=question_group_id,
                    video_segment_id=row['video'],
                    video_source_id=source_id,
                    video_source_type=source,
                    options=option_list,
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class PlmFGQATrain(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA")
    coin_video_path = os.path.join(VIDEO_DATA_HOME, "coin/videos")
    coin_video_segments_path = os.path.join(VIDEO_DATA_HOME, "coin/video_segments")
    youcook2_path = os.path.join(VIDEO_DATA_HOME, "YouCook2/all_data")
    youcook2_video_path = os.path.join(VIDEO_DATA_HOME, "YouCook2/all_data/raw_videos/training")
    youcook2_video_wo_ext_to_video_path = os.path.join(VIDEO_DATA_HOME, "YouCook2/all_data/video_wo_ext_to_video_path.json")
    crosstask_path = os.path.join(VIDEO_DATA_HOME, "crosstask")
    ego_4d_data_path = os.path.join(VIDEO_DATA_HOME, "Ego4d/ego4d_data/v2/full_scale")
    ht100m_data_path = os.path.join(VIDEO_DATA_HOME, "ht100m")
    ht100m_non_h264_path = os.path.join(VIDEO_DATA_HOME, "ht100m/non_h264_ht100m_may_6_2025_57k_filtered.txt")

    def __init__(self, split):
        assert split in ["train"]
        super().__init__(split)

    @staticmethod
    def probe_for_h264(video_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip() == 'h264'

    def load(self):
        data = []

        youcook2_video_wo_ext_to_video = json.load(open(self.youcook2_video_wo_ext_to_video_path))
        ht100m_non_h264 = open(self.ht100m_non_h264_path).read().splitlines()
        ht100m_non_h264 = set([row.strip() for row in ht100m_non_h264])

        fgqa_df = pd.read_parquet(os.path.join(self.data_path, "plm_fgqa_train_w_src_with_src_id_duration.parquet"))
        # fgqa_df = fgqa_df[fgqa_df['source'] == "ht100m"]
        # fgqa_df = fgqa_df[:1000]

        segment_id_to_message_list = {}
        for index, row in fgqa_df.iterrows():
            metadata = row['metadata']
            source_id = metadata['source_video_id']
            source = metadata['source_dataset']

            start_time = metadata['source_start_time']
            end_time = metadata['source_end_time']
            if start_time < 0 or (end_time - start_time) < 0.05:
                continue

            if source == "coin":
                video_path = os.path.join(self.coin_video_segments_path, f"{source_id}_{round(start_time, 2)}_{round(end_time, 2)}.mp4")
                if not os.path.exists(video_path):
                    continue
            elif source == "youcook2":
                video_path = youcook2_video_wo_ext_to_video[source_id]
                video_path = reformat_webm_to_mp4_if_needed(video_path)
                video_path = save_bounded_video(video_path, start_time, end_time, task_type="default")
            elif source == "crosstask":
                video_path = os.path.join(self.crosstask_path, source_id, f"{source_id}.mp4")
                if not os.path.exists(video_path):
                    continue
                video_path = save_bounded_video(video_path, start_time, end_time, task_type="default")
            elif source == "ego4d":
                video_path = os.path.join(self.ego_4d_data_path, source_id + ".mp4")
                if not os.path.exists(video_path):
                    continue
                # video_path = save_bounded_video(video_path, start_time, end_time, task_type="default")
            elif source == "ht100m":
                video_parent_dir = os.path.join(self.ht100m_data_path, source_id)
                if not os.path.exists(video_parent_dir):
                    continue
                video_path = None
                for file_name in os.listdir(video_parent_dir):
                    if ".mkv" in file_name or ".mp4" in file_name or ".webm" in file_name:
                        video_path = os.path.join(video_parent_dir, file_name)
                        break
                if video_path is None:
                    continue
                # video_path = reformat_webm_to_mp4_if_needed(video_path)
                if ".webm" in video_path:
                    continue
                if video_path in ht100m_non_h264:
                    continue
                if start_time >= row['duration']:
                    continue
            else:
                continue

            if row['segment_id'] not in segment_id_to_message_list:
                segment_id_to_message_list[row['segment_id']] = {
                    "video": video_path,
                    "metadata": dict(
                        video_segment_id=row['segment_id'],
                        video_source_id=source_id,
                        video_source_dataset=source,
                    ),
                    "message_list": [dict(
                            question=row['question'],
                            answer=row['answer'],
                            style="llava_video_da",
                        )
                    ]
                }
                if source in ["ht100m", "ego4d"]:
                    segment_id_to_message_list[row['segment_id']]['metadata']['clip_start_time'] = start_time
                    segment_id_to_message_list[row['segment_id']]['metadata']['clip_end_time'] = end_time
            else:
                segment_id_to_message_list[row['segment_id']]['message_list'].append(
                    dict(
                        question=row['question'],
                        answer=row['answer'],
                        style="llava_video_da",
                    )
                )

        for segment_id, example in segment_id_to_message_list.items():
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], )


class NeXTQA(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "NeXTQA")

    def __init__(self, split, task):
        # implement MC task for now
        if task == "multiple-choice":
            assert split in ["test"]
        else:
            raise NotImplementedError(f"Task {task} not implemented")
        self.task = task
        super().__init__(split)
    
    def mc_qoa_template(self, data):
        options = [data[f'a{idx}'].strip() for idx in range(5)]
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}. {options[idx]}" for idx in range(5)
        )
        answer = f"{chr(ord('A') + int(data['answer']))}"
        question = "\n".join(
            [
                data["question"].strip(),
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, options, answer
    
    def load(self):
        task = self.task
        data = []
        if task == "multiple-choice":
            df = pd.read_parquet(os.path.join(self.data_path, "MC", "test-00000-of-00001.parquet"))
            for idx, row in df.iterrows():
                video_path = os.path.join(self.data_path, "NExTVideo", f"{row['video']}.mp4")
                quesiton, options, answer = self.mc_qoa_template(row)
                example = {
                    "question": quesiton,
                    "answer": answer,
                    "video": video_path,
                    "style": "video_eval_multiple_choice",
                    "metadata": dict(
                        question_id=str(idx),
                        question_type=row["type"],
                        video_id=row["video"],
                        options=options,
                    )
                }
                data.append(example)
        else:
            raise NotImplementedError(f"Task {task} not implemented")
        return data

    def get(self, idx, rng):
        return self.data[idx]


class LongVideoBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LongVideoBench")

    def __init__(self, split, allow_subtitle=True):
        assert split in ["validation"]
        self.allow_subtitle = allow_subtitle
        super().__init__(split)
    
    def qa_template(self, qa_data):
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(qa_data['candidates']))
        answer = f"{chr(ord('A') + qa_data['correct_choice'])}"
        question = "\n".join(
            [
                qa_data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer
    
    def load(self):
        json_data = json.loads(read_file(os.path.join(self.data_path, "lvb_val.json")))
        data = []
        for qa_data in json_data:
            question, answer = self.qa_template(qa_data)
            if not self.allow_subtitle and "subtitle" in question:
                continue

            video_path = os.path.join(self.data_path, "videos", qa_data["video_path"])
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=qa_data["id"],
                    video_id=qa_data["video_id"],
                    level=qa_data["level"],
                    options=qa_data["candidates"],
                    question_category=qa_data["question_category"],
                    duration_group=qa_data["duration_group"],
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class MLVU(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MVLU", "MLVU")
    val_mc_tasks = [
        "plotQA",
        "needle",
        "ego",
        "count",
        "order",
        "anomaly_reco",
        "topic_reasoning"
    ]
    vak_gen_tasks = ["sub_scene", "summary"]

    def __init__(self, split, task):
        assert split in ["validation"]
        assert task in ["multiple-choice", "generation"]
        self.task = task
        super().__init__(split)
    
    def mc_qa_template(self, data):
        """lmms-eval uses the MVBench's template, but llava-video uses the different one, so just follow the PerceptionTest's template"""
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(data['answer'])
        answer = f"{chr(ord('A') + answer_idx)}"
        question = "\n".join(
            [
                data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer
        """
        option_text = "\n".join(f"({chr(ord('A') + idx)}) {opt}" for idx, opt in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(data['answer'])
        answer = f"{chr(ord('A') + answer_idx)}"
        question = "\n".join([data["question"], option_text, "Only give the best option.", "Best option: ("])
        return question, answer
        """
    
    def load(self):
        task = self.task
        data = []
        if task == "multiple-choice":
            question_id = 0
            for idx, task_type in enumerate(self.val_mc_tasks, 1):
                name = f"{idx}_{task_type}"
                json_data = json.loads(read_file(os.path.join(self.data_path, "json", f"{name}.json")))
                for qa_data in json_data:
                    video_path = os.path.join(self.data_path, "video", name, f"{qa_data['video']}")
                    question, answer = self.mc_qa_template(qa_data)
                    example = {
                        "question": question,
                        "answer": answer,
                        "video": video_path,
                        "style": "video_eval_multiple_choice",
                        "metadata": dict(
                            question_id=str(question_id),
                            video_id=qa_data['video'],
                            task_type=task_type,
                        )
                    }
                    data.append(example)
                    question_id += 1
        else:
            question_id = 0
            for idx, task_type in enumerate(self.vak_gen_tasks, 1):
                name = f"{len(self.val_mc_tasks) + idx}_{task_type}"
                json_data = json.loads(read_file(os.path.join(self.data_path, "json", f"{name}.json")))
                for qa_data in json_data:
                    video_path = os.path.join(self.data_path, "video", name, f"{qa_data['video']}")
                    example = {
                        "question": qa_data['question'],
                        "answer": qa_data['answer'],
                        "video": video_path,
                        "style": "demo",
                        "metadata": dict(
                            question_id=str(question_id),
                            question=qa_data['question'],
                            answer=qa_data['answer'],
                            video_id=qa_data['video'],
                            task_type=task_type,
                        )
                    }
                    if "scoring_points" in qa_data:
                        example["metadata"]["scoring_points"] = qa_data["scoring_points"]
                    data.append(example)
                    question_id += 1
        return data

    def get(self, idx, rng):
        return self.data[idx]
                

class PerceptionTest(DatasetBase):
    """PerceptionTest Multiple-Choice Video QA task"""
    data_path = os.path.join(VIDEO_DATA_HOME, "PerceptionTest_Val")

    def __init__(self, split):
        assert split in ["validation"]
        super().__init__(split)
    
    def qa_template(self, question, options, answer_id):
        prefixes = "abcdefg".upper()
        option_text = "\n".join(
            f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options)
        )
        question = "\n".join(
            [
                question,
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        answer = prefixes[answer_id]
        return question, answer
    
    def load(self):
        df = pd.read_parquet(os.path.join(self.data_path, "mc_question_val/validation-00000-of-00001.parquet"))
        data = []
        for idx, row in df.iterrows():
            video_path = os.path.join(self.data_path, "videos", row["video_name"] + ".mp4")
            question, answer = self.qa_template(row["question"], row["options"], int(row["answer_id"]))
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=row["question_id"],
                    video_id=row["video_name"],
                    answer_idx=int(row["answer_id"]),
                    area=row["area"],
                    reasoning=row["reasoning"],
                )
            }
            data.append(example)
        return data
    
    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class EgoSchema(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "egoschema")

    def __init__(self, split):
        assert split in ["validation"]
        super().__init__(split)
    
    def question_template(self, question, options):
        question = "\n".join(
            [
                question,
                "\n".join(options),
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question
    
    def load(self):
        df = pd.read_parquet(os.path.join(self.data_path, "Subset/test-00000-of-00001.parquet"))
        data = []
        for idx, row in df.iterrows():
            video_path = os.path.join(self.data_path, "videos", row["video_idx"] + ".mp4")
            question = self.question_template(row["question"], row["option"])
            answer = "abcdefg".upper()[int(row["answer"])]
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=row["question_idx"],
                    video_id=row["video_idx"],
                    options=row["option"],
                    answer_idx=int(row["answer"]),
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class VideoMME(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "Video-MME")
    duration = ["short", "medium", "long"]

    def __init__(self, split, duration="all"):
        assert split in ["validation"]
        assert duration in (self.duration + ["all"])
        self.target_duration = duration
        super().__init__(split)
    
    def question_template(self, question, options):
        prompt = "Select the best answer to the following multiple-choice question based on the video."
        prompt += " Respond with only the letter (A, B, C, or D) of the correct option."
        question = "\n".join(
            [
                prompt,
                question,
                "\n".join(options),
                "The best answer is:"
            ]
        )
        return question
    
    def load(self):
        df = pd.read_parquet(os.path.join(self.data_path, "videomme/test-00000-of-00001.parquet"))
        if self.target_duration != "all":
            df = df[df["duartion"] == self.target_duration]
        data = []
        video_dir = os.path.join(self.data_path, "data")
        for idx, row in df.iterrows():
            question = self.question_template(row["question"], row["options"])
            video_path = os.path.join(video_dir, row["videoID"] + ".mp4")
            example = {
                "question": question,
                "answer": row["answer"],
                "video": video_path,
                "metadata": dict(
                    video_id=row["video_id"],
                    question_id=row["question_id"],
                    duration=row["duration"],
                    domain=row["domain"],
                    sub_category=row["sub_category"],
                    task_type=row["task_type"],
                )
            }
            data.append(example)
        
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class TempCompass(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "TempCompass")
    tasks = ["multi-choice", "yes_no", "caption_matching", "captioning"]
    answer_prompt = {
        "multi-choice": "Please directly give the best option:",
        "yes_no": "Please answer yes or no:",
        "caption_matching": "Please directly give the best option:",
        "captioning": "" # The answer "Generated Caption:" is already contained in the question
    }

    def question_template(self, question, task):
        question = "\n".join([question, self.answer_prompt[task]])
        return question

    def __init__(self, split, task="all"):
        assert split in ["validation"]
        assert task in (self.tasks + ["all"])
        self.target_task = task
        super().__init__(split)
    
    def load(self):
        target_tasks = self.tasks if self.target_task == "all" else [self.target_task]
        meta_infos = json.loads(read_file(os.path.join(self.data_path, "meta_info.json")))
        data = []
        for task in target_tasks:
            parquet_file = resource_path(join(self.data_path, task), "test-00000-of-00001.parquet")
            df = pd.read_parquet(parquet_file)
            if task == "captioning":
                style = "demo"
            elif task in ["multi-choice", "caption_matching"]:
                style = "video_eval_multiple_choice"
            else:
                style = "video_eval_short_answer"
            for idx, row in df.iterrows():
                vid = row["video_id"]
                question = self.question_template(row["question"], task)
                video_path = os.path.join(
                    self.data_path, "videos", f"{vid}.mp4"
                )
                temp_asp = row["dim"]
                fine_grained_temp_asp = meta_infos[
                    vid.replace('.jpg', '').replace('.mp4', '') # Follow the original evaluation script
                ]["eval_dim"][temp_asp]["type"] if temp_asp != "order" else "order"
                example = {
                    "question": question,
                    "answer": row["answer"],
                    "video": video_path,
                    "metadata": dict(
                        video_id=vid,
                        task=task,
                        question=row["question"],
                        temporal_aspect=temp_asp,
                        fine_grained_temporal_aspect=fine_grained_temp_asp,
                    ),
                    "style": style,
                }
                if "mc_question" in row:
                    example["metadata"]["mc_question"] = row["mc_question"]
                    example["metadata"]["mc_answer"] = row["mc_answer"]
                data.append(example)
        
        return data
    
    def get(self, idx, rng):
        return self.data[idx]


class MVBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MVBench")
    data_list = {
        "Action Sequence": ("action_sequence.json", f"{data_path}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", f"{data_path}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", f"{data_path}/video/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", f"{data_path}/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", f"{data_path}/video/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{data_path}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", f"{data_path}/video/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{data_path}/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", f"{data_path}/video/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{data_path}/video/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{data_path}/video/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{data_path}/video/nturgbd_convert/", "video", False),
        "Character Order": ("character_order.json", f"{data_path}/video/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{data_path}/video/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", f"{data_path}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
    }
    data_types_with_bound = {"Action Sequence", "Action Prediction", "Object Interaction", "Action Localization", "Episodic Reasoning"}

    def __init__(self, split):
        assert split in ["validation", "val"]
        if split == "validation":
            split = "val"
        super().__init__(split)

    def qa_template(self, data):
        # question = f"Question: {data['question']}\n"
        # question += "Options:\n"
        question = data['question']
        answer = data['answer']
        answer_idx = -1
        options = "\n".join(f"{chr(ord('A') + idx)}. {c}" for idx, c in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(answer)
        answer = f"{chr(ord('A') + answer_idx)}."
        question = "\n".join(
            [
                question,
                options,
                "Please respond with only the letter of the correct answer.",
            ]
        )
        return question, answer
    
    def load(self):
        data = []
        for k, v in self.data_list.items():
            json_data = json.loads(read_file(os.path.join(self.data_path, "json", v[0])))
            
            for qa_idx, qa_data in enumerate(json_data):
                question, answer = self.qa_template(qa_data)
                if k == "Fine-grained Pose":
                    video_name = qa_data['video']
                    converted_video_name = video_name.replace(".avi", ".mp4")
                    video_path = os.path.join(v[1], converted_video_name)
                else:
                    video_path = os.path.join(v[1], qa_data['video'])

                if k in self.data_types_with_bound:
                    if is_url(video_path):
                        # Assume the bounded video has already been saved since even calling
                        # `file_exists` on each example can be slow if they are URLs
                        base_path, ext = os.path.splitext(video_path)
                        start_time, end_time = qa_data['start'], qa_data['end']
                        if k == "Episodic Reasoning":
                            fps = 3
                            start_frame = int(start_time * fps) + 1  # +1 because frames start at 1
                            end_frame = int(end_time * fps)
                            video_path = os.path.join(video_path, f"video_{start_frame:05d}_{end_frame:05d}.mp4")
                        else:
                            start_time = round(start_time, 2)
                            end_time = round(end_time, 2)
                            video_path = f"{base_path}_bounded_decimal_2_{start_time}_{end_time}{ext}"
                    else:
                        video_path = save_bounded_video(video_path, qa_data['start'], qa_data['end'], k)

                data.append({
                    'question': question,
                    'answer': answer,
                    'video': video_path,
                    "metadata": dict(
                        example_id=f"{k}_{qa_idx}",
                        video_path=video_path,
                        task_type=k,
                        prefix=v[1],
                        data_type=v[2],
                        start_time=qa_data['start'] if 'start' in qa_data else None,
                        end_time=qa_data['end'] if 'end' in qa_data else None
                    )
                })
        return data
        
    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class LLaVAVideo178K(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K")
    file_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K", "data_subset_config.yaml")
    shuffled_video_names_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K", "shuffled_llava_video_names.json")

    files_not_found = set(["ytb_GqeRnxSuLFI.mp4", "ytb_y6ReUXtm_VE.mp4"])
    corrupt_files = set([
        "ytb_3ujEaKQBqqE.mp4",
        "ytb_93RkWNK3BZc.mp4",
        "ytb_FwoZBsssEXg.mp4",
        "v_iB20nDf5yJs.mp4",
        "ytb_-CTxMb7fsWE.mp4",
        "ytb_F0IdifHpXRc.mp4",
        "ytb_bRwdpNx6bdM.mp4",
        "v_ZTHsS5lQyvQ.mp4",
        "ytb_pvf5ykfo5Ko.mp4",
        "ytb_nJ11r1kVt14.mp4",
        "ytb_pWRqmt6EEqw.mp4",
        "ytb_ZIGajSaQQLM.mp4",
        "ytb_4s2QqSla2CA.mp4",
        "ytb_UKLnTkIzsxs.mp4",
        "ytb_KWmrJ_jxozc.mp4"
    ])

    def __init__(self, split, answer_type="multi_choice", flat=False, max_per_video=None,
                 id_source="", cap_source="lv"):
        if split == "val":
            split = "validation"    
        assert split in ["train", "validation"]
        assert answer_type in ["multi_choice", "open_ended", "caption", "all"]
        assert cap_source in ["lv", "human"]
        self.answer_type = answer_type
        self.flat = flat
        self.max_per_video = max_per_video
        self.id_source = id_source
        self.cap_source = cap_source
        super().__init__(split)

    def load(self):
        config = yaml.safe_load(read_file(join(self.data_path, "data_subset_config.yaml")))

        shuffled_video_names = json.loads(read_file(self.shuffled_video_names_path))
        if self.split == "train":
            subset_video_names = set(shuffled_video_names[:int(len(shuffled_video_names) * 0.95)])
        elif self.split == "validation":
            subset_video_names = set(shuffled_video_names[int(len(shuffled_video_names) * 0.95):])
        else:
            raise NotImplementedError(self.split)

        # Add human caption data as the captions in training
        allowed_subset_to_caption = {}
        if self.id_source:
            data_frame = pd.read_parquet(self.id_source)
            for _, row in data_frame.iterrows():
                allowed_subset_to_caption[row['video_path']] = row['merged_caption']

        data = {}
        data_list_format = []
        self.video_paths = []
        for config_item in config.get('configs', []):
            for data_file in config_item['data_files']:
                question_type = data_file['split']
                if self.answer_type != "all" and question_type != self.answer_type:
                    continue

                if question_type == "caption":
                    style = "video_long_caption"
                else:
                    style = "llava_video_" + ("da" if question_type == "open_ended" else "mc")

                config_path = os.path.join(self.data_path, data_file['path'])
                first_file_data = None
                for file in glob(config_path):
                    first_file_data = json.loads(read_file(file))
                    break

                for qa_data in first_file_data:
                    relative_video_path = os.path.join(qa_data['data_source'], qa_data['video'])
                    if self.id_source and relative_video_path not in allowed_subset_to_caption:
                        continue

                    video_path = os.path.join(self.data_path, qa_data['data_source'], qa_data['video'])
                    video_name = os.path.basename(video_path)
                    if video_name in self.files_not_found or video_name in self.corrupt_files:
                        continue
                    video_id = os.path.join(qa_data['data_source'], qa_data['video'])
                    if video_id not in subset_video_names:
                        continue

                    self.video_paths.append(video_path)
                    example_id = f"{qa_data['id']}_{qa_data['data_source']}_{qa_data['video']}_{question_type}"

                    conversations = qa_data['conversations']
                    if example_id not in data:
                        messages = []
                    else:
                        messages = data[example_id]['message_list']

                    for conv_idx in range(0, len(conversations), 2):
                        question = conversations[conv_idx]['value']
                        if tokenizer.IMAGE_PROMPT in question:
                            raise ValueError()
                        if question.startswith("<image>\n"):
                            question = question[len("<image>\n"):]
                        answer = conversations[conv_idx + 1]['value']
                        answer = answer.lstrip().strip()

                        if self.id_source and self.cap_source == "human":
                            answer = allowed_subset_to_caption[relative_video_path]

                        messages.append(dict(
                            question=question,
                            answer=answer,
                            style=style,
                        ))

                    data[example_id] = {
                        'video': video_path,
                        'prefix': data_file['path'],
                        'message_list': messages
                    }

        for example_id, example in data.items():
            data_list_format.append({
                "video": example["video"],
                "metadata": dict(
                    example_id=example_id,
                    prefix=example["prefix"],
                ),
                "message_list": example["message_list"],
            })

        if self.flat:
            data_list_format = flatten_lists(
                [dict(ex, message_list=[message]) for message in ex["message_list"]]
                for ex in data_list_format
            )
        elif self.max_per_video:
            flat = []
            for ex in tqdm(data_list_format):
                for msg in split_into_groups(ex["message_list"], self.max_per_video):
                    flat.append(dict(ex, message_list=msg))
            logging.info(f"Split {len(data_list_format)} in {len(flat)} examples")
            data_list_format = flat

        return data_list_format

    def __len__(self):
        return len(self.data)

    def get_shuffle_subset(self, set_video_paths):
        """
        Code to create a shuffled video names file that can be used to get train/val split
        """
        video_id_list = []
        for video_path in set_video_paths:
            video_id_list.append(video_path.split("LLaVA-Video-178K/")[1])
        print(f"Unique video names: {len(video_id_list)}")

        # save all names to a dictionary
        sorted_video_names = sorted(video_id_list)
        random.seed(42)
        random.shuffle(sorted_video_names)
        json.dump(sorted_video_names, open(self.shuffled_video_names_path, 'w'))

    def get(self, idx, rng):
        return self.data[idx]


class InternVid(DatasetBase):
    SPLITS = ["train", "validation"]
    INTERN_VID = join(VIDEO_DATA_HOME, "intern_vid")
    
    def __init__(self, split, n_val=20):
        assert split in self.SPLITS
        self.n_val = n_val
        super().__init__(split)

    def load(self):
        metadata_source = join(self.INTERN_VID, "metadata")
        files = [f for f in os.listdir(metadata_source) if f.startswith('internvid_10m_flt-seed42-')]
        files = sorted([f for f in files if not f.endswith("_filtered.csv")])

        if self.split == "train":
            files = files[:-self.n_val]
        else:
            files = files[-self.n_val:]
        
        data = []
        for file in files:
            video_folder_id = file.split("flt-")[1].split(".csv")[0]
            video_folder_path = join(self.INTERN_VID, "videos", video_folder_id)
            video_files = [f for f in os.listdir(video_folder_path) if f.endswith(".mp4") or f.endswith(".mkv")]

            df = pd.read_csv(join(metadata_source, file), index_col=0)
            for video_file in video_files:
                row_id = video_file.split("_")[0]
                row = df.loc[int(row_id)]
                identifier = row['YoutubeID']
                caption = row['Caption']
                start_time = pd.to_timedelta(row['Start_timestamp'])
                end_time = pd.to_timedelta(row['End_timestamp'])

                video_path = join(video_folder_path, video_file)

                data.append(
                    dict(
                        video=video_path,
                        messages=dict(text=caption, style="video_short_caption"),
                        metadata=dict(
                            example_id=f"{row_id}_{identifier}",
                            start_time=start_time,
                            end_time=end_time
                        )
                    )
                )

        return data
    
    def get(self, item, rng):
        return self.data[item]


class Koala(DatasetBase):
    SPLITS = ["train", "validation"]
    KOALA_SRC = join(VIDEO_DATA_HOME, "koala_36m")

    def __init__(self, split, n_val=4):
        assert split in self.SPLITS
        self.n_val = n_val
        super().__init__(split)

    def load(self):
        metadata_source = join(self.KOALA_SRC, "metadata")
        files = sorted([f for f in os.listdir(metadata_source) if f.startswith('koala_36m-seed42-')])

        if self.split == "train":
            files = files[:-self.n_val]
        else:
            files = files[-self.n_val:]
        
        data = []
        for file in files:
            video_folder_id = file.split("koala_36m-")[1].split(".csv")[0]
            video_folder_path = join(self.KOALA_SRC, "videos", video_folder_id)
            video_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(".mp4") or f.endswith(".mkv")])

            df = pd.read_csv(join(metadata_source, file), index_col="videoID")
            for video_file in video_files:
                video_id = video_file.split(".")[0]
                row = df.loc[video_id]
                caption = row["caption"]
                st, et = ast.literal_eval(row['timestamp'])
                start_time = pd.to_timedelta(st)
                end_time = pd.to_timedelta(et)

                video_path = join(video_folder_path, video_file)

                data.append(
                    dict(
                        video=video_path,
                        messages=dict(text=caption, style="video_long_caption"),
                        metadata=dict(
                            example_id=video_id,
                            start_time=start_time,
                            end_time=end_time
                        )
                    )
                )

        return data
    
    def get(self, item, rng):
        return self.data[item]


def load_all_frames_decord_or_pyav(video_path: str) -> np.ndarray:
    try:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)  # Total frame count
        frame_indices = np.arange(0, total_frames)
        return vr.get_batch(frame_indices).asnumpy()

    except Exception as e:
        frames = []
        for frame in iio.imiter(video_path, plugin="pyav"):
            frames.append(frame)
        return np.stack(frames)


class PeVideo(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "PE-Video")

    @classmethod
    def build_index(cls):
        for split in ["train", "test"]:
            split_path = join(cls.data_path, split)
            index_path = join(cls.data_path, f"{split_path}_index.json")
            if exists(index_path):
                continue
            indices = {}
            logging.info(f"Building index for {split}")
            video_ids = Counter()
            for file in tqdm(os.listdir(split_path)):
                index = []
                example_to_files = {}
                with tarfile.open(join(split_path, file), 'r') as db:
                    it = iter(db)
                    while True:
                        try:
                            json_tarinfo = next(it)
                        except StopIteration:
                            break
                        mp4_tarinfo = next(it)
                        json_data = json.load(db.extractfile(json_tarinfo))
                        video_id = json_data["video_id"]
                        video_ids[video_id] += 1
                        example_id = int(json_tarinfo.name.split('.')[0])
                        assert example_id == int(mp4_tarinfo.name.split('.')[0])
                        assert video_id == example_id
                        index.append((
                            example_id,
                            json_tarinfo.offset_data, json_tarinfo.size,
                            mp4_tarinfo.offset_data, mp4_tarinfo.size,
                        ))
                        db.members = []
                indices[file] = index
            with open(index_path, "w") as f:
                json.dump(indices, f)

    def load(self):
        if self.split == "test":
            with open(join(self.data_path, f"test_index.json")) as f:
                data = json.load(f)
            data = flatten_lists(((file,) + val for val in vals) for file, vals in data.items())
        else:
            with open(join(self.data_path, f"train_index.json")) as f:
                data = json.load(f)
            data = flatten_lists(([file] + val for val in vals) for file, vals in data.items())
            np.random.RandomState(1321).shuffle(data)
            if self.split == "validation":
                data = data[:4096]
            else:
                data = data[4096:]
        return data

    def get(self, item, rng):
        file, example_id, json_off, json_sz, mp_off, mp_size = self.data[item]
        if self.split == "test":
            file = join(self.data_path, "test", file)
        else:
            file = join(self.data_path, "train", file)
        data = json.loads(get_bytes_range(file, json_off, json_sz).decode("utf-8"))
        video = get_bytes_range(file, mp_off, mp_size)
        return dict(
            text=data["model_caption"],
            video=video,
            style="long_caption",
            metadata=dict(example_id=data["video_id"]),
        )


if __name__ == "__main__":

    # input_path = "/weka/oe-training-default/mm-olmo/video_datasets/Ego4d/ego4d_data/v2/full_scale/bdab681e-1d1b-49f7-983a-92a563ae4dfd.mp4"
    # output_folder = "/weka/oe-training-default/rohunt/data_viz"
    # start = 2.0
    # end = 6.0
    # start_timer = time.time()
    # save_bounded_video(input_path, start, end, output_folder=output_folder,
    #                    clip_tag="use_video_file_clip", use_video_file_clip=True)
    # end_timer = time.time()
    # first_time = end_timer - start_timer
    #
    # start_timer = time.time()
    # save_bounded_video(input_path, start, end, output_folder=output_folder,
    #                    clip_tag="use_ffmpeg_set_pts", use_video_file_clip=False)
    # end_timer = time.time()
    # second_time = end_timer - start_timer
    # print(f"First time: {first_time}, second time: {second_time}")

    # Test decord for trimming and loading of clips
    dataset = PlmFGQATrain("train")

    # for example in dataset.data:
    #     try:
    #         if "clip_start_time" in example['metadata']:
    #             load_decord_video(video_path=example['video'],
    #                               clip_start_time=example['metadata']['clip_start_time'],
    #                               clip_end_time=example['metadata']['clip_end_time'],
    #                               max_frames=48)
    #         else:
    #             load_decord_video(video_path=example['video'], max_frames=48)
    #     except Exception as e:
    #         import pdb; pdb.set_trace()

    print("Number of unique videos - ", len(set([ex["video"] for ex in dataset])))
    print("Number of unique segments - ", len(dataset))
    print("Number of unique QA - ", sum([len(ex['message_list']) for ex in dataset]))

    # dataset = LLaVAVideo178K("train", "caption",
    #                          id_source="/weka/oe-training-default/mm-olmo/video_captions/video-captions-9k.parquet",
    #                          cap_source="lv")
    # print(f"Total samples: {len(dataset)}")
    #
    # set_video_paths = set(dataset.video_paths)
    # print(f"Unique video names: {len(set_video_paths)}")
    #
    # caption_length = []
    # for i in range(len(dataset)):
    #     ex = dataset[i]
    #     caption_length.append(len(ex["message_list"][0]['answer'].split(" ")))
    #
    # print("LLava annotations on annotated train set. Max and mean length of words")
    # print(np.max(caption_length))
    # print(np.mean(caption_length))

    # # Code to test loading of the videos
    # from multiprocessing import Pool
    # from tqdm import tqdm
    #
    # def process_video(example):
    #     try:
    #         if "metadata" in example and "clip_start_time" in example["metadata"]:
    #             clip_start_time = example["metadata"]["clip_start_time"]
    #             clip_end_time = example["metadata"]["clip_end_time"]
    #             frames, frame_times, chosen_fps = load_video_decord_or_pyav(example["video"], max_frames=48, clip_start_time=clip_start_time, clip_end_time=clip_end_time)
    #
    #         else:
    #             frames, frame_times, chosen_fps = load_video_decord_or_pyav(example["video"], max_frames=48)
    #         if len(frames) == 0:
    #             raise ValueError("len(frames)==0")
    #     except Exception as e:
    #         print(f"Error loading video {example['video']}: {e}")
    #
    # ex_list = dataset.data
    # with Pool(processes=48) as pool:
    #     result = list(tqdm(pool.imap(process_video, ex_list), total=len(ex_list)))

    # # Run ffprobe to find the videos that won't be loaded by decord
    # for video in tqdm(list(set([ex["video"] for ex in dataset]))):
    #     try:
    #         if not PlmFGQATrain.probe_for_h264(video):
    #             print(video)
    #     except:
    #         print(video)
