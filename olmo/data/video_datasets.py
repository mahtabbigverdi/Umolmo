# This file is used to load the video dataset for the MVBench task.
# Code is adapted from https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb

import os
import json
# import glob
import yaml
import pandas as pd
import numpy as np

import imageio.v3 as iio
from moviepy import VideoFileClip

import ast
import decord
from torchvision.datasets.utils import list_dir
from tqdm import tqdm

from olmo.io import read_file, is_url, glob, file_exists
from olmo.util import flatten_lists, resource_path

decord.logging.set_level(2)
from decord import VideoReader, cpu

from os.path import join

from olmo.data.dataset import DatasetBase, VIDEO_DATA_HOME, DatasetBase


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


def save_bounded_video(video_path, start_time, end_time, task_type):
    """
    Creates a new video file containing only the segment between start_time and end_time.
    
    Args:
        video_path (str): Path to the original video file or frames directory
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        task_type (str): Type of task, determines handling of frames vs video
    
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
    output_path = f"{base_path}_bounded_decimal_2_{start_time}_{end_time}{ext}"

    if file_exists(output_path):
        return output_path

    metadata = iio.immeta(video_path)
    duration = metadata['duration']

    # Load the video and extract the subclip
    video = VideoFileClip(video_path)
    clip = video.subclipped(start_time, min(end_time, duration))
    
    # Write the subclip to file
    clip.write_videofile(output_path, codec='libx264')
    
    # Close the video to free up resources
    video.close()
    clip.close()
    
    return output_path


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
        with open(os.path.join(self.data_path, "meta_info.json"), "r") as f:
            meta_infos = json.load(f)
        data = []
        for task in target_tasks:
            parquet_file = os.path.join(self.data_path, task, "test-00000-of-00001.parquet")
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

    def __init__(self, split, answer_type="multi_choice", flat=False):
        if split == "val":
            split = "validation"    
        assert split in ["train", "validation"]
        assert answer_type in ["multi_choice", "open_ended", "caption", "all"]
        self.answer_type = answer_type
        self.flat = flat
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
                    if self.split == "train":
                        if example_id not in data:
                            messages = []
                        else:
                            messages = data[example_id]['message_list']

                        for conv_idx in range(0, len(conversations), 2):
                            question = conversations[conv_idx]['value']
                            if question.startswith("<image>\n"):
                                question = question[len("<image>\n"):]
                            answer = conversations[conv_idx + 1]['value']
                            answer = answer.lstrip().strip()

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

                    else:  # for validation set, we only use one QA
                        if example_id in data:
                            continue

                        question = conversations[0]['value']
                        if question.startswith("<image>\n"):
                            question = question[len("<image>\n"):]
                        answer = conversations[1]['value']
                        answer = answer.lstrip().strip()

                        data[example_id] = {
                            "video": video_path,
                            "prefix": data_file['path'],
                            "question": question,
                            "answer": answer,
                            "style": style,
                        }

        if self.split == "validation":
            for example_id, example in data.items():
                data_list_format.append({
                    "video": example["video"],
                    "metadata": dict(
                        example_id=example_id,
                        prefix=example["prefix"],
                    ),
                    "question": example["question"],
                    "answer": example["answer"],
                    "style": example["style"],
                })
        else:
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
            data_list_format = flatten_lists([dict(ex, message_list=[message]) for message in ex["message_list"]] for ex in data_list_format)
        return data_list_format

    def __len__(self):
        return len(self.data)
    
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


if __name__ == "__main__":
    dataset = LLaVAVideo178K("train", "all")
    print(f"Total samples: {len(dataset)}")

    set_video_paths = set(dataset.video_paths)
    print(f"Unique video names: {len(set_video_paths)}")

    # Code to create a shuffled video names file that can be used to get train/val split
    # # get list of video names
    # video_id_list = []
    # for video_path in set_video_paths:
    #     video_id_list.append(video_path.split("LLaVA-Video-178K/")[1])
    # print(f"Unique video names: {len(video_id_list)}")

    # # save all names to a dictionary
    # sorted_video_names = sorted(video_id_list)
    # random.seed(42)
    # random.shuffle(sorted_video_names)
    # with open(os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K", "shuffled_llava_video_names.json"), 'w') as f:
    #     json.dump(sorted_video_names, f)

    # # Code to test loading of the videos
    # from multiprocessing import Pool
    # from tqdm import tqdm

    # def process_video(video_path):
    #     try:
    #         frames = load_all_frames_decord_or_pyav(video_path)
    #     except Exception as e:
    #         print(f"Error loading video {video_path}: {e}")

    # video_path_list = list(set_video_paths)
    # with Pool(processes=32) as pool:
    #     result = list(tqdm(pool.imap(process_video, video_path_list), total=len(video_path_list)))
