import argparse
from os.path import join

from tqdm import tqdm

import numpy as np

from olmo.data.get_dataset import get_dataset_by_name
from olmo.html_utils import example_to_html_dict, build_html_table
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.data.dataset import DeterministicDataset
from olmo.models.molmo.model_preprocessor import Preprocessor, MolmoPreprocessor
from olmo.tokenizer import build_tokenizer

from olmo.models.video_olmo.video_preprocessor import VideoPreprocessor
from olmo.models.video_olmo.video_preprocessor import VideoTextPreprocessor


def build_qualitative_table(name, split, n, preprocessor, is_training=None, for_inference=False, shuffle=True,
                            show_patches=False, show_crops=False):
    if is_training is None:
        is_training = True if split == "train" else False,
    seq_len = {
        "is_training": is_training,
        "target_tokens": 2048 + 512,
    }
    if split != "train":
        seq_len["seed"] = 42

    dataset = get_dataset_by_name(name, split)
    data = DeterministicDataset(dataset, preprocessor, 0)
    if shuffle:
        ix = list(range(len(data)))
        np.random.shuffle(ix)
    else:
        ix = range(n)
    it = (data[i] for i in ix[:n])
    voc = preprocessor.tokenizer

    table = []
    n_images = []
    n_tokens = []
    for ix, ex in enumerate(tqdm(it, total=n)):
        n_tokens.append((ex["target_tokens"] != -1).sum())
        n_images.append(ex["images"].shape[0])
        table.append(example_to_html_dict(ex, preprocessor, show_patches, show_crops))
    print("Mean num tokens: " + str(np.mean(n_tokens)))
    print("Mean num crops: " + str(np.mean(n_images)))
    return build_html_table(table)


def main():
    parser = argparse.ArgumentParser(prog="Visualize a dataste used in Molmo")
    parser.add_argument("task", help="Task name")
    parser.add_argument("output_dir", default=".",
                        help="Directory to save the visualization")
    parser.add_argument("--output_name", default=None,
                        help="Override the default file name")
    parser.add_argument("--debug", action="store_true",
                        help="Turn on tf.data.dataset debugging mode")
    parser.add_argument("--eval", action="store_true",
                        help="Run in eval model")
    parser.add_argument("--inference", action="store_true",
                        help="Run in inference model (so responses will not be included)")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--show_patches", action="store_true",
                        help="Visualize how the patch features are interleaved with the text")
    parser.add_argument("--show_crops", action="store_true",
                        help="Show the crops used by the preprocessor")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_examples", default=100, type=int,
                        help="Number of examples to show")

    # Changes pre-processing for visualizing crops/patches/prompts
    parser.add_argument("--prompt_templates", default="uber_model",
                        help="Prompt mode for the preprocessor")
    parser.add_argument("--system_prompt", default="demo_or_style",
                        help="System prompt mode for the preprocessor")
    parser.add_argument("--message_format", default="role",
                        help="Message format mode for the preprocessor")
    parser.add_argument("--crop_mode", default="resize",
                        help="How to build crops")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2-7B",
                        help="Tokenizer to use")
    parser.add_argument("--max_frames", type=int, default=4,
                        help="Max crops to select")
    parser.add_argument("--max_crops", type=int, default=4,
                        help="Max crops to select")
    parser.add_argument("--frame_sample_mode", type=str, default="fps",
                        help="How to sample frames")
    parser.add_argument("--loss_token_weighting", type=str, default=None,
                        help="re-weighting of loss tokens")
    args = parser.parse_args()

    name = args.task
    output_name = args.output_name if args.output_name is not None else f"{name}.html"
    output_file = join(args.output_dir, output_name)
    print(f"Getting qual. examples for {name}")

    def is_video_task(name):
        if name in ["pe_video", "mvbench", "intern_vid", "koala", "ego_schema", "perception_test",
                    "long_video_bench", "nextqa_mc", "plm_fgq_eval", "plm_fgqa_train", "long_video_bench_no_subtitle"]:
            return True
        if any(name.startswith(x) for x in
               ["llava_video_178k", "temp_compass", "video_mme", "mlvu"]):
            return True
        return False

    if is_video_task(name):
        pre = VideoPreprocessor(
            DataFormatter(
                prompt_templates=args.prompt_templates,
                message_format=args.message_format,
                system_prompt=args.system_prompt,
                always_start_with_space=True,
            ),
            VideoTextPreprocessor(
                tokenizer=build_tokenizer(args.tokenizer),
                crop_mode=args.crop_mode,
                max_crops=args.max_crops,
                periodic_high_res_frame=2,
                image_pooling_w=4,
                image_pooling_h=4,
                high_res_pooling_h=2,
                high_res_pooling_w=2
            ),
            for_inference=args.inference,
            frame_sample_mode=args.frame_sample_mode,
            include_image=True,  # include the image in the metadata so we can visualize it
            max_frames=args.max_frames,
        )
    else:
        pre = Preprocessor(
            DataFormatter(
                prompt_templates=args.prompt_templates,
                message_format=args.message_format,
                system_prompt=args.system_prompt,
                always_start_with_space=True,
            ),
            MolmoPreprocessor(
                tokenizer=build_tokenizer(args.tokenizer),
                crop_mode=args.crop_mode,
                max_crops=args.max_crops,
            ),
            for_inference=args.inference,
            include_image=True  # include the image in the metadata so we can visualize it
        )

    html = build_qualitative_table(
        args.task, args.split, args.num_examples, pre, is_training=not args.eval,
        for_inference=args.inference, show_patches=args.show_patches,
        show_crops=args.show_crops, shuffle=args.shuffle)
    print(f"Save examples to {output_file}")
    with open(output_file, "w") as f:
        f.write(html)
    print("Done")


if __name__ == '__main__':
    main()
