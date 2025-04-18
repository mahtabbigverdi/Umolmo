"""Evals a checkpoint on multiple tasks, run this script with 'torchrun'."""
import argparse
import logging
from dataclasses import replace
from typing import cast

from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation
from olmo.train.trainer_config import FSDPConfig, FSDPPrecision
from olmo.models.model import FSDPWrapStrategy
from olmo.util import (
    clean_opt,
    prepare_torchrun_environment, select_checkpoint, )
from scripts.mm_eval import ModelEvaluator, DatasetEvaluatorConfig, EvalConfig

log = logging.getLogger(__name__)


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Evaluate a model on downstream tasks")
    parser.add_argument("checkpoint",
                        help="Checkpoint to evaluate, should contain a config file and unshared model file")
    parser.add_argument("tasks", nargs="+", help="Tasks to evaluate")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Override models default number of crops")
    parser.add_argument("--max_crops", type=int, default=None,
                        help="Override models default number of crops")
    parser.add_argument("--candidate_sampling_fps", type=float, nargs="+", default=None,
                        help="Override models default candidate sampling fps")
    parser.add_argument("--seq_len", default=1536, type=int,
                        help="Max sequence length to use")
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--eval_name",
                        help="Name to use as a prefix when saving results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true",
                        help="Load with FSDP, can be used to avoid OOMs")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    parser.add_argument("--include_image", action="store_true",
                        help="Include image in the evaluation outputs")
    args, other_args = parser.parse_known_args()

    tasks = []
    for task in args.tasks:
        if task == "short-video":
            tasks += [
                "mvbench",
                "temp_compass",
                "perception_test",
                "ego_schema",
                "nextqa_mc:test",
            ]
        elif task == "long-video":
            tasks += [
                "video_mme",
                "mlvu_mc",
                "long_video_bench",
            ]
        elif "," in task:
            tasks += task.split(",")   # support comma seperator just because the jax code does
        else:
            tasks.append(task)
    tasks = list({k: None for k in tasks})  # de-duplicate but keep order

    inf_evaluators = []
    for task in tasks:
        base_config = get_evaluation(name=task, seq_len=args.seq_len, max_examples=args.max_examples)
        eval_config = DatasetEvaluatorConfig(
            label=base_config.label,
            data=replace(base_config.data, pad="to_max" if args.fsdp else None),
            generative_evaluator=replace(
                base_config.evaluator,
                n_to_log=4,
                num_wandb_examples=300,
                save_predictions="_default",
            ),
            device_batch_size=args.device_batch_size,
            subset_num_batches=None,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens or base_config.max_new_tokens,
        )
        inf_evaluators.append(eval_config)

    checkpoint_dir = "debug" if args.checkpoint == "debug" else select_checkpoint(args.checkpoint)

    cfg = EvalConfig(
        max_frames_override=args.max_frames,
        max_crops_override=args.max_crops,
        candidate_sampling_fps_override=args.candidate_sampling_fps,
        evaluations=inf_evaluators,
        load_path=checkpoint_dir,
        console_log_interval=10,
        precision="amp_bf16",
        pbar=False,
        eval_name=args.eval_name,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
            fsdp2=True
        ) if args.fsdp else None,
        skip_if_metrics_cached=not args.overwrite,
        include_image=args.include_image,
    )

    config = OmegaConf.create(cfg)
    config.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(EvalConfig, OmegaConf.to_object(config))
    cfg.build().run()


if __name__ == "__main__":
    main()
