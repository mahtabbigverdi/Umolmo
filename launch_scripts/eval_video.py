"""Evals a checkpoint on multiple tasks, run this script with 'torchrun'."""
import argparse
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from olmo.config import EvalConfig, FSDPConfig, FSDPWrapStrategy, FSDPPrecision
from olmo.torch_util import get_world_size
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment, )
from scripts.mm_eval import ModelEvaluator
from launch_scripts.utils import get_evaluation

log = logging.getLogger(__name__)


def main():
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
    parser.add_argument("--seed", default=6198, type=int)
    parser.add_argument("--seq_len", default=6400, type=int,
                        help="Max sequence length to use")
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save the evaluation results")
    parser.add_argument("--save_to_checkpoint_dir", action="store_true",
                        help="Save to the checkpoint directory")
    parser.add_argument("--eval_name",
                        help="Name to use as a prefix when saving results")
    parser.add_argument("--pbar", action="store_true",
                        help="Show a progress bar")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true",
                        help="Load with FSDP, can be used to avoid OOMs")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    # Set num_workers to 0 to debug without multiprocessing
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers to use for evaluation")
    args, other_args = parser.parse_known_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    add_cached_path_clients()
    prepare_cli_environment()

    tasks = []
    for task in args.tasks:
        if "," in task:
            tasks += task.split(",")   # support comma seperator just because the jax code does
        else:
            tasks.append(task)
    tasks = list({k: None for k in tasks})  # de-duplicate but keep order

    inf_evaluators = []
    for task in tasks:
        eval_config = get_evaluation(
            name=task, seq_len=args.seq_len,
            batch_size=args.device_batch_size * get_world_size(),
            max_examples=args.max_examples,
            num_workers=args.num_workers,
        )
        if args.max_new_tokens:
            eval_config = replace(eval_config, max_new_tokens=args.max_new_tokens)
        inf_evaluators.append(replace(
            eval_config,
            mm_evaluator=replace(
                eval_config.mm_evaluator,
                n_to_log=4,
                num_wandb_examples=300,
                save_predictions="_default",
            ),
            save_to_checkpoint_dir=args.save_to_checkpoint_dir,
            save_dir=args.save_dir,
            eval_name=args.eval_name,
            skip_if_metrics_cached=not args.overwrite,
        ))

    checkpoint_dir = Path(args.checkpoint)
    if not (checkpoint_dir / "model.pt").exists() and args.checkpoint != "debug":
        candidates = []
        for file in checkpoint_dir.iterdir():
            match = re.match("^step([0-9]+)-unsharded.*", file.name)
            if match:
                candidates.append((file, int(match.group(1))))
        if len(candidates) == 0:
            raise FileNotFoundError(f"{checkpoint_dir} is a directory but it did not "
                                    f"contain any unsharded checkpoints")
        checkpoint_dir = max(candidates, key=lambda x: x[1])[0].absolute().as_posix()
        logging.info(f"Selected {checkpoint_dir} as oldest checkpoint in {checkpoint_dir}")
    else:
        checkpoint_dir = args.checkpoint

    cfg = EvalConfig(
        max_frames_override=args.max_frames,
        max_crops_override=args.max_crops,
        candidate_sampling_fps_override=args.candidate_sampling_fps,
        evaluations=inf_evaluators,
        load_path=checkpoint_dir,
        seed=args.seed,
        device_inf_eval_batch_size=args.device_batch_size,
        pbar=args.pbar,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ) if args.fsdp else None,
    )

    if other_args:
        config = OmegaConf.create(cfg)
        overrides = [clean_opt(arg) for arg in other_args]
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        cfg = cast(EvalConfig, OmegaConf.to_object(config))
    ModelEvaluator(cfg).run()


if __name__ == "__main__":
    main()
