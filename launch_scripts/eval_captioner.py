import argparse
import logging
import re
from pathlib import Path
from typing import cast

import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from transformers import CompileConfig

from launch_scripts.utils import select_checkpoint
from olmo.config import EvalConfig, FSDPConfig, FSDPWrapStrategy, FSDPPrecision, \
    DatasetEvaluatorConfig, \
    EvaluatorConfig, DataConfig, CompilerConfig
from olmo.torch_util import get_world_size
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment, prepare_torchrun_environment, )
from scripts.mm_eval import ModelEvaluator

log = logging.getLogger(__name__)


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Script to generate dense captions")
    parser.add_argument("checkpoint")
    parser.add_argument("--task", default="dense_caption_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_crops", type=int, default=None)
    parser.add_argument("--seq_len", default=1536, type=int)
    parser.add_argument("--max_examples", default=None, type=int)
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--eval_name")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("--loss", action="store_true",
                        help="Compute loss/accuracy metrics instead of doing inference")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=448,
                        help="Override max new tokens, otherwise use task-specific default")
    args, other_args = parser.parse_known_args()


    if args.max_examples:
        batch_size = get_world_size()*args.device_batch_size
        n_batches = args.max_examples//batch_size
        logging.info(f"Evaluating on {n_batches} batches ({batch_size*n_batches} examples)")
    else:
        n_batches = -1

    checkpoint_dir = select_checkpoint(args.checkpoint)

    eval_config = DatasetEvaluatorConfig(
        data=DataConfig(
            args.task, split=args.split,
            for_inference=not args.loss,
            sequence_length=args.seq_len,
            drop_last=False,
            shuffle=False,
            num_workers=2, pin_memory=True,
        ),
        max_new_tokens=args.max_new_tokens,
        mm_evaluator=EvaluatorConfig(
            n_to_log=10,
            num_wandb_examples=300,
            save_predictions="_default",
        ),
        loss=args.loss,
        save_to_checkpoint_dir=args.save_dir is None,
        save_dir=args.save_dir,
        eval_name=args.eval_name,
        skip_if_metrics_cached=not args.overwrite,
        label=args.task,
        subset_num_batches=n_batches,
    )

    cfg = EvalConfig(
        max_crops_override=args.max_crops,
        evaluations=[eval_config],
        load_path=checkpoint_dir,
        seed=6198,
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
        config.merge_with_dotlist(overrides)
        cfg = cast(EvalConfig, OmegaConf.to_object(config))
    ModelEvaluator(cfg).run()


if __name__ == '__main__':
    main()