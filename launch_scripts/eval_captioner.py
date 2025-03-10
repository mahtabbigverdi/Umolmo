import argparse
import logging
from typing import cast

from omegaconf import OmegaConf

from olmo.eval.inf_evaluator import EvaluatorConfig
from olmo.train.trainer_config import FSDPConfig, FSDPPrecision
from olmo.models.model import FSDPWrapStrategy
from olmo.data.data_loader import DataConfig
from olmo.torch_util import get_world_size
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint
from scripts.mm_eval import ModelEvaluator, EvalConfig, DatasetEvaluatorConfig

log = logging.getLogger(__name__)


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Script to generate dense captions")
    parser.add_argument("checkpoint")
    parser.add_argument("--task", default="dense_caption_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq_len", default=1536, type=int)
    parser.add_argument("--max_examples", default=None, type=int)
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--loss", action="store_true",
                        help="Compute loss/accuracy metrics instead of doing inference")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=448,
                        help="Override max new tokens, otherwise use task-specific default")
    args, other_args = parser.parse_known_args()

    checkpoint_dir = select_checkpoint(args.checkpoint)

    eval_config = DatasetEvaluatorConfig(
        data=DataConfig(
            args.task, split=args.split,
            sequence_length=args.seq_len,
            drop_last=False, seed=95818,
            shuffle=False,
            num_workers=2, pin_memory=True,
        ),
        device_batch_size=args.device_batch_size,
        max_new_tokens=args.max_new_tokens,
        generative_evaluator=None if args.loss else EvaluatorConfig(
            n_to_log=10,
            num_wandb_examples=300,
            save_predictions="_default",
        ),
        label=args.task,
        max_examples=args.max_examples
    )

    cfg = EvalConfig(
        pbar=False,
        evaluations=[eval_config],
        load_path=checkpoint_dir,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
            fsdp2=True,
        ) if args.fsdp else None,
        save_to_checkpoint_dir=args.save_dir is None,
        save_dir=args.save_dir,
    )

    config = OmegaConf.create(cfg)
    config.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(EvalConfig, OmegaConf.to_object(config))
    ModelEvaluator(cfg).run()


if __name__ == '__main__':
    main()