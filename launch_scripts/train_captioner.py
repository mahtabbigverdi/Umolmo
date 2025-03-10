import argparse
import logging
from dataclasses import replace
from typing import cast

from omegaconf import omegaconf, OmegaConf

from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.model_preprocessor import MultiModalPreprocessorConfig
from olmo.data.pixmo_datasets import PixMoCap
from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.model import FSDPWrapStrategy
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.nn.vision_backbone import VisionBackboneConfig, ImagePaddingEmbed
from scripts.train import run_trainer

from olmo.data.data_loader import DataConfig
from olmo.train.trainer_config import BatchDivisor, SpeedMonitorConfig, \
    CompilerConfig, TrainConfig, WandbConfig, FSDPConfig, FSDPPrecision
from olmo.util import clean_opt, prepare_torchrun_environment


log = logging.getLogger("train")


if __name__ == "__main__":
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a captioner")
    parser.add_argument("llm", choices=["debug"] + list(LLMS.keys()))
    parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="openai")
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--n_eval_examples", default=2048, type=int)
    parser.add_argument("--device_eval_batch_size", default=4, type=int)
    parser.add_argument("--seq_len", default=2304, type=int)
    parser.add_argument("--dataset", default="pixmo_cap_with_transcripts")
    args, other_args = parser.parse_known_args()

    seq_len = args.seq_len
    debug = args.llm in ["debug", "debug-12crop"]
    if debug:
        model_cfg = DEBUG_MODEL
        if args.llm == "debug-12crop":
            model_cfg.mm_preprocessor.max_crops = 12
            model_cfg.mm_preprocessor.crop_mode = "overlap-and-resize-c2"
        model_cfg.data_formatter.system_prompt = 'style_and_length'

        global_batch_size = 8
        model_init = None
        eval_interval = 20
        log_interval = 5
        eval_examples = 64
        duration = 200
    else:
        eval_examples = args.n_eval_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        n = len(PixMoCap("train", "captions"))
        duration = 4 * (n + global_batch_size - 1) // global_batch_size
        eval_interval = 1000
        vit_layers = [-2, -9] if args.vision_backbone == "openai" else [-3, -9]

        image_vit = VISION_BACKBONES[args.vision_backbone]
        model_cfg = MolmoConfig(
            llm=replace(
                LLMS[args.llm],
                residual_dropout=0.0,
                response_residual_dropout=0.1,
                additional_vocab_size=128,
            ),
            vision_backbone=VisionBackboneConfig(
                vit=VISION_BACKBONES[args.vision_backbone],
                vit_layers=vit_layers,
                image_padding_embed=ImagePaddingEmbed.pad_and_partial_pad
            ),
            data_formatter=DataFormatter(
                system_prompt='style_and_length',
            ),
            mm_preprocessor=MultiModalPreprocessorConfig(
                crop_mode="overlap-and-resize-c2",
                max_crops=12,
            )
        )

    evaluator = LossDatasetEvaluatorConfig(
        label="val",
        max_examples=eval_examples,
        device_batch_size=args.device_eval_batch_size,
        console_log_interval="${console_log_interval}",
        data=DataConfig(
            seed="${seed}",
            dataset=args.dataset,
            shuffle=False,
            split="validation",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    cfg = TrainConfig(
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        compile_loss=True,
        model=model_cfg,
        data=DataConfig(
            dataset=args.dataset,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            seed=95818,
            num_workers=2,
            pad="to_max",
            pin_memory=True,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=2e-4,
            vit_learning_rate=6e-6,
            llm_learning_rate=2e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            metrics_log_interval=-1
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=2000,
            llm_t_warmup=2000,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=None,
        save_overwrite=debug,
        save_interval=4000,
        allow_resume=True,
        save_num_checkpoints_to_keep=1,
        save_final_unsharded_checkpoint=False,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=4,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        eval_interval=eval_interval,
        evaluators=[
            evaluator,
            replace(evaluator, data=replace(evaluator.data, dataset="pixmo_cap"), label="caption_val")
        ]
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)



