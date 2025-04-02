import argparse
import logging
from os.path import join, exists
from typing import cast, List

import omegaconf
from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, VIDEO_DEBUG_MODEL
from launch_scripts.train_multitask_model import get_training_mixture
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.video_olmo.video_preprocessor import MultiModalVideoPreprocessorConfig
from olmo.nn.image_vit import VitConfig

from olmo.train.optim import OptimizerType, OptimizerConfig, SchedulerConfig, SchedulerType
from olmo.train.trainer_config import (
    WandbConfig, BatchDivisor, SpeedMonitorConfig,
    FSDPConfig, FSDPPrecision, CompilerConfig, TrainConfig
)

from olmo.models.model import FSDPWrapStrategy
from olmo.models.video_olmo.video_olmo import VideoOlmoConfig
from olmo.data.data_loader import DataLoaderConfig, RootSizeMixture
from olmo.torch_util import get_world_size
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint
from scripts.train import run_trainer

log = logging.getLogger("train")


if __name__ == "__main__":
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--seq_len", default="auto", type=str)
    parser.add_argument("--max_eval_examples", default=512, type=int)
    parser.add_argument("--max_eval_examples_inf", default=-1, type=int)
    parser.add_argument("--crop_mode", default="resize", type=str)
    parser.add_argument("--max_frames", default=96, type=int)
    parser.add_argument("--candidate_sampling_fps", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--max_crops", default=12, type=int)
    parser.add_argument("--time_mode", default="fps-prefix", type=str)
    parser.add_argument("--frame_sample_mode", default="fps_uniform", type=str)
    parser.add_argument("--bi_directional_attn", default=None, type=str)
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--device_eval_batch_size", default=1, type=int)
    parser.add_argument("--device_inf_batch_size", default=1, type=int)
    parser.add_argument("--device_train_batch_size", default=1, type=int)
    parser.add_argument("--llm_learning_rate", default=1e-5, type=float)
    parser.add_argument("--vit_learning_rate", default=5e-6, type=float)
    parser.add_argument("--connector_learning_rate", default=5e-6, type=float)
    parser.add_argument("--duration", default=6000, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--prefetch_factor", default=8, type=int)
    parser.add_argument("--freeze_vit", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--image_pooling_h", default=2, type=int)
    parser.add_argument("--image_pooling_w", default=2, type=int)
    parser.add_argument("--high_res_pooling_h", default=2, type=int)
    parser.add_argument("--high_res_pooling_w", default=2, type=int)
    parser.add_argument("--periodic_high_res_frame", default=None, type=int)
    args, other_args = parser.parse_known_args()
    
    if args.mixture == "intern_vid":
        eval_tasks = ['intern_vid']
        tasks = [["aux", ["intern_vid"], 1.0]]

    elif args.mixture in ["lv_intern_vid"]:
        tasks = [["short_cap", ["intern_vid"], 0.1],
                ["long_cap_oe_mc", ["llava_video_178k"], 0.9]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_mc"]:
        tasks = [["lv_mc", ["llava_video_178k_mc"], 1.0]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_oe"]:
        tasks = [["lv_oe", ["llava_video_178k_oe"], 1.0]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_long_cap"]:
        tasks = [["lv_long_cap", ["llava_video_178k_cap"], 1.0]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_intern_vid"]:
        tasks = [["lv_mc", ["llava_video_178k_mc"], 0.2],
                 ["lv_oe", ["llava_video_178k_oe"], 0.2],
                 ["intern_vid_short_cap", ["intern_vid"], 0.4]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_koala"]:
        tasks = [["lv_mc", ["llava_video_178k_mc"], 0.2],
                 ["lv_oe", ["llava_video_178k_oe"], 0.2],
                 ["koala_long_cap", ["koala"], 0.4]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv"]:
        tasks = [["lv_mc", ["llava_video_178k_mc"], 0.2],
                 ["lv_oe", ["llava_video_178k_oe"], 0.4],
                 ["lv_long_cap", ["llava_video_178k_cap"], 0.4]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_split"]:
        tasks = [["lv_mc", ["llava_video_178k_mc_split"], 0.2],
                 ["lv_oe", ["llava_video_178k_oe"], 0.4],
                 ["lv_long_cap", ["llava_video_178k_cap"], 0.4]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_flat"]:
        tasks = [["lv_mc", ["llava_video_178k_mc_flat"], 0.2],
                 ["lv_oe", ["llava_video_178k_oe_flat"], 0.4],
                 ["lv_long_cap", ["llava_video_178k_cap_flat"], 0.4]]
        eval_tasks = ["mvbench"]

    else:
        raise NotImplementedError(args.mixture)

    debug = args.checkpoint in ["debug"]
    if debug:
        checkpoint = None
        model_cfg = VIDEO_DEBUG_MODEL
        global_batch_size = args.global_batch_size
        model_init = None
        inf_eval_interval = 20
        eval_interval = 20
        save_interval = 500
        log_interval = args.log_interval
        max_eval_examples = 16
        duration = 30000
        num_workers = 0
    else:
        global_batch_size = args.global_batch_size
        max_eval_examples = args.max_eval_examples
        log_interval = args.log_interval
        eval_interval = 1000
        save_interval = 1000
        duration = args.duration
        inf_eval_interval = 2000
        checkpoint = select_checkpoint(args.checkpoint)
        if exists(join(args.checkpoint, "model.yaml")):
            model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
        else:
            model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")
        model_cfg = VideoOlmoConfig(
            llm=model_cfg.llm,
            vision_backbone=model_cfg.vision_backbone,
            data_formatter=model_cfg.data_formatter,
            mm_preprocessor=MultiModalVideoPreprocessorConfig(
                time_mode=args.time_mode,
                high_res_pooling_h=args.high_res_pooling_h,
                high_res_pooling_w=args.high_res_pooling_w,
                periodic_high_res_frame=args.periodic_high_res_frame,
                max_frames=args.max_frames,
                frame_sample_mode=args.frame_sample_mode,
                candidate_sampling_fps=args.candidate_sampling_fps,
                **dict(
                    model_cfg.mm_preprocessor.asdict(),
                    crop_mode=args.crop_mode,
                    max_crops=args.max_crops,
                    pooling_h=args.image_pooling_h,
                    pooling_w=args.image_pooling_w,
                )
            ),
            bi_directional_attn=args.bi_directional_attn,
        )
        num_workers = args.num_workers

    if args.seq_len == "auto":
        max_for_image = model_cfg.mm_preprocessor.get_max_image_tokens(model_cfg.vision_backbone)
        if args.mixture in ["lv_flat"]:
            seq_len = 256 + max_for_image
        else:
            seq_len = 768 + max_for_image
        seq_len = ((seq_len  + 128 - 1) // 128) * 128
        log.info(f"Setting seq len to {seq_len}")
    else:
        seq_len = int(args.seq_len)
    model_cfg.llm.max_sequence_length = seq_len

    # Fine-tuning settings
    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments"

    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    evaluations = []
    inf_evaluators = []
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            seq_len,
            max_examples=max_eval_examples,
            num_workers=num_workers,
            for_inference=False,
            device_batch_size=args.device_eval_batch_size,
        )
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        evaluations.append(evaluation)

        inf_evaluation = get_evaluation(
            task,
            seq_len,
            max_examples=args.max_eval_examples_inf,
            num_workers=num_workers,
            for_inference=True,
            device_batch_size=args.device_inf_batch_size,
        )
        inf_evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        inf_evaluators.append(inf_evaluation)

    cfg = TrainConfig(
        run_name="multitask_video",
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
        compile_loss=True,
        compile=CompilerConfig(mode="default"),
        allow_resume=True,
        model=model_cfg,
        save_overwrite=debug,
        data=DataLoaderConfig(
            root_size_mixture=root_size_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            seed=50189,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=not args.freeze_vit,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=args.connector_learning_rate,
            vit_learning_rate=args.vit_learning_rate,
            llm_learning_rate=args.llm_learning_rate,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=args.device_train_batch_size,
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
        evaluators=evaluations,
        eval_interval=eval_interval,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=inf_evaluators,
        save_final_unsharded_checkpoint=False,
    )

    conf = OmegaConf.create(cfg)
    if other_args:
        conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)
