import argparse
import logging
from os.path import join, exists
from typing import cast, List

import omegaconf
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, DEBUG_MODEL
from launch_scripts.train_multitask_model import get_training_mixture

from olmo import TrainConfig
from olmo.config import DataConfig, \
    ModelConfig, WandbConfig, OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType, \
    BatchDivisor, SpeedMonitorConfig, ActivationCheckpointingStrategy, FSDPConfig, FSDPWrapStrategy, \
    FSDPPrecision, RootSizeMixture, CompilerConfig
from olmo.torch_util import get_world_size
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
from scripts.train import main as train

log = logging.getLogger("train")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Initialize process group.
    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--wandb_run_name", default="multitask_video", type=str)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--max_inf_examples", default=1024, type=int)
    parser.add_argument("--crop_mode", default="resize", type=str)
    parser.add_argument("--max_frames", default=8, type=int)
    parser.add_argument("--candidate_sampling_fps", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--max_crops", default=4, type=int)
    parser.add_argument("--frame_sample_mode", default="fps", type=str)
    parser.add_argument("--bi_directional_attn", default=None, type=str)
    parser.add_argument("--global_batch_size", default=32, type=int)
    parser.add_argument("--device_eval_batch_size", default=2, type=int)
    parser.add_argument("--device_inf_batch_size", default=2, type=int)
    parser.add_argument("--device_train_batch_size", default=2, type=int)
    parser.add_argument("--llm_learning_rate", default=1e-5, type=float)
    parser.add_argument("--vit_learning_rate", default=5e-6, type=float)
    parser.add_argument("--connector_learning_rate", default=5e-6, type=float)
    parser.add_argument("--duration", default=14000, type=int)
    parser.add_argument("--image_pooling_h", default=2, type=int)
    parser.add_argument("--image_pooling_w", default=2, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    args, other_args = parser.parse_known_args()
    
    if args.mixture == "intern_vid":
        eval_tasks = ['intern_vid']
        tasks = [["aux", ["intern_vid"], 1.0]]

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
        tasks = [["lv_mc", ["llava_video_178k_mc"], 0.33],
                 ["lv_oe", ["llava_video_178k_oe"], 0.33],
                 ["intern_vid_short_cap", ["intern_vid"], 0.34]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv_koala"]:
        tasks = [["lv_mc", ["llava_video_178k_mc"], 0.33],
                 ["lv_oe", ["llava_video_178k_oe"], 0.33],
                 ["koala_long_cap", ["koala"], 0.34]]
        eval_tasks = ["mvbench"]

    elif args.mixture in ["lv"]:
        tasks = [["lv_mc", ["llava_video_178k_mc"], 0.2],
                 ["lv_oe", ["llava_video_178k_oe"], 0.4],
                 ["lv_long_cap", ["llava_video_178k_cap"], 0.4]]
        eval_tasks = ["mvbench"]
    else:
        raise NotImplementedError(args.mixture)

    debug = args.checkpoint in ["debug"]
    if debug:
        model_cfg = DEBUG_MODEL
        global_batch_size = args.global_batch_size
        model_init = None
        inf_eval_interval = 20000
        eval_interval = 20
        save_interval = 500
        log_interval = args.log_interval
        eval_examples = 16
        max_inf_examples = 16
        duration = 30000
        eval_subset_batches = 4
        num_workers = 0
    else:
        global_batch_size = args.global_batch_size
        max_inf_examples = args.max_inf_examples
        eval_examples = 256
        log_interval = args.log_interval
        eval_interval = 2000
        save_interval = 2000
        duration = args.duration
        inf_eval_interval = 2000  # Hack - never trigger inf eval
        model_init = args.checkpoint
        if exists(join(args.checkpoint, "model.yaml")):
            model_cfg = ModelConfig.load(join(args.checkpoint, "model.yaml"))
        else:
            model_cfg = ModelConfig.load(join(args.checkpoint, "config.yaml"), key="model")

        eval_subset_batches = eval_examples//(args.device_eval_batch_size*get_world_size())
        logging.info(f"Setting eval subset batches to {eval_subset_batches}")
        assert eval_subset_batches > 0
        num_workers = 2

    model_cfg.bi_directional_attn = args.bi_directional_attn
    if model_cfg.bi_directional_attn:
        log.info(f"Setting bi-directional attention to {model_cfg.bi_directional_attn}")

    model_cfg.image_pooling_h = args.image_pooling_h
    model_cfg.image_pooling_w = args.image_pooling_w
    log.info(f"Setting image pooling to {model_cfg.image_pooling_h}x{model_cfg.image_pooling_w}")

    # Setting for the video training
    model_cfg.crop_mode = args.crop_mode
    model_cfg.max_frames = args.max_frames
    model_cfg.max_crops = args.max_crops
    model_cfg.frame_sample_mode = args.frame_sample_mode
    model_cfg.candidate_sampling_fps = args.candidate_sampling_fps
    max_crops = model_cfg.get_max_crops()
    log.info(
        f"Sample_mode: {model_cfg.frame_sample_mode}, max frames: {model_cfg.max_frames}, candidate_fps: {model_cfg.candidate_sampling_fps}, "
        f"crop_mode: {model_cfg.crop_mode}, max_crops: {max_crops}"
    )

    if args.seq_len >= model_cfg.max_sequence_length:
        model_cfg.max_sequence_length = args.seq_len
    log.info(f"Max sequence length to {model_cfg.max_sequence_length}")

    # Fine-tuning settings
    model_cfg.residual_dropout = 0.1
    model_cfg.response_residual_dropout = 0.0
    model_cfg.prompt_type = "uber_model"
    model_cfg.message_formatting = "role"
    model_cfg.system_prompt_kind = "demo_or_style"
    model_cfg.multi_annotation_weighting = "root_subsegments"

    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    evaluations = []
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            args.seq_len,
            max_examples=max_inf_examples,
            num_workers=num_workers,
            for_inference=True,
        )
        evaluation.data.persistent_workers = True
        evaluations.append(evaluation)

    # import os
    # wandb_entity = os.environ.get("WANDB_ENTITY", "prior-ai2")
    # wandb_project = os.environ.get("WANDB_PROJECT", "video_olmo")

    cfg = TrainConfig(
        run_name=args.wandb_run_name,
        no_pre_train_checkpoint=True,
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
        compile=CompilerConfig(mode="default", target="blocks", dynamic=False),
        allow_resume=True,
        model=model_cfg,
        save_overwrite=debug,
        save_dataloader_state=False,
        data=DataConfig(
            root_size_mixture=root_size_mixture,
            for_inference=False,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=num_workers,
            pad="to_max",
            shuffle_messages=True,
            pin_memory=True,
            seed=50189,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
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
            metrics_log_interval=20
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
        initial_model_checkpoint=None if "debug" in args.checkpoint else args.checkpoint,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=1,
        save_interval_unsharded="${max_duration}",
        global_train_batch_size=global_batch_size,
        device_inf_eval_batch_size=args.device_inf_batch_size,
        device_eval_batch_size=args.device_eval_batch_size,
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
        activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
        eval_interval=eval_interval,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=evaluations,
        eval_subset_num_batches=eval_subset_batches,
        evaluators=[],
    )

    conf = OmegaConf.create(cfg)
    if other_args:
        overrides = [clean_opt(arg) for arg in other_args]
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    train(cfg)
