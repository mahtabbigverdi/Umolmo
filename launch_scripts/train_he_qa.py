import argparse
import logging
import time
from dataclasses import replace
from datetime import timedelta
from os import environ
from os.path import join, exists
from typing import cast

from cached_path import cached_path
from omegaconf import omegaconf, OmegaConf

from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, \
    get_evaluator, get_evaluation
from olmo.data.data_loader import RootSizeMixture, DataConfig
from olmo.eval.inf_evaluator import InfDatasetEvaluatorConfig
from olmo.models.he_molmo.he_data_formater import HeDataFormatter
from olmo.models.he_molmo.he_molmo import TokenScorerConfig, HeMolmoConfig
from olmo.models.he_molmo.he_preprocessor import HePreprocessorConfig
from olmo.models.he_molmo.token_selector import TokenSelectionConfig
from olmo.models.model import FSDPWrapStrategy
from olmo.nn.vision_backbone import VisionBackboneConfig
from olmo.torch_util import get_world_size
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.trainer_config import TrainConfig, WandbConfig, FSDPConfig, FSDPPrecision, \
    CompilerConfig, BatchDivisor, SpeedMonitorConfig
from scripts.train import run_trainer

from olmo.util import (
    clean_opt,
    prepare_cli_environment, prepare_torchrun_environment, select_checkpoint,
)
from olmo.io import add_cached_path_clients
import torch.multiprocessing as mp
import torch.distributed as dist


log = logging.getLogger("train")


if __name__ == "__main__":
    prepare_torchrun_environment()
    parser = argparse.ArgumentParser(prog="Train a captioner")
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--global_batch_size", default=256, type=int)
    parser.add_argument("--n_eval_examples", default=None, type=int)
    parser.add_argument("--n_high_res", default=None, type=int)
    args, other_args = parser.parse_known_args()

    if args.checkpoint in ["debug", "debug_he"]:
        debug = True
        model_cfg = HeMolmoConfig(
            DEBUG_MODEL.llm,
            DEBUG_MODEL.vision_backbone,
            data_formatter=HeDataFormatter(
                system_prompt="style_and_length_v2"
            ),
            mm_preprocessor=HePreprocessorConfig(
                crop_mode="overlap-and-resize-c2",
                max_crops=8 if args.vision_backbone == "siglip" else 12,
                overlap_margins=(4, 4)
            ),
            token_scorer=TokenScorerConfig(
                source="all_layers",
                low_res_features_drop=0.1,
                high_res_patch_prior_drop=0.1
            ),
            token_selector=TokenSelectionConfig(
                loss="batch-mean"
            ),
        )
        global_batch_size = 4
        model_init = None
        eval_interval = 20
        log_interval = 5
        eval_examples = 64
        duration = 200
    if args.checkpoint in ["debug_large"]:
        debug = True
        model_cfg = HeMolmoConfig(
            replace(LLMS["qwen2.5_3b"], init_path=None, n_layers=8),
            vision_backbone=VisionBackboneConfig(
                vit=replace(VISION_BACKBONES["metaclip_l14_336"], init_path=None, image_num_layers=4),
                vit_layers=[-3, -1],
                image_padding_embed=None,
            ),
            data_formatter=HeDataFormatter(
                system_prompt="style_and_length_v2"
            ),
            mm_preprocessor=HePreprocessorConfig(
                crop_mode="overlap-and-resize-c2",
                max_crops=12,
                overlap_margins=(4, 4)
            ),
            token_scorer=TokenScorerConfig(
                source="all_layers",
                low_res_features_drop=0.1,
                high_res_patch_prior_drop=0.1
            ),
            token_selector=TokenSelectionConfig(
                loss="batch-mean"
            ),
        )
        global_batch_size = 4
        model_init = None
        eval_interval = 20
        log_interval = 2
        eval_examples = 64
        duration = 200
    else:
        debug = False
        eval_examples = args.n_eval_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        duration = 6000
        eval_interval = 500
        args.checkpoint = select_checkpoint(args.checkpoint)
        model_init = args.checkpoint
        model_cfg = HeMolmoConfig.load(cached_path(join(args.checkpoint, "config.yaml")), key="model")

    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    if args.n_high_res:
        model_cfg.mm_preprocessor.num_high_res_features = args.n_high_res

    if isinstance(model_cfg, HeMolmoConfig):
        seq_len = 384 + model_cfg.mm_preprocessor.num_high_res_features
        model_cfg.mm_preprocessor.max_query_len = 128 + 64
    else:
        seq_len = 1664

    mixture = None
    root_size_mixture = None
    multi_res_mixture = None
    evaluators = []
    evals = ["chart_qa"]
    eval_config = DataConfig(
        dataset="",
        shuffle=False,
        split="validation",
        drop_last=True,
        sequence_length="${data.sequence_length}",
        num_workers="${data.num_workers}",
        pin_memory=True,
        persistent_workers=True,
    )

    if args.data == "chart_qa":
        mixture = dict(
            chart_qa_weighted=1.0,
        )
    elif args.data == "natural":
        mixture = dict(
            pixmo_ask_model_anything_flat=0.50,
            pixmo_cap_qa_flat=0.50
        )
        if model_cfg.token_selection:
            # model_cfg.token_selection.offset = 0
            seq_len = 512 + model_cfg.token_selection.num_high_res_features
            model_cfg.max_one_image_query_len = 128
        else:
            seq_len = 2048
        evals = []
        evaluators.append(InfDatasetEvaluatorConfig(
            label="pixmo_ama",
            max_examples=32 if debug else 2048,
            data=DataConfig(
                dataset="pixmo_ask_model_anything_flat",
                shuffle=False,
                split="validation",
                drop_last=True,
                sequence_length=seq_len,
                num_workers="${data.num_workers}",
                pin_memory=True,
                persistent_workers=True,
            ),
        ))
        evaluators.append(InfDatasetEvaluatorConfig(
            label="pixmo_cap_qa",
            max_examples=32 if debug else 2048,
            data=DataConfig(
                dataset="pixmo_cap_qa_flat",
                shuffle=False,
                split="validation",
                drop_last=True,
                sequence_length=seq_len,
                num_workers="${data.num_workers}",
                pin_memory=True,
                persistent_workers=True,
            ),
        ))
    elif args.data == "v1":
        mixture = dict(
            chart_qa_weighted=0.15,
            doc_qa=0.15,
            pixmo_docs_charts_flat=0.25,
            pixmo_docs_other_flat=0.25,
            pixmo_docs_tables_flat=0.1,
            pixmo_docs_diagrams_flat=0.1
        )
        evals.append("doc_qa")
    elif args.data == "v2":
        root_size_mixture = [RootSizeMixture(1.0, dict(
            chart_qa_weighted=None,
            doc_qa=None,
            info_qa=None,
            pixmo_docs_charts_flat=0.2,
            pixmo_docs_other_flat=0.2,
            pixmo_docs_tables_flat=0.2,
            pixmo_docs_diagrams_flat=0.2
        ))]
        evals.append("doc_qa")
        evals.append("info_qa")
    elif args.data == "v2-ex":
        root_size_mixture = [RootSizeMixture(1.0, dict(
            chart_qa_weighted=None,
            doc_qa=None,
            info_qa=None,
            pixmo_docs_charts_flat=None,
            pixmo_docs_other_flat=None,
            pixmo_docs_tables_flat=None,
            pixmo_docs_diagrams_flat=None
        ))]
        evals.append("doc_qa")
        evals.append("info_qa")
    else:
        raise ValueError(args.data)

    evaluator = InfDatasetEvaluatorConfig(
        label="val",
        subset_num_batches=2048//(8*get_world_size()),
        data=DataConfig(
            mixture=mixture,
            pad="to_max",
            shuffle=True,
            split="validation",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    cfg = TrainConfig(
        run_name="multitask_train",
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        initial_model_checkpoint=model_init,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        model=model_cfg,
        data=DataConfig(
            mixture=mixture,
            root_size_mixture=root_size_mixture,
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
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
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
            vit_t_warmup=200,
            llm_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            fsdp2=True,
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        reset_optimizer_state=True,
        reset_trainer_state=True,
        allow_resume=False,
        save_overwrite=True,
        load_path=None,
        compile=CompilerConfig(),
        fused_loss=False,
        save_interval=40000,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=8,
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
        activation_checkpointing=True,
        eval_interval=eval_interval,
        inf_eval_interval=eval_interval,
        evaluators=evaluators,
        inf_evaluators=[
            get_evaluation(x, seq_len, device_batch_size=4, max_examples=32 if debug else None)
            for x in evals
        ],
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)
