import argparse
import logging
import warnings
from dataclasses import replace
from typing import cast

from omegaconf import omegaconf, OmegaConf
from transformers import CompileConfig

from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS
from olmo.data.data_formatter import DataFormatter
from olmo.data.data_loader import DataConfig
from olmo.data.model_preprocessor import MultiModalPreprocessorConfig
from olmo.data.pixmo_datasets import PixMoCap
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.he_molmo.data_formater import HeDataFormatter
from olmo.he_molmo.he_molmo import TokenScorerConfig, HeMolmoConfig
from olmo.he_molmo.he_molmo_trainer import HeMolmoTrainerConfig
from olmo.he_molmo.hierarchical_preprocessor import HePreprocessorConfig
from olmo.he_molmo.token_selector import TokenSelectionConfig
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig
from olmo.nn.model import FSDPWrapStrategy, ModelConfig
from olmo.nn.vision_backbone import VisionBackboneConfig
from olmo.tokenizer import TokenizerConfig
from olmo.torch_util import get_world_size
from olmo.train.optim import OptimizerType, OptimizerConfig, SchedulerConfig, SchedulerType
from olmo.train.trainer_config import SpeedMonitorConfig, WandbConfig, FSDPConfig, FSDPPrecision, \
    CompilerConfig, BatchDivisor

from olmo.util import clean_opt, prepare_torchrun_environment
from olmo.io import add_cached_path_clients
import torch.multiprocessing as mp
import torch.distributed as dist

from scripts.mm_eval import DatasetEvaluatorConfig
from scripts.train import run_trainer

log = logging.getLogger("train")


if __name__ == "__main__":
    prepare_torchrun_environment()
    parser = argparse.ArgumentParser(prog="Train a captioner")
    parser.add_argument("llm", choices=["debug", "debug-12crop"] + list(LLMS.keys()))
    parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="metaclip_l14_336")
    parser.add_argument("--global_batch_size", default=256, type=int)
    parser.add_argument("--n_eval_examples", default=2048, type=int)
    parser.add_argument("--num_high_res_features", default=512, type=int)
    parser.add_argument("--device_eval_batch_size", default=4, type=int)
    parser.add_argument("--two_epochs", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    args, other_args = parser.parse_known_args()

    debug = args.llm in ["debug", "debug-12crop"]
    if debug:
        model_cfg = HeMolmoConfig(
            llm=LlmConfig(
                d_model=128,
                n_heads=2,
                n_layers=1,
                max_sequence_length=4096,
                additional_vocab_size=128,
                vocab_size=152064,
                rope=True,
                embedding_size=None,
                weight_tying=False,
                tokenizer=TokenizerConfig(
                    identifier="Qwen/Qwen2-7B",
                )
            ),
            vision_backbone=VisionBackboneConfig(
                vit=VitConfig(image_num_layers=1, resize_mode="metaclip"),
            ),
            token_scorer=TokenScorerConfig(
              n_features=512,
              source="all-layers"
            ),
            token_selection=TokenSelectionConfig(),
            data_formatter=HeDataFormatter(),
            mm_preprocessor=HePreprocessorConfig(crop_mode="overlap-and-resize-c2", max_crops=6)
        )

        global_batch_size = max(8, get_world_size())
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
        duration = 2 * (2 if args.two_epochs else 4) * (n + global_batch_size - 1) // global_batch_size
        eval_interval = 1000
        vit_layers = [-2, -9] if args.vision_backbone == "openai" else [-3, -9]
        raise ValueError()
        # model_cfg = replace(
        #     LLMS[args.llm],
        #     vision_backbone=VISION_BACKBONES[args.vision_backbone],
        #     llm_load_path=DEFAULT_LOAD_PATHS.get(args.llm, omegaconf.MISSING),
        #     vit_load_path=DEFAULT_LOAD_PATHS.get(args.vision_backbone, omegaconf.MISSING),
        #     crop_mode="overlap-and-resize-c2",
        #     system_prompt_kind='style_and_length_v2',
        #     residual_dropout=0.0,
        #     response_residual_dropout=0.1,
        #     max_crops=12,
        #     vit_layers=vit_layers,
        #     additional_vocab_size=128,
        # )

    model_cfg.vision_backbone.image_padding_embed = None
    model_cfg.bi_directional_attn = "within_image"

    model_cfg.token_selection.n_features = 512
    seq_len = 768 + 512

    evaluator = LossDatasetEvaluatorConfig(
        label="val",
        max_examples=eval_examples,
        device_batch_size=args.device_eval_batch_size,
        data=DataConfig(
            seed="${seed}",
            dataset="pixmo_cap_transcript",
            shuffle=False,
            split="validation",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    warmup_scale = 2 if args.two_epochs else 1
    cfg = HeMolmoTrainerConfig(
        run_name="multitask_train",
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
        fused_loss=False,
        compile_loss=True,
        model=model_cfg,
        data=DataConfig(
            mixture=dict(
                pixmo_cap=0.5,
                pixmo_transcript=0.5
            ),
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
            metrics_log_interval=20
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200//warmup_scale,
            vit_t_warmup=2000//warmup_scale,
            llm_t_warmup=2000//warmup_scale,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        allow_resume=not debug,
        save_overwrite=True,
        load_path=None,
        compile=CompilerConfig(mode="default", dynamic=False),
        initial_model_checkpoint=None,
        save_interval=100000 if args.two_epochs else 4000,
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
        evaluators=[
            # Evaluate loss on data with and without the transcripts
            evaluator,
            replace(
                evaluator,
                label="caption_val",
                data=replace(evaluator.data, dataset="pixmo_cap")
            )
        ]
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(HeMolmoTrainerConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)
