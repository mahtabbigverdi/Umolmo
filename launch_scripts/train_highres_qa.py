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
from olmo.torch_util import get_world_size
from scripts.train import run_trainer

from olmo.util import (
    clean_opt,
    prepare_cli_environment, prepare_torchrun_environment,
)
from olmo.io import add_cached_path_clients
import torch.multiprocessing as mp
import torch.distributed as dist


log = logging.getLogger("train")


if __name__ == "__main__":
    prepare_torchrun_environment()
    parser = argparse.ArgumentParser(prog="Train a captioner")
    parser.add_argument("checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="metaclip_l14_336")
    parser.add_argument("--global_batch_size", default=256, type=int)
    parser.add_argument("--n_eval_examples", default=None, type=int)
    parser.add_argument("--n_high_res", default=None, type=int)
    args, other_args = parser.parse_known_args()

    if args.checkpoint in ["debug", "debug_he"]:
        debug = True
        model_cfg = DEBUG_MODEL
        model_cfg.vision_backbone.resize_mode = "metaclip"
        model_cfg.max_crops = 12
        model_cfg.crop_mode = "overlap-and-resize-c2"
        model_cfg.bi_directional_image_attn = "within_image"
        model_cfg.system_prompt_kind = 'style_and_length'
        if args.checkpoint == "debug_he":
            model_cfg.token_selection = TokenSelectionConfig(
                num_high_res_features=128,
                high_res_patch_selector= "token_scaling",
                high_res_patch_prior=True,
                one_score_per_patch=False,
                fully_offset_position_ids=True,
                mask_invalid=True,
                importance_scores_source= "all_layer",
            )

        global_batch_size = 4
        model_init = None
        eval_interval = 20
        log_interval = 5
        eval_examples = 64
        duration = 200
    elif args.checkpoint in ["3b_dbg", "llm_3b_dbg"]:
        debug = False
        global_batch_size = args.global_batch_size
        model_init = None
        eval_interval = 20000
        log_interval = 2
        eval_examples = 64
        duration = 200
        model_cfg = replace(
            LLMS["qwen2.5_3b"],
            vision_backbone=replace(VISION_BACKBONES["metaclip_l14_336"], image_num_layers=12),
            llm_load_path=None,
            vit_load_path=None,
            bi_directional_image_attn="within_image",
            crop_mode="overlap-and-resize-c2",
            system_prompt_kind='style_and_length_v2',
            residual_dropout=0.0,
            response_residual_dropout=0.1,
            max_crops=12,
            vit_layers=[-2],
            additional_vocab_size=128,
        )
        if args.checkpoint == "llm_3b_dbg":
            model_cfg.llm_token_selection = LlmTokenSelectionConfig(
                k=args.n_high_res,
                dropout=0.1,
            )
        else:
            model_cfg.token_selection = TokenSelectionConfig(
                num_high_res_features=args.n_high_res,
                high_res_patch_selector= "token_scaling",
                high_res_patch_prior_drop=0.1,
                one_score_per_patch=False,
                high_res_patch_prior=True,
                importance_scores_source="all_layers",
                fully_offset_position_ids=True,
                mask_invalid=True,
                high_res_col_tokens=True
            )
    else:
        debug = False
        eval_examples = args.n_eval_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        duration = 6000
        eval_interval = 500
        args.checkpoint = select_checkpoint(args.checkpoint)
        model_init = args.checkpoint
        model_cfg = ModelConfig.load(cached_path(join(args.checkpoint, "config.yaml")), key="model")

    model_cfg.residual_dropout = 0.1
    model_cfg.response_residual_dropout = 0.0
    model_cfg.prompt_type = "uber_model"
    model_cfg.message_formatting = "role"
    model_cfg.system_prompt_kind = "demo_or_style"
    model_cfg.multi_annotation_weighting = "root_subsegments"
    if args.n_high_res:
        if model_cfg.token_selection is not None:
            model_cfg.token_selection.num_high_res_features = args.n_high_res
        elif model_cfg.llm_token_selection is not None:
            model_cfg.llm_token_selection.k = args.n_high_res

    if model_cfg.token_selection:
        seq_len = 384 + model_cfg.token_selection.num_high_res_features
        model_cfg.token_selection.max_query_len = 128
        model_cfg.token_selection.multi_res_min = None
        model_cfg.token_selection.multi_res_selection = None
    else:
        seq_len = 1664

    mixture = None
    root_size_mixture = None
    multi_res_mixture = None
    evaluators = []
    evals = ["chart_qa"]
    eval_config = DataConfig(
        dataset="",
        for_inference=False,
        shuffle=False,
        split="validation",
        drop_last=True,
        sequence_length="${data.sequence_length}",
        num_workers="${data.num_workers}",
        pin_memory=True,
        persistent_workers=True,
        shuffle_messages=False,
    )

    if args.data == "chart_qa":
        mixture = dict(
            chart_qa_weighted=1.0,
        )
    elif args.data == "chart_qa_multi_res":
        multi_res_mixture = dict(
            chart_qa_weighted=MultiResDataset(
                rate=1.0,
                preprocessor_args=PreprocessorKwargs(min_k=32, max_k=64, num_k=3)
            ),
        )
    elif args.data in ["pixmo_simple_ocr_qa"]:
        seq_len = 512 + model_cfg.token_selection.num_high_res_features
        evals = []
        model_cfg.max_one_image_query_len = 128
        mixture = dict(pixmo_simple_ocr_qa=1.0)
        base_eval = DatasetEvaluatorConfig(
            label="val",
            max_examples=32 if debug else 2048,
            data=replace(eval_config, dataset="pixmo_simple_ocr_qa"),
        )
        evaluators = [base_eval]
    elif args.data in ["pixmo_simple_ocr_qa_multires"]:
        seq_len = 512 + model_cfg.token_selection.num_high_res_features
        evals = []
        model_cfg.max_one_image_query_len = 128
        multi_res_mixture = dict(
            pixmo_simple_ocr_qa=MultiResDataset(
                rate=1.0,
                preprocessor_args=PreprocessorKwargs(min_k=16, max_k=64)
            ),
        )
        evaluators = [
            DatasetEvaluatorConfig(
                label="val",
                max_examples=32 if debug else 2048,
                data=replace(eval_config, dataset="pixmo_simple_ocr_qa", preprocessor_args=PreprocessorKwargs(max_k=32)),
            ),
            DatasetEvaluatorConfig(
                label="val-16",
                max_examples=32 if debug else 2048,
                data=replace(eval_config, dataset="pixmo_simple_ocr_qa", preprocessor_args=PreprocessorKwargs(max_k=16)),
            ),
            DatasetEvaluatorConfig(
                label="val-64",
                max_examples=32 if debug else 2048,
                data=replace(eval_config, dataset="pixmo_simple_ocr_qa", preprocessor_args=PreprocessorKwargs(max_k=64)),
            )
        ]
        model_cfg.token_selection.indicate_k = "before-low-res"
        model_cfg.token_selection.loss = "example-mean"
    elif args.data in ["simple_v1", "simple_v1_multi_res"]:
        evals = []
        base_eval = DatasetEvaluatorConfig(
            label="val",
            max_examples=32 if debug else 2048,
            data=replace(eval_config, dataset="pixmo_simple_cap_qa"),
        )
        if args.data == "simple_v1_multi_res":
            multi_res_mixture = dict(
                pixmo_simple_cap_qa=MultiResDataset(
                    rate=1.0, preprocessor_args=PreprocessorKwargs(min_k=32, max_k=128)))
            evaluators = [replace(
                base_eval,
                data=replace(base_eval.data, preprocessor_args=PreprocessorKwargs(max_k=64)),
                label="val",
            )]
            evaluators.append(replace(
                base_eval,
                data=replace(base_eval.data, preprocessor_args=PreprocessorKwargs(max_k=128)),
                label="val_fe128",
            ))
            evaluators.append(replace(
                base_eval,
                data=replace(base_eval.data, preprocessor_args=PreprocessorKwargs(max_k=32)),
                label="val_f32",
            ))
            model_cfg.token_selection.loss = "example-mean"
            model_cfg.token_selection.indicate_k = "before-low-res"
        else:
            mixture = dict(pixmo_simple_cap_qa=1.0)
            evaluators.append(base_eval)
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
        evaluators.append(DatasetEvaluatorConfig(
            label="pixmo_ama",
            max_examples=32 if debug else 2048,
            data=DataConfig(
                dataset="pixmo_ask_model_anything_flat",
                for_inference=False,
                shuffle=False,
                split="validation",
                drop_last=True,
                sequence_length=seq_len,
                num_workers="${data.num_workers}",
                pin_memory=True,
                persistent_workers=True,
                shuffle_messages=False,
            ),
        ))
        evaluators.append(DatasetEvaluatorConfig(
            label="pixmo_cap_qa",
            max_examples=32 if debug else 2048,
            data=DataConfig(
                dataset="pixmo_cap_qa_flat",
                for_inference=False,
                shuffle=False,
                split="validation",
                drop_last=True,
                sequence_length=seq_len,
                num_workers="${data.num_workers}",
                pin_memory=True,
                persistent_workers=True,
                shuffle_messages=False,
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
    elif args.data == "scifi":
        root_size_mixture = [RootSizeMixture(1.0, dict(
            pixmo_docs_charts_flat=None,
            pixmo_docs_other_flat=None,
            pixmo_docs_tables_flat=None,
            pixmo_docs_diagrams_flat=None
        ))]
        evals = []
        base_eval = DatasetEvaluatorConfig(
            label="val",
            max_examples=32 if debug else 2048,
            data=replace(eval_config, dataset="pixmo_docs_charts_flat"),
        )
        evaluators = [base_eval]
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
    elif args.data == "v2-192-train":
        kwargs = PreprocessorKwargs(min_k=None, max_k=192)
        multi_res_mixture = dict(
            chart_qa_weighted=MultiResDataset(None, preprocessor_args=kwargs),
            doc_qa=MultiResDataset(None, preprocessor_args=kwargs),
            info_qa=MultiResDataset(None, preprocessor_args=kwargs),
            pixmo_docs_charts_flat=MultiResDataset(0.2, preprocessor_args=kwargs),
            pixmo_docs_other_flat=MultiResDataset(0.2, preprocessor_args=kwargs),
            pixmo_docs_tables_flat=MultiResDataset(0.2, preprocessor_args=kwargs),
            pixmo_docs_diagrams_flat=MultiResDataset(0.2, preprocessor_args=kwargs)
        )
        seq_len += 64
        evals.append("doc_qa")
        evals.append("info_qa")
    elif args.data == "v2-pointing":
        kwargs = PreprocessorKwargs(min_k=None, max_k=128)
        multi_res_mixture = dict(
            pixmo_points_flat=MultiResDataset(0.15, preprocessor_args=kwargs),
            chart_qa_weighted=MultiResDataset(None, preprocessor_args=kwargs),
            doc_qa=MultiResDataset(None, preprocessor_args=kwargs),
            info_qa=MultiResDataset(None, preprocessor_args=kwargs),
            pixmo_docs_charts_flat=MultiResDataset(0.2, preprocessor_args=kwargs),
            pixmo_docs_other_flat=MultiResDataset(0.2, preprocessor_args=kwargs),
            pixmo_docs_tables_flat=MultiResDataset(0.2, preprocessor_args=kwargs),
            pixmo_docs_diagrams_flat=MultiResDataset(0.2, preprocessor_args=kwargs)
        )
        seq_len = 640 + model_cfg.token_selection.num_high_res_features
        model_cfg.max_one_image_query_len = 128
        evals.append("doc_qa")
        evals.append("info_qa")
    elif args.data == "v2-multi-res":
        kwargs = PreprocessorKwargs(min_k=64, max_k=256, sample_k="lin0.2")
        model_cfg.token_selection.indicate_k = "before-low-res"
        model_cfg.token_selection.loss = "example-mean"
        multi_res_mixture = dict(
            chart_qa_weighted=MultiResDataset(None, preprocessor_args=kwargs),
            doc_qa=MultiResDataset(None, preprocessor_args=kwargs),
            info_qa=MultiResDataset(None, preprocessor_args=kwargs),
            pixmo_docs_charts_flat=MultiResDataset(root_rate=0.2, preprocessor_args=kwargs),
            pixmo_docs_other_flat=MultiResDataset(root_rate=0.2, preprocessor_args=kwargs),
            pixmo_docs_tables_flat=MultiResDataset(root_rate=0.2, preprocessor_args=kwargs),
            pixmo_docs_diagrams_flat=MultiResDataset(root_rate=0.2, preprocessor_args=kwargs)
        )
        evals.append("doc_qa")
        evals.append("info_qa")
        seq_len = 384 + 256
    else:
        raise ValueError(args.data)

    evaluator = DatasetEvaluatorConfig(
        label="val",
        subset_num_batches=2048//(8*get_world_size()),
        data=DataConfig(
            mixture=mixture,
            for_inference=False,
            pad="to_max",
            shuffle=True,
            split="validation",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            shuffle_messages=False,
        ),
    )

    cfg = TrainConfig(
        run_name="multitask_train",
        no_pre_train_checkpoint=True,
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
            multi_res_mixture=multi_res_mixture,
            for_inference=False,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            seed=95818,
            num_workers=2,
            pad="to_max",
            pin_memory=True,
            shuffle_messages=False,
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
        allow_resume=False,
        save_overwrite=True,
        load_path=None,
        compile=CompilerConfig(target="image_blocks", dynamic=False),
        fused_loss=False,
        save_dataloader_state=False,
        save_interval=40000,
        save_num_checkpoints_to_keep=1,
        save_interval_unsharded="${max_duration}",
        global_train_batch_size=global_batch_size,
        device_eval_batch_size="${device_train_microbatch_size}",
        device_inf_eval_batch_size=8,
        device_train_microbatch_size=8,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        save_unsharded_optim=False,
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
        eval_interval=eval_interval,
        inf_eval_interval=eval_interval,
        evaluators=evaluators,
        inf_evaluators=[
            get_evaluation(x, seq_len, 4,
                           max_examples=32 if debug else None,
                           persistent_workers=True)
            for x in evals
        ],
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    train(cfg)
