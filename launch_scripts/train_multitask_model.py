import argparse
import logging
from os.path import join, exists
from typing import cast, List

import omegaconf
from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, DEBUG_MODEL
from olmo.train.optim import OptimizerType, OptimizerConfig, SchedulerConfig, SchedulerType
from olmo.train.trainer_config import (
    WandbConfig, BatchDivisor, SpeedMonitorConfig,
    FSDPConfig, FSDPPrecision, CompilerConfig, TrainConfig
)
from olmo.models.model import FSDPWrapStrategy
from olmo.models.molmo.molmo import MolmoConfig
from olmo.data.data_loader import DataLoaderConfig, RootSizeMixture
from olmo.torch_util import get_world_size
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint
from scripts.train import run_trainer

log = logging.getLogger("train")


AUX_EXCEPT_DOCS = [
    # Supervised datasets we want eval on
    "coco_2014_vqa_multi",
    "text_vqa",
    "okvqa",
    "chart_qa_weighted",
    "doc_qa",
    "info_qa",
    "ai2_diagram_v2_mix_transparent",
    "a_okvqa_mc",
    "a_okvqa_da",
    "android_control",

    # Some other datasets we might want to eval on
    "science_qa_img",
    "tabwmp_da",
    "st_qa",
    "tally_qa",

    ("pixmo_clocks", 250000),  # Downsample since it is huge

    # # Other synthetic data, also downsampled since they are huge
    ("dv_qa", 10000),
    ("figure_qa", 10000),
    ("plot_qa", 20000),
]


AUX = AUX_EXCEPT_DOCS + [
    "pixmo_docs_charts",
    "pixmo_docs_tables",
    "pixmo_docs_other",
    "pixmo_docs_diagrams",
]


AUX_COSYN_V1 = AUX_EXCEPT_DOCS + [
    "cosyn_chart_exp",
    "cosyn_chemical_exp",
    # "cosyn_circuit_exp", # quality not good
    "cosyn_diagram_exp",
    "cosyn_document",
    # "cosyn_graphic_exp", # quality not good
    "cosyn_math_exp",
    "cosyn_music_exp",
    # "cosyn_nutrition_exp", # zero-shot evaluation dataset
    "cosyn_table_exp",
]


def get_training_mixture(submixture):
    resolved_weights = {}
    for task_name in submixture:
        mix = {}
        if isinstance(task_name, tuple):
            task_name, size = task_name
        else:
            size = None
        resolved_weights[task_name] = size
    return resolved_weights


if __name__ == "__main__":
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--seq_len", default=2304, type=int)
    parser.add_argument("--inf_seq_len", default=1792, type=int)
    parser.add_argument("--duration", default=30000, type=int)
    parser.add_argument("--max_inf_examples", default=16, type=int)
    parser.add_argument("--global_batch_size", default=32, type=int)
    parser.add_argument("--device_eval_batch_size", default=4, type=int)
    parser.add_argument("--device_inf_batch_size", default=4, type=int)
    parser.add_argument("--device_train_batch_size", default=4, type=int)
    parser.add_argument("--include_image", action="store_true",
                        help="Include image in the evaluation outputs")
    parser.add_argument("--turn_off_inference", action="store_true",
                        help="Turn off inference during training")
    parser.add_argument("--max_crops", default=None, type=int)
    parser.add_argument("--image_pooling_h", default=None, type=int)
    parser.add_argument("--image_pooling_w", default=None, type=int)
    parser.add_argument("--max_images", default=None, type=int)
    args, other_args = parser.parse_known_args()

    if args.mixture.startswith("single"):
        task_name = args.mixture.split("_", 1)[1]
        eval_tasks = [task_name,]
        tasks = [["eval", eval_tasks, 1.0]]
    elif args.mixture == "android":
        eval_tasks = ["android_control_ll"]
        tasks = [["eval", ["android_control"], 1.0]]
    elif args.mixture in ["small1", "debug"]:
        eval_tasks = ["depth"]
        tasks = [["aux", ["depth"], 1.0]]
    elif args.mixture in ["smallmahtab"]:
        eval_tasks = ["depth"]
        tasks = [["aux", ["depth"], 1.0]]
    elif args.mixture in ["pointing"]:
        eval_tasks = ["pointing_eval:test"]
        tasks = [["pointing", [
            "pixmo_points",
            "pixmo_count",
            "pixmo_points_high_freq",
            "pixmo_points_counting",
            "pixmo_points_high_freq_counting",
            "pixmo_count_counting",
        ], 1.0]]

    elif args.mixture == "small2":
        eval_tasks = ["chart_qa", "doc_qa", "info_qa"]
        tasks = [["aux", [("chart_qa", 4*4),
                          ("doc_qa", 2*2), ("info_qa", 1)], 1.0]]
    elif args.mixture in ["3.2-synthetic"]:
        aux = list(AUX)
        eval_tasks = [
            "chart_qa",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            "pixmo_clocks",
            "android_control_ll",
            "pointing_eval:test",
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points_train",
                "pixmo_count_train",
                "pixmo_points_high_freq_train",
            ], 0.35]
        ]
    elif args.mixture in ["3.3-synthetic"]:
        aux = list(AUX_COSYN_V1)
        eval_tasks = [
            "chart_qa",
            "chart_qa_exp",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            "pixmo_clocks",
            "android_control_ll",
            "pointing_eval:test",
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points",
                "cosyn_point",
                "pixmo_count",
                "pixmo_points_high_freq",
                "pixmo_points_counting",
                "pixmo_points_high_freq_counting",
                "pixmo_count_counting",
            ], 0.35]
        ]
    elif args.mixture in ["3.4-synthetic"]:
        aux = list(AUX_COSYN_V1)
        eval_tasks = [
            "chart_qa",
            "chart_qa_exp",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            "pixmo_clocks",
            "android_control_ll",
            "pointing_eval:test",
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points_train",
                "pixmo_count_train",
                "pixmo_points_high_freq_train",
                "cosyn_point",
            ], 0.35]
        ]
    elif args.mixture in ["multi-image"]:
        eval_tasks = ["muir_bench:test", "muir_bench_mc:test"]
        tasks = [["multi-image", ["correction_qa_multi_only_train"], 1.0]]
    else:
        raise NotImplementedError(args.mixture)

    debug = args.checkpoint in ["debug", "debug2"]
    if debug:
        checkpoint = None
        model_cfg = DEBUG_MODEL
        if args.checkpoint == "debug2":
            model_cfg.max_crops = 12
            model_cfg.crop_mode = "overlap-and-resize-c2"
            model_cfg.tokenizer.identifier = "mm:hf-Qwen/Qwen2-7B"
            model_cfg.embedding_size = 152064
            model_cfg.vocab_size = 152064
            model_cfg.pad_tokenizer = True
        global_batch_size = 8
        model_init = None
        inf_eval_interval = 1000
        eval_interval = 1000
        log_interval = 10
        eval_examples = 16
        max_inf_examples = 16
        duration = '2ep'
        eval_subset_batches = 4
    
    else:
        eval_examples = 16
        max_inf_examples = 16
        ## was 2048 before for both max_inf_examples and eval_examples
        # max_inf_examples = args.max_inf_examples

        log_interval = 5
        global_batch_size = args.global_batch_size
        inf_eval_interval = 300
        eval_interval = 300
        duration =  600
        checkpoint = select_checkpoint(args.checkpoint)
        if exists(join(checkpoint, "model.yaml")):
            model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
        else:
            model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")

        eval_subset_batches = eval_examples//(args.device_eval_batch_size*get_world_size())
        logging.info(f"Setting eval subset batches to {eval_subset_batches}")
        assert eval_subset_batches > 0

    # Fine-tuning settings
    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments"

    # Overriding model config
    model_cfg.mm_preprocessor.max_crops = args.max_crops or model_cfg.mm_preprocessor.max_crops
    model_cfg.mm_preprocessor.pooling_w = args.image_pooling_w or model_cfg.mm_preprocessor.pooling_w
    model_cfg.mm_preprocessor.pooling_h = args.image_pooling_h or model_cfg.mm_preprocessor.pooling_h
    model_cfg.mm_preprocessor.max_images = args.max_images or model_cfg.mm_preprocessor.max_images
    if model_cfg.llm.max_sequence_length < args.seq_len:
        model_cfg.llm.max_sequence_length = args.seq_len
    
    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    num_workers = 0
    evaluations = []
    if not args.turn_off_inference:
        for task in eval_tasks:
            evaluation = get_evaluation(
                task,
                args.inf_seq_len,
                device_batch_size=args.device_inf_batch_size,
                max_examples=max_inf_examples,
                num_workers=num_workers,
                include_image=args.include_image,
            )
            evaluation.data.persistent_workers = True
            evaluations.append(evaluation)
    cfg = TrainConfig(
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
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=debug,
        data=DataLoaderConfig(
            root_size_mixture=root_size_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            seed=50189,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
            gen_learning_rate=1e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            gen_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            gen_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            gen_eps=1e-6,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            gen_t_warmup=200,
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
        save_interval=80,
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
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        eval_interval=eval_interval,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=evaluations,
        save_final_unsharded_checkpoint=False,
        evaluators=[]
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)
