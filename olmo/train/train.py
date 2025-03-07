"""Run this script with 'torchrun'."""

import logging
import os
import sys
from os import listdir
from os.path import join
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from beaker import Beaker
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from olmo.train.trainer_config import TrainConfig, CheckpointType
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.nn.model import Molmo
from olmo.train.optim import BoltOnWarmupScheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    freeze_parameters_by_name,
)
from olmo.train.trainer import Trainer, BeakerLogger
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


def run(cfg: TrainConfig) -> None:
    if cfg.run_name is None:
        log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    if cfg.load_path is not None and cfg.model.low_cpu_fsdp:
        log.warning(
            "When loading a checkpoint to resume/finetune, the `low_cpu_fsdp` will be ignored."
        )
        cfg.model.low_cpu_fsdp = False
    
    if "BEAKER_EXPERIMENT_ID" in os.environ and "BEAKER_TOKEN" in os.environ:
        if get_global_rank() == 0:
            experiment_id = os.environ["BEAKER_EXPERIMENT_ID"]
            client = Beaker.from_env()
            experiment = client.experiment.get(experiment_id)
            beaker_logger = BeakerLogger(client, experiment, cfg.beaker_log_interval, experiment.description)
            beaker_logger.log_init()
        else:
            beaker_logger = None
    else:
        if cfg.beaker_log_interval > 0:
            logging.info(f"Beaker log interval set to {cfg.beaker_log_interval}, but beaker env variables are missing")
        beaker_logger = None

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    barrier()

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    # Display and save configuration.
    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)

        if cfg.allow_resume:
            config_path = Path(cfg.save_folder) / "config.yaml"
            if config_path.exists():
                lastest_checkpoint = Path(cfg.save_folder) / "latest"
                if lastest_checkpoint.exists():
                    logging.info(f"Resuming from {lastest_checkpoint}")
                    saved_config = TrainConfig.load(config_path)
                    if saved_config.model != cfg.model:
                        logging.warning("Model config does not match the one resuming from")
                    if saved_config.optimizer != cfg.optimizer:
                        logging.warning("Optimizer config does not match the one resuming from")
                    if saved_config.data != cfg.data:
                        logging.warning("Data config does not match the one resuming from")
                    cfg.load_path = str(lastest_checkpoint)
                else:
                    logging.info("Not resuming since no latest checkpoint found")

        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    barrier()

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_cfg = cfg.asdict(exclude=["wandb"])

        if "BEAKER_EXPERIMENT_ID" in os.environ:
            wandb_cfg["beaker_experiment_id"] = os.environ["BEAKER_EXPERIMENT_ID"]
            if beaker_logger is not None:
                wandb_cfg["beaker_url"] = beaker_logger.beaker.experiment.url(beaker_logger.experiment)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=wandb_cfg,
        )
        wandb_url = wandb.run.get_url()
        beaker_logger.add_wandb(wandb_url)  # add to beaker description

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = cfg.data.build_train_dataloader(cfg.model, cfg.global_train_batch_size, device)

    # Construct evaluators.
    if cfg.eval_interval > 0 or cfg.eval_on_load:
        evaluators = [v.build_dataset_evaluator(cfg.model, device) for v in cfg.evaluators]
    else:
        evaluators = None

    if cfg.inf_eval_interval > 0 or cfg.eval_on_load:
        inf_evaluators = [v.build_dataset_evaluator(cfg.model, device) for v in cfg.inf_evaluators]
    else:
        inf_evaluators = None
    barrier()

    # Initialize the model.
    logging.info(f"Building model")
    olmo_model = Molmo(cfg.model)

    # Freeze model components.
    if cfg.model.vision_backbone is not None and not cfg.ft_connector:
        freeze_parameters_by_name(olmo_model, Molmo.get_connector_parameters(), warn=False)
    if cfg.model.vision_backbone is not None and not cfg.ft_vit:
        log.info(f"Freezing vision backbone")
        freeze_parameters_by_name(olmo_model, Molmo.get_vit_parameters(), warn=False)
    if not cfg.ft_llm:
        log.info(f"Freezing LLM")
        freeze_parameters_by_name(olmo_model, Molmo.get_llm_parameters(), warn=False)
    if cfg.ft_embedding != "all":
        if cfg.ft_embedding == "ln_f":
            log.info(f"Freezing LLM: wte.embedding, ff_out")
            freeze_names = ["transformer.wte.embedding", "transformer.wte.weight"]
            freeze_names += ["transformer.ff_out"]
        elif cfg.ft_embedding == "lm_head":
            log.info(f"Freezing LLM: wte.embedding")
            freeze_names = ["transformer.wte.embedding", "transformer.wte.weight"]
        else:
            assert cfg.ft_embedding == "wte"
            log.info(f"Freezing LLM: ln_f, ff_out")
            freeze_names = ["transformer.ln_f", "transformer.ff_out"]
        freeze_parameters_by_name(olmo_model, tuple(freeze_names), warn=False)

    if cfg.compile is not None:
        log.info(f"Compiling {cfg.compile.target}...")
        if cfg.compile.target == "model":
            torch.compile(olmo_model, **cfg.compile.compile_args())
        elif cfg.compile.target == "blocks":
            for block_idx, block in enumerate(olmo_model.transformer.blocks):
                block.compile_vit(**cfg.compile.compile_args())
            for block_idx, block in enumerate(olmo_model.vision_backbone.image_vit.transformer.resblocks):
                block.compile_vit(**cfg.compile.compile_args())
        elif cfg.compile.target == "image_blocks":
            for block_idx, block in enumerate(olmo_model.vision_backbone.image_vit.transformer.resblocks):
                block.compile_vit(**cfg.compile.compile_args())
        elif cfg.compile.target == "llm_blocks":
            for block_idx, block in enumerate(olmo_model.transformer.blocks):
                block.compile_vit(**cfg.compile.compile_args())
        elif cfg.compile.target == "transformers":
            olmo_model.transformer.compile(**cfg.compile.compile_args())
            olmo_model.vision_backbone.image_vit.transformer.compile_vit(**cfg.compile.compile_args())
        else:
            raise NotImplementedError(cfg.compile.target)

    listdir(cfg.save_folder)

    sync_module_states = True
    if cfg.load_path is None:
        # Sine we typically load some parameters from a pre-trained checkpoint, we init the rank0
        # model on the cpu and then use `sync_module_states` in FSDP to sync the parameters
        # with the rest of the devices
        init_weights = False
        if get_local_rank() == 0:
            if cfg.initial_model_checkpoint:
                logging.warning(f"Loading model checkpoint {cfg.initial_model_checkpoint}")
                state_dict = torch.load(join(cfg.initial_model_checkpoint, "model.pt"), map_location="cpu")
                olmo_model.load_state_dict(state_dict)
                del state_dict
            else:
                olmo_model.reset_with_pretrained_weights()
    else:
        init_weights = True

    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    if olmo_model.config.llm.block_type == "moe":
        log.info(f"Number of active parameters: {olmo_model.num_params(include_inactive_params=False):,d}")    
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

    if init_weights or version.parse(torch.__version__) >= version.parse("2.1.0"):
        # Model is already initialized, so give FSDP a do-nothing init function
        # so it doesn't re-initialize the parameters
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_default_device(), recurse=False)

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
    device_mesh = None
    hybrid_sharding_fsdp_kwargs = {}
    if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        if version.parse(torch.__version__) < version.parse("2.2.0"):
            # Device mesh was not added to PyTorch until v2.2.0
            raise OLMoConfigurationError(
                "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
            )

        from torch.distributed.device_mesh import init_device_mesh

        num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
            get_world_size() // get_local_world_size()
        )

        if num_model_replicas <= 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

        if get_world_size() % num_model_replicas != 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")

        device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
        hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=cfg.fsdp_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        sync_module_states=sync_module_states,
        param_init_fn=param_init_fn,
        **hybrid_sharding_fsdp_kwargs,
    )

    # This can prevent OOMs if loading a LLM checkpoint, presumably due to
    # reducing memory fragmentation
    torch.cuda.empty_cache()
    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = cfg.optimizer.build_optimizer(cfg.max_grad_norm, cfg.max_grad_norm_ratio, fsdp_model)
    scheduler = cfg.scheduler.build()
    checkpointer = cfg.checkpointer_config.build()

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=fsdp_model,
        checkpointer=checkpointer,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        inference_evaluators=inf_evaluators,
        beaker_logger=beaker_logger
    ) as trainer:
        if not cfg.dry_run and not cfg.no_pre_train_checkpoint and cfg.load_path is None:
            # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
            log.info("Saving pre-train checkpoint...")
            checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
            log.info(f"Checkpoint saved to {checkpoint_path}")

        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            trainer.restore_checkpoint(
                cfg.load_path,
                load_optimizer_state=not cfg.reset_optimizer_state,
                load_trainer_state=not cfg.reset_trainer_state,
            )
            log.info("Checkpoint successfully loaded")

            # If we have to, set a new scheduler:
            if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
                trainer.scheduler = BoltOnWarmupScheduler.wrap(
                    trainer.scheduler,
                    trainer.global_step,
                    int(trainer.global_step + cfg.scheduler.t_warmup),
                )

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")
