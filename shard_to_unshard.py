from olmo.train.distributed_checkpointing import unshard_checkpoint, UnshardStrategy
# Basic usage: produce a single model.pt file without optimizer state.
model_path, _ = unshard_checkpoint(
    "/mmfs1/gscratch/krishna/mahtab/Umolmo/step30000/model_and_optim",  # directory containing the sharded checkpoint
    "/mmfs1/gscratch/krishna/mahtab/Umolmo/pretrained/3B-step30000-unsharded",    # target directory to write unsharded files
    optim=False,                    # whether to unshard optimizer state
    use_safetensors=False,          # whether to save using safetensors format
)
print(f"Unsharded model saved to {model_path}")

