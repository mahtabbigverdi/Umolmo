import logging
from typing import Dict

from olmo.models.molmo.data_formatter import DataFormatter
from olmo.data.data_loader import DataConfig
from olmo.models.molmo.model_preprocessor import MultiModalPreprocessorConfig
from olmo.eval.inf_evaluator import InfDatasetEvaluatorConfig, EvaluatorConfig
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig, AttentionType, LayerNormType
from olmo.models.molmo.molmo import MolmoConfig
from olmo.tokenizer import TokenizerConfig
from olmo.nn.vision_backbone import VisionBackboneConfig

log = logging.getLogger(__name__)


DEBUG_MODEL = MolmoConfig(
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
        vit=VitConfig(image_num_layers=1)
    ),
    data_formatter=DataFormatter(),
    mm_preprocessor=MultiModalPreprocessorConfig(crop_mode="resize", max_crops=1)
)


def get_evaluator(name) -> EvaluatorConfig:
    """Gets the default evaluator for task `name`"""

    if name in ["text_vqa", "okvqa", "coco_2014_vqa", "coco_2014_vqa_multi"]:
        return EvaluatorConfig(vqa_eval="vqa_score")
    elif name.startswith("math_vista"):
        return EvaluatorConfig(math_vista_eval=True)
    elif name == "a_okvqa_da":
        return EvaluatorConfig(vqa_eval="a_okvqa_score")
    elif name.startswith("android_control"):
        return EvaluatorConfig(android_eval=True)
    elif name == "vqa_v2_test":
        return EvaluatorConfig()
    elif name.startswith("chart_qa"):
        return EvaluatorConfig(vqa_eval="relaxed_correctness,em")
    elif name in ["doc_qa", "info_qa", "st_qa"]:
        return EvaluatorConfig(vqa_eval="ansl,em")
    elif name in ["gqa", "tally_qa"]:
        return EvaluatorConfig(vqa_eval="em")
    elif name in ["science_qa", "a_okvqa_mc", "science_qa_img", "ai2_diagram", "ai2_diagram_v2", "ai2_diagram_v2_transparent"]:
        return EvaluatorConfig(vqa_eval="mc")
    elif name in ["ai2_diagram_v2_mix_transparent", "ai2_diagram_v2_mix_transparent_one_style"]:
        return EvaluatorConfig(vqa_eval="mc_ai2d_opaque,mc_ai2d_transparent")
    elif name.startswith("mmmu"):
        return EvaluatorConfig(vqa_eval="mmmu_score")
    elif name.startswith("countbench_qa") or name.startswith("pixmo_count"):
        return EvaluatorConfig(point_count_eval=True)
    elif name.startswith("real_world_qa"):
        return EvaluatorConfig(vqa_eval="real_world_qa_score")
    elif name == "pixmo_clocks":
        return EvaluatorConfig(clock_eval=True)
    elif name == "pointing_eval":
        return EvaluatorConfig(pointing_eval=True)
    elif name == "clock_bench":
        return EvaluatorConfig(clock_bench_eval=True)
    elif name in ["countbench_qa"]:
        return EvaluatorConfig(count_eval=True)
    elif name in ["dense_caption_eval", "user_qa", "vqa_v2_test"]:
        # No metrics, but still save prediction file
        return EvaluatorConfig()
    else:
        raise NotImplementedError(name)


def get_evaluation(name, seq_len, max_examples, num_workers=2, device_batch_size=None) -> InfDatasetEvaluatorConfig:
    """Gets the default evaluation config for task (or task:split string) `name`"""
    if ":" in name:
        name, split = name.split(":")
    else:
        split = None

    if name == "chart_qa_weighted":
        name = "chart_qa"
    if name == "coco_2014_vqa_multi":
        name = "coco_2014_vqa"

    evaluator = get_evaluator(name)
    evaluator.num_wandb_examples = 64

    eval_only_tasks = ["mmmu", "mme", "math_vista", "real_world_qa", "seed_bench",
                       "mmbench", "sugar_crepe", "blink"]
    eval_only_tasks += [task_name + "_test" for task_name in eval_only_tasks]
    if name == "tall_qa_count":
        task_name = "tally_qa"
    elif name in eval_only_tasks:
        task_name = name + "_test" if not name.endswith("_test") else name
    else:
        task_name = name
    evaluator.num_wandb_examples = 32
    evaluator.n_to_log = 0
    evaluator.save_predictions = None
    test_eval_tasks = ["mme_test", "real_world_qa_test", "real_world_qa_test", "count_bench",
                       "seed_bench_test", "sugar_crepe_test", "count_bench_from_caption", "pointing_test"]
    if split is None:
        split = "test" if task_name in test_eval_tasks else "validation"

    if name.startswith("named_entity"):
        max_new_tokens = 256
    elif name == "math_vista_demo":
        max_new_tokens = 384
    elif name in ["chart_qa_scifi", "chart_qa_ex", "chart_qa_prompting_explanation"] or name.endswith("_demo"):
        max_new_tokens = 256
    elif name.startswith("user_questions_for_elo"):
        max_new_tokens = 768  # Can have counts of 20+ so make sure there is room
    elif name in ["pointing_eval", "pointing"]:
        max_new_tokens = 192  # 192 is enought for counts <=10 in the point tag format
    elif "countbench_qa" in name or "pixmo_count" in name:
        max_new_tokens = 192
    elif name == "android_control_hl_cot":
        max_new_tokens = 64
    elif name.startswith("android_control"):
        max_new_tokens = 16
    elif "refc" in name:
        max_new_tokens = 32
    else:
        max_new_tokens = 12

    ds = DataConfig(
        dataset=task_name, sequence_length=seq_len, split=split, shuffle=True,
        num_workers=num_workers, pad="to_max", pin_memory=True, drop_last=False,
        seed=691203
    )

    return InfDatasetEvaluatorConfig(
        max_examples=max_examples,
        device_batch_size=device_batch_size,
        max_new_tokens=max_new_tokens,
        evaluator=evaluator,
        label="ai2_diagram" if "ai2_diagram" in name else name,
        data=ds,
        console_log_interval="${console_log_interval}"  # Use log interval in top-level config
    )


DEFAULT_VISION_BACKBONE = VitConfig(
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


SIGLIP_VISION_BACKBONE = VitConfig(
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=27,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=729, # no CLS token
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
)


DINOV2_LARGE_336_VISION_BACKBONE = VitConfig(
    image_model_type="dino",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
)


METACLIP_L14_336_VISION_BACKBONE = VitConfig(
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


OLMOE = LlmConfig(
    d_model=2048,
    n_heads=16,
    n_layers=16,
    mlp_ratio=1,
    activation_type='swiglu',
    block_type='moe',
    rope=True,
    rope_full_precision=True,
    rope_theta=10000.0,
    attention_type='sdpa',
    attention_layer_norm=True,
    residual_dropout=0.1,
    response_residual_dropout=0.0,
    embedding_dropout=0.0,
    layer_norm_type='rms',
    layer_norm_with_affine=True,
    layer_norm_eps=1e-05,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    max_position_embeddings=32768,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=50280,
    embedding_size=50304,
    additional_vocab_size=128,
    new_embedding_init_range=0.02,
    weight_tying=False,
    normalize_input_embeds=False,
    use_position_ids=True,

    # MOE parameters
    moe_num_experts=64,
    moe_top_k=8,
    moe_mlp_impl='sparse',
    moe_log_expert_assignment=False,
    moe_shared_expert=False,
    moe_lbl_in_fp32=False,
    moe_interleave=False,
    moe_loss_weight=0.0,
    moe_zloss_weight=0.0,
    moe_dropless=True,
    moe_capacity_factor=1.25,

    tokenizer=TokenizerConfig(
        identifier='allenai/OLMoE-1B-7B-0924',
    ),
)


OLMO_1024_PREVIEW = LlmConfig(
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/dolma2-tokenizer",
    ),
    embedding_dropout=0,
)


QWEN2_7B = LlmConfig(
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
    ),
)

QWEN25_15B = LlmConfig(
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=1536,
    mlp_hidden_size=8960*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=12,
    n_kv_heads=2,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-1.5B",
    ),
)


QWEN25_3B = LlmConfig(
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2048,
    mlp_hidden_size=11008*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=16,
    n_kv_heads=2,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-3B",
    ),
)


QWEN2_72B = LlmConfig(
    additional_vocab_size=128,
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=8192,
    mlp_hidden_size=29568*2,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-72B",
    ),
)


DEFAULT_LOAD_PATHS = {
    "openai": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    "siglip": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    "dinov2_large_336": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    "metaclip_l14_336": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    "olmoe": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmoe.pt",
    "olmo_1024_preview": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo-1024-preview.pt",
    "qwen2.5_1.5b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-1.5b.pt",
    "qwen2.5_3b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-3b.pt",
    "qwen2_7b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-7b.pt",
    "qwen2_72b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-70b.pt",
}


VISION_BACKBONES: Dict[str, VitConfig] = {
    "openai": DEFAULT_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_LARGE_336_VISION_BACKBONE,
    "metaclip_l14_336": METACLIP_L14_336_VISION_BACKBONE,
}


LLMS: Dict[str, LlmConfig] = {
    "olmoe": OLMOE,
    "olmo_1024_preview": OLMO_1024_PREVIEW,
    "qwen2_7b": QWEN2_7B,
    "qwen2_72b": QWEN2_72B,
    "qwen2.5_3b": QWEN25_3B,
    "qwen2.5_1.5b": QWEN25_15B,
}


