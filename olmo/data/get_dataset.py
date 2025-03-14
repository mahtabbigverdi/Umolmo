from olmo.data.academic_datasets import (
    ScienceQAImageOnly, TextVqa, OkVqa,
    AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, TabWMPDirectAnswer,
    AndroidControl, TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench
)
from olmo.data.academic_datasets_manual import ChartQa, InfoQa, SceneTextQa, DocQa
from olmo.data.video_datasets import (
    InternVid, Koala, LLaVAVideo178K, MVBench, TempCompass
)
from olmo.data.dataset import Dataset
from olmo.data.pixmo_datasets import (
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations,
    PixMoAskModelAnything, PixMoPointsEval, DenseCaptionEval, PixMoClocks,
    CoSyn, CoSynPoint
)
import itertools


def get_dataset_by_name(dataset_name, split) -> Dataset:
    if dataset_name == "intern_vid":
        return InternVid(split=split)
    if dataset_name == "koala":
        return Koala(split=split)
    if dataset_name == "llava_video_178k_mc":
        return LLaVAVideo178K(split=split, answer_type="multi_choice")
    if dataset_name == "llava_video_178k_oe":
        return LLaVAVideo178K(split=split, answer_type="open_ended")
    if dataset_name == "llava_video_178k_cap":
        return LLaVAVideo178K(split=split, answer_type="caption")
    if dataset_name == "mvbench":
        return MVBench(split=split)
    if dataset_name.startswith("temp_compass"):
        dataset_name = dataset_name.replace("_disable_api", "")
        task = '_'.join(dataset_name.split("_")[2:]) if len(dataset_name.split("_")) > 2 else "all"
        return TempCompass(split=split, task=task)
    if dataset_name in ["scifi_document_qa", "pixmo_docs_other"]:
        return PixMoDocs("other", split=split)
    elif dataset_name in ["scifi_table_qa", "pixmo_docs_tables"]:
        return PixMoDocs("tables", split=split)
    elif dataset_name in ["scifi_diagram_qa", "pixmo_docs_diagrams"]:
        return PixMoDocs("diagrams", split=split)
    elif dataset_name in ["scifi_charts_qa", "pixmo_docs_charts"]:
        return PixMoDocs("charts", split=split)

    elif dataset_name in ["pixmo_docs_other_flat"]:
        return PixMoDocs("other", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_charts_flat"]:
        return PixMoDocs("charts", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_tables_flat"]:
        return PixMoDocs("tables", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_diagrams_flat"]:
        return PixMoDocs("diagrams", split=split, flat=True)

    # CoSyn-400K / CoSyn-point
    doc_types = [
        "chart", "chemical", "circuit", "diagram",
        "document", "graphic", "math", "music",
        "nutrition", "table"
    ]
    cosyn_dataset_names = [f"cosyn_{doc_type}{suffix}" for doc_type, suffix in itertools.product(doc_types, ["", "_exp"])]
    if dataset_name == "cosyn_point":
        return CoSynPoint(split=split)
    elif dataset_name in cosyn_dataset_names:
        doc_type = dataset_name.split("_")[1]
        return CoSyn(doc_type, split=split, use_exp=dataset_name.endswith("_exp"))

    # PixMo-Pointing
    elif dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=False)
    elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=True)
    elif dataset_name in ["pointing", "pixmo_points"]:
        return PixMoPoints(kind="basic", split=split, counting=False)
    elif dataset_name in ["point_count", "pixmo_points_counting"]:
        return PixMoPoints(kind="basic", split=split, counting=True)

    # PixMo-Point-Explanations
    elif dataset_name in ["point_qa", "pixmo_pointing_explanations"]:
        return PixMoPointExplanations(split=split, split_groups=True)

    # PixMo-Count
    elif dataset_name in ["fast_flickr_count_qa_point_count", "pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["fast_flickr_count_qa_pointing", "pixmo_count"]:
        return PixMoCount(split=split, counting=False)

    # PixMo-AskModelAnything
    elif dataset_name in ["user_qa", "pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)

    # PixMo-CapQa
    elif dataset_name in ["synthetic_qa_v3", "pixmo_cap_qa"]:
        return PixMoCapQa(split=split)
    elif dataset_name in ["synthetic_qa_v3_as_user_qa", "pixmo_cap_qa_as_user_qa"]:
        return PixMoCapQa(split=split, style="user_qa")

    # PixMo-Cap
    if dataset_name in ["cockatoo_and_transcript_712k_sept6", "pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
        return PixMoCap(split, mode="captions")
    if dataset_name in ["pixmo_cap_transcript", "pixmo_transcript"]:
        return PixMoCap(split, mode="transcript")
    # if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
    #     return PixMoCap(split, mode="captions")
    # if dataset_name in ["pixmo_transcript"]:
    #     return PixMoCap(split, mode="transcript")

    elif dataset_name in ["pixmo_clocks"]:
        return PixMoClocks(split=split)

    if dataset_name == "pointing_eval":
        assert split == "test"
        return PixMoPointsEval()

    # Academic datasets
    if dataset_name == "android_control":
        return AndroidControl(split)
    if dataset_name == "android_control_ll":
        return AndroidControl(split, mode="ll")
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "chart_qa_exp":
        return ChartQa(split, weighted=False, use_exp=True)
    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split, multi_question=True)
    if dataset_name == "coco_2014_vqa":
        return Vqa2(split, multi_question=False)
    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "plot_qa":
        return PlotQa(split, in_memory=False)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split, in_memory=False)
    if dataset_name == "okvqa":
        return OkVqa(split)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "tally_qa":
        return TallyQa(split=split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "clock_bench":
        return ClockBench(split=split)
    if dataset_name == "dense_caption_eval":
        assert split == "test"
        return DenseCaptionEval()
    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    raise NotImplementedError(dataset_name, split)