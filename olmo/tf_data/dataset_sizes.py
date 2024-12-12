DATASET_SIZES = {
    ("cockatoo_qa_v2", "train"): 194820,
    ("user_qa", "train"): 71172,

    ("text_vqa", "train"): 34602,
    ("chart_qa", "train"): 28299,
    ("chart_qa_prompting", "train"): 28299,
    ("chart_qa_weighted", "train"): 28299,
    ("tally_qa", "train"): 132981,
    ("doc_qa", "train"): 39463,
    ("info_qa", "train"): 23946,
    ("okvqa", "train"): 9009,
    ("gqa", "train"): 943000,
    ("gqa_multi", "train"): 72140,
    ("coco_2014_vqa", "train"): 443757,  # (82783, 443757)
    ("coco_captioning_karpathy", "train"): 414113,  # (82783, 414113)
    ("coco_captioning_karpathy_multi", "train"): 82783,
    ("coco_2014_vqa_multi", "train"): 82783,
    ("science_qa_img", "train"): 6218,
    ("ai2_diagram", "train"): 11389,
    ("a_okvqa_mc", "train"): 17056,
    ("a_okvqa_da", "train"): 17056,
    ("ocr_vqa", "train"): 166043,
    ("st_qa", "train"): 25050,
    ("ocr_qa", "train"): 166043,

    ("dv_qa", "train"): 200000,
    ("tabwmp_da", "train"): 23059,
    ("figure_qa", "train"): 100000,
    ("figure_qa_zero_shot", "train"): 100000,
    ("plot_qa", "train"): 157070,
    ('clocks', 'train'): 800269,
    ('clocks', 'validation'): 25600,

    ("st_qa", "test"): 4070,
    ('text_vqa', "test"): 5734,
    ('okvqa', "test"): 5046,
    ('chart_qa', "test"): 1250,
    ('doc_qa', "test"): 5188,
    ('info_qa', "test"): 3288,
    ('gqa', "test"): 95336,
    ('coco_captioning_karpathy', "test"): 25010,
    ("science_qa_img", "test"): 2017,
    ("ai2_diagram", "test"): 3088,
    ("a_okvqa_mc_eval", "test"): 6702,
    ("a_okvqa_da_eval", "test"): 6109,

    ("ai2_diagram_v2", "train"): 10950,
    ("ai2_diagram_v2", "validation"): 1463,
    ("ai2_diagram_v2", "test"): 3088,
    ("vqa_v2_test", "test2015"): 555187,

    ("ai2_diagram_v2_transparent", "train"): 10950,
    ("ai2_diagram_v2_transparent", "validation"): 1463,
    ("ai2_diagram_v2_transparent", "test"): 3088,

    # splits in mix_data include both transparent + opaque boxes
    ("ai2_diagram_v2_mix_transparent", "train"): 15042,
    ("ai2_diagram_v2_mix_transparent", "validation"): 1980,
    ("ai2_diagram_v2_mix_transparent", "test"): 4272,

    # vaia_qa
    ('vaia_qa', 'train'): 477052,
    ('vaia_qa', 'validation'): 1024,

    ('vaia_qa_latex_image', 'train'): 477052,
    ('vaia_qa_latex_image', 'validation'): 1024,
    ('vaia_qa_latex_image_only', 'train'): 42605,
    ('vaia_qa_latex_image_only', 'validation'): 1024,
    ('vaia_qa_latex_all_image_only', 'train'): 154266,
    ('vaia_qa_latex_all_image_only', 'validation'): 1024,

    ("vaia_qa_latex_image_math_subset_short_answer", 'train'): 198161,
    ("vaia_qa_latex_image_math_subset_short_answer", 'validation'): 419,
    ("vaia_qa_latex_image_math_subset_mc_only_short_answer", "train"): 57568,
    ("vaia_qa_latex_image_math_subset_mc_only_short_answer", "validation"): 118,
    ("vaia_qa_latex_image_math_subset_mc_only_short_answer_first", "train"): 57568,
    ("vaia_qa_latex_image_math_subset_mc_only_short_answer_first", "validation"): 118,
    
    ("vaia_qa_latex_image_all_image_only_short_answer", "train"): 86752,
    ("vaia_qa_latex_image_all_image_only_short_answer", "validation"): 92,
    ("vaia_qa_latex_image_all_image_only_short_answer_first", "train"): 86752,
    ("vaia_qa_latex_image_all_image_only_short_answer_first", "validation"): 92,
    ("vaia_qa_latex_image_math_subset_image_only_short_answer", "train"): 21726,
    ("vaia_qa_latex_image_math_subset_image_only_short_answer", "validation"): 48,

    ('vqa_online', 'train'): 62722,
    ('vqa_online', 'validation'): 1024,
    ('vqa_online', 'test'): 1024,

    ('vqa_online_gpt_longQ_longA', 'train'): 62722,
    ('vqa_online_gpt_longQ_longA', 'validation'): 1024,
    ('vqa_online_gpt_longQ_longA', 'test'): 1024,

    ("tally_qa", "validation"): 38589,
    ('text_vqa', "validation"): 5000,
    ('okvqa', "validation"): 5046,
    ('chart_qa', "validation"): 960*2,
    ('chart_qa_prompting_explanation', "validation"): 960*2,
    ('chart_qa_ex', "validation"): 960*2,
    ('chart_qa_human', "validation"): 960,
    ('chart_qa_aug', "validation"): 960,
    ('doc_qa', "validation"): 5349,
    ('info_qa', "validation"): 2801,
    ('coco_2014_vqa', "validation"): 214354,  # 40504 images
    ('coco_2014_vqa_multi', "validation"): 214354,
    ('coco_captioning_karpathy', "validation"): 25010,
    ('gqa', "validation"): 132062,
    ("science_qa_img", "validation"): 2097,
    ("ai2_diagram", "validation"): 1024,
    ("a_okvqa_mc", "validation"): 1145,
    ("a_okvqa_da", "validation"): 1075,
    ("charxiv_descriptive", "validation"): 1000,
    ("charxiv_descriptive", "test"): 1320,
    ("charxiv_reasoning", "validation"): 1000,
    ("charxiv_reasoning", "test"): 1320,
    ("fintabnetqa", "validation"): 125,
    ("fintabnetqa", "test"): 250,
    ("vwtq", "validation"): 125,
    ("vwtq", "test"): 750,
    ("vwtq_syn", "validation"): 125,
    ("vwtq_syn", "test"): 250,
    ("vtabfact", "validation"): 125,
    ("vtabfact", "test"): 250,
    ("nutrition_fact", "validation"): 100,
    ("nutrition_fact", "test"): 100,

    ("mmmu_test", "validation"): 900,
    ("count_bench", "test"): 500,
    ("mmmu_test", "test"): 10500,
    ("real_world_qa_test", "test"): 765,
    ("real_world_qa_no_instruction", "test"): 765,
    ("real_world_qa_dbg", "test"): 765,
    ("real_world_qa_as_user_qa", "test"): 765,

    ("seed_bench_test", "test"): 19241,
    ("pope_test", "test"): 9000,
    ("mme_test", "test"): 2374,
    ("math_vista_test", "validation"): 1000,
    ("math_vista_demo", "validation"): 1000,
    ("math_vista_v2", "validation"): 1000,

    ("math_vista_test", "test"): 5141,
    ("mmbench_test", "validation"): 4329,
    ("mmbench_test", "test"): 6666,
    ("sugar_crepe_test", "test"): 15022,
    ("blink_test", "validation"): 1901,
    ("dense_caption_eval_dbg", "validation"): 1,

    ("refclef_unc", "train"): 17978,
    ("refclef_unc", "validation"): 12029,
    ("refcoco_unc", "train"): 16994,
    ("refcoco_unc", "validation"): 10834,
    ("refcocoplus_unc", "train"): 16992,
    ("refcocoplus_unc", "validation"): 10758,
    ("refcocog_umd", "train"): 21899,
    ("refcocog_umd", "validation"): 4896,
    ("refclef_unc", "testA"): 3449,
    ("refclef_unc", "testB"): 3221,
    ("refclef_unc", "testC"): 2664,
    ("refclef_unc", "testAB"): 116,
    ("refclef_unc", "testBC"): 86,
    ("refcoco_unc", "testA"): 5657,
    ("refcoco_unc", "testB"): 5095,
    ("refcocoplus_unc", "testA"): 5726,
    ("refcocoplus_unc", "testB"): 4889,
    ("refcocog_umd", "test"): 9602,
    ("countbench_qa_point_count", "huggingface"): 490,
    ("countbench_qa_pointing", "huggingface"): 490,
    ("countbench_qa_count_then_point", "huggingface"): 490,
    ("countbench_qa_only_count", "huggingface"): 490,
    ('countbench_qa', 'huggingface'): 490,

    ('cockatoo_712k_sept6', 'train'): 712121,
    ('cockatoo_712k_sept6', 'validation'): 5120,
    ('user_qa', 'train'): 71172,
    ('user_qa', 'validation'): 2048,

    # pointing
    ("pointing_test", "test"): 436,

    ("fast_flickr_count_qa_point_count", "train"): 36916,
    ("fast_flickr_count_qa_point_count", "validation"): 163,
    ("fast_flickr_count_qa_point_count", "test"): 540,
    ("fast_flickr_count_qa_point_count_random_points", "train"): 36916,
    ("fast_flickr_count_qa_point_count_random_points_and_length", "train"): 36916,
    ("fast_flickr_count_qa_point_count_random_order", "train"): 36916,
    ("fast_flickr_count_qa_pointing", "train"): 36916,
    ("fast_flickr_count_qa_pointing", "validation"): 163,
    ("fast_flickr_count_qa_pointing", "test"): 540,
    ("fast_flickr_count_qa_count_then_point", "train"): 36916,
    ("fast_flickr_count_qa_count_then_point", "validation"): 163,
    ("fast_flickr_count_qa_count_then_point", "test"): 540,
    ("fast_flickr_count_qa_only_count", "train"): 36916,
    ("fast_flickr_count_qa_only_count", "validation"): 163,
    ("fast_flickr_count_qa_only_count", "test"): 540,
    ('point_count', 'train'): 309216,
    ('point_count', 'validation'): 2054,
    ('pointing', 'train'): 309216,
    ('pointing', 'validation'): 2054,
    ("only_count", "train"): 309216,
    ("only_count", "validation"): 2054,
    ("count_then_point", "train"): 309216,
    ("count_then_point", "validation"): 2054,
    ('point_count_high_freq', 'train'): 113840,
    ('point_count_high_freq', 'validation'): 3969,
    ('pointing_high_freq', 'train'): 113840,
    ('pointing_high_freq', 'validation'): 3969,
    ("only_count_high_freq", "train"): 113840,
    ("only_count_high_freq", "validation"): 3969,
    ("count_then_point_high_freq", "train"): 113840,
    ("count_then_point_high_freq", "validation"): 3969,
    ("point_count_random_points", "train"): 309216,
    ("point_count_random_points_and_length", "train"): 309216,
    ("point_count_random_order", "train"): 309216,
    ("point_count_high_freq_random_points", "train"): 113840,
    ("point_count_high_freq_random_points_and_length", "train"): 113840,
    ("point_count_high_freq_random_order", "train"): 113840,
    ('point_qa', 'train'): 27856,
    ('point_qa', 'validation'): 978,
    ("a_okvqa_da", "test"): 6109,
    ("a_okvqa_mc", "test"): 6702,
    ("user_questions_for_elo", "test"): 14851,
    ("user_questions_for_elo_long", "test"): 1368,
    ("user_questions_for_elo_9_to_12", "test"): 3000,

    ("sim_point_count_qa", "train"): 522611,
    ("sim_point_count_qa", "validation"): 800,
    ("sim_point_count_qa", "test"): 800,
    ("sim_count_qa", "train"): 522611,
    ("sim_count_qa", "validation"): 800,
    ("sim_count_qa", "test"): 800,

    ("scifi_charts_qa", "validation"): 1024,
    ("scifi_table_qa", "validation"): 1024,
    ("scifi_natural_qa", "validation"): 128,
    ("scifi_nutrition_qa", "validation"): 128,
    ("scifi_document_qa", "validation"): 1024,
    ("scifi_diagram_qa", "validation"): 1024,
    ("scifi_charts_qa", "train"): 233622,
    ("scifi_table_qa", "train"): 93036,
    ("scifi_document_qa", "train"): 142559,
    ("scifi_diagram_qa", "train"): 33102,

    ("scifi_charts_qa_split", "train"): 116814,
    ("scifi_table_qa_split", "train"): 46518,
    ("scifi_document_qa_split", "train"): 71282,
    ("scifi_diagram_qa_split", "train"): 16551,

    ("scifi_charts_qa_exp_split", "train"): 116814,
    ("scifi_table_qa_exp_split", "train"): 46518,
    ("scifi_document_qa_exp_split", "train"): 71282,
    ("scifi_diagram_qa_exp_split", "train"): 16551,

    ("android_control", "train"): 74714,
    ("android_control", "validation"): 690,
    ("android_control", "test"): 3897,

    ("synthetic_qa_v3_multi_turn", "train"): 9824,
    ("synthetic_qa_v3", "train"): 162855,
    ("synthetic_qa_v3_style_tag", "train"): 162855,
    ("synthetic_qa_v3_as_user_qa", "train"): 162855,
}


for (name, split), count in list(DATASET_SIZES.items()):
    if name in ["chart_qa"]:
        DATASET_SIZES[(name + "_scifi", split)] = count
    if name in ["android_control"]:
        for k in ["ll", "hl", "hl_ll", "hl_cot"]:
            DATASET_SIZES[(f"{name}_{k}", split)] = count
    if name in ["scifi_charts_qa" ,"scifi_table_qa", "scifi_document_qa", "scifi_diagram_qa", "scifi_datikz_qa"]:
        DATASET_SIZES[(name + "_exp", split)] = count
        DATASET_SIZES[(name[:-3] + "_exp", split)] = count
        DATASET_SIZES[(name[:-3] + "_demo", split)] = count
    if name in ["ai2_diagram_v2_mix_transparent"]:
        DATASET_SIZES[("ai2_diagram_v2_mix_transparent_one_style", split)] = count
    if name in ["chart_qa", "info_qa", "doc_qa", "text_vqa", "coco_2014_vqa",
                "ai2_diagram_v2_mix_transparent", "countbench_qa", "chart_qa_human"]:
        DATASET_SIZES[(name + "_demo", split)] = count


def get_dataset_size(name, split):
    if name.endswith("_eval"):
        if (name, split) in DATASET_SIZES:
            return DATASET_SIZES[(name, split)]
        name = name[:-len('_eval')]
    return DATASET_SIZES[(name, split)]
