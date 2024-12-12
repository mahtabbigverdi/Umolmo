import re

import gin
import tensorflow as tf

IMAGE_PROMPT = "<|image|>"


GENERAL_PROMPTS_V1 = {
    "short_answer": [
        "Answer this question very briefly\n{question}",
        "{question} Answer with a few words",
        "{question} Response very briefly",
        "{question} Answer directly without any details, explanation, or elaboration",
        "I have a question about this image, please answer it very briefly: {question}",
        "Question: {question} Short Answer:",
        "Question: {question}\nShort Answer:",
        '{question}\nAnswer the question as briefly as possible.',
        'Answer very briefly:\n{question}',
        'The question "{question}" can be answered using the image. A short answer is',
        "{question} Based on the image, respond to this question with a short answer:",
        "{question} Short answer:",
        "{question} A short answer to the question is",
        "Give a short, matter-of-fact answer to this question: {question}",
        "Give me a simple, direct answer to this question, do not elaborate or explain your answer:\n{question}"
    ],
    "short_caption": [
        'Caption the image with 1 or two sentences',
        'Write a very short description of this image.',
        'Briefly describe the image.',
        'Look and this image, and then summarize it in a sentence or two.',
        'Write a brief caption describing the image',
        'Brief Caption:'
        'A short image caption:',
        'A short image description',
        'Briefly describe the content of the image.',
        'Can you give me one sentence summary of the picture?',
        'How would you describe this image in a sentence or two?',
    ],
    "long_caption": [
        'Describe this image.',
        'Describe this image',
        'describe the image',
        'Write a long description of this image.',
        'caption the picture',
        'Caption',
        'caption',
        'Construct a long caption for this image',
        'Generate a caption',
        'Create a detailed caption',
        'Write a long caption',
        'Describe this image in detail',
        'Describe this',
        'describe this',
        'Caption this',
        'What can be seen in this image?',
        'What do you see in the image?',
        'Look at this photo carefully and then tell me about it in detail',
        'Write a long description of this image',
        'Tell me about this picture.',
        'Write a paragraph about this image.',
        'Look at this image carefully and then describe it in detail',
        'Generate a long caption about this image.'
    ],
    "long_caption_no_pointing": [
        'Describe this image in detail, but without any pointing.',
        'Write a long description of this image, do not produce any points.',
        'Tell me about this picture, use plain text only.',
        'Generate a plain text description of this caption',
        "What is in this image?\nNo pointing\nGive lots of detail"
        "Write a long caption.\nDo not use image coordinates\nOutput a full paragraph"
    ],
    "transcript": [
        'Describe this image as if you are a person speaking',
        'Imagine you are a person talking about this image. Generate a transcript of what you would say.',
        "Generate an audio transcript of a person describing this image",
        "Create a transcript of a human describing this image out load",
        "Describe this in this style of a human talking",
    ],
    "refexp": [
        'What region does \"{refexp}\" refer to?',
    ],
    "count_bench": [
        'How many {object} are there?',
    ],
    "refexp_pointing": [
        'Where is the \"{refexp}\"?',
        'Point to {refexp}',
        'point at {refexp}',
        'Find the {refexp}.',
        'Which object in the image does \"{refexp}\" refer to?',
        'Locate the object \"{refexp}\" refers to.',
        'Point to the object that best matches the expression:\n{refexp}\n',
        'What object could be described as: {refexp}.\nPoint:',
        'Referring Expression: {refexp}.\nPoint:',
        'Expression: {refexp}\nPoint to the refexp',
        'Task: Point to the object that best matches the expression.\nExpression: {refexp}\nPoint:',
        'Instruction: Locate the object that matches the expression by returning a point.\nReferring Expression: {refexp}\n',
        'Help me find an object in this image by pointing to the {refexp}',
        'What point of the image might the expression \'{refexp}\' refer to?',
    ],
    "plain": ["{question}"],
    "multiple_choice": [
        "{question}\n{options}\nReturn only the letter of the best answer option",
        "Answer this question by naming one of the provided options:\n{question}\n{options}",
        "{question}\n{options}\nWhat option best answers the question?",
        "{question}\n{options}\nReturn the best answer option",
        "Look at the options, then return the letter of the option that best answers the question.\nQuesiton: {question}\nOptions: {options}",
        "{question}? Select an answer option from:\n{options}",
        "{question}\nSelect an answer option from:\n{options}\n\n",
        "Question: {question}? Options: {options} Answer:",
        "Answer the question by selecting an answer options\nQuestion: {question}\nOptions: {options}",
        "{question}?\n{options}\nReturn only the letter of the correct answer",
        "Help me answer this question: \"{question}\", by stating which of the following options is correct\n{options}."
    ],
    "binary": ["{question}\nAnswer with 'yes' or 'no'"],
    "pointing": [
        "Point to {entity}\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{entity}\"",
        "Point to any {entity} in the image",
        "Point to any {entity} in the image.",
        "Point: Where are the {entity}",
        "Show me where the {entity} are",
        "Can you show me where the {entity} are?",
        "Show me where the {entity} are",
        "Show me where a {entity} is",
        "Show me where a {entity} is.",
        "If there are any {entity} in the image? Show me where they are.",
        "Where are the {entity}?",
        "Generate a list of points showing where the {entity} are.",
        "Find the \"{entity}\".",
        "Find a \"{entity}\".",
        "Locate all {entity}.",
        "Locate an {entity}.",
        "Locate a {entity}.",
        "Locate every {entity}.",
        "Locate {entity}.",
        "Locate the {entity}.",
        "Object: {entity}\nInstruction: Point to the object.",
        "find {entity}",
        "find {entity}.",
        "Point to every {entity}",
        "find any {entity} in the picture",
        "Find the {entity}",
        "Find any {entity}",
        "Point to a {entity}",
        "Point to an {entity}",
        "Look for {entity} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {entity}.",
        "I am looking for {entity}, where can they be found in the image?",
        "Can you see any {entity} in the image? Point to them.",
        "Point out each {entity} in the image.",
        "Point out every {entity} in the image.",
        "Point to the {entity} in the image.",
        "Locate each {entity} in the image.",
        "Can you point out all {entity} in this image?",
        "Please find {entity} and show me where they are.",
        "If there are any {entity} present, indicate their positions.",
        "If there is a {entity} present, indicate its positions.",
        "show me all visible {entity}",
    ],
    "point_count": [
        "How many {entity} are there?",
        "How many {entity}?",
        "How many {entity}.",
        "how many {entity}.",
        "how many {entity}?",
        "How many {entity} are there in the image?",
        "Tell me how many {entity} there are",
        "Tell me how many {entity} there are and point to them.",
        "how many {entity}",
        "Tell me where each {entity} is.",
        "Tell me how many {entity} are in the image",
        "count {entity}",
        "count every {entity}",
        "count each {entity}",
        "count {entity}.",
        "Count the {entity}.",
        "How many {entity} do you see?",
        "How many {entity} are visible?",
        "Count all the {entity}",
        "how mmny {entity}?",
        "Count every {entity} in the picture.",
        "Count all the {entity}",
        "Count each {entity}",
        "Point to and count the {entity} in the picture.",
        "Point and count {entity}",
        "Point to every {entity}",
        "Locate the {entity} and count them",
        "Locate every {entity} and count them",
        "Find all the {entity}. How many are there?",
        "Find each {entity}. How many are there?",
        "Point at {entity} and then tell me the count.",
        "What is the total number of {entity} in the image?",
        "In all the picture, how many {entity} are there?",
        "Point at the {entity} and then count them.",
        "Point to all the visible {entity} output the total count.",
        "Point to all the {entity} visible and output the total count. \nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{entity}\" and output the total count.",
        "Show me where the {entity} are and output the total count.",
        "Where are the {entity}? How many are there?",
        "Generate list of points showing where the {entity} are and output the total count.",
        "Object: {entity}\nInstruction: Point to the object and output the total count.",
        "find any {entity} in the picture and output the total count.",
        "Can you see any {entity} in the image? Point to them and output the total count.",
        "Can you point out all {entity} in this image? How many are there?",
        "If there are any {entity} present, indicate their positions and output the total count.",
        "How many {entity} are there in the image? Point to them and output the total count.",
        "How many {entity} are there in the image?",
        "Give me the count of {entity} in the image.",
        "How many {entity} are visible in the image?",
        "How many {entity} are there?",
        "In the image, how many {entity} are there?",
        "Can you count the number of {entity} in the image?",
        "Can you count every {entity} in the picture?",
        "Can you see any {entity} in the image? How many are there?",
        "Are there any {entity} in the image? How many are there?",
        "If you see any {entity} in the image, give me the count. Otherwise, say 'This isn't in the image.'",
        "Object: {entity}\nInstruction: How many are there?",
    ],
    "count_then_point": [
        "Count the {entity} in the image, then point to them.",
        "How many {entity} are there? Point to them.",
        "Count every {entity} in the picture, then point to them.",
        "Locate the {entity} and count them, then point to them.",
        "Find all the {entity}. How many are there? Point to them.",
        "Find each {entity}. How many are there? Point to them.",
        "Point to and count the {entity} in the picture.",
    ],
    "only_count": [
        "Count the {entity} in the image.",
        "How many {entity} are there?",
        "Count every {entity} in the picture.",
        "Locate the {entity} and count them.",
        "Find all the {entity}. How many are there?",
        "Find each {entity}. How many are there?",
    ],
    # vaia
    "detailed_solution": [
        "Answer the question providing a step by step solution and answer in the end.\n"
        "Provide a step-by-step solution to the question, ending with your final answer.\n",
        "Please provide a step-by-step solution to the question shown in the image.\n",
        "Give a detailed explanation for the question, concluding with your final answer.\n",
        "Solve the problem presented in the question with a thorough explanation. Give me your final answer at the end.\n",
        "Please analyze the question and provide a complete solution, finishing with your final answer.\n",
        "Work through the problem, offering detailed reasoning before stating your final answer.\n",
        "Interpret the question and guide me through the solution, concluding with your answer.\n",
        "Review the question and deliver a well-explained solution, making sure to include your final answer.\n",
        "Examine the question: provide a detailed explanation followed by your final answer.\n" 
    ],

    # vaia first answer with short_answer
    "detailed_solution_answer_first": [
        "Answer the question directly, then provide a step-by-step solution.\n",
        "Please provide the answer first, followed by a step-by-step solution to the question shown in the image.\n",
        "Give the final answer first, then provide a detailed explanation for the question.\n",
        "Provide the final answer, then solve the problem presented in the question with a thorough explanation.\n",
        "First, give the final answer, then analyze the question and provide a complete solution.\n",
        "State the final answer first, then work through the problem, offering detailed reasoning.\n",
        "Provide the final answer, then interpret the question and guide me through the solution.\n",
        "Give the final answer first, then review the question and deliver a well-explained solution.\n",
        "First, provide the final answer, then examine the question and give a detailed explanation.\n"
    ],
    
    # vqa_online
    "detailed_answer": [
        "Answer the question providing a step-by-step explanation and answer in the end.\n",
        "Provide a step-by-step explanation to the question, ending with your final answer.\n",
        "Please provide a step-by-step explanation to the question shown in the image.\n",
        "Give a detailed explanation for the question, concluding with your final answer.\n",
        "Address the problem presented in the question with a thorough explanation. Give me your final answer at the end.\n",
        "Please analyze the question and provide a complete explanation, finishing with your final answer.\n",
        "Work through the problem, offering detailed reasoning before stating your final answer.\n",
        "Interpret the question and guide me through the explanation, concluding with your answer.\n",
        "Review the question and deliver a well-explained answer, making sure to include your final answer.\n",
        "Examine the question: provide a detailed explanation followed by your final answer.\n"
    ],
}

GENERAL_PROMPTS_V1["pointing_tag"] = [txt + " Make the alt text and the inside of the tag the target label." for txt in GENERAL_PROMPTS_V1["pointing"]]

STYLE_TO_GENERAL_PROMPT = {
    "vqa2": "short_answer",
    "coco_captioning": "short_caption",
    "gqa": "short_answer",
    "ocr_vqa": "short_answer",
    "tally_qa": "short_answer",
    "text_vqa": "short_answer",
    "okvqa": "short_answer",
    "chart_qa": "short_answer",
    "doc_qa": "short_answer",
    "info_qa": "short_answer",
    "science_qa": "multiple_choice",
    "ai2_diagram": "multiple_choice",
    "a_okvqa_mc": "multiple_choice",
    "a_okvqa_da": "short_answer",
    "long_caption": "long_caption",
    "web_pointing": "plain",
    "count_bench": "count_bench",
    "refexp": "refexp",
    "refexp_pointing": "refexp_pointing",
    "vtabfact": "binary",
    "vwtq": "short_answer",
    "vwtq_syn": "short_answer",
    "fintabnetqa": "short_answer",
    "scifi_charts": "short_answer",
    "scifi_charts_qa": "short_answer",
    "charxiv_descriptive": "short_answer",
    "charxiv_reasoning": "short_answer",
    "pointing": "pointing",
    "pointing_tag": "pointing_tag",
    "point_count": "point_count",
    "count_then_point": "count_then_point",
    "only_count": "only_count",
    "plain": "plain",
}


# def maybe_format_options(example, option_style="basic"):
#     abc = tf.constant(list("abcdefg".upper()))
#     if option_style == "random-v1":
#         letter_option_sep = [": ", ". ", ")"]
#         option_sep = ["\n", "\n", "\n", " ", ". ", ".\n", "; ", ", "]
#         option_sep = tf.constant(option_sep)[tf.random.uniform((), 0, len(option_sep), tf.int32)]
#     elif option_style == "basic":
#         letter_option_sep = ": "
#         option_sep = "\n"
#     else:
#         raise NotImplementedError(option_style)
#
#     options = example["options"]
#     short_options = abc[:tf.shape(options)[0]]
#     sep = tf.constant(letter_option_sep)[tf.random.uniform((), 0, len(letter_option_sep), tf.int32)]
#
#     options = tf.stack([short_options, options,], 1)
#
#     options = tf.strings.reduce_join(options, axis=-1, separator=sep)
#
#     options = tf.strings.reduce_join(options, separator=option_sep)
#     example["options"] = options
#     tf.debugging.assert_equal(tf.reduce_any(tf.strings.regex_full_match(options, ".*\|\|\|.*")), False)
#     example["metadata/option_names"] = tf.strings.reduce_join(short_options, separator="|||")
#
#     if "answer_idx" in example:
#         if example["answer_idx"] < 0:
#             example["text"] = "?"
#         else:
#             example["text"] = short_options[example["answer_idx"]]
#         example["metadata/answer_idx"] = example["answer_idx"]
#     return example


@gin.configurable
def apply_keyword_prompt(prompts, example, seed=None, weights=None, keywords=None, dbg=False):
    if dbg:
        prompts = prompts[:1]
    if isinstance(prompts, list):
        assert keywords is None
        all_keywords = [sorted(re.findall("{([^{}]+)}", x)) for x in prompts]
        keywords = all_keywords[0]
        assert len(keywords) == len(set(keywords)), f"Repeated keywords in {keywords}"
        assert all(keywords == x for x in all_keywords), f"Inconsistent keywords in prompts {all_keywords}"
        assert not any("{" not in word[1:-1] and "}" in word[1:-1] for word in keywords)

        for k in keywords:
            assert k in example, f"Example missing expected field {k}, example={example}"
        prompts = tf.constant(prompts)

    multiple = False
    if "text" in example and len(example["text"].shape) > 0:
        multiple = True

    if weights is not None:
        weights = tf.expand_dims(tf.math.log(weights), 0)

    if seed is None:
        raise ValueError()

    if not multiple:
        if weights is None:
            prompt = prompts[tf.random.stateless_uniform((), seed, 0, len(prompts), dtype=tf.int32)]
        else:
            prompt = prompts[tf.random.stateless_categorical(weights, 1, seed, 0, len(prompts), dtype=tf.int32)][0, 0]
        for keyword in keywords:
            # We use split not regex_replace because regex_replace has issues with
            # value strings with backslashes
            res = tf.strings.split(prompt, "{"+keyword+"}", maxsplit=2)
            prompt = tf.strings.join([res[0], example[keyword], res[1]])
        return prompt
    else:
        n_prompts = tf.shape(example["text"])[0]
        if weights is None:
            ix = tf.random.stateless_uniform(
                (n_prompts,), seed, 0, tf.shape(prompts)[0], dtype=tf.int32)
        else:
            ix = tf.random.stateless_categorical(
                weights, tf.shape(prompts)[0], seed, 0, len(prompts), dtype=tf.int32)[0]
        prompt = tf.gather(prompts, ix)
        out = tf.TensorArray(dtype=tf.string, size=n_prompts, element_shape=())
        for i in range(n_prompts):
            modified = prompt[i]
            for keyword in keywords:
                res = tf.strings.split(modified, "{"+keyword+"}", maxsplit=2)
                modified = tf.strings.join([res[0], example[keyword][i], res[1]])
            out = out.write(i, modified)
        return out.stack()

