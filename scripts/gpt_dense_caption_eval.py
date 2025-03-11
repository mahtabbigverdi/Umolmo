# %%
import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from json import JSONEncoder
from os.path import exists, join, dirname, isdir, isfile
from pathlib import Path
from typing import Dict, Tuple, List, Union

import numpy as np
import wandb
from openai import OpenAI, BadRequestError
from torchvision.datasets.utils import list_dir
from tqdm import tqdm

from olmo.io import read_json, is_url, write_json, file_exists, dir_is_empty
from olmo.util import prepare_cli_environment

METRIC_ORDER = ["name", "wandb", "step", "checkpoint", "src", "num_statements", "is_repeating",
                "consistency", "recall", "recall_at_10", "loss", "acc"]


class Gpt4WithCache:
    def __init__(self, model, cache_dir, cache_only=False):
        self.model = model
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        import openai  # import here so dependency is optional
        self.client = openai.OpenAI()

    def __call__(self, message, **kwargs):
        import openai
        if isinstance(message, str) and len(kwargs) == 0:
            query_hash = compute_hash(self.model + "::::" + message)
        elif len(kwargs) == 0:
            query_hash = compute_hash(self.model + "::::" + json.dumps(message))
        else:
            query_hash = compute_hash(self.model + "::::" + json.dumps(message) + "::::" + json.dumps(kwargs, sort_keys=True))
        use_cache = self.cache_dir

        if use_cache:
            cache_file = join(self.cache_dir, f"{query_hash}-v1.json")
            if exists(cache_file):
                with open(cache_file) as f:
                    try:
                        return json.load(f), True
                    except Exception as e:
                        raise ValueError(f"Error loading {cache_file}", e)

        if self.cache_only:
            raise ValueError("Not cached")

        if isinstance(message, str):
            message = [{"role": "user", "content": message}]

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                **kwargs
            )
        except openai.BadRequestError as e:
            if "We\'ve encountered an issue with repetitive patterns in your prompt" in e.message:
                # I have seen this error rarely with GPT 3.5, we allow evaluation to continue
                return e, False
            raise RuntimeError(e)
        completion = completion.dict()

        if use_cache:
            # Write to tmp file and rename to avoid incorrect data if interrupted
            # Use `dir=self.cache_dir` to make sure the tmp file is on the same device as the
            # the output file to ensure it can be re-renamed
            (fd, tmp_file) = tempfile.mkstemp(
                ".tmp", prefix=f"{query_hash}-v1.json",
                text=True, dir=self.cache_dir)
            os.close(fd)

            with open(tmp_file, "w") as f:
                json.dump(completion, f)
            os.rename(tmp_file, cache_file)
        return completion, False


@dataclass
class ConsistencyEval:
    consistency_str: str
    statements_str: str
    num_statements: int
    num_consistent: int
    statement_errors: List
    statement_scores: List
    error: str = None
    consistency_error_msg: str = None

    @property
    def valid(self):
        return self.error is None and self.num_statements > 0

    @property
    def consistency(self):
        return self.num_consistent / self.num_statements

    @property
    def inconsistency(self):
        return 1.0 - self.num_consistent / self.num_statements

    @property
    def name(self):
        return "consistency"


@dataclass
class RecallEval:
    recall_str: str
    mturk_statements_str: str
    num_statements: int
    num_covered: int
    error: str = None
    statement_errors: List=None
    statement_scores: List=None
    error_msg: str = None

    @property
    def valid(self):
        return self.error is None and self.num_statements > 0

    @property
    def recall(self):
        return self.num_covered / self.num_statements

    def recall_at(self, n):
        return min(self.num_covered, n) / min(self.num_statements, n)

    @property
    def name(self):
        return "recall"


@dataclass
class IsRepeatingEval:
    is_repeating: bool
    is_repeating_str: str
    error: str = None
    error_msg: str = None
    statement_error: str=None

    @property
    def name(self):
        return "repeating"

    @property
    def valid(self):
        return self.error is None and self.is_repeating is not None


@dataclass
class FullEval:
    image_url: str
    caption: str
    recall: RecallEval=None
    consistency: ConsistencyEval=None
    repeating: IsRepeatingEval=None

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["image_url"],
            data["caption"],
            RecallEval(**data["recall"]) if data.get("recall") else None,
            ConsistencyEval(**data["consistency"]) if data.get("consistency") else None,
            IsRepeatingEval(**data["repeating"]) if data.get("repeating") else None,
        )


def compute_hash(string: str) -> str:
    """Computes the hash of a string."""
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def _do_task(arg):
    target, model_caption, image_info, evaluator = arg
    if target == "repeat":
        result, usage = evaluator.eval_repeat(model_caption)
    elif target == "recall":
        result, usage = evaluator.eval_recall(model_caption, image_info["image"])
    elif target == "consistency":
        result, usage = evaluator.eval_consistency(model_caption, image_info)
    else:
        raise NotImplementedError(target)
    return image_info["image"], result, usage


class DenseCaptionEvaluator:
    def __init__(self, data_dir, gpt_to_query, target_metrics,
                 sample=None):
        self.data_dir = data_dir
        self.gpt_to_query = gpt_to_query
        self.target_metrics = target_metrics
        self.client = OpenAI()
        with open(join(data_dir, "final-data.json")) as f:
            data = json.load(f)
        for ex in data:
            ex["image_id"] = compute_hash(ex["image"])
        data.sort(key=lambda x: x["image_id"])
        np.random.RandomState(12312).shuffle(data)
        self.data = data[:sample]
        self.data_idx = {x["image"]: x for x in data}
        self._mturk_cache = {}

    def get_mturk_statements(self, image_url):
        if image_url in self._mturk_cache:
            return self._mturk_cache[image_url]
        else:
            image_id = compute_hash(image_url)
            with open(join(self.data_dir, f"mturk-eval-statements/{image_id}.json")) as f:
                data = json.load(f)
            self._mturk_cache[image_url] = data
            return data

    def query_gpt(self, message, **kwargs):
        msg, was_cached = self.gpt_to_query(message, **kwargs)
        if isinstance(msg, Exception):
            return msg, Counter(was_cached=False)
        usage = Counter({k: msg["usage"][k] for k in ["completion_tokens", "prompt_tokens", "total_tokens"]})
        del usage["completion_tokens_details"]
        del usage["prompt_tokens_details"]
        usage["was_cached"] = was_cached
        return msg["choices"][0]["message"]["content"], usage

    def compute_if_stated_from_mturk_statements(self, mturk_statements: str, caption: str) -> str:
        prompt = (
            f"Here are statements that annotators gave for an image.\n\n"
            + (
                # captions
                mturk_statements.strip()
            )
            + (
                '\n\nNext, consider the following caption of the image. For each statement above, state whether the fact is "Stated" or "Not Stated" in the caption. The output should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\nDo not output anything other than an ordered list of Stated and Not Stated.\n\n Here is the caption: '
            )
            + (
                # statements
                caption.strip()
                if caption
                else "No caption provided."
            )
        )
        return self.query_gpt(prompt, temperature=0)

    def get_canonical_statements(self, caption: str) -> str:
        # logging.warning("DEBUGGING")
        canonical_statements_prompt = (
            f"Based on the description of the image, come up with a list of the MOST canonical statements that are mentioned in it. Each statement should be broken down as much as possible."
            " The statements should be an ordered list, where each item is separated a newline. For instance, the rseponse may look like:\n\n1. Statement A\n2. Statement B\n3. Statement C\n\n\n"
            f"\n\n\nHere is the image description: {caption}"
        )
        # msg, was_cached = Gpt4WithCache("gpt-4o-2024-05-13", join("/home/chris/data/dense-caption-eval/gpt4-cache"),False)(canonical_statements_prompt)
        # usage = Counter(msg["usage"])
        # usage["was_cached"] = was_cached
        # return msg["choices"][0]["message"]["content"], usage
        return self.query_gpt(canonical_statements_prompt, temperature=0)

    def get_consistency_statements(
        self, num_transcripts: int, transcripts_str: str, statements_str,
    ) -> str:
        prompt = (
            f"Here are {num_transcripts} captions people gave for an image using their voice.\n\n"
            + (
                # captions
                transcripts_str
            )
            + (
                '\n\nHere are statements that a captioning model made about the image. For each statement, state whether it\'s "Consistent" or "Inconsistent" with the statements provided above. The output should be in the form\n\n1. Consistent\n2. Inconsistent\n3. Consistent\n\nDo not output anything other than an ordered list of Consistent and Inconsistent.\n\n'
            )
            + (
                # statements
                statements_str
            )
        )
        return self.query_gpt(prompt, temperature=0)

    def get_consistency_statements_v2(
        self, num_transcripts: int, transcripts_str: str, statements_str
    ) -> str:
        instruction = """
Your job is to classify a list of statements about the image using the captions. For each statement, use your 
best judgment to decide if the statement could be true given the transcripts, in which case it 
should be labeled as \"Consistent\", or if it is cannot be true given the transcripts, 
in which case they should be labeled as \"Inconsistent\". 
If impossible to even make a guess, such if the statement is primarily about a detail in the image
that is not even indirectly mentioned in the transcripts, label the statement as \"Unclear\",
but this should happen rarely and only if it is really impossible to make a reasonable guess.
"""
        instruction = " ".join(instruction.split())

        prompt = f"""
f"Here are {num_transcripts} captions people gave for an image using their voice:

{transcripts_str}

{instruction}

The output should be in the form like:
1. Consistent
2. Inconsistent
3. Unclear

Do not output anything other than an ordered list that only contains the words Consistent, Inconsistent, or Unclear

Statements to analyze:
{statements_str}
""".strip()
        return self.query_gpt(prompt, temperature=0)

    def is_repeating(self ,caption):
        prompt = f"""Determine if a caption is constantly repeating itself. Only reply with "Yes" or "No" and nothing else.\n\nHere is an example of a caption that repeats itself:\n
        The image is a detailed scene of a hallway in a school, featuring a group of men dressed in formal attire. The hallway is adorned with green and white tiles, and there are two rows of lockers on either side. The men are standing in front of a mirror, with one man in a gray suit and black shoes standing on a white step ladder, reaching up to touch the mirror. He is holding a black briefcase in his left hand. The man in the gray suit is wearing a black tie and has a black briefcase in his left hand. He is standing next to a man in a black suit, who is also wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is wearing a black tie and has a briefcase in his left hand. The man in the orange suit is standing next to a man in a black suit, who is wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is wearing a black tie and has a briefcase in his left hand. The man in the orange suit is standing next to a man in a black suit, who is wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is wearing a black tie and has a briefcase in his left hand. The man in the orange suit is standing next to a man in a black suit, who is wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is we\n\nHere is the caption that you are evaluating: {caption}"""
        return self.query_gpt(prompt)

    def eval_recall(self, model_caption, image) -> RecallEval:
        mturk_statements = self.get_mturk_statements(image)
        statements_str, usage = self.get_canonical_statements(model_caption)
        recall_output, usage2 = self.compute_if_stated_from_mturk_statements(
            mturk_statements["canonical_statements"], model_caption
        )
        usage += usage2

        lines = [x.strip() for x in recall_output.split("\n") if x.strip()]
        scores = []
        statement_errors = []
        all_scores = []
        for line in lines:
            valid = None
            # GPT is mispells "not stated" sometimes, give it some slack
            if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
                valid = False
            elif " stated" in line.lower():
                valid = True
            else:
                statement_errors.append(f"Bad recall output {line} for {image}, id={image}")
                all_scores.append(None)
                continue
            all_scores.append(valid)
        scores = [x for x in all_scores if x is not None]
        return RecallEval(
            recall_str=recall_output,
            mturk_statements_str=mturk_statements["canonical_statements"],
            num_statements=len(scores),
            num_covered=int(np.sum(scores)),
            statement_errors=statement_errors,
            statement_scores=all_scores
        ), usage

    def get_scored_statements(self, model_caption, image_url):
        image_info = self.data_idx[image_url]
        statements_str, usage = self.get_canonical_statements(model_caption)
        if isinstance(statements_str, BadRequestError):
            return None
        transcripts_str = "\n\n".join(
            [x["whisperTranscript"] for x in image_info["transcripts"]]
        )
        consistency, usage2 = self.get_consistency_statements(
            len(image_info["transcripts"]), transcripts_str, statements_str
        )
        consistency_lines = [x.strip() for x in consistency.split("\n") if x.strip()]
        statements = [x.strip() for x in statements_str.split("\n") if x.strip()]
        if len(statements) > len(consistency_lines):
            logging.warning("Some statements do not have a consistency score")
            statements = statements[:len(consistency)]
        elif len(consistency_lines) > len(statements):
            logging.warning("Have too many consistency scores")
            consistency_lines = consistency_lines[:len(statements)]
        scored_statements = []
        for line, statement in zip(consistency_lines, statements):
            inconsistent = None
            # GPT 4 is surprisingly bad at following in consistent/inconsistent format exactly,
            # do some fuzzy matching for mispellings and other variations
            if re.fullmatch(r".*\b((i?inconsis?ten(t|cy)?)|incorrect|inconsiistent|inconsisent|incomplete|contradictory)\.?$", line, flags=re.IGNORECASE):
                inconsistent = True
            if re.fullmatch(r".*\b(consistent(ly)?|constistent|correct)\.?$", line, flags=re.IGNORECASE):
                if inconsistent:
                    raise ValueError(line)
                inconsistent = False
            if ("not enough information" in line.lower()) or ("ambiguous" in line.lower()):
                # Its not supposed to return this, but if it does we will just skip it
                continue
            if inconsistent is None:
                logging.warning(f"Bad consistency output {line}")
                continue
            scored_statements.append((not inconsistent, statement))
        return scored_statements

    def eval_consistency(self, model_caption, image_info):
        statements_str, usage = self.get_canonical_statements(model_caption)
        error = None
        if isinstance(statements_str, BadRequestError):
            return ConsistencyEval(
                consistency_str=None, statements_str=None, num_statements=0,
                num_consistent=None, statement_errors=[], statement_scores=[],
                error="bad-request", consistency_error_msg=statements_str.message), {}
        transcripts_str = "\n\n".join(
            [x["whisperTranscript"] for x in image_info["transcripts"]]
        )
        consistency, usage2 = self.get_consistency_statements(
            len(image_info["transcripts"]), transcripts_str, statements_str
        )
        usage += usage2
        lines = [x.strip() for x in consistency.split("\n") if x.strip()]
        scores = []
        statement_errors = []
        for line in lines:
            inconsistent = None
            # GPT 4 is surprisingly bad at following in consistent/inconsistent format exactly,
            # do some fuzzy matching for mispellings and other variations
            if re.fullmatch(r".*[^a-z]((i?inconsis?ten(t|cy)?)|incorrect|inconsistence|iconsistent|inconsisent|incomplete|contradictory).*", line, flags=re.IGNORECASE):
                inconsistent = True

            if re.fullmatch(r".*[^a-z](consistent(ly)?|constistent|correct).*$", line, flags=re.IGNORECASE):
                if inconsistent:
                    inconsistent = None
                else:
                    inconsistent = False

            scores.append(inconsistent)
            if inconsistent is None:
                statement_errors.append(f"Bad consistency output {line}")
                # Model is not instructed to output these unknown options, but does anyway
                unknown = [
                    "not specified",
                    "cannot determine",
                    "not determinable",
                    "no verification",
                    "N/A",
                    "not confirmed",
                    "neither",
                    "not stated",
                    "no judgement",
                    "unable to determine",
                    "inconclusive",
                    "undetermined",
                    "insufficient information",
                    "no relevant information",
                    "no conclusion",
                    "not clear",
                    "unknown",
                    "uncertain",
                    "ambiguous",
                    "not addressed",
                    "not enough information",
                    "not mentioned",
                    "not enough info",
                    "no information",
                    "not verifiable",
                    "not applicable"
                ]
                if not re.fullmatch(r".*\b(" + "|".join(unknown) + r").*$", line, flags=re.IGNORECASE):
                    # Warn if it is something very unexpected
                    logging.warning(statement_errors[-1])
        all_scores = scores
        scores = [x for x in all_scores if x is not None]
        return ConsistencyEval(
            consistency_str=consistency,
            statements_str=statements_str,
            num_statements=len(scores), num_consistent=sum(not x for x in scores),
            error=None, statement_errors=statement_errors,
            statement_scores=all_scores
        ), usage

    def eval_repeat(self, model_caption):
        ans_full, usage = self.is_repeating(model_caption)
        if isinstance(ans_full, BadRequestError):
            return IsRepeatingEval(is_repeating_str=None, is_repeating=False, error=ans_full.message), usage
        ans = ans_full.strip().lower()
        error = None
        statement_error = None
        if "yes" in ans:
            is_repeat = True
        elif "no" in ans:
            is_repeat = False
        else:
            statement_error = f"Bad output {ans}"
            logging.warning(statement_error)
            is_repeat = None
        return IsRepeatingEval(is_repeating_str=ans_full, is_repeating=is_repeat, statement_error=statement_error), usage

    def eval_captions(self, url_to_caption: Dict[str, str], n_threads=24) -> Tuple[Dict[str, FullEval], Counter]:
        data = {x["image"]: x for x in self.data}

        # In case we are evaluating on a subsample
        # Strip since de-tokenization can add leading space for some tokenizers
        url_to_caption = {k: v.strip() for k, v in url_to_caption.items() if k in data}
        if len(url_to_caption) != len(data):
            raise ValueError("Missing urls!")

        tasks = []
        for image_url in sorted(url_to_caption):
            image_info = data[image_url]
            for metric_name in self.target_metrics:
                tasks.append((metric_name, url_to_caption[image_url], image_info, self))
        scores = defaultdict(dict)
        total_usage = Counter()
        if n_threads > 1:
            with ThreadPoolExecutor(n_threads) as pool:
                for image_url, result, usage in tqdm(pool.map(_do_task, tasks), total=len(tasks), ncols=100):
                    total_usage += usage
                    if result.error is not None:
                        logging.warning(f"Got error {result.kind}: {result.message}, skipping")
                    scores[image_url][result.name] = result
        else:
            for task in tqdm(tasks, ncols=100):
                image_url, result, usage = _do_task(task)
                total_usage += usage
                if result.error is not None:
                    logging.warning(f"Got error {result.error}: {result.consistency_error_msg}, skipping")
                scores[image_url][result.name] = result
        full_eval = {}
        for k, v in scores.items():
            full_eval[k] = FullEval(**v, image_url=k, caption=url_to_caption[k])
        return full_eval, total_usage


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return JSONEncoder.default(self, obj)


def get_wandb_run_id(prediction_file):
    model_dir = dirname(prediction_file)
    wandb_dir = join(model_dir, 'wandb')
    if exists(join(wandb_dir, "wandb")):
        wandb_dir = join(wandb_dir, "wandb")
    if not exists(wandb_dir):
        logging.warning(f"Wandb directory not found in {model_dir}")
        return None

    run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith('run-')]
    if not run_dirs:
        logging.warning(f"No run directories found in {wandb_dir}")
        return None

    latest_run_dir = max(run_dirs, key=lambda d: os.stat(join(wandb_dir, d)).st_mtime)
    latest_run_id = latest_run_dir.split('-')[-1]
    return latest_run_id


def get_wandb_run(prediction_file):
    run_id = get_wandb_run_id(prediction_file)
    if run_id is None:
        return
    wandb_entity = os.environ.get("WANDB_ENTITY", "prior-ai2")
    wandb_project = os.environ.get("WANDB_PROJECT", "cockatoo")
    return wandb.Api().run(f"{wandb_entity}/{wandb_project}/{run_id}") # load wandb run from run_id


def _format(val):
    if isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        return val
    else:
        return f"{val:0.2f}"


def list_of_dict_to_string(table: List[Dict[str, Union[str, int, float]]], filler="", rows=None) -> str:
    keys = dict()
    for row in table:
        keys.update(row)
    if rows is not None:
        keys = [k for k in rows if k in keys] + [k for k in keys if k not in rows]
    raw_table = [list(keys)]
    raw_table += [[_format(row.get(key, filler)) for key in keys] for row in table]
    return table_string(raw_table)


def table_string(table: List[List[str]]) -> str:
    """Table as listoflists to evenly spaces string"""
    # print while padding each column to the max column length
    if len(table) == 0:
        return ""
    col_lens = [0] * len(table[0])
    for row in table:
        for i, cell in enumerate(row):
            col_lens[i] = max(len(cell), col_lens[i])

    formats = ["{0:<%d}" % x for x in col_lens]
    out = []
    for row in table:
        out.append(" ".join(formats[i].format(row[i]) for i in range(len(row))))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file", nargs="+")
    parser.add_argument("--data_dir", default="/weka/oe-training-default/mm-olmo/dense_caption_eval/")
    parser.add_argument(
        "--metrics", nargs="+", choices=["recall", "repeat", "consistency", "all"],
        default=["recall", "consistency"],
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--model", default="gpt-4o-2024-05-13")
    parser.add_argument("--n_threads", type=int, default=36)
    parser.add_argument("--find_wandb", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--upload_to_wandb", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--show_with_hyperparameters", action="store_true")

    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--metric_dir")
    args = parser.parse_args()
    prepare_cli_environment()

    if "all" in args.metrics:
        args.metrics = ["recall", "repeat", "consistency"]

    logging.getLogger('httpx').setLevel(logging.WARNING)  # info level logs every request
    metrics = args.metrics

    evaluator = DenseCaptionEvaluator(
        args.data_dir,
        Gpt4WithCache(
            args.model,
            None if args.no_cache else join(args.data_dir, "gpt4-cache"),
            args.cache_only
        ),
        metrics,
        sample=args.sample,
    )

    # See if we were given a files containing a list of prediction files to process
    target_files = None
    if len(args.prediction_file) == 1 and isfile(args.prediction_file[0]):
        source_files = read_json(args.prediction_file[0])
        if isinstance(source_files, dict):
            target_files = list(source_files.items())

    if target_files is None:
        target_files = []
        for file in args.prediction_file:
            if "::" in file:
                name, file = file.split("::", 1)
            elif ":" in file and not file.startswith("gs:"):
                name, file = file.split(":", 1)
            else:
                name = None
            target_files.append((name, file))

    resolved_targets = []
    for name, file in target_files:
        if not dir_is_empty(file):
            candidates = [x for x in list_dir(file) if "dense_caption_eval-test" in x]
            if len(candidates) == 1:
                logging.info(f"Selecting {candidates[0]} for {file}")
                file = join(file, candidates[0], "predictions.json")
            else:
                raise ValueError(f"Unable to auto-select predictions in directory {file}")
        resolved_targets.append((name, file))
    target_files = resolved_targets

    runs = None
    header = None
    per_example_results = defaultdict(list)
    all_results = []
    results_with_hyperparameters = []
    for name, file in target_files:
        prefix = []
        if args.sample:
            prefix.append(f"s{args.sample}")
        if args.metrics != ["recall", "repeat", "consistency"]:
            metric_short_name = "-".join(sorted(args.metrics))
            prefix.append(f"m{metric_short_name}")
        prefix = '-'.join(prefix) + ("-" if prefix else "")

        if args.metric_dir:
            prefix = args.metric_dir.rstrip("/") + f"/{prefix}"
        else:
            prefix = dirname(file) + f"/{prefix}"

        results_file = prefix + "results_v3.json"
        if args.save_metrics and exists(results_file):
           logging.info(f"Loading metrics from {results_file}")
           results = read_json(results_file)
        else:
            try:
                captions = read_json(file)
            except Exception as e:
                if len(target_files) == 1:
                    raise ValueError(f"Error reading {file}", e)
                else:
                    logging.warning(f"Error reading {file}: {e}")
                    continue
            captions = {x["image_url"]: x["prediction"] for x in captions}

            stripped_eos = 0
            for k, v in captions.items():
                if v.endswith("<|endoftext|>"):
                    stripped_eos += 1
                    captions[k] = v[:-len("<|endoftext|>")]
            if stripped_eos > 0:
                logging.warning("Stripped EOS from %s examples", stripped_eos)

            if name is not None:
                logging.info(f"Evaluating {name}: {file}")
            else:
                logging.info(f"Evaluating {file}")
            full_eval, usage = evaluator.eval_captions(captions, n_threads=args.n_threads)
            full_eval: Dict[str, FullEval] = full_eval

            if args.save_metrics:
                metric_file = prefix + "all-results-v3.json"
                logging.info(f"Saving eval to {metric_file}")
                write_json(metric_file, dict(
                    metrics=metrics,
                    sample=args.sample,
                    date=datetime.now().strftime("%Y%m%d-%H%M%S"),
                    results={k: asdict(v) for k, v in full_eval.items()}
                ), indent=2)

            results = dict()
            if "consistency" in metrics:
                consistency = [x.consistency for x in full_eval.values() if x.consistency.valid]
                results.update(dict(
                    num_statements=np.mean([x.num_statements for x in consistency]),
                    consistency=np.mean([x.consistency for x in consistency])*100,
                ))
            if "repeat" in metrics:
                results["is_repeating"] = np.mean([
                    x.repeating.is_repeating for x in full_eval.values() if x.repeating.valid
                ])*100
            if "recall" in metrics:
                recall = [x.recall for x in full_eval.values() if x.recall.valid]
                results.update(dict(
                    recall=np.mean([x.recall for x in recall])*100,
                    recall_at_10=np.mean([x.recall_at(10) for x in recall])*100
                ))
            results = {k: float(v) for k, v in results.items()}
            if args.save_metrics:
                logging.info(f"Saving scores to {results_file}")
                write_json(results_file, dict(results), indent=2)

        if name is not None:
            results["name"] = name

        # Figure out the step evaluated using the fie name, or a hard-coded mapping
        matches = re.findall("-ck([0-9]+)", file)
        if len(matches) != 1:
            logging.warning(f"Unable to detect step for {file}")
            step = " -"
        else:
            assert len(matches) == 1
            step = int(matches[0])
        results["step"] = step

        run = None
        if args.find_wandb:
            # Look up the wandb run and find the val loss
            run = get_wandb_run(file)
            if run is None:
                url = ''
                loss = ''
                acc = ''
            else:
                wandb_keys = {
                    "caption_val/Loss": "loss",
                    "caption_val/Accuracy": "acc",
                    "val/Loss": "mix_loss",
                    "val/Accuracy": "mix_acc",
                }
                hist = run.scan_history(
                    keys=["_step"] + list(wandb_keys), min_step=step, max_step=step+1)
                for summary in hist:
                    assert summary["_step"] == step
                    for wandb_key, out_key in wandb_keys.items():
                        val = summary[wandb_key]
                        results[out_key] = val*100
                    break
                else:
                    logging.warning(f"Unable to find loss for {run.id} {step}")
                url = f"https://wandb.ai/prior-ai2/cockatoo/runs/{run.id}"

        config_file = join(dirname(dirname(file)), "config.yaml")
        all_results.append(results)

        if args.show_with_hyperparameters:
            assert args.find_wandb
            assert run is not None
            cfg = {}
            if name is not None:
                cfg["Name"] = name
            cfg["checkpoint"] = step
            cfg["pred file"] = file
            for k in ["num_statements", "is_repeating", "consistency", "recall", "recall_at_10", "loss", "acc"]:
                cfg[k] = results.get(k, "")
            results_with_hyperparameters.append(cfg)

    if args.show_with_hyperparameters:
        print("*"*10 + " TSV results with hyper-parameters " + "*"*10)
        print("\t".join(results_with_hyperparameters[0].keys()))
        print("\n".join(";".join(str(x) for x in r.values()) for r in results_with_hyperparameters))
        print("*"*50)
        print()

    print(list_of_dict_to_string(all_results, rows=METRIC_ORDER))


if __name__ == '__main__':
    main()