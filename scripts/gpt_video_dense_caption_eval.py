# %%
import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from json import JSONEncoder
from os.path import dirname, exists, isfile, join, basename
from typing import Dict, List, Tuple, Union
import pandas as pd

import numpy as np
import wandb
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from olmo.io import is_dir, list_directory, read_json, write_json, resource_path, file_exists
from olmo.train.trainer_config import RuntimeData, WandbConfig
from olmo.util import prepare_cli_environment

METRIC_ORDER = ["name", "wandb", "step", "checkpoint", "src", "num_statements", "is_repeating",
                "consistency", "recall", "recall_at_10", "loss", "acc"]


class Gpt4WithCache:
    def __init__(self, model, statement_model, judge_model, cache_dir, cache_only=False):
        self.model = model
        self.statement_model = model if statement_model is None else statement_model
        self.judge_model = model if judge_model is None else judge_model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_only = cache_only
        import openai  # import here so dependency is optional
        self.client = openai.OpenAI()

    def _query(self, model, message, **kwargs):
        import openai

        if isinstance(message, str) and len(kwargs) == 0:
            query_hash = compute_hash(model + "::::" + message)
        elif len(kwargs) == 0:
            query_hash = compute_hash(model + "::::" + json.dumps(message))
        else:
            kwargs_ = kwargs.copy()
            if 'response_format_str' in kwargs:
                # use when response format is not hashable
                kwargs_.pop('response_format')
                kwargs.pop('response_format_str')
            query_hash = compute_hash(model + "::::" + json.dumps(message) + "::::" + json.dumps(kwargs_, sort_keys=True))
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
            if 'response_format' in kwargs:
                completion = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=message,
                    **kwargs
                )
            else:
                completion = self.client.chat.completions.create(
                    model=model,
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

    def __call__(self, message, **kwargs):
        return self._query(self.model, message, **kwargs)

    def judge(self, message, **kwargs):
        return self._query(self.judge_model, message, **kwargs)

    def get_statement(self, message, **kwargs):
        return self._query(self.statement_model, message, **kwargs)


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
    usage: dict = None

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
    statement_errors: List = None
    statement_scores: List = None
    error_msg: str = None
    usage: dict = None

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
    statement_error: str = None

    @property
    def name(self):
        return "repeating"

    @property
    def valid(self):
        return self.error is None and self.is_repeating is not None


@dataclass
class FullEval:
    video: str
    caption: str
    recall: RecallEval = None
    consistency: ConsistencyEval = None
    repeating: IsRepeatingEval = None

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["video"],
            data["caption"],
            RecallEval(**data["recall"]) if data.get("recall") else None,
            ConsistencyEval(**data["consistency"]) if data.get("consistency") else None,
            IsRepeatingEval(**data["repeating"]) if data.get("repeating") else None,
        )


def compute_hash(string: str) -> str:
    """Computes the hash of a string."""
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def _do_task(arg):
    target, model_caption, data_info, evaluator = arg
    if target == "repeat":
        result, usage = evaluator.eval_repeat(model_caption)
    elif target == "recall":
        result, usage = evaluator.eval_recall(model_caption, data_info)
    elif target == "consistency":
        result, usage = evaluator.eval_consistency(model_caption, data_info)
    else:
        raise NotImplementedError(target)
    return data_info["video_path"], result, usage


class DenseCaptionEvaluator:
    def __init__(self, data_dir, data_file, gpt_to_query, target_metrics,
                 sample=None, eval_mode='text'):
        self.data_dir = data_dir
        self.gpt_to_query: Gpt4WithCache = gpt_to_query
        self.target_metrics = target_metrics
        self.client = OpenAI()

        df = pd.read_parquet(data_file)
        if sample == "heldout_2k":
            data = df.to_dict(orient='records')
            np.random.RandomState(4931).shuffle(data)
            data = data[:2048]
        else:
            sample = int(sample)
            if sample is not None and sample > 0:
                df = df.sample(sample, random_state=12312)
            data = df.to_dict(orient='records')

        for ex in data:
            ex["video_id"] = compute_hash(ex["video_path"])
        data.sort(key=lambda x: x["video_id"])

        self.data = data
        self.data_idx = {x["video_path"]: x for x in data}
        self._mturk_cache = {}

        self.eval_mode = eval_mode

    def get_mturk_statements(self, data_info):
        video = data_info['video_path']
        if video in self._mturk_cache:
            return self._mturk_cache[video]
        else:
            video_id = compute_hash(video)
            path = join(self.data_dir, f"mturk-eval-statements/{video_id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
            else:
                statements_str, usage = self.get_canonical_statements(data_info['merged_caption'])
                data = {
                        'video_path'          : data_info['video_path'],
                        'caption'             : data_info['merged_caption'],
                        'canonical_statements': statements_str,
                    }
                json.dump(
                    data,
                    open(path, 'w')
                )

            self._mturk_cache[video] = data
            return data

    def query_gpt(self, mode, message, **kwargs):
        assert  mode in ['statement', 'judge', 'other']
        if mode == 'statement':
            msg, was_cached = self.gpt_to_query.get_statement(message, **kwargs)
        elif mode == 'judge':
            msg, was_cached = self.gpt_to_query.judge(message, **kwargs)
        else:
            msg, was_cached = self.gpt_to_query(message, **kwargs)
        if isinstance(msg, Exception):
            return msg, Counter(was_cached=False)
        model = msg["model"]
        usage = Counter({f'{model}|{k}': msg["usage"][k] for k in ["completion_tokens", "prompt_tokens", "total_tokens"]})
        usage[f"{model}|was_cached"] = was_cached
        return msg["choices"][0]["message"]["content"], usage

    def compute_if_stated_from_mturk_statements(self, mturk_statements: str, caption: str):
        if self.eval_mode == 'per_statement':
            lines = []
            total_usage = None
            for i, line in enumerate(mturk_statements.split('\n')):
                statement = line.split('. ', 1)[1]
                prompt = (
                        f"Here is a statement that annotators gave for a video.\n\n"
                        + (
                            # captions
                            statement.strip()
                        )
                        + (
                            '\n\nNext, consider the following caption of the video. For the statement above, state whether the fact is "Stated" or "Not Stated" in the caption. The output should be Stated or Not Stated. Do not output anything other than Stated and Not Stated.\n\n Here is the caption: '
                        )
                        + (
                            # statements
                            caption.strip()
                            if caption
                            else "No caption provided."
                        )
                )
                res, usage = self.query_gpt('judge', prompt, temperature=0)
                if total_usage is None:
                    total_usage = usage
                else:
                    total_usage += usage
                lines.append(res)
            return lines, total_usage

        elif self.eval_mode == 'structured_output':
            prompt = (
                    f"Here are statements that annotators gave for a video.\n\n"
                    + (
                        # captions
                        mturk_statements.strip()
                    )
                    + (
                        '\n\nNext, consider the following caption of the video. For each statement above, state whether the fact is "Stated" or "Not Stated" in the caption. The output should be a list of "Stated" or "Not Stated". Do not output anything other than an ordered list of Stated and Not Stated.\n\n Here is the caption: '
                    )
                    + (
                        # statements
                        caption.strip()
                        if caption
                        else "No caption provided."
                    )
            )

            n_statement = len(mturk_statements.split('\n'))

            response_format = {
                "type"       : "json_schema",
                "json_schema": {
                    "name"  : "Result",
                    "strict": True,
                    "schema": {
                        "type"                : "object",
                        "properties"          : {
                            f'result_{i}' : {
                                "type": "string",
                                "enum": ["Stated", "Not Stated"]
                            } for i in range(1, n_statement+1)
                        },
                        "required"            : [f'result_{i}' for i in range(1, n_statement+1)],
                        "additionalProperties": False
                    }
                }
            }

            response_format_str = json.dumps(response_format)

            return self.query_gpt('judge', prompt, temperature=0, response_format=response_format, response_format_str=response_format_str)
        elif self.eval_mode == 'text':
            prompt = (
                    f"Here are statements that annotators gave for a video.\n\n"
                    + (
                        # captions
                        mturk_statements.strip()
                    )
                    + (
                        '\n\nNext, consider the following caption of the video. For each statement above, state whether the fact is "Stated" or "Not Stated" in the caption. The output should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\nDo not output anything other than an ordered list of Stated and Not Stated.\n\n Here is the caption: '
                    )
                    + (
                        # statements
                        caption.strip()
                        if caption
                        else "No caption provided."
                    )
            )
            return self.query_gpt('judge', prompt, temperature=0)
        else:
            raise NotImplementedError(f"unsupport eval mode: {self.eval_mode}")

    def get_canonical_statements(self, caption: str):
        canonical_statements_prompt = (
            f"Based on the description of the video, come up with a list of the MOST canonical statements that are mentioned in it. Each statement should be broken down as much as possible."
            " The statements should be an ordered list, where each item is separated a newline. For instance, the response may look like:\n\n1. Statement A\n2. Statement B\n3. Statement C\n\n\n"
            f"\n\n\nHere is the video description: {caption}"
        )
        return self.query_gpt('statement', canonical_statements_prompt, temperature=0)

    def get_consistency_statements(
            self, num_transcripts: int, transcripts_str: str, statements_str: str,
    ):
        if self.eval_mode == 'per_statement':
            lines = []
            total_usage = None
            for i, line in enumerate(statements_str.split('\n')):
                statement = line.split('. ', 1)[1]
                prompt = (
                        f"Here are {num_transcripts} captions people gave for a video using their voice.\n\n"
                        + (
                            # captions
                            transcripts_str
                        )
                        + (
                            '\n\nHere is one statement that a captioning model made about the video. State whether it\'s "Consistent" or "Inconsistent" with the captions provided above. The output should be "Consistent" or "Inconsistent". Do not output anything other than Consistent and Inconsistent.\n\n'
                        )
                        + (
                            # statements
                            statement
                        )
                )
                res, usage = self.query_gpt('judge', prompt, temperature=0)
                if total_usage is None:
                    total_usage = usage
                else:
                    total_usage += usage
                lines.append(res)
            return lines, total_usage

        elif self.eval_mode == 'structured_output':
            prompt = (
                    f"Here are {num_transcripts} captions people gave for a video using their voice.\n\n"
                    + (
                        # captions
                        transcripts_str
                    )
                    + (
                        '\n\nHere are statements that a captioning model made about the video. For each statement, state whether it\'s "Consistent" or "Inconsistent" with the captions provided above. The output should be a list of "Consistent" or "Inconsistent". Do not output anything other than an ordered list of Consistent and Inconsistent.\n\n'
                    )
                    + (
                        # statements
                        statements_str
                    )
            )

            n_statement = len(statements_str.split('\n'))

            response_format = {
                "type"       : "json_schema",
                "json_schema": {
                    "name"  : "Result",
                    "strict": True,
                    "schema": {
                        "type"                : "object",
                        "properties"          : {
                            f'result_{i}': {
                                "type": "string",
                                "enum": ["Consistent", "Inconsistent"]
                            } for i in range(1, n_statement + 1)
                        },
                        "required"            : [f'result_{i}' for i in range(1, n_statement + 1)],
                        "additionalProperties": False
                    }
                }
            }

            response_format_str = json.dumps(response_format)

            return self.query_gpt('judge', prompt, temperature=0, response_format=response_format, response_format_str=response_format_str)

        elif self.eval_mode == 'text':
            prompt = (
                    f"Here are {num_transcripts} captions people gave for a video using their voice.\n\n"
                    + (
                        # captions
                        transcripts_str
                    )
                    + (
                        '\n\nHere are statements that a captioning model made about the video. For each statement, state whether it\'s "Consistent" or "Inconsistent" with the captions provided above. The output should be in the form\n\n1. Consistent\n2. Inconsistent\n3. Consistent\n\nDo not output anything other than an ordered list of Consistent and Inconsistent.\n\n'
                    )
                    + (
                        # statements
                        statements_str
                    )
            )
            return self.query_gpt('judge', prompt, temperature=0)

    def is_repeating(self, caption):
        prompt = f"""Determine if a caption is constantly repeating itself. Only reply with "Yes" or "No" and nothing else.\n\nHere is an example of a caption that repeats itself:\n
        the video is a detailed scene of a hallway in a school, featuring a group of men dressed in formal attire. The hallway is adorned with green and white tiles, and there are two rows of lockers on either side. The men are standing in front of a mirror, with one man in a gray suit and black shoes standing on a white step ladder, reaching up to touch the mirror. He is holding a black briefcase in his left hand. The man in the gray suit is wearing a black tie and has a black briefcase in his left hand. He is standing next to a man in a black suit, who is also wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is wearing a black tie and has a briefcase in his left hand. The man in the orange suit is standing next to a man in a black suit, who is wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is wearing a black tie and has a briefcase in his left hand. The man in the orange suit is standing next to a man in a black suit, who is wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is wearing a black tie and has a briefcase in his left hand. The man in the orange suit is standing next to a man in a black suit, who is wearing a black tie and has a briefcase in his left hand. The man in the black suit is standing next to a man in a blue suit, who is wearing a black tie and has a briefcase in his left hand. The man in the blue suit is standing next to a man in an orange suit, who is we\n\nHere is the caption that you are evaluating: {caption}"""
        return self.query_gpt('judge', prompt)

    def eval_recall(self, model_caption, data_info) -> RecallEval:
        mturk_statements = self.get_mturk_statements(data_info)
        statements_str, usage = self.get_canonical_statements(model_caption)
        recall_output, usage2 = self.compute_if_stated_from_mturk_statements(
            mturk_statements["canonical_statements"], model_caption
        )
        usage += usage2

        if self.eval_mode == 'per_statement':
            statement_errors = []
            all_scores = []
            for line in recall_output:
                valid = None
                # GPT is mispells "not stated" sometimes, give it some slack
                if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
                    valid = False
                elif "stated" in line.lower():
                    valid = True
                else:
                    statement_errors.append(f"Bad recall output {line}")
                all_scores.append(valid)
            scores = [x for x in all_scores if x is not None]

        elif self.eval_mode == 'structured_output':
            lines = json.loads(recall_output)
            statement_errors = []
            all_scores = []
            for k, line in lines.items():
                valid = None
                if line == 'Stated':
                    valid = True
                elif line == 'Not Stated':
                    valid = False
                else:
                    statement_errors.append(f"Bad recall output {line}")
                all_scores.append(valid)
            scores = [x for x in all_scores if x is not None]

        elif self.eval_mode == 'text':
            lines = [x.strip() for x in recall_output.split("\n") if x.strip()]
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
                    statement_errors.append(f"Bad recall output {line}")
                all_scores.append(valid)
            scores = [x for x in all_scores if x is not None]
        else:
            raise NotImplementedError(self.eval_mode)

        return RecallEval(
            recall_str=recall_output,
            mturk_statements_str=mturk_statements["canonical_statements"],
            num_statements=len(scores),
            num_covered=int(np.sum(scores)),
            statement_errors=statement_errors,
            statement_scores=all_scores,
            usage=dict(usage)
        ), usage

    def eval_consistency(self, model_caption, data_info):
        statements_str, usage = self.get_canonical_statements(model_caption)
        error = None
        if isinstance(statements_str, BadRequestError):
            return ConsistencyEval(
                consistency_str=None, statements_str=None, num_statements=0,
                num_consistent=None, statement_errors=[], statement_scores=[],
                error="bad-request", consistency_error_msg=statements_str.message), {}
        transcripts = list(data_info["clip_captions"]) + [data_info["video_transcript"]]
        transcripts_str = "\n\n".join([t for t in transcripts if t is not None])
        consistency, usage2 = self.get_consistency_statements(
            len(transcripts), transcripts_str, statements_str
        )
        usage += usage2
        if self.eval_mode == 'per_statement':
            statement_errors = []
            all_scores = []
            for line in consistency:
                inconsistent = None
                if line == 'Inconsistent':
                    inconsistent = True
                elif line == 'Consistent':
                    inconsistent = False
                else:
                    statement_errors.append(f"Bad consistency output {line}")
                all_scores.append(inconsistent)
            scores = [x for x in all_scores if x is not None]

        elif self.eval_mode == 'structured_output':

            lines = json.loads(consistency)
            statement_errors = []
            all_scores = []
            for k, line in lines.items():
                inconsistent = None
                if line == 'Inconsistent':
                    inconsistent = True
                elif line == 'Consistent':
                    inconsistent = False
                else:
                    statement_errors.append(f"Bad consistency output {line}")
                all_scores.append(inconsistent)
            scores = [x for x in all_scores if x is not None]


        elif self.eval_mode == 'text':
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
        else:
            raise NotImplementedError(self.eval_mode)

        return ConsistencyEval(
            consistency_str=consistency,
            statements_str=statements_str,
            num_statements=len(scores), num_consistent=sum(not x for x in scores),
            error=None, statement_errors=statement_errors,
            statement_scores=all_scores,
            usage=dict(usage)
        ), usage

    def eval_repeat(self, model_caption):
        raise NotImplementedError

    def eval_captions(self, video_to_caption: Dict[str, str], n_threads=24) -> Tuple[Dict[str, FullEval], Counter]:
        data = {x["video_path"]: x for x in self.data}

        # In case we are evaluating on a subsample
        # Strip since de-tokenization can add leading space for some tokenizers
        # _video_to_caption = {k: v.strip() for k, v in video_to_caption.items() if k in data}
        # if len(_video_to_caption) == 0:
            # FIXME remove this hack
        data = {k.replace("/", "_"): v for k, v in data.items()}
        video_to_caption = {k.replace("/", "_"): v for k, v in video_to_caption.items()}

        if len(video_to_caption) != len(data):
            raise ValueError("Missing urls!")

        tasks = []
        for video in sorted(video_to_caption):
            data_info = data[video]
            for metric_name in self.target_metrics:
                tasks.append((metric_name, video_to_caption[video], data_info, self))
        scores = defaultdict(dict)
        total_usage = Counter()
        if n_threads > 1:
            with ThreadPoolExecutor(n_threads) as pool:
                for video, result, usage in tqdm(pool.map(_do_task, tasks), total=len(tasks), ncols=100):
                    total_usage += usage
                    if result.error is not None:
                        logging.warning(f"Got error {result.kind}: {result.message}, skipping")
                    scores[video][result.name] = result
        else:
            for task in tqdm(tasks, ncols=100):
                video, result, usage = _do_task(task)
                total_usage += usage
                if result.error is not None:
                    logging.warning(f"Got error {result.error}: {result.consistency_error_msg}, skipping")
                scores[video][result.name] = result
        full_eval = {}
        for k, v in scores.items():
            k = k.replace("/", "_")
            full_eval[k] = FullEval(**v, video=k, caption=video_to_caption[k])
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


def get_wandb_run(prediction_file):
    checkpoint = int(re.match(".*/predictions-ck([0-9]+)-", prediction_file).group(1))
    checkpoint_dir = join(dirname(dirname(prediction_file)), f"step{checkpoint}")
    data = RuntimeData.load(resource_path(checkpoint_dir, "config.yaml"), key="runtime_data")
    wandb_data = WandbConfig.load(resource_path(checkpoint_dir, "config.yaml"), key="wandb")
    return wandb.Api().run(f"{wandb_data.entity}/{wandb_data.project}/{data.wandb_id}")


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
    parser.add_argument("--data_dir", default="/weka/oe-training-default/mm-olmo/video_dense_caption_eval/")
    parser.add_argument("--data_file", default="/weka/oe-training-default/mm-olmo/video_captions/video-captions-9k.parquet")
    parser.add_argument(
        "--metrics", nargs="+", choices=["recall", "repeat", "consistency", "all"],
        default=["recall", "consistency"],
    )
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--eval_mode", default='text', choices=['text', 'structured_output', 'per_statement'])
    parser.add_argument("--model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--statement_model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--judge_model", default="gpt-4.1-nano-2025-04-14")
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
        args.data_file,
        Gpt4WithCache(
            args.model,
            args.statement_model,
            args.judge_model,
            None if args.no_cache else join(args.data_dir, "gpt4-cache"),
            args.cache_only
        ),
        metrics,
        sample=args.sample,
        eval_mode=args.eval_mode,
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
        if is_dir(file):
            candidates = [x for x in list_directory(file) if "cap_v2-human_annotated_val-len-null" in x]
            if len(candidates) == 1:
                logging.info(f"Selecting {candidates[0]} for {file}")
                file = join(candidates[0], "predictions.json")
            else:
                logging.warning(f"Unable to auto-select predictions in directory {file}")
                continue
        resolved_targets.append((name, file))
    target_files = resolved_targets

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
        if args.save_metrics and file_exists(results_file):
            logging.info(f"Loading metrics from {results_file}")
            with open(resource_path(dirname(results_file), basename(results_file))) as f:
                results = json.load(f)
        else:
            try:
                captions = read_json(file)
            except Exception as e:
                if len(target_files) == 1:
                    e.add_note(f"Error reading {file}")
                    raise e
                else:
                    logging.warning(f"Error reading {file}: {e}")
                    continue
            if "video" in captions[0]:
                captions = {x["video"]: x["prediction"] for x in captions}
            elif captions[0]["example_id"].endswith("/caption"):
                # example_id's where build with video_id and type in the iID
                captions = {x["example_id"].rsplit("/", 1)[0].split("/", 1)[1]: x["prediction"] for x in captions}
            else:
                raise ValueError()

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
                    consistency=np.mean([x.consistency for x in consistency]) * 100,
                ))
            if "repeat" in metrics:
                results["is_repeating"] = np.mean([
                    x.repeating.is_repeating for x in full_eval.values() if x.repeating.valid
                ]) * 100
            if "recall" in metrics:
                recall = [x.recall for x in full_eval.values() if x.recall.valid]
                results.update(dict(
                    recall=np.mean([x.recall for x in recall]) * 100,
                    recall_at_10=np.mean([x.recall_at(10) for x in recall]) * 100
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
                    "loss": "llava_cap/CrossEntropyLoss",
                    "acc": "llava_cap/Accuracy",
                }
                hist = run.scan_history(
                    keys=["_step"] + list(wandb_keys.values()), min_step=step-1, max_step=step + 1)
                for summary in hist:
                    assert summary["_step"] == step
                    for out_key, wandb_key in wandb_keys.items():
                        val = summary[wandb_key]
                        results[out_key] = val * 100
                    break
                else:
                    import pdb; pdb.set_trace()
                    logging.warning(f"Unable to find loss for {name}, run={run.id} step={step}")

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
        print("*" * 10 + " TSV results with hyper-parameters " + "*" * 10)
        print("\t".join(results_with_hyperparameters[0].keys()))
        print("\n".join(";".join(str(x) for x in r.values()) for r in results_with_hyperparameters))
        print("*" * 50)
        print()

    print(list_of_dict_to_string(all_results, rows=METRIC_ORDER))


if __name__ == '__main__':
    main()
