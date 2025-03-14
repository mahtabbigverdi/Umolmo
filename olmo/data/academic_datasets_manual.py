"""Datasets the load directly from source files,
Currently not used in favour of using HF datasets"""
import json
import logging
from os import environ
from os.path import exists
from os.path import join
import numpy as np
from cached_path import cached_path

from olmo.data.dataset import DATA_HOME, DatasetBase


if DATA_HOME is not None:
    DOWNLOADS = join(DATA_HOME, "downloads")
    CHARTQA_SOURCE = join(DATA_HOME, "chartqa")
    DOCQA_SOURCE = join(DATA_HOME, "docqa")
    INFOQA_SOURCE = join(DATA_HOME, "info_qa")
    ST_QA_SRC = join(DATA_HOME, "scene-text")
else:
    CHARTQA_SOURCE = None
    DOCQA_SOURCE = None
    INFOQA_SOURCE = None
    ST_QA_SRC = None


class InfoQa(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                filename = "infographicsVQA_val_v1.0_withQT.json"
            else:
                filename = f"infographicsVQA_{split}_v1.0.json"
            if not exists(join(INFOQA_SOURCE, filename)):
                raise ValueError(
                    "InfoQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 3)"
                    f" please download and unzip the data into `{INFOQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            filename = "infographicsVQA_val_v1.0_withQT.json"
        else:
            filename = f"infographicsVQA_{split}_v1.0.json"
        filename = join(INFOQA_SOURCE, filename)
        logging.info(f"Loading docqa data from {filename}")
        with open(cached_path(filename, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            image_path = join(INFOQA_SOURCE, "images", ex.pop("image_local_name"))
            out.append(dict(
                image=image_path,
                question=ex["question"],
                answers=ex.get("answers", []),
                metadata=dict(example_id=ex["questionId"]),
            ))
        return out

    def get(self, item, rng):
        return dict(**self.data[item], style="info_qa")


class DocQa(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                split = "val"
            if split == "test":
                src = join(DOCQA_SOURCE, f"{split}_v1.0.json")
            else:
                src = join(DOCQA_SOURCE, f"{split}_v1.0_withQT.json")
            if not exists(src):
                raise ValueError(
                    "DocQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 1)"
                    f" please download and unzip the data into `{DOCQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        if self.split == "test":
            src = join(DOCQA_SOURCE, f"{split}_v1.0.json")
        else:
            src = join(DOCQA_SOURCE, f"{split}_v1.0_withQT.json")
        logging.info(f"Loading docqa data from {src}")
        with open(cached_path(src, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            assert ex.pop("data_split") == split
            image_path = join(DOCQA_SOURCE, ex["image"])
            if self.split == "test":
                for k in ["answers", "question_types"]:
                    assert k not in ex
                    ex[k] = []
            out.append(dict(
                image=join(DOCQA_SOURCE, ex["image"]),
                question=ex["question"],
                answers=ex.get("answers"),
                metadata=dict(
                    doc_id=ex["docId"],
                    question_types=ex.get("question_types"),
                    example_id=ex["questionId"],
                ),
            ))
        return out

    def get(self, item, rng):
        return dict(self.data[item], style="doc_qa")


class ChartQa(DatasetBase):
    def __init__(self, split, parts="both", weighted=False, use_exp=False):
        self.weighted = weighted
        assert split in ["train", "validation", "test"]
        assert parts in ["human", "augmented", "both"]
        self.parts = parts
        self.use_exp = use_exp
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        examples = []
        if self.parts == "both":
            parts = ["human", "augmented"]
        else:
            parts = [self.parts]
        for part in parts:
            src = f"{CHARTQA_SOURCE}/{split}/{split}_{part}.json"
            logging.info(f"Loading chartqa data from {src}")
            with open(cached_path(src, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
                data = json.load(f)
            for ex_id, ex in enumerate(data):
                ex = dict(
                    image=join(CHARTQA_SOURCE, split, "png", ex.pop("imgname")),
                    question=ex["query"],
                    answers=ex["label"],
                    metadata=dict(
                        is_human=part == "human",
                        example_id=ex_id
                    )
                )
                examples.append(ex)
        return examples

    def get(self, item, rng):
        ex = dict(self.data[item], style="chart_qa_exp" if self.use_exp else "chart_qa")
        if self.weighted:
            is_human = ex["metadata"]["is_human"]
            # Weight to balanced human/augmented sets
            if is_human:
                w = 2*20901/(20901+7398)
            else:
                w = 2*7398/(20901+7398)
            ex["weight"] = w
        return ex


class SceneTextQa(DatasetBase):

    @classmethod
    def download(cls, n_procs=1):
        for split in ["train", "test"]:
            if not exists(join(join(ST_QA_SRC, f"{split}_task_3.json"))):
                raise ValueError(
                    "SceneTextQa requires manually downloading https://rrc.cvc.uab.es/?ch=11"
                    f" please download and unzip the data into `{ST_QA_SRC}`"
                )

    def __init__(self, split):
        assert split in ["train", "test", "validation"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "train"
        src = join(ST_QA_SRC, f"{self.split}_task_3.json")
        logging.info(f"Loading scene text data from {src}")
        with open(src) as f:
            data = json.load(f)["data"]
        out = []
        for question in data:
            out.append(dict(
                image=join(ST_QA_SRC, question["file_path"]),
                question=question["question"],
                metadata=dict(example_id=question["question_id"]),
                answers=question.get("answers", []),
            ))
        if self.split in ["train", "validation"]:
            # Custom val split since the data doesn't have one
            out.sort(key=lambda x: x["metadata"]["example_id"])
            np.random.RandomState(63069).shuffle(out)
            if self.split == "train":
                return out[1024:]
            else:
                return out[:1024]
        else:
            return out

    def get(self, item, rng):
        return dict(self.data[item], style="st_qa")