import json
import logging
from os import environ
from os.path import exists
from os.path import join
import numpy as np
import copy
import pandas as pd
import io
from PIL import Image
from collections import defaultdict
from cached_path import cached_path

from olmo.data.dataset import DATA_HOME, DatasetBase, Dataset
from olmo.io import read_file, write_json, file_exists

if DATA_HOME is not None:
    DEPTH_SOURCE = join(DATA_HOME, "depth")
    AURORA_SOURCE = join(DATA_HOME, "aurora")
else:
    DEPTH_SOURCE = None
    AURORA_SOURCE = None


class Aurora(DatasetBase):
    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        examples = []
        
        src = f"{AURORA_SOURCE}/{split}/{split}.json"
        logging.info(f"Loading aurora data from {src}")
        with open(cached_path(src, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
            data = json.load(f)
        for ex_id, ex in enumerate(data):
            ex = dict(
                image= ex.pop("imgname"),
                question=ex["query"],
                answers=ex["label"],
                image_outputs = ex['image_output_paths'],
                metadata=dict(
                    example_id=ex['id']
                )
            )
            examples.append(ex)
        
        return examples

    def get(self, item, rng):
        ex = dict(self.data[item], style="aurora")
        return ex





class Depth(DatasetBase):
    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        examples = []
        
        src = f"{DEPTH_SOURCE}/{split}/{split}.json"
        logging.info(f"Loading depth data from {src}")
        with open(cached_path(src, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
            data = json.load(f)
        for ex_id, ex in enumerate(data):
            ex = dict(
                image= ex.pop("imgname"),
                question=ex["query"],
                answers=ex["label"],
                image_outputs = ex['image_output_paths'],
                metadata=dict(
                    example_id=ex['id']
                )
            )
            examples.append(ex)
        
        return examples

    def get(self, item, rng):
        ex = dict(self.data[item], style="depth")
        return ex

