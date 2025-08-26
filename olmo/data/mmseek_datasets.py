import logging
from os import environ
from os.path import join

from olmo.data.dataset import DATA_HOME, DatasetBase
from olmo.io import read_json, get_cached_path, prepare_cached_data_to_local


class Aurora(DatasetBase):
    def __init__(self, split, name="aurora", include_image_outputs=True):
        assert split in ["train", "validation", "test"]
        if name in ["frozenlake_debug", "frozenlake_debug_answer", "frozenlake_debug+answer"]:
            self.dataset_name = "frozen-lake-action-safe-single-image-tag-20k-cot"
        elif name in ["frozenlake_plan", "frozenlake_plan_answer", "frozenlake_plan+answer"]:
            self.dataset_name = "frozen-lake-action-plan-single-image-tag-10k-cot"
        else:
            self.dataset_name = name
        if name == "frozenlake_debug_answer" or name == "frozenlake_plan_answer":
            include_image_outputs = False
        elif name == "frozenlake_debug+answer" or name == "frozenlake_plan+answer":
            include_image_outputs = "both"
        if split == "validation":
            split = "val"
        # if name == "aurora_small" and split == "train":
        #     split = "val"
        self.dataset_dir = join(DATA_HOME, self.dataset_name, split)
        self.include_image_outputs = include_image_outputs
        prepare_cached_data_to_local(self.dataset_dir)
        super().__init__(split)
		

    def load(self):
        split = self.split
        examples = []
        
        src = f"{self.dataset_dir}/{split}.json"
        logging.info(f"Loading {self.dataset_name} data from {src}")
        all_images = set()
        src_load_path = get_cached_path(src, cache_dir=environ.get("MOLMO_CACHE_DIR"))
        data = read_json(src_load_path)
        for ex_id, ex in enumerate(data):
            ex_new = dict(
                image= ex.pop("imgname"),
                question=ex["query"],
                answers=ex["label"],
                metadata=dict(
                    example_id=ex['id']
                )
            )
            if not self.include_image_outputs:
                ex_new['image_outputs'] = []
                ex_new["answers"] = ex_new['answers'].replace("<im_gen_start>", "")
                ex_new["answers"] = ex_new["answers"].replace("<im_gen_end>", "")
                ex_new["answers"] = ex_new["answers"].replace("Here is the visualization of the path taken by the agent in the maze:\n\n", "").strip()
                examples.append(ex_new)
            elif self.include_image_outputs == "both":
                ex_new['image_outputs'] = ex['image_output_paths']

                ex_copy = ex_new.copy()
                ex_copy['image_outputs'] = []
                ex_copy["answers"] = ex_copy['answers'].replace("<im_gen_start>", "")
                ex_copy["answers"] = ex_copy["answers"].replace("<im_gen_end>", "")
                ex_copy["answers"] = ex_copy["answers"].replace("Here is the visualization of the path taken by the agent in the maze:\n\n", "").strip()

                examples.append(ex_new)
                examples.append(ex_copy)
            else:
                ex_new['image_outputs'] = ex['image_output_paths']
                examples.append(ex_new)
                
            all_images.add(ex_new["image"])
            all_images.update(ex_new['image_outputs'])
        prepare_cached_data_to_local(list(all_images))
        
        logging.info(f"Cached {len(all_images)} unique images in {self.dataset_name} {split} split.")
        return examples

    def get(self, item, rng):
        ex = dict(self.data[item], style="aurora")
        return ex


class Depth(Aurora):
    def __init__(self, split, dataset_name="depth"):
        assert split in ["train", "validation", "test"]
        super().__init__(split, dataset_name=dataset_name)

    def get(self, item, rng):
        ex = dict(self.data[item], style="depth")
        return ex

