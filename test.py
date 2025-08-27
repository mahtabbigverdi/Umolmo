# from olmo.tokenizer import build_tokenizer

# # 1. Build tokenizer for Qwen
# tok = build_tokenizer("Qwen/Qwen2.5-3B", pad_tokenizer_to=151936    )

# 2. Text with some of the special tokens you defined


# import json
# with open('/gscratch/krishna/mahtab/Umolmo/Data/torch_datasets/frozen-lake-action-safe-single-image-tag-20k-cot/train/train.json' , 'r') as f:
#     data = json.load(f)

# max_len = 0
# for el in data:
#     text = el['query'] + el['label']
#     tokens = tok.encode(text)
#     print(len(tokens))
#     if len(tokens) > max_len:
#         max_len = len(tokens)
# print("Max Token Length in Dataset:", max_len)

# import numpy as np
# data = np.load("../sideUmolmo/VSP_planning_BFS_text.npy", allow_pickle=True).item()
# max_len = 0
# for el in data.values():
#     text = el
#     tokens = tok.encode(text)
#     print(len(tokens))
#     if len(tokens) > max_len:
#         max_len = len(tokens)
# print("Max Token Length in Dataset:", max_len)

import json
with open('//mmfs1/gscratch/krishna/mahtab/Umolmo/Data/torch_datasets/frozen-lake-action-plan-single-image-tag-10k-cot/train/train.json' , 'r') as f:
    data = json.load(f)
len(data)

import numpy as np
d = np.load("../sideUmolmo/VSP_planning_BFS_text.npy", allow_pickle=True).item()

for el in data:
    idx = el['id']
    el['label'] = d[idx]
    data[el['id']] = el

with open('//mmfs1/gscratch/krishna/mahtab/Umolmo/Data/torch_datasets/frozen-lake-action-plan-single-image-tag-10k-cot/train/train.json', "w") as f:
    json.dump(data, f, indent=4)