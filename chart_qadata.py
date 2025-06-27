from datasets import DatasetDict, Dataset, DatasetInfo, Features, Value
from datasets.arrow_dataset import Dataset as ArrowDataset
import pyarrow as pa
import os

# Path to your cached folder (adjust this!)
base_path = "/gscratch/krishna/mahtab/Umolmo/huggingface/datasets/HuggingFaceM4___chart_qa/default/0.0.0"
data_dir = os.path.join(base_path, "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5")

# Define paths to Arrow files
train_files = [
    os.path.join(data_dir, f"chart_qa-train-0000{i}-of-00003.arrow") for i in range(3)
]
val_file = os.path.join(data_dir, "chart_qa-val.arrow")
test_file = os.path.join(data_dir, "chart_qa-test.arrow")
print(train_files)
# Load Arrow tables
train_table = pa.ipc.RecordBatchFileReader(pa.memory_map(train_files[0], "r")).read_all()
for f in train_files[1:]:
    next_table = pa.ipc.RecordBatchFileReader(pa.memory_map(f, "r")).read_all()
    train_table = pa.concat_tables([train_table, next_table])

val_table = pa.ipc.RecordBatchFileReader(pa.memory_map(val_file, "r")).read_all()
test_table = pa.ipc.RecordBatchFileReader(pa.memory_map(test_file, "r")).read_all()

# Create Hugging Face datasets from Arrow tables
train_dataset = ArrowDataset(train_table)
val_dataset = ArrowDataset(val_table)
test_dataset = ArrowDataset(test_table)

# Wrap as a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print(dataset)
