<div align="center">
  <img src="assets/Molmo-logo.svg" alt="Molmo Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>Molmo: Multimodal Open Language Model</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/mm_olmo/blob/release/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://molmo.allenai.org/blog">
    <img alt="Blog Post" src="https://img.shields.io/badge/Molmo-blog-F0529C">
  </a>
  <a href="https://arxiv.org/pdf/2409.17146">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2409.17146-blue">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b">
    <img alt="PixMo (Datasets)" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-PixMo (Datasets)-yellow">
  </a>
</p>

Molmo is a repository for training and using Ai2's state-of-the-art multimodal open language models.


## Installation
We recommend using python 3.10.
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install dependencies, run:

```bash
git clone https://github.com/allenai/molmo.git
cd molmo
pip install -e .[all]
```

For training and evaluating MolmoE-1B, please install megablocks by running `pip install git+https://github.com/Muennighoff/megablocks.git@olmoe`.

For running on beaker, `Dockerfile` can build an image to use. See [beaker](##Beaker).

## Data Downloading and Setup
If you are using Ai2 compute, you can probably skip this part as the data is already downloaded.
See running with [beaker](##Beaker)

Molmo uses huggingface datasets for most data, therefore most 
data will be stored in the default huggingface cache. See [here](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
for how to set it. Some additional data is stored separately in the path
set by `MOLMO_DATA_DIR`. 

For example, if you want to store the data in `/data/molmo` you could set

```bash
export MOLMO_DATA_DIR=/data/molmo
export HF_HOME=/data/molmo/huggingface
```

Data can then be downloaded with:

```bash
python3 scripts/download.py all --n_proc 12
```

Downloading the pixmo datasets requires downloading images from URLs. The download script
will do this automatically, but it will take some time.
Downloading everything from scratch can take up to a day.
More processes can make it faster, but it also increases the risk of getting rate-limited.

Downloading can be resumed if canceled or an error occurs mid-download.

Some datasets (InfoQa and Scene-Text) require manually downloading the files.
The download scripts will throw an error if those files are not found.

Downloading the android control dataset requires additional dependencies
since it requires parsing the original tfrecords.

To download a specific dataset pass in the dataset name run:
```bash
python3 scripts/download_data.py ChartQa --n_proc 12
```

## Visualizing Data
Once downloaded, datasets can be visualized by using `scripts/dataset_visualize.py` script:

```bash
python3 scripts/dataset_visualize.py chart_qa /path/to/viz/dir
```

## Trained Models
On weka, our existing model can be found in `/weka/oe-training-default/mm-olmo/released-models-0924/`


## Captioner Evaluation
We generally evaluate captioning offline on the `dense_caption_eval` task, the prediction file
can be built with:

`torchrun --nproc-per-node 8 launch_scripts/eval_captioner.py /weka/oe-training-default/mm-olmo/released-models-0924/qwen2-7b-dense-captioner`

Then the eval script can be run like this (the OPENAI_API_KEY must be set in the environment)

`python3 scripts/gpt_dense_caption_eval.py /weka/oe-training-default/chrisc/cockatoo/models/dense-captioner-v21-olmo1.8/lr3-9-3/predictions-ck14700-dense_caption_eval-validation/predictions.json --sample 1500 --metrics all`

The `eval_captioner.py` script also supports computing the cross-entropy loss on the val set:

`torchrun --nproc-per-node 8 launch_scripts/eval_captioner.py /weka/oe-training-default/mm-olmo/released-models-0924/qwen2-7b-dense-captioner --loss --seq_len=2048 --task=pixmo_cap --split=validation`

By default, results are saved to the directory containing the target checkpoints.

## Downstream Evaluation
Evaluation is done with the `launch_scripts/eval_downstream.py` script. 
FSDP can be used to evaluate large models, or for high-resolution processing. 
Note that the vLLM version of Molmo will be significantly faster for inference, but most of 
our numbers were reported using the results of this local evaluation. 

To eval on a single task pass the `task name`, or `task_name:split`:

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 text_vqa --save_to_checkpoint_dir
```

For most tasks, we evaluate with high resolution:

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 text_vqa --save_to_checkpoint_dir --high_res --fsdp --device_batch_size=2
```

The `--fsdp` flag will use FSDP which is needed for to avoid OOMs when using high resolution.

To evaluate on our default eval set (including the 11 tasks in the paper):
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 low-res --save_to_checkpoint_dir
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 high-res --save_to_checkpoint_dir --high_res --fsdp --device_batch_size=2
```

To get test numbers, use `low-res-test` and `high-res-test`. Some test numbers will require
re-formatting the prediction files and then submitting to test servers.

To evaluate the 72B model with this codebase you will need to run on multiple nodes
and might need to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

These scripts will save the metrics and predictions in the save directory. Future calls to the 
eval script will re-use cached metrics if they exist, to overwrite these cached metrics use
the `--overwrite` flag.


### Evaluation with VLMEvalkit
Evaluation of the HF models is also supported via [open-compass/VLMEvalkit](https://github.com/open-compass/VLMEvalKit). Check [PR#648](https://github.com/open-compass/VLMEvalKit/pull/648) for supported prompts and evaluation settings to reproduce results from the paper.
However a few datasets (e.g., PixMo-Count) are not supported.

## Pretrained Models for Initialization
Training end-to-end requires downloading the pre-trained models used to initialize Molmo.
This can be done with the script `scripts/convert_hf_to_molmo.py`

For example, to load the Qwen2 LLM and OpenAI CLIP model, run:

```bash
python3 scripts/convert_hf_to_molmo.py qwen2_7b
python3 scripts/convert_hf_to_molmo.py openai
```

The model will be downloaded from huggingface, converted into a compatible format,
and then saved into the `MOLMO_DATA_DIR` directory.

## Pre-Training
The main training script is `scripts/train.py`. To train a model you can either construct a config
file to pass to it, or call one of the higher-level helper scripts in `launch_scripts` which
will construct a low-level config from some higher-level settings and then invoke the train script for you.

To start a debugging run:

`torchrun --nproc-per-node=1 launch_scripts/train_captioner.py debug
--save_folder=/path/to/save/folder`

To train with the Qwen2 LLM and the CLIP vision encoder:

`WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b
--wandb.name=run_name --wandb.entity=entity --wandb.project=project --save_folder=/path/to/save/folder`

You can use other vision encoders including SigLIP, MetaCLIP and DINOv2 with the option `--vision_backbone=model_name`.

Under-the-hood, the `launch_scripts/train_captioner.py` constructs a `TrainerConfig` object 
and then runs it. For fine-grained control, CLI args can be used to override parts of
the `TrainerConfig`, for example:

To run without wandb, use:

`torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b
--wandb=null --save_folder=/path/to/save/folder`

To turn off dropout:

`torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b
--wandb=null --save_folder=/path/to/save/folder --model.residual_dropout=0`

Or to use 4 workers for the train and eval data loaders:

`torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b
--wandb=null --save_folder=/path/to/save/folder --evaluations.0.data.num_workers=4 
--evaluations.1.data.num_workers=4 --data.num_wokers=4`


## Multitask Training
Multitask training can be done with `launch_scripts/multtask_train.py`, for example:

`WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py 3.2-synthetic /path/to/checkpoint
--wandb.name=run_name --wandb.entity=entity --wandb.project=project
--save_folder=/path/to/save/folder
`

Here `3.2-synthetic` refers to what training mixture to use and `/path/to/checkpoint` points to a
model checkpoint to start from, typically a dense captioning model.

To launch a debug run:

`
torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py debug debug 
--save_folder=dbg --save_overwrite
`

## Throughput Optimization
Be default full activation checkpointing is used, you can
experiment with `--activation_checkpointing=one_in_two` to get more performance,
but I have generally found increasing the batch size is better than using less activation
checkpointing.

By default, batches are padded to fixed max sequence length, removing this can improve 
performance a bit, although I have had mixed results with it if using more then a few nodes so
it is not turned on by default.
For example, use: `torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py debug debug
--save_folder=dbg --save_overwrite --data.pad=to_128`

Using torch 2.5.1 with `torch.compile` can significant performance benefits, so far the `default`
or `max-autotune-no-cudagraphs` modes need to be used since cudagraphs can cause issues.
Generally I have not seen much benefit to using `max-autotune-no-cudagraphs` over `default` and
it can take 20+ minutes to compile, so default seems to work best.
The caption and multitask scripts now automatically compile in `default` mode.
I have not seen compilation improve inference speed.

For `torch.compile` I recommend using `dynamic=False` since if the model gets accidentally re-compiled in 
dynamic mode you will see a significant performance drop. Autoregressive decoding is 
done in a non-compiled path so it does not trigger an excessive number of re-compilations.

## Remote Files
Generally remote files path should work seamlessly as long as credentials are setup correctly.
For example, you could use:
`--save_folder=gs://mm-olmo/chrisc-models/run_name` for train scripts.

### GCP
Augusta machines start with access to some google cloud buckets (including `gs://mm-olmo/`), so you
you should not include your own GCP credentials, however if you want to write to a new bucket you will have to
give permission for the GCP service account used by augusta to write to that bucket.

On other machines credentials can be passed in through a ENV variable `GOOGLE_APPLICATION_CREDENTIALS_JSON`
so it can be set with a beaker-secret. It should contain a raw JSON google 
credential file.

Sharded checkpoints will be much more efficient on weka since they can be downloaded/uploaded
in parallel by different nodes.

### Weka
Non-cirrascale machines can still directly access weka if you setup an AWS account,
one-pass contains the needed credentials. The credential file can be passed as a raw string
to `AWS_CREDENTIALS`. Then some other flags need to be set, as documented 
[here](https://beaker-docs.apps.allenai.org/compute/data-storage.html#s3-access):

```
export WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000",
export WEKA_PROFILE="weka",
export AWS_CREDENTIALS=YOUR_CREDENTIAL_DATA
```

In this case paths like `weka://oe-training-default/chrisc/models` will work.
On Augusta, that will be slower and less reliable than using GFS.

## Preemption/Restarting
Train runs with `--allow_resume` (usually true by default) should auto-recover
if restarted as long as a checkpoint has been saved. Restarted runs will create a new wandb run entry.
Resumed runs are expected to nearly exactly match what would have happened without restarting.

Evaluations on multiple datasets will skip evaluating already evaluated dataset
as long as `--skip_if_metrics_cached` is set.


## Beaker
`Dockerfile` can be used to build a beaker image. I have one built at `chrisc/molmo-torch2.6.0-cuda12.4`.
Some gantry settings to use:
 
### Environment
Generally beaker jobs should use these flags:

```
--env HF_DATASETS_OFFLINE=1
--env OLMO_SHARED_FS=1
--env OMP_NUM_THREADS=8
--env-secret HF_ACCESS_TOKEN=YOUR_HF_KEY_SECRET_NAME
--env-secret OPENAI_API_KEY=YOUR_OPENAI_API_KEY_SECRET_NAME
```

`HF_DATASETS_OFFLINE` stops HF issues tons of requests to the HF dataset hub even though the data
is already download, I think to check the data is up-to-date.

`OLMO_SHARED_FS` tell the codes to assume, for multi-nodes jobs, are saving to a shared
file system, meaning they either save to weka or a remote FS. This could be turned off if writing 
data locally, but generally there is no reason to prefer doing that.

`HF_ACCESS_TOKEN` might be used to download the tokenizer, and
`OPENAI_API_KEY` might be used in some evaluations.

`OMP_NUM_THREADS` is for torch.

### Cirrascale machines
Setup access to the data in weka
```
--env MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo
--weka oe-training-default:/weka/oe-training-default
```

For jupiter, also set the environment variables from [here](https://beaker-docs.apps.allenai.org/experiments/distributed-training.html#ai2jupiter-cirrascale-2)

### Augusta
To run on Augusta, instead use:
```
--env MOLMO_DATA_DIR="gs://mm-olmo"
--env NCCL_TIMEOUT_MINUTES=30
 ```

The `NCCL_TIMEOUT_MINUTES` can prevent `barrier()` from timing out while 
loading/writing large files from remote storage, although a better solution is
to use sharded checkpoints only.

Be sure also set the environment variables from [here](https://beaker-docs.apps.allenai.org/compute/augusta.html)

Augusta jobs should generally use GFS to save/load models, see [remote files](###GCP).

Only some datasets are support on GFS, including all the Pixmo datasets.

### Wandb
Setup wandb and access keys (first store your keys as beaker secrets):
```
--env WANDB_ENTITY=prior-ai2 
--env WANDB_PROJECT=cockatoo
--env-secret WANDB_API_KEY=YOUR_WANDB_KEY_SECRET_NAME
```

### Experiment flags
Runs for research on Molmo can use:
```
--budget ai2/oe-training
--workspace ai2/mm-olmo
```

### Examples
I have been using the script `examples/run_gantry.py` to make launching jobs
easier, feel free to use it for reference, but do not use it directly since 
it will use my personal beaker secrets.
