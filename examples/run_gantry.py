import argparse
import os
import subprocess
import sys
from datetime import datetime, time
from time import perf_counter

import click
from gantry.commands import run as gantry_run


GANTRY_HOME = "/Users/chrisc/Programming/molmo-gantry"


AUGUSTA_ENV = dict(
    LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}",
    NCCL_CROSS_NIC="0",
    NCCL_ALGO="Ring,Tree",
    NCCL_MIN_NCHANNELS="4",
    NCCL_P2P_NET_CHUNKSIZE="524288",
    NCCL_P2P_PCI_CHUNKSIZE="524288",
    NCCL_P2P_NVL_CHUNKSIZE="1048576",
    NCCL_FASTRAK_NUM_FLOWS="2",
    NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL="0",
    NCCL_BUFFSIZE="8388608",
    NCCL_FASTRAK_USE_SNAP="1",
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    NCCL_NET_GDR_LEVEL="PIX",
    NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING="0",
    NCCL_TUNER_PLUGIN="libnccl-tuner.so",
    NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS="600000",
    NCCL_NVLS_ENABLE="0",
    NCCL_DEBUG="WARN",
    NCCL_FASTRAK_CTRL_DEV="enp0s12",
    NCCL_FASTRAK_IFNAME="enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
    NCCL_SOCKET_IFNAME="enp0s12",
    NCCL_USE_SNAP="1",
    NCCL_FASTRAK_USE_LLCM="1",
    NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="/dev/aperture_devices",
    # NCCL_PROTO="Simple",
    # NCCL_TUNER_CONFIG_PATH="/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto",
    # NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE="/var/lib/tcpxo/lib64/a3plus_guest_config.textproto",
    NCCL_PROTO="Simple,LL128",
    NCCL_TUNER_CONFIG_PATH="/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto",
    NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE="/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="torchrun command to run")
    parser.add_argument("--name", help="For train scripts, wandb name/directory name")
    parser.add_argument("--group", help="For train scripts, wandb group/super-directory name")
    parser.add_argument("--beaker_name", default=None, help="Override default name")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--saturn", action="store_true")
    parser.add_argument("--augusta", action="store_true")
    parser.add_argument("--priority", default="high")
    args = parser.parse_args()

    command = args.command
    group = args.group
    name = args.name

    # This will rsync code to separate folder and commit it to an "experiments" branch in github
    # for gantry to use
    # We do this so gantry can run without filling out our working branch with small commits,
    # and since commiting our code to a separate "experiment" branch ended up being tricky
    # to implement with git and made me nervous since it can mess with our working files
    print("Syncing code")
    rsync_command = f"rsync -rzv --delete --exclude .git --exclude .gitmodules --exclude .idea --exclude '*.html' --exclude '*.pyc' --exclude __pycache__ --exclude .cache --exclude .pytest_cache --exclude '*.ipynb' /Users/chrisc/Programming/mm_olmo_dev/ {GANTRY_HOME}"
    subprocess.call(rsync_command, shell=True)
    os.chdir(GANTRY_HOME)
    status_out = subprocess.check_output(["git", "status", "-s"]).decode("utf-8")
    if status_out.strip():
        print("File have changed, pushing...")
        subprocess.call(["git", "add", "."])
        subprocess.call(["git", "commit", "-m", "update"])
        subprocess.call(["git", "push"])
    else:
        print("No changes, not comitting...")

    if args.beaker_name:
        beaker_name = args.beaker_name
    elif group is None:
        beaker_name = f"{name}"
    else:
        beaker_name = f"{group}_{name}"

    gantry_kwargs = dict(
        name=beaker_name,
        task_name=beaker_name,
        priority=args.priority,
        budget="ai2/oe-training",
        gpus=8,
        shared_memory="16GiB",
        venv="base",
        beaker_image="chrisc/molmo-torch2.6.0-cuda12.6-video",
        workspace="ai2/mm-olmo",
        gh_token_secret="CHRISC_GITHUB_TOKEN",
        conda=False,
        description=command.split("--nproc-per-node 8")[-1]
    )
    env = dict(
        HF_DATASETS_OFFLINE=1,
        OMP_NUM_THREADS=8,
        WANDB_ENTITY="prior-ai2",
        WANDB_PROJECT="cockatoo",
        LOG_FILTER_TYPE="rank0_only",
        TORCH_LOGS_RANK0="recompiles,graph_breaks",
        OLMO_NUM_THREADS_ENV_VAR=8,

        # Allows remote access to weka
        WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000",
        WEKA_PROFILE="weka",
    )
    env_secret = dict(
        OPENAI_API_KEY="CHRISC_OPENAI_API_KEY",
        WANDB_API_KEY="CHRISC_WANDB_API_KEY",
        HF_ACCESS_TOKEN="CHRISC_HF_ACCESS_TOKEN",
        AWS_CREDENTIALS="CHRISC_AWS_CREDENTIALS",
        BEAKER_TOKEN ="CHRISC_BEAKER_TOKEN_ID"
    )
    clusters = []
    rdz_id = str(perf_counter())

    if args.replicas:
        # beaker replicated experiment args
        gantry_kwargs.update(
            replicas=args.replicas,
            leader_selection=True,
            host_networking=True,
            propagate_failure=True,
            propagate_preemption=True
        )

        # hack torchrun command to add disributed args
        torch_run_args = [
            "--nnodes", f"{args.replicas}",
            # Specify the rank so beaker node 0 is always also global rank 0
            "--node-rank", "$BEAKER_REPLICA_RANK",
            "--rdzv_backend=static",  # Must be static for "--node-rank" to work
            f"--rdzv_id={rdz_id}",
            f"--rdzv_conf=\"read_timeout=600\"",
            "--rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29401"
        ]
        a, b = command.split("--nproc-per-node 8")
        command = a + " ".join(torch_run_args) + " --nproc-per-node 8" + b

        if args.saturn:
            raise ValueError("You probably don't want to do that")
        if args.augusta:
            clusters = ["ai2/augusta-google-1"]
        else:
            clusters = ["ai2/jupiter-cirrascale-2"]
    else:
        if args.augusta:
            clusters = ["ai2/augusta-google-1"]
        else:
            clusters = ["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]
        if args.saturn:
            clusters.append("ai2/saturn-cirrascale")

    if args.augusta:
        env.update(
            MOLMO_DATA_DIR="gs://mm-olmo",
            # Helpful when downloading/uploading large files to remote folders
            OLMO_SHARED_FS="1",  # Assume we are writing to a remote FS
            MOLMO_CACHE_DIR="/data/molmo-cache",
            MODEL_DIR="gs://oe-training-chrisc/molmo-models"
        )
        env.update(AUGUSTA_ENV)
        gantry_kwargs["preemptible"] = True
    else:
        gantry_kwargs["weka"] = ["oe-training-default:/weka/oe-training-default"]
        gantry_kwargs["preemptible"] = args.preemptible
        env.update(
            MOLMO_DATA_DIR="/weka/oe-training-default/mm-olmo",
            OLMO_SHARED_FS="1",
            HF_DATASETS_CACHE="/weka/oe-training-default/mm-olmo/hf_datasets",
            MODEL_DIR="/weka/oe-training-default/chrisc/cockatoo/models"
        )
        # To allow GCP access,
        # Note this is not needed for augusta since it has GCP access by default
        env.update(GCLOUD_PROJECT="ai2-oe-training",)
        env_secret.update(GOOGLE_APPLICATION_CREDENTIALS_JSON="CHRISC_GCLOUD_CREDENTIALS")
        if args.replicas:
            env.update(NCCL_SOCKET_IFNAME="ib", NCCL_IB_HCA="^=mlx5_bond_0")

    if any(x in command for x in ["train_", "train.py"]):
        # hack torchrun command to add wandb/save locations for training commands
        if "--wandb=null" not in command:
            if group:
                command += f" --wandb.group={group}"
            if name:
                command += f" --wandb.name={name}"
        if group and name:
            if args.augusta:
                command += f" --save_folder=gs://oe-training-chrisc/molmo-models/{group}/{name}"
            else:
                command += f" --save_folder=/weka/oe-training-default/chrisc/cockatoo/models/{group}/{name}"

    # Call the gantry run command programmatically from our gantry repo
    ctx = click.Context(gantry_run)
    # Trick gantry into validating our args since it checks `sys.argv` directly
    sys.argv = ["--", "/bin/bash", "-c", command]
    ctx.forward(
        gantry_run,
        arg=["/bin/bash", "-c", command],
        cluster=clusters,
        env=[f"{k}={v}" for k, v in env.items()],
        env_secret=[f"{k}={v}" for k, v in env_secret.items()],
        **gantry_kwargs,
    )


if __name__ == '__main__':
    main()