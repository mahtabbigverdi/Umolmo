# Defines a CUDA-enabled Docker image suitable for running this project's experiments
# via beaker-gantry.

FROM ubuntu:20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV TZ="America/Los_Angeles"
ARG DEBIAN_FRONTEND="noninteractive"

# Install conda
RUN  apt-get update && \
     apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        jq \
        language-pack-en \
        make \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        sudo \
        unzip \
        vim \
        wget \
        fish \
        parallel \
        iputils-ping \
        htop \
        zsh \
        rsync \
        tmux \
        ca-certificates \
        libxml2-dev \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    apt-get clean
ENV PATH /opt/conda/bin:$PATH


# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade wheel

# Install a few additional utilities via pip
RUN /opt/conda/bin/pip install --no-cache-dir \
    gpustat \
    jupyter \
    beaker-gantry \
    oocmap

# Ensure users can modify their container environment.
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Make the base image friendlier for interactive workloads. This makes things like the man command
# work.
RUN yes | unminimize

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ENV MOFED_VER 5.8-1.1.2.1
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz

# Install dependencies
RUN pip install pyopenssl --upgrade
RUN conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
COPY . ./src
RUN pip install --no-cache-dir -e "./src[all]" --upgrade --force-reinstall --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip uninstall -y ai2-olmo && \
    rm -rf src/
RUN pip install --upgrade fabric dataclasses tqdm cloudpickle smart_open[gcs] func_timeout aioredis==1.3.1
RUN pip uninstall -y tensorflow tensorflow-cpu
RUN pip install --no-cache-dir tensorflow-cpu
RUN pip uninstall -y protobuf
RUN pip install --no-binary=protobuf protobuf==3.20.3
RUN pip install httplib2 chardet
RUN pip uninstall -y uvloop
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN conda clean -ay
RUN mkdir -p /root/.config/gcloud
COPY application_default_credentials.json /root/.config/gcloud/application_default_credentials.json

WORKDIR /app/olmo