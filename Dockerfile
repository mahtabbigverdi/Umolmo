# From https://github.com/allenai/OLMo-core/blob/e279fd8191a8cc466481f521a3cab10c27deb693/src/Dockerfile
# NOTE: make sure CUDA_VERSION and TORCH_CUDA_VERSION always match, except for punctuation
ARG CUDA_VERSION="12.4"
ARG TORCH_CUDA_VERSION="124"
ARG TORCH_VERSION="2.6.0"

#########################################################################
# Build image
#########################################################################

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-devel as build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        libxml2-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install/upgrade Python build dependencies.
RUN pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja

# Build megablocks, grouped-gemm, stanford-stk
#ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"
#ENV GROUPED_GEMM_CUTLASS="1"
#ARG MEGABLOCKS_VERSION="megablocks[gg] @ git+https://git@github.com/epwalsh/megablocks.git@epwalsh/deps"
#RUN pip wheel --no-build-isolation --no-cache-dir "${MEGABLOCKS_VERSION}"

# Build flash-attn.
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN wget ${FLASH_ATTN_WHEEL}

# Only keep the target wheels and dependencies with CUDA extensions.
RUN echo "Built wheels:" \
    && ls -lh . \
#    && ls -1 | grep -Ev 'megablocks|grouped_gemm|stanford_stk|flash_attn' | xargs rm \
    && echo "Final wheels:" \
    && ls -lh .

#########################################################################
# Stable image
#########################################################################

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime

# Install system dependencies.
# Include some that might be useful in interactive sessions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        language-pack-en \
        make \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        rsync \
        vim \
        sudo \
        unzip \
        fish \
        parallel \
        zsh \
        htop \
        tmux \
        libxml2-dev \
        git \
        wget \
        nano \
        emacs \
        libxml2-dev \
        apt-transport-https \
        gnupg \
        jq \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ENV MOFED_VER="24.01-0.3.3.1"
ENV OS_VER="ubuntu22.04"
ENV PLATFORM="x86_64"
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz

# Install/upgrade Python build dependencies.
RUN pip install --upgrade --no-cache-dir pip wheel packaging

# Install torchao.
ARG TORCH_CUDA_VERSION
ARG TORCHAO_VERSION="0.6.1"
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION} \
    torchao==${TORCHAO_VERSION}

# Copy and install wheels from build image.
COPY --from=build /app/build /app/build
RUN pip install --no-cache-dir /app/build/*

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY olmo/__init__.py olmo/__init__.py
COPY olmo/version.py olmo/version.py
RUN pip install --no-cache-dir '.[all]' && rm -rf *

# aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
 unzip awscliv2.zip && \
 sudo ./aws/install && \
 rm -rf aws

# gsutil/gcloud
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
 echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
 sudo apt-get update && sudo apt-get -y install google-cloud-cli

# Install a few additional utilities via pip
RUN /opt/conda/bin/pip install --no-cache-dir \
    gpustat \
    jupyter \
    beaker-gantry \
    oocmap

WORKDIR /app/olmo-core

