# Each RUN creates a new layer
ARG BASE_IMAGE=arm64v8/ubuntu:bionic-20230530

FROM --platform=linux/arm64 ${BASE_IMAGE}

ARG BRANCH_VERSION="r32.7"
ARG BRANCH_NAME="main"
ARG SOC="t210"
ARG TIMEZONE="Africa/Casablanca"

ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/Paris
# ENV TZ=Africa/Dakar
# Africa/Douala
ENV TZ=${TIMEZONE}

# 1. Dépôts NVIDIA et dépendances système (incluant OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 ca-certificates curl python3-pip python3-dev jq wget \
    libopenblas-base libopenmpi-dev \
    python3-opencv \
    && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | apt-key add - \
    && echo "deb https://repo.download.nvidia.com/jetson/common ${BRANCH_VERSION} ${BRANCH_NAME}" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/${SOC} ${BRANCH_VERSION} ${BRANCH_NAME}" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-get update -y

# https://repo.download.nvidia.com/jetson/
# https://repo.download.nvidia.com/jetson/l4t/r32.7/main/binary-aarch64/Packages 
# 2. CUDA et Bibliothèques
RUN apt-get install -y --no-install-recommends \
    cuda-libraries-10-2 cuda-nvcc-10-2 cuda-cudart-10-2 libcudnn8 cuda-nvtx-10-2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV OPENBLAS_CORETYPE=ARMV8

# 3. PyTorch 1.10.0
# wheel compatible: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir numpy==1.19.5 Cython Pillow==8.4.0 && \
    curl -L -C - https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -o torch-1.10.0-cp36-cp36m-linux_aarch64.whl && \
    python3 -m pip install --no-cache-dir torch-1.10.0-cp36-cp36m-linux_aarch64.whl && \
    rm torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 4. Torchvision 0.11.1 + Modèles récents
# Note: ConvNeXt et EfficientNetV2 demandent parfois une version plus récente, 
# si l'import échoue dans torchvision, on installe 'timm' qui les contient tous.
RUN python3 -m pip install --no-cache-dir torchvision==0.11.1 timm==0.6.12

WORKDIR /app

# Test de chargement d'un modèle pour vérifier
CMD ["python3", "-c", "import torch; import torchvision.models as models; m = models.resnet50(pretrained=False); print('ResNet50 loaded for test !')"]
