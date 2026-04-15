# Each RUN creates a new layer
ARG BASE_IMAGE=arm64v8/ubuntu:bionic-20230530

FROM --platform=linux/arm64 ${BASE_IMAGE}

ARG BRANCH_VERSION="r32.7"
ARG BRANCH_NAME="main"
ARG SOC="t210"
ARG TIMEZONE="Africa/Casablanca"

# Empêcher les questions interactives lors de l'install
ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/Paris
# ENV TZ=Africa/Dakar
# Africa/Douala
ENV TZ=${TIMEZONE}

# Empêcher Python d'écrire des fichiers .pyc et forcer l'affichage des logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances minimales pour la vidéo et CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl jq wget \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    gnupg2 ca-certificates \
    libopenblas-base libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 libegl1 \
    ffmpeg \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Installation des librairies Python pour le traitement (ex: Ultralytics/YOLO ou OpenCV seul)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir numpy ultralytics

# Dépôts NVIDIA et dépendances système
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | apt-key add - \
    && echo "deb https://repo.download.nvidia.com/jetson/common ${BRANCH_VERSION} ${BRANCH_NAME}" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/${SOC} ${BRANCH_VERSION} ${BRANCH_NAME}" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-get update -y

# https://repo.download.nvidia.com/jetson/
# https://repo.download.nvidia.com/jetson/l4t/r32.7/main/binary-aarch64/Packages 
# CUDA et Bibliothèques
RUN apt-get install -y --no-install-recommends \
    cuda-libraries-10-2 cuda-nvrtc-10-2 cuda-cudart-10-2 libcudnn8 libcublas10 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV OPENBLAS_CORETYPE=ARMV8

# Définition des variables d'environnement NVIDIA pour le runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all,utility,compute,video,graphics

#RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

WORKDIR /app
#COPY processor.py .

# Commande pour lancer votre script de traitement
CMD ["python3", "--version"]
