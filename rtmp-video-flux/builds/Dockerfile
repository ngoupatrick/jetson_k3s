# Each RUN creates a new layer
ARG BASE_IMAGE=arm64v8/ubuntu:bionic-20230530

FROM --platform=linux/arm64 ${BASE_IMAGE}

# Empêcher les questions interactives lors de l'install
ENV DEBIAN_FRONTEND=noninteractive

# Empêcher Python d'écrire des fichiers .pyc et forcer l'affichage des logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Installation des dépendances minimales pour la vidéo et CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl jq wget \
    python3-pip \
    python3-opencv \
    libopencv-dev \
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

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra

# Installation des librairies Python pour le traitement (ex: Ultralytics/YOLO ou OpenCV seul)
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir numpy

# Définition des variables d'environnement NVIDIA pour le runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all,utility,compute,video,graphics

WORKDIR /app
#COPY processor.py .

# Commande pour lancer votre script de traitement
CMD ["python3", "--version"]
