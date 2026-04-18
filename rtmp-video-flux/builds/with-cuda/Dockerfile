# purpose
# dependency management
# https://deps.dev

# Image de base certifiee NVIDIA pour JetPack 4.6.6
FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime

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

# Configuration de l'environnement pour eviter les questions interactives
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV GST_DEBUG=0

# Définition des variables d'environnement NVIDIA pour le runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all,utility,compute,video,graphics
ENV OPENBLAS_CORETYPE=ARMV8

RUN rm /etc/apt/sources.list.d/cuda.list || true && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-opencv \
    python3-gst-1.0  libv4l-dev libopencv-dev
 
RUN apt-get install -y --no-install-recommends \
    gstreamer1.0-plugins-good libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gir1.2-gst-plugins-base-1.0

RUN apt-get install -y --no-install-recommends \
    libgpgme11-dev swig build-essential

# Mise a jour des outils pip, setuptools, wheel, six et pycuda
RUN python3.6 -m pip install --upgrade --no-cache-dir pip setuptools wheel
RUN python3.6 -m pip install --no-cache-dir six
RUN python3.6 -m pip install --no-cache-dir pycuda --user

# Nettoyage des paquets de developpement pour alleger l'image
RUN apt-get purge -y python3-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libgpgme11-dev libgpg-error-dev \
        swig build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
#COPY processor.py .

# Commande pour lancer votre script de traitement
# CMD ["python3.6", "processor-with-cuda.py"]
CMD ["python3.6", "--version"]
