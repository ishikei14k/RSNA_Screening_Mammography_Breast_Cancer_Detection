FROM nvcr.io/nvidia/pytorch:22.02-py3
ARG PYTORCH="1.11.0"
ARG TORCHVISION="0.12.0"
ARG CUDA="113"
# Avoiding user interaction with libopencv-dev
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    vim \
    libopencv-dev

# Install python package.
ADD docker/requirements.txt  /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==${PYTORCH}+cu${CUDA} torchvision==${TORCHVISION}+cu${CUDA} -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install opencv-python==4.5.5.62 && \
    pip install opencv-python-headless==4.5.5.62 && \
    pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html && \
    pip install mmdet==2.25.1 && \
    pip install albumentations && \
    pip install timm && \
    pip install -r /tmp/requirements.txt

WORKDIR /home/working

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US