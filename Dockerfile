# Python 3.8을 포함한 CUDA 11.3 이미지 사용
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8을 기본 python3로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# pip 업그레이드
RUN python3 -m pip install --upgrade pip

# PyTorch 설치 (CUDA 11.3용)
RUN pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 나머지 패키지 설치
RUN pip3 install numpy==1.21.5 \
    pandas==1.4.2 \
    Pillow==9.0.1 \
    imageio==2.9.0 \
    tqdm==4.61.0 \
    scipy==1.7.3 \
    seaborn==0.11.2 \
    pyyaml==6.0 \
    imutils==0.5.4 \
    opencv-python==4.6.0.66 \
    scikit-image==0.19.2 \
    scikit-learn==1.0.2 \
    albumentations==1.1.0 \
    efficientnet-pytorch==0.7.1 \
    timm==0.6.12 \
    segmentation-models-pytorch==0.3.2 \
    torchtoolbox==0.1.8.2 \
    tensorboard==2.10.1 \
    tensorboardX \
    setuptools==59.5.0 \
    loralib \
    einops \
    transformers \
    filterpy \
    simplejson \
    kornia \
    fvcore \
    imgaug==0.4.0

# dlib 설치
RUN pip3 install dlib==19.24.0

# CLIP 설치
RUN pip3 install git+https://github.com/openai/CLIP.git

# gdown 설치
RUN pip3 install gdown

# 프로젝트 코드 복사
COPY . /workspace/

# 필요한 디렉토리 생성
RUN mkdir -p /workspace/DeepfakeBench/training/weights \
    && mkdir -p /workspace/checkpoints \
    && mkdir -p /workspace/outputs

# 기본 명령어 설정
CMD ["/bin/bash"]