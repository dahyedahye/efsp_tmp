# 기본 패키지
pip install numpy==1.21.5
pip install scipy==1.7.3
pip install pandas==1.4.2
pip install Pillow==9.0.1
pip install scikit-learn==1.0.2
pip install scikit-image==0.19.2

# 이미지 처리
pip install opencv-python==4.6.0.66
pip install imageio==2.9.0
pip install imgaug==0.4.0
pip install albumentations==1.1.0
pip install imutils==0.5.4

# 시각화
pip install seaborn==0.11.2
pip install tqdm==4.61.0

# 기타 유틸리티
pip install pyyaml==6.0
pip install setuptools==59.5.0
pip install simplejson
pip install filterpy
pip install fvcore

# PyTorch (기존 설치 확인)
python -c "import torch; print(f'PyTorch installed: {torch.__version__}')" || \
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 딥러닝 관련
pip install efficientnet-pytorch==0.7.1
pip install timm==0.6.12
pip install segmentation-models-pytorch==0.3.2
pip install torchtoolbox==0.1.8.2
pip install tensorboard==2.10.1
pip install kornia

# LoRA, einops, transformers
pip install loralib
pip install einops
pip install transformers

# CLIP (마지막에 설치)
pip install git+https://github.com/openai/CLIP.git

# dlib (설치 실패 가능성 있음)
pip install dlib==19.24.0 || echo "dlib installation failed, continuing..."