# from .clip_models import ClipModel


# VALID_NAMES = {
#     'CLIP:ViT-B/16_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch16/',
#     'CLIP:ViT-B/32_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch32/',
#     'CLIP:ViT-L/14_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/', 
#     'SigLIP:ViT-L/16_256_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/google/siglip-large-patch16-256/',
#     'BEiTv2:ViT-L/16_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/BEiT-v2/beitv2_large_patch16_224_pt1k_ft21k.pth',
# }


# def get_model(name, opt):
#     assert name in VALID_NAMES.keys()
#     if name.startswith("CLIP:"):
#         return ClipModel(VALID_NAMES[name], opt)
#     else:
#         assert False 

from .clip_models import ClipModel
import os

# 우리 환경의 로컬 모델 경로
LOCAL_CLIP_PATH = "/workspace/Effort-AIGI-Detection/DeepfakeBench/training/models--openai--clip-vit-large-patch14"

# 경로가 존재하면 로컬, 없으면 Hugging Face에서 다운로드
VALID_NAMES = {
    'CLIP:ViT-B/16_svd': 'openai/clip-vit-base-patch16',
    'CLIP:ViT-B/32_svd': 'openai/clip-vit-base-patch32',
    'CLIP:ViT-L/14_svd': LOCAL_CLIP_PATH if os.path.exists(LOCAL_CLIP_PATH) else 'openai/clip-vit-large-patch14',
    'SigLIP:ViT-L/16_256_svd': 'google/siglip-large-patch16-256',
    'BEiTv2:ViT-L/16_svd': 'microsoft/beit-large-patch16-224-pt22k-ft22k',
}

def get_model(name, opt):
    assert name in VALID_NAMES.keys(), f"Invalid model name: {name}. Available: {list(VALID_NAMES.keys())}"
    
    model_path = VALID_NAMES[name]
    print(f"Loading model {name} from: {model_path}")
    
    if name.startswith("CLIP:"):
        return ClipModel(model_path, opt)
    else:
        raise NotImplementedError(f"Model type {name} not implemented")