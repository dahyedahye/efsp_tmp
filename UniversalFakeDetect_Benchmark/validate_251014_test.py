import argparse
import os
import math
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
from random import shuffle
import shutil
from scipy.ndimage.filters import gaussian_filter
import pandas as pd # pandas 라이브러리 추가
from collections import OrderedDict

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073],
    "beitv2": [0.485, 0.456, 0.406],
    "siglip": [0.5, 0.5, 0.5],
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711],
    "beitv2": [0.229, 0.224, 0.225],
    "siglip": [0.5, 0.5, 0.5],
}


def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"
    N = y_true.shape[0]
    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2
    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1
        temp[temp<thres] = 0
        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc
    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


# 'validate' 함수 수정: 개별 이미지 결과 저장을 위해 'result_folder'와 'subset_name' 인자 추가
def validate(model, loader, result_folder, subset_name, find_thres=False):
    with torch.no_grad():
        y_true, y_pred = [], []
        print (f"'{subset_name}' 데이터셋 평가 중... (총 {len(loader.dataset)}개 이미지)")
        for img, label in tqdm(loader, desc=f"  - 추론 진행"):
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    image_paths = loader.dataset.total_list # 이미지 경로 목록 가져오기

    # --- 개별 이미지 결과 저장 ---
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'label': y_true.astype(int),
        'score': y_pred
    })
    
    # 결과를 CSV 파일로 저장
    individual_results_path = os.path.join(result_folder, f'{subset_name}_scores.csv')
    results_df.to_csv(individual_results_path, index=False)
    print(f"  - 개별 이미지 결과가 '{individual_results_path}'에 저장되었습니다.")
    # -----------------------------

    # Get AP
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return {'AP': ap, 'ACC': acc0}

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    # 서브셋별 요약 결과 반환
    return {'AP': ap, 'ACC_0.5': acc0, 'ACC_best': acc1, 'Best_Threshold': best_thres}


def recursively_read(rootdir, must_contain, classes=[], exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[-1].lower() in exts)  and  (must_contain in os.path.join(r, file)):
                if len(classes) == 0:
                    out.append(os.path.join(r, file))
                elif os.path.join(r, file).split('/')[-3] in classes:
                    out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain='', classes=[]):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain, classes)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self,  real_path,
                        fake_path,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        real_list = get_list(real_path)
        fake_list = get_list(fake_path)
        
        # 실제 이미지는 항상 fake 이미지 수와 동일하게 샘플링
        if len(real_list) > len(fake_list):
            real_list = random.sample(real_list, len(fake_list))
        
        self.total_list = real_list + fake_list
        shuffle(self.total_list)

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        if arch.lower().startswith("imagenet"):
            stat_from = "imagenet"
        elif arch.lower().startswith("clip"):
            stat_from = "clip"
        elif arch.lower().startswith("siglip"):
            stat_from = "siglip"
        elif arch.lower().startswith("beitv2"):
            stat_from = "beitv2"
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: translate_duplicate(img, 256)),
            transforms.CenterCrop(224) if stat_from != "siglip" else transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)
        img = self.transform(img)
        return img, label


# --- 새로운 메인 실행 블록 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- 모델 및 기본 설정 ---
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14_svd', help='사용할 모델 아키텍처')
    parser.add_argument('--ckpt', type=str, required=True, help='학습된 모델 체크포인트 경로')
    parser.add_argument('--use_svd', action='store_true', help='SVD 모델을 사용할 경우 지정')
    parser.add_argument('--batch_size', type=int, default=128, help='추론 시 배치 크기')
    
    # --- 데이터셋 및 결과 경로 설정 ---
    parser.add_argument('--dataset_root', type=str, required=True, help="'images' 폴더가 있는 상위 경로 (예: /dev/shm/dahye_tmp/datasets/LDMFakeDetect/dataset)")
    parser.add_argument('--result_folder', type=str, default='results/inference_results', help='결과가 저장될 폴더')

    opt = parser.parse_args()

    # 결과 폴더 생성
    os.makedirs(opt.result_folder, exist_ok=True)

    # 모델 로딩
    print("모델을 로딩합니다...")
    model = get_model(opt.arch, opt)
    state_dict = torch.load(opt.ckpt, map_location='cpu')

    # 체크포인트에서 실제 모델 가중치 추출
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # 'module.' 접두사 제거
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 'module.' 제거
        new_state_dict[name] = v

    # strict=False 옵션으로 누락된 키는 무시하고 가중치 로드
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    model.cuda()
    print("모델 로딩 완료.")

    # 데이터셋 경로 설정
    image_base_path = os.path.join(opt.dataset_root, 'images')
    real_path = os.path.join(image_base_path, '0_real')
    
    # '0_real'을 제외한 모든 하위 폴더를 fake 데이터셋으로 간주
    fake_subsets = [d for d in os.listdir(image_base_path) if os.path.isdir(os.path.join(image_base_path, d)) and d != '0_real']

    # 전체 결과를 저장할 리스트
    summary_results = []
    
    # 각 서브셋에 대해 평가 진행
    for subset in fake_subsets:
        set_seed()
        fake_path = os.path.join(image_base_path, subset)

        dataset = RealFakeDataset(real_path, fake_path, opt.arch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        
        # 수정된 validate 함수 호출
        subset_summary = validate(model, loader, opt.result_folder, subset, find_thres=True)
        subset_summary['Dataset'] = subset # 데이터셋 이름 추가
        summary_results.append(subset_summary)
        print("-" * 50)

    # --- 최종 결과 종합 및 출력 ---
    print("\n" + "=" * 25 + " 최종 요약 결과 " + "=" * 25)
    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df[['Dataset', 'AP', 'ACC_0.5', 'ACC_best', 'Best_Threshold']] # 컬럼 순서 정리
    
    # 평균 계산
    mean_scores = summary_df[['AP', 'ACC_0.5', 'ACC_best']].mean()
    mean_df = pd.DataFrame(mean_scores).T
    mean_df['Dataset'] = 'Average'
    
    # 최종 DataFrame 생성 및 출력
    final_df = pd.concat([summary_df, mean_df], ignore_index=True)
    print(final_df.to_string(index=False, float_format="%.4f"))

    # 최종 결과를 CSV 파일로 저장
    summary_csv_path = os.path.join(opt.result_folder, 'summary_results.csv')
    final_df.to_csv(summary_csv_path, index=False, float_format="%.4f")
    print(f"\n최종 요약 결과가 '{summary_csv_path}'에 저장되었습니다.")
    print("=" * 70)