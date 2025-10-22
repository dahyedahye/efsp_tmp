import numpy as np
import cv2
import yaml
from tqdm import tqdm
from PIL import Image as pil_image
import dlib
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from detectors import DETECTOR
from imutils import face_utils
from skimage import transform as trans
import os
from pathlib import Path
import argparse
import sys
from typing import Tuple, List
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score
import random # ---! ADDED: random 라이브러리 추가 !---

# --- 스크립트 상단에 추가 ---
# 이 스크립트는 DeepfakeBench 폴더에서 실행되어야 합니다.
# detectors 모듈을 찾을 수 있도록 경로를 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    data = data_dict['image'].to(device)
    predictions = model({'image': data}, inference=True)
    return predictions


def extract_aligned_face_dlib(face_detector, predictor, image, res=224):
    def get_keypts(image, face):
        shape = predictor(image, face)
        leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
        reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
        nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
        lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
        rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
        return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    def img_align_crop(img, landmark, outsize=(res, res)):
        dst = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        dst = dst * (outsize[0] / 112.0)
        tform = trans.SimilarityTransform()
        tform.estimate(landmark.astype(np.float32), dst)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(img, M, (outsize[1], outsize[0]), borderValue=0.0)
        return warped

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if len(faces):
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = get_keypts(rgb, face)
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res))
        return cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
    return None

def load_detector(detector_cfg: str, weights: str):
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    try:
        ckpt = torch.load(weights, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        if 'model' in state:
            state = state['model']
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
    except Exception as e:
        print(f"가중치 로딩 중 오류 발생: {e}")
        if 'head.weight' in new_state_dict:
            new_state_dict['fc.weight'] = new_state_dict.pop('head.weight')
            new_state_dict['fc.bias'] = new_state_dict.pop('head.bias')
            model.load_state_dict(new_state_dict, strict=False)
        else:
            raise e

    model.eval()
    print("[✓] Detector loaded.")
    return model


def preprocess_face(img_bgr: np.ndarray, res: int):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (res, res), interpolation=cv2.INTER_LINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(pil_image.fromarray(img_resized)).unsqueeze(0)


@torch.inference_mode()
def infer_single_image(
    img_bgr: np.ndarray, model, face_detector=None, landmark_predictor=None
) -> float:
    res = model.config.get('resolution', 224)
    if face_detector and landmark_predictor:
        face_aligned = extract_aligned_face_dlib(face_detector, landmark_predictor, img_bgr, res=res)
        if face_aligned is None:
            face_aligned = img_bgr
    else:
        face_aligned = img_bgr
    
    face_tensor = preprocess_face(face_aligned, res).to(device)
    preds = inference(model, {"image": face_tensor})
    prob = preds["prob"].squeeze().cpu().item()
    return prob

def print_subset_summary(df: pd.DataFrame):
    scores = df['score']
    total_images = len(df)
    
    print("\n" + "="*10 + " Subset Summary " + "="*10)
    
    print("📊 Statistics:")
    print(f"  • Mean probability: {scores.mean():.4f}")
    print(f"  • Std deviation: {scores.std():.4f}")
    print(f"  • Min probability: {scores.min():.4f}")
    print(f"  • Max probability: {scores.max():.4f}")
    print(f"  • Median probability: {scores.median():.4f}")

    fake_count = (scores > 0.5).sum()
    real_count = total_images - fake_count
    print("\n🎯 Detection Results (threshold=0.5):")
    print(f"  • Detected as FAKE: {fake_count}/{total_images} ({fake_count/total_images:.1%})")
    print(f"  • Detected as REAL: {real_count}/{total_images} ({real_count/total_images:.1%})")
    
    print("\n📈 Probability Distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0)"]
    hist = pd.cut(scores, bins=bins, right=False, labels=labels).value_counts().sort_index()
    for label, count in hist.items():
        percentage = count / total_images
        bar = '█' * int(percentage * 50)
        print(f"  {label}: {count:4d} images ({percentage:6.1%}) {bar}")

    df_sorted = df.sort_values(by='score', ascending=False)
    print("\n🔝 Top 5 (Most likely to be fake):")
    for _, row in df_sorted.head(5).iterrows():
        filename = Path(row['image_path']).name
        label = "FAKE" if row['label'] == 1 else "REAL"
        print(f"  {filename} | Prob: {row['score']:.4f} | GT: {label}")

    print("\n🔻 Bottom 5 (Least likely to be fake):")
    for _, row in df_sorted.tail(5).iloc[::-1].iterrows():
        filename = Path(row['image_path']).name
        label = "FAKE" if row['label'] == 1 else "REAL"
        print(f"  {filename} | Prob: {row['score']:.4f} | GT: {label}")
    print("="*36 + "\n")


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def parse_args():
    p = argparse.ArgumentParser(description="Batch inference for AI-generated images using demo.py structure")
    p.add_argument("--detector_config", default='training/config/detector/effort.yaml', help="YAML config file path")
    p.add_argument("--weights", required=True, help="Pre-trained weights for the detector")
    p.add_argument("--dataset_root", required=True, help="Root directory of the dataset containing 'images' folder")
    p.add_argument("--sub_folder", default="images", choices=["images", "with_postprocessing"], help="Subfolder to test: 'images' or 'with_postprocessing'")
    p.add_argument("--result_folder", default="results/demo_batch_inference", help="Folder to save results")
    p.add_argument("--landmark_model", default=None, help="dlib 81 landmarks .dat file path. If not provided, face alignment is skipped.")
    # ---! MODIFIED: 샘플링을 위한 인자 추가 !---
    p.add_argument("--num_samples", type=int, default=None, help="Number of fake samples to test per subset. Real samples will be matched. Default is all.")
    return p.parse_args()

def main():
    args = parse_args()

    # 결과 폴더를 서브 폴더 이름에 따라 자동으로 생성
    result_folder_with_subset = os.path.join(args.result_folder, args.sub_folder)
    os.makedirs(result_folder_with_subset, exist_ok=True)
    
    model = load_detector(args.detector_config, args.weights)
    face_det, shape_predictor = None, None
    if args.landmark_model:
        try:
            face_det = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor(args.landmark_model)
            print("[✓] Dlib face detector loaded.")
        except Exception as e:
            print(f"[Error] Failed to load dlib model: {e}")
            return

    # 데이터셋 경로 설정            
    image_base_path = os.path.join(args.dataset_root, args.sub_folder)
    print(f"\nTargeting dataset folder: {image_base_path}") # 현재 어떤 폴더를 테스트하는지 출력

    real_path = os.path.join(image_base_path, '0_real')
    fake_subsets = [d for d in os.listdir(image_base_path) if os.path.isdir(os.path.join(image_base_path, d)) and d != '0_real']

    all_results_summary = []

    for subset_name in fake_subsets:
        print(f"\n--- Processing subset: {subset_name} ---")
        
        subset_results = []
        
        # 이미지 경로 로드
        real_image_paths_all = sorted([p for p in Path(real_path).glob('*') if p.suffix.lower() in IMG_EXTS])
        fake_path = os.path.join(image_base_path, subset_name)
        fake_image_paths_all = sorted([p for p in Path(fake_path).glob('*') if p.suffix.lower() in IMG_EXTS])

        # ---! MODIFIED: 샘플링 로직 추가 !---
        real_image_paths = real_image_paths_all
        fake_image_paths = fake_image_paths_all
        if args.num_samples and args.num_samples > 0:
            print(f"  - Sampling {args.num_samples} real and fake images...")
            if len(fake_image_paths_all) > args.num_samples:
                fake_image_paths = random.sample(fake_image_paths_all, args.num_samples)
            
            num_to_sample = len(fake_image_paths) # 실제 샘플링된 fake 이미지 수만큼 real 이미지 샘플링
            if len(real_image_paths_all) > num_to_sample:
                real_image_paths = random.sample(real_image_paths_all, num_to_sample)
        # -----------------------------------

        # Real 이미지 처리
        for img_path in tqdm(real_image_paths, desc=f"  - Real images"):
            img = cv2.imread(str(img_path))
            if img is None: continue
            score = infer_single_image(img, model, face_det, shape_predictor)
            subset_results.append({'image_path': str(img_path), 'label': 0, 'score': score})

        # Fake 이미지 처리
        for img_path in tqdm(fake_image_paths, desc=f"  - Fake images"):
            img = cv2.imread(str(img_path))
            if img is None: continue
            score = infer_single_image(img, model, face_det, shape_predictor)
            subset_results.append({'image_path': str(img_path), 'label': 1, 'score': score})

        if not subset_results:
            print(f"[Warning] No images found for subset: {subset_name}")
            continue

        # 1. 서브셋 처리 직후 DataFrame 생성
        subset_df = pd.DataFrame(subset_results)
        
        # 2. 서브셋별 CSV 파일 저장 경로 설정 및 저장
        subset_csv_path = os.path.join(result_folder_with_subset, f'{subset_name}_scores.csv')
        subset_df.to_csv(subset_csv_path, index=False, float_format="%.6f")
        print(f"\n[✓] Subset results saved to '{subset_csv_path}'")
        
        # 3. 터미널에 상세 요약 출력
        print_subset_summary(subset_df)

        y_true = subset_df['label']
        y_score = subset_df['score']
        ap = average_precision_score(y_true, y_score)
        acc = accuracy_score(y_true, y_score > 0.5)
        all_results_summary.append({'Dataset': subset_name, 'AP': ap, 'ACC': acc})

    if not all_results_summary:
        print("\n[Error] No subsets were processed. Please check your dataset path.")
        return

    summary_df = pd.DataFrame(all_results_summary)
    
    if len(summary_df) > 1:
        mean_scores = summary_df[['AP', 'ACC']].mean()
        mean_df = pd.DataFrame(mean_scores).T
        mean_df['Dataset'] = 'Average'
        final_df = pd.concat([summary_df, mean_df], ignore_index=True)
    else:
        final_df = summary_df
    
    print("\n" + "=" * 25 + " 최종 요약 결과 " + "=" * 25)
    print(final_df.to_string(index=False, float_format="%.4f"))
    
    # 3. 최종 요약 결과 파일 저장
    summary_csv_path = os.path.join(result_folder_with_subset, 'summary_results.csv')
    final_df.to_csv(summary_csv_path, index=False, float_format="%.4f")
    print(f"\n[✓] Final summary saved to '{summary_csv_path}'")
    print("=" * 70)


if __name__ == "__main__":
    main()