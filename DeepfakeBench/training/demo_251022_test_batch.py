import numpy as np
import cv2
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as pil_image
# import dlib
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
from collections import defaultdict
from sklearn import metrics
from tqdm import tqdm
import csv
import torch.utils.data as data #(NEW) DataLoader 사용을 위해 import

# Import detector modules
from trainer.trainer import Trainer
from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ... (inference, load_detector, infer_single_image 함수는 변경 없음) ...
# ... (run_batch_inference, load_images_from_json, print_statistics 함수도 변경 없음) ...


@torch.no_grad()
def inference(model, data_dict):
    data = data_dict['image']
    data_dict['image'] = data.to(device)
    
    if 'label' in data_dict and data_dict['label'] is not None:
        label = data_dict['label']
        data_dict['label'] = label.to(device)
    else:
        data_dict['label'] = None
        
    predictions = model(data_dict, inference=True)
    return predictions


def load_detector(detector_cfg: str, weights: str):
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[✓] Detector loaded.")
    return model


def preprocess_image(img_path: str) -> torch.Tensor:
    if isinstance(img_path, str) or isinstance(img_path, Path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_path
    
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])
    
    return transform(pil_image.fromarray(img_rgb))


@torch.inference_mode()
def infer_single_image(img_path: str, model) -> Tuple[int, float]:
    face_tensor = preprocess_image(img_path).unsqueeze(0).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)
    
    cls_out = preds["cls"]
    prob_out = preds["prob"]
    
    if cls_out.dim() > 1 and cls_out.shape[1] > 1:
        _, cls_out = torch.max(cls_out, dim=1)
    
    cls_val = cls_out.squeeze().cpu().item()
    prob_val = prob_out.squeeze().cpu().item()
    
    return cls_val, prob_val


@torch.inference_mode()
def run_batch_inference(model, batch_tensor: torch.Tensor) -> Tuple[List[int], List[float]]:
    batch_size = batch_tensor.shape[0]
    dummy_labels = torch.zeros(batch_size, dtype=torch.long) 
    
    data_dict = {
        'image': batch_tensor,
        'label': dummy_labels
    }
    
    preds = inference(model, data_dict)

    cls_out = preds["cls"]
    prob_out = preds["prob"]

    if cls_out.dim() > 1 and cls_out.shape[1] > 1:
        _, cls_out = torch.max(cls_out, dim=1)

    pred_classes = cls_out.reshape(-1).cpu().tolist()
    pred_probs = prob_out.reshape(-1).cpu().tolist()
    
    if len(pred_classes) != batch_size:
        print(f"Warning: Class prediction count ({len(pred_classes)}) doesn't match batch size ({batch_size}).")
    if len(pred_probs) != batch_size:
        print(f"Warning: Probability prediction count ({len(pred_probs)}) doesn't match batch size ({batch_size}).")

    return pred_classes, pred_probs


def load_images_from_json(json_path: str, dataset_name: str, num_samples: int) -> Dict[str, List[Path]]:
    print(f"  • Loading from JSON: {json_path}")
    print(f"  • Using dataset key: {dataset_name}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not read or parse JSON file: {json_path}. Error: {e}")

    if dataset_name not in data:
        raise KeyError(f"Dataset name '{dataset_name}' not found in JSON file. Available keys: {list(data.keys())}")

    dataset_data = data[dataset_name]
    real_key = f"{dataset_name}_Real"
    fake_key = f"{dataset_name}_Fake"

    if real_key not in dataset_data or fake_key not in dataset_data:
        raise KeyError(f"Expected keys '{real_key}' and '{fake_key}' not found in JSON for dataset '{dataset_name}'")

    real_videos = dataset_data[real_key].get("test", {})
    fake_videos = dataset_data[fake_key].get("test", {})
    
    if not real_videos and not fake_videos:
        print("⚠️  Warning: No 'test' entries found in the JSON for this dataset.")
        
    real_image_paths = [Path(video_data["frames"][0]) for video_data in real_videos.values() if video_data["frames"]]
    fake_image_paths = [Path(video_data["frames"][0]) for video_data in fake_videos.values() if video_data["frames"]]

    print(f"  • Total fake images found in JSON: {len(fake_image_paths)}")
    print(f"  • Total real images found in JSON: {len(real_image_paths)}")
    
    if num_samples == -1:
        num_fake_samples = len(fake_image_paths)
        num_real_samples = len(real_image_paths)
        print("  • Using all available test images (-1 specified).")
    else:
        num_fake_samples = min(num_samples, len(fake_image_paths))
        num_real_samples = min(num_samples, len(real_image_paths))
    
    sampled_fake = random.sample(fake_image_paths, num_fake_samples)
    sampled_real = random.sample(real_image_paths, num_real_samples)
    
    return {
        'fake': sampled_fake,
        'real': sampled_real
    }

# ... (print_statistics 함수는 변경 없음) ...
def print_statistics(results: List[Dict], subset_name: str = "Test"):
    if not results:
        print(f"\n⚠️  No successful results to analyze")
        return
    print(f"\n{'=' * 40}")
    print(f" {subset_name} Summary ")
    print('=' * 40)
    probs = np.array([r['prob'] for r in results])
    labels = np.array([r['label'] for r in results])
    preds = (probs > 0.5).astype(int)
    print("\n📊 Statistics:")
    print(f"  • Total samples processed: {len(results)}")
    num_actual_fake = np.sum(labels == 1)
    num_actual_real = np.sum(labels == 0)
    print(f"\n📍 Ground Truth Distribution:")
    print(f"  • Actual FAKE: {num_actual_fake}/{len(results)} ({100*num_actual_fake/len(results):.1f}%)")
    print(f"  • Actual REAL: {num_actual_real}/{len(results)} ({100*num_actual_real/len(results):.1f}%)")
    num_detected_fake = np.sum(preds == 1)
    num_detected_real = np.sum(preds == 0)
    total = len(results)
    print(f"\n🎯 Detection Results (threshold=0.5):")
    print(f"  • Detected as FAKE: {num_detected_fake}/{total} ({100*num_detected_fake/total:.1f}%)")
    print(f"  • Detected as REAL: {num_detected_real}/{total} ({100*num_detected_real/total:.1f}%)")
    if len(np.unique(labels)) > 1:
        accuracy = metrics.accuracy_score(labels, preds)
        ap = metrics.average_precision_score(labels, probs)
        real_mask = labels == 0
        fake_mask = labels == 1
        real_acc = metrics.accuracy_score(labels[real_mask], preds[real_mask]) if real_mask.any() else 0
        fake_acc = metrics.accuracy_score(labels[fake_mask], preds[fake_mask]) if fake_mask.any() else 0
        print(f"\n📈 Performance Metrics:")
        print(f"  • Overall Accuracy: {accuracy:.4f} ({np.sum(preds == labels)}/{total})")
        print(f"  • Real Accuracy: {real_acc:.4f} ({np.sum((preds == labels) & real_mask)}/{np.sum(real_mask)})")
        print(f"  • Fake Accuracy: {fake_acc:.4f} ({np.sum((preds == labels) & fake_mask)}/{np.sum(fake_mask)})")
        print(f"  • Average Precision (AP): {ap:.4f}")
        try:
            auc = metrics.roc_auc_score(labels, probs)
            print(f"  • AUC: {auc:.4f}")
        except:
            pass
        print(f"\n📊 Confusion Matrix:")
        print(f"               Predicted")
        print(f"               Real | Fake")
        print(f"   Actual Real:   {np.sum((labels==0) & (preds==0)):4d} | {np.sum((labels==0) & (preds==1)):4d}")
        print(f"   Actual Fake:   {np.sum((labels==1) & (preds==0)):4d} | {np.sum((labels==1) & (preds==1)):4d}")
    print(f"\n📈 Probability Distribution (Fake Score):")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = np.sum((probs >= low) & (probs <= high if high == 1.0 else probs < high))
        pct = 100 * count / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"   [{low:.1f}-{high:.1f}]: {count:4d} images ({pct:5.1f}%) {bar}")
    sorted_results = sorted(results, key=lambda x: x['prob'], reverse=True)
    print(f"\n🔝 Top 5 (Most likely to be fake):")
    for r in sorted_results[:min(5, len(sorted_results))]:
        gt = "FAKE" if r['label'] == 1 else "REAL"
        pred = "FAKE" if r['prob'] > 0.5 else "REAL"
        correct = "✓" if (r['label'] == 1 and r['prob'] > 0.5) or (r['label'] == 0 and r['prob'] <= 0.5) else "✗"
        print(f"   {r['filename']:30s} | Prob: {r['prob']:.4f} | GT: {gt:4s} | Pred: {pred:4s} {correct}")
    print(f"\n🔻 Bottom 5 (Least likely to be fake):")
    for r in sorted_results[-min(5, len(sorted_results)):]:
        gt = "FAKE" if r['label'] == 1 else "REAL"
        pred = "FAKE" if r['prob'] > 0.5 else "REAL"
        correct = "✓" if (r['label'] == 1 and r['prob'] > 0.5) or (r['label'] == 0 and r['prob'] <= 0.5) else "✗"
        print(f"   {r['filename']:30s} | Prob: {r['prob']:.4f} | GT: {gt:4s} | Pred: {pred:4s} {correct}")
    print('=' * 40)


# (NEW) PyTorch Dataset 클래스 정의
class TestDataset(data.Dataset):
    def __init__(self, all_paths, all_labels):
        self.all_paths = all_paths
        self.all_labels = all_labels

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.all_paths[idx]
            label = self.all_labels[idx]
            
            # preprocess_image 함수를 여기서 호출
            tensor = preprocess_image(str(img_path))
            
            filename = img_path.name
            full_path = str(img_path)
            
            return tensor, label, filename, full_path
        except Exception as e:
            # 이미지 로딩/전처리 중 오류 발생 시
            print(f"Warning: Error loading {self.all_paths[idx]}: {e}")
            return None # (NEW) 오류 발생 시 None 반환

# (NEW) None을 거르기 위한 collate_fn
def custom_collate_fn(batch):
    # batch는 __getitem__이 반환한 (tensor, label, filename, full_path) 튜플의 리스트
    # (None, None, None, None)이 아닌, None 자체를 반환했으므로, None을 필터링
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # 이 배치의 모든 항목이 오류였다면
        return None, None, None, None
    
    # torch.utils.data.dataloader.default_collate를 사용하여 정상적인 항목들을 배치로 만듦
    return data.dataloader.default_collate(batch)


# (MODIFIED) DataLoader를 사용하도록 test_on_samples 수정
def test_on_samples(model, sampled_images: Dict[str, List[Path]], batch_size: int, num_workers: int):
    """Test model on sampled images in batches using DataLoader"""
    results = []
    errors = []
    
    # 1. Combine all images and labels
    fake_paths = sampled_images['fake']
    real_paths = sampled_images['real']
    
    all_paths = fake_paths + real_paths
    all_labels = ([1] * len(fake_paths)) + ([0] * len(real_paths))
    
    num_fake = len(fake_paths)
    num_real = len(real_paths)
    total_images = len(all_paths)
    
    print(f"\n🔍 Testing on {total_images} total images ({num_fake} fake, {num_real} real)...")
    print(f"📦 Batch Size: {batch_size}")
    print(f"⚙️ Num Workers: {num_workers}")

    # 2. (NEW) Dataset 및 DataLoader 생성
    dataset = TestDataset(all_paths, all_labels)
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # 테스트 시에는 셔플 불필요
        num_workers=num_workers, # (NEW) 병렬 로딩 워커 수
        pin_memory=True, # (NEW) CPU -> GPU 전송 속도 향상
        collate_fn=custom_collate_fn # (NEW) 오류 항목을 거르기 위함
    )

    # 3. (NEW) DataLoader로 루프 실행
    for batch_data in tqdm(data_loader, desc="Testing Batches"):
        
        # collate_fn에서 (None, ...)을 반환한 경우 (배치 전체가 오류)
        if batch_data[0] is None:
            errors.append(f"Skipped an entire batch due to read errors.")
            continue

        stacked_batch, batch_labels, batch_filenames, batch_fullpaths = batch_data
        
        try:
            # 4. Run inference (이미 GPU 메모리에 올라갈 준비가 된 배치)
            batch_preds, batch_probs = run_batch_inference(model, stacked_batch)
            
            # 5. Collect results
            for path, filename, label, pred, prob in zip(batch_fullpaths, batch_filenames, batch_labels.tolist(), batch_preds, batch_probs):
                results.append({
                    'filename': filename,
                    'path': str(path),
                    'prob': float(prob),
                    'pred': int(pred),
                    'label': int(label)
                })
                
        except Exception as e:
            error_msg = f"Error during batch inference (starting with {batch_filenames[0]}): {str(e)}"
            errors.append(error_msg)

    if errors:
        print(f"\n⚠️  Total errors during processing: {len(errors)}")
        if len(errors) <= 20:
            for err in errors:
                print(f"  • {err}")
        else:
            print(f"  • (Showing first 20 errors) {errors[0]}")
            
    print(f"✓ Successfully processed {len(results)}/{total_images} images.")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test deepfake detector on dataset samples from JSON")
    
    parser.add_argument("--json_path", required=True,
                        help="Path to the dataset JSON file (e.g., TestDataset_1.json)")
    parser.add_argument("--dataset_name", required=True,
                        help="Name of the dataset key in the JSON file (e.g., TestDataset_1)")
    
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to test from each class (default: 20). Use -1 for all images.")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
                        
    # (NEW) num_workers 인자 추가
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for data loading (default: 4)")

    parser.add_argument("--detector_config", default='training/config/detector/effort.yaml',
                        help="Path to detector configuration file")
    parser.add_argument("--weights", required=True,
                        help="Path to model weights")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--save_results", action='store_true',
                        help="Save detailed results to a text file")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save CSV results (e.g., results.csv)")
    return parser.parse_args()


def save_results_to_csv(results: List[Dict], output_path: str):
    # (No changes to this function)
    try:
        with open(output_path, 'w', newline='') as csvfile:
            csvfile.write("image_path,label,score\n")
            for r in results:
                csvfile.write(f"{r['path']},{r['label']},{r['prob']:.6f}\n")
        print(f"💾 CSV results saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving CSV to {output_path}: {e}")


def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"🚀 Starting Dataset Testing from JSON")
    print(f"📄 JSON File: {args.json_path}")
    print(f"🔑 Dataset Name: {args.dataset_name}")
    print(f"🔢 Samples per class: {'ALL' if args.num_samples == -1 else args.num_samples}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"⚙️  Num Workers: {args.num_workers}") # (NEW)
    print(f"🎲 Random seed: {args.seed}")
    print(f"🖥️   Device: {device}")
    
    print("\n📦 Loading model...")
    model = load_detector(args.detector_config, args.weights)
    
    print("\n🎯 Loading and sampling images from JSON...")
    try:
        sampled_images = load_images_from_json(args.json_path, args.dataset_name, args.num_samples)
    except (FileNotFoundError, KeyError, IOError) as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("Aborting.")
        return

    num_fake = len(sampled_images['fake'])
    num_real = len(sampled_images['real'])

    if num_fake == 0 and num_real == 0:
        print("❌ No images were loaded. Please check your JSON file and paths.")
        return
        
    print(f"  • Will test {num_fake} fake samples")
    print(f"  • Will test {num_real} real samples")
    
    # (MODIFIED) args.num_workers 전달
    results = test_on_samples(model, sampled_images, args.batch_size, args.num_workers)
    
    if results:
        test_summary_name = f"{args.dataset_name} Test"
        print_statistics(results, subset_name=test_summary_name)
        
        if args.output_csv:
            save_results_to_csv(results, args.output_csv)
        
        if args.save_results:
            output_file = f"{args.dataset_name}_test_results_seed{args.seed}.txt"
            with open(output_file, 'w') as f:
                f.write(f"{args.dataset_name} Test Results\n")
                f.write(f"========================\n\n")
                f.write(f"JSON File: {args.json_path}\n")
                f.write(f"Samples (Fake/Real): {num_fake}/{num_real}\n")
                f.write(f"Batch Size: {args.batch_size}\n")
                f.write(f"Num Workers: {args.num_workers}\n") # (NEW)
                f.write(f"Model weights: {args.weights}\n")
                f.write(f"Random seed: {args.seed}\n\n")
                
                f.write("Detailed Results:\n")
                f.write("-" * 80 + "\n")
                for r in sorted(results, key=lambda x: x['prob'], reverse=True):
                    gt = "FAKE" if r['label'] == 1 else "REAL"
                    pred = "FAKE" if r['prob'] > 0.5 else "REAL"
                    correct = "CORRECT" if (r['label'] == 1 and r['prob'] > 0.5) or (r['label'] == 0 and r['prob'] <= 0.5) else "WRONG"
                    f.write(f"{r['filename']:35s} | Prob: {r['prob']:.6f} | GT: {gt:4s} | Pred: {pred:4s} | {correct}\n")
            
            print(f"💾 Detailed results saved to {output_file}")
    else:
        print("\n❌ No successful results to analyze. Please check your dataset and model.")


if __name__ == "__main__":
    main()