import numpy as np
import cv2
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as pil_image
# import dlib  # (dlib is imported but not used in the script, can be removed)
import os
import json  # Added import
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
from collections import defaultdict
from sklearn import metrics
from tqdm import tqdm
import csv

# Import detector modules
from trainer.trainer import Trainer
from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    data, label = data_dict['image'], data_dict['label']
    data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    predictions = model(data_dict, inference=True)
    return predictions


def load_detector(detector_cfg: str, weights: str):
    """Load the deepfake detector model"""
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


def preprocess_face(img_path: str) -> torch.Tensor:
    """Preprocess image for the model"""
    # Read image
    if isinstance(img_path, str) or isinstance(img_path, Path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_path
    
    # Resize to 224x224
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])
    
    return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)


@torch.inference_mode()
def infer_single_image(img_path: str, model) -> Tuple[int, float]:
    """Infer a single image"""
    face_tensor = preprocess_face(img_path).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)
    
    # Handle different output formats
    cls_out = preds["cls"]
    prob_out = preds["prob"]
    
    # Ensure we get scalar values
    if cls_out.dim() > 1:
        # If cls_out is [batch_size, num_classes], get the predicted class
        _, cls_out = torch.max(cls_out, dim=1)
    
    # Convert to scalar
    cls_val = cls_out.squeeze().cpu().item()
    prob_val = prob_out.squeeze().cpu().item()
    
    return cls_val, prob_val


def load_images_from_json(json_path: str, dataset_name: str, num_samples: int) -> Dict[str, List[Path]]:
    """Load and sample image paths from the dataset JSON file."""
    print(f"  • Loading from JSON: {json_path}")
    print(f"  • Using dataset key: {dataset_name}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not read or parse JSON file: {json_path}. Error: {e}")

    # Check if dataset_name exists
    if dataset_name not in data:
        raise KeyError(f"Dataset name '{dataset_name}' not found in JSON file. Available keys: {list(data.keys())}")

    dataset_data = data[dataset_name]
    real_key = f"{dataset_name}_Real"
    fake_key = f"{dataset_name}_Fake"

    if real_key not in dataset_data or fake_key not in dataset_data:
        raise KeyError(f"Expected keys '{real_key}' and '{fake_key}' not found in JSON for dataset '{dataset_name}'")

    # Extract 'test' paths
    real_videos = dataset_data[real_key].get("test", {})
    fake_videos = dataset_data[fake_key].get("test", {})
    
    if not real_videos and not fake_videos:
        print("⚠️  Warning: No 'test' entries found in the JSON for this dataset.")
        
    # Get all image paths (assuming 1 frame per entry as per JSON creation script)
    # Use Path() to create Path objects, as expected by test_on_samples
    real_image_paths = [Path(video_data["frames"][0]) for video_data in real_videos.values() if video_data["frames"]]
    fake_image_paths = [Path(video_data["frames"][0]) for video_data in fake_videos.values() if video_data["frames"]]

    print(f"  • Total fake images found in JSON: {len(fake_image_paths)}")
    print(f"  • Total real images found in JSON: {len(real_image_paths)}")
    
    if num_samples == -1:
        # Use all images if num_samples is -1
        num_fake_samples = len(fake_image_paths)
        num_real_samples = len(real_image_paths)
        print("  • Using all available test images (-1 specified).")
    else:
        # Sample images
        num_fake_samples = min(num_samples, len(fake_image_paths))
        num_real_samples = min(num_samples, len(real_image_paths))
    
    sampled_fake = random.sample(fake_image_paths, num_fake_samples)
    sampled_real = random.sample(real_image_paths, num_real_samples)
    
    return {
        'fake': sampled_fake,
        'real': sampled_real
    }


def print_statistics(results: List[Dict], subset_name: str = "Test"):
    """Print detailed statistics for the results"""
    
    if not results:
        print(f"\n⚠️  No successful results to analyze")
        return
    
    print(f"\n{'=' * 40}")
    print(f" {subset_name} Summary ")
    print('=' * 40)
    
    # Extract data
    probs = np.array([r['prob'] for r in results])
    labels = np.array([r['label'] for r in results])
    preds = (probs > 0.5).astype(int)
    
    # Basic statistics
    print("\n📊 Statistics:")
    print(f"  • Total samples processed: {len(results)}")
    
    # Ground truth distribution
    num_actual_fake = np.sum(labels == 1)
    num_actual_real = np.sum(labels == 0)
    print(f"\n📍 Ground Truth Distribution:")
    print(f"  • Actual FAKE: {num_actual_fake}/{len(results)} ({100*num_actual_fake/len(results):.1f}%)")
    print(f"  • Actual REAL: {num_actual_real}/{len(results)} ({100*num_actual_real/len(results):.1f}%)")
    
    # Detection results
    num_detected_fake = np.sum(preds == 1)
    num_detected_real = np.sum(preds == 0)
    total = len(results)
    
    print(f"\n🎯 Detection Results (threshold=0.5):")
    print(f"  • Detected as FAKE: {num_detected_fake}/{total} ({100*num_detected_fake/total:.1f}%)")
    print(f"  • Detected as REAL: {num_detected_real}/{total} ({100*num_detected_real/total:.1f}%)")
    
    # Metrics
    if len(np.unique(labels)) > 1:  # Only calculate if we have both classes
        accuracy = metrics.accuracy_score(labels, preds)
        ap = metrics.average_precision_score(labels, probs)
        
        # Separate accuracy for real and fake
        real_mask = labels == 0
        fake_mask = labels == 1
        
        real_acc = metrics.accuracy_score(labels[real_mask], preds[real_mask]) if real_mask.any() else 0
        fake_acc = metrics.accuracy_score(labels[fake_mask], preds[fake_mask]) if fake_mask.any() else 0
        
        print(f"\n📈 Performance Metrics:")
        print(f"  • Overall Accuracy: {accuracy:.4f} ({np.sum(preds == labels)}/{total})")
        print(f"  • Real Accuracy: {real_acc:.4f} ({np.sum((preds == labels) & real_mask)}/{np.sum(real_mask)})")
        print(f"  • Fake Accuracy: {fake_acc:.4f} ({np.sum((preds == labels) & fake_mask)}/{np.sum(fake_mask)})")
        print(f"  • Average Precision (AP): {ap:.4f}")
        
        # Calculate AUC if possible
        try:
            auc = metrics.roc_auc_score(labels, probs)
            print(f"  • AUC: {auc:.4f}")
        except:
            pass
        
        # Confusion Matrix
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Real | Fake")
        print(f"  Actual Real:   {np.sum((labels==0) & (preds==0)):4d} | {np.sum((labels==0) & (preds==1)):4d}")
        print(f"  Actual Fake:   {np.sum((labels==1) & (preds==0)):4d} | {np.sum((labels==1) & (preds==1)):4d}")

    # Probability distribution
    print(f"\n📈 Probability Distribution (Fake Score):")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = np.sum((probs >= low) & (probs <= high if high == 1.0 else probs < high))
        pct = 100 * count / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  [{low:.1f}-{high:.1f}]: {count:4d} images ({pct:5.1f}%) {bar}")
    
    # Top 5 and Bottom 5
    sorted_results = sorted(results, key=lambda x: x['prob'], reverse=True)
    
    print(f"\n🔝 Top 5 (Most likely to be fake):")
    for r in sorted_results[:min(5, len(sorted_results))]:
        gt = "FAKE" if r['label'] == 1 else "REAL"
        pred = "FAKE" if r['prob'] > 0.5 else "REAL"
        correct = "✓" if (r['label'] == 1 and r['prob'] > 0.5) or (r['label'] == 0 and r['prob'] <= 0.5) else "✗"
        print(f"  {r['filename']:30s} | Prob: {r['prob']:.4f} | GT: {gt:4s} | Pred: {pred:4s} {correct}")
    
    print(f"\n🔻 Bottom 5 (Least likely to be fake):")
    for r in sorted_results[-min(5, len(sorted_results)):]:
        gt = "FAKE" if r['label'] == 1 else "REAL"
        pred = "FAKE" if r['prob'] > 0.5 else "REAL"
        correct = "✓" if (r['label'] == 1 and r['prob'] > 0.5) or (r['label'] == 0 and r['prob'] <= 0.5) else "✗"
        print(f"  {r['filename']:30s} | Prob: {r['prob']:.4f} | GT: {gt:4s} | Pred: {pred:4s} {correct}")
    
    print('=' * 40)


def test_on_samples(model, sampled_images: Dict[str, List[Path]]):
    """Test model on sampled images"""
    results = []
    errors = []
    
    num_fake_to_test = len(sampled_images['fake'])
    num_real_to_test = len(sampled_images['real'])
    
    print(f"\n🔍 Testing on {num_fake_to_test} fake and {num_real_to_test} real samples...")
    
    # Process fake images
    if num_fake_to_test > 0:
        print("\nProcessing FAKE images...")
        successful_fake = 0
        for img_path in tqdm(sampled_images['fake'], desc="Fake"):
            try:
                cls, prob = infer_single_image(str(img_path), model)
                results.append({
                    'filename': img_path.name,
                    'path': str(img_path),
                    'prob': float(prob),
                    'pred': int(cls),
                    'label': 1  # Fake
                })
                successful_fake += 1
            except Exception as e:
                error_msg = f"Error processing {img_path.name}: {str(e)}"
                errors.append(error_msg)
        print(f"  ✓ Successfully processed: {successful_fake}/{num_fake_to_test}")
    
    # Process real images
    if num_real_to_test > 0:
        print("\nProcessing REAL images...")
        successful_real = 0
        for img_path in tqdm(sampled_images['real'], desc="Real"):
            try:
                cls, prob = infer_single_image(str(img_path), model)
                results.append({
                    'filename': img_path.name,
                    'path': str(img_path),
                    'prob': float(prob),
                    'pred': int(cls),
                    'label': 0  # Real
                })
                successful_real += 1
            except Exception as e:
                error_msg = f"Error processing {img_path.name}: {str(e)}"
                errors.append(error_msg)
        print(f"  ✓ Successfully processed: {successful_real}/{num_real_to_test}")
    
    if errors:
        print(f"\n⚠️  Total errors: {len(errors)}")
        if len(errors) <= 10:
            for err in errors:
                print(f"  • {err}")
        else:
            print(f"  • (Showing first 10 errors) {errors[0]}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test deepfake detector on dataset samples from JSON")
    
    # --- MODIFIED ARGUMENTS ---
    parser.add_argument("--json_path", required=True,
                        help="Path to the dataset JSON file (e.g., TestDataset_1_DeepFakeCeleb.json)")
    parser.add_argument("--dataset_name", required=True,
                        help="Name of the dataset key in the JSON file (e.g., TestDataset_1_DeepFakeCeleb)")
    # --- END MODIFIED ARGUMENTS ---
    
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to test from each class (default: 20). Use -1 for all images.")
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
    """Save results to CSV file"""
    try:
        with open(output_path, 'w', newline='') as csvfile:
            # Write header
            csvfile.write("image_path,label,score\n")
            
            # Write data
            for r in results:
                csvfile.write(f"{r['path']},{r['label']},{r['prob']:.6f}\n")
        
        print(f"💾 CSV results saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving CSV to {output_path}: {e}")


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"🚀 Starting Dataset Testing from JSON")
    print(f"📄 JSON File: {args.json_path}")
    print(f"🔑 Dataset Name: {args.dataset_name}")
    print(f"🔢 Samples per class: {'ALL' if args.num_samples == -1 else args.num_samples}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"🖥️  Device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = load_detector(args.detector_config, args.weights)
    
    # Sample images from dataset
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
        
    print(f"  • Will test {num_fake} fake samples")
    print(f"  • Will test {num_real} real samples")
    
    # Test on samples
    results = test_on_samples(model, sampled_images)
    
    # Print statistics
    if results:
        test_summary_name = f"{args.dataset_name} Test"
        print_statistics(results, subset_name=test_summary_name)
        
        # Save results to CSV if specified
        if args.output_csv:
            save_results_to_csv(results, args.output_csv)
        
        # Save detailed results to text file if requested
        if args.save_results:
            output_file = f"{args.dataset_name}_test_results_seed{args.seed}.txt"
            with open(output_file, 'w') as f:
                f.write(f"{args.dataset_name} Test Results\n")
                f.write(f"========================\n\n")
                f.write(f"JSON File: {args.json_path}\n")
                f.write(f"Samples (Fake/Real): {num_fake}/{num_real}\n")
                f.write(f"Model weights: {args.weights}\n")
                f.write(f"Random seed: {args.seed}\n\n")
                
                f.write("Detailed Results:\n")
                f.write("-" * 80 + "\n")
                # Sort by prob (high to low)
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