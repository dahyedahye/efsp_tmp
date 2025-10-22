#!/usr/bin/env python
"""
Prepare HiDF dataset for training with DeepfakeBench
- Creates train/test split
- Generates JSON configuration file
- Updates label dictionary
- Creates training script
"""

import os
import json
import yaml
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


class HiDFDatasetPreparer:
    """Prepare HiDF dataset for DeepfakeBench training"""
    
    def __init__(self, dataset_path: str, output_base: str = "./datasets/HiDF", 
                 train_ratio: float = 0.8, seed: int = 42):
        """
        Args:
            dataset_path: Path to original HiDF dataset (containing Fake-img and Real-img)
            output_base: Base path for organized dataset
            train_ratio: Ratio for train/test split
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.output_base = Path(output_base)
        self.train_ratio = train_ratio
        self.seed = seed
        
        random.seed(seed)
        
        # Validate dataset structure
        self.fake_dir = self.dataset_path / "Fake-img"
        self.real_dir = self.dataset_path / "Real-img"
        
        if not self.fake_dir.exists() or not self.real_dir.exists():
            raise FileNotFoundError(
                f"Expected 'Fake-img' and 'Real-img' directories in {dataset_path}"
            )
    
    def get_image_files(self) -> Tuple[List[Path], List[Path]]:
        """Get all image files from dataset"""
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        fake_images = sorted([
            f for f in self.fake_dir.iterdir() 
            if f.suffix.lower() in img_exts
        ])
        real_images = sorted([
            f for f in self.real_dir.iterdir() 
            if f.suffix.lower() in img_exts
        ])
        
        print(f"📊 Found {len(fake_images)} fake images and {len(real_images)} real images")
        return fake_images, real_images
    
    def split_dataset(self, images: List[Path]) -> Tuple[List[Path], List[Path]]:
        """Split images into train and test sets"""
        # Shuffle for random split
        images_copy = images.copy()
        random.shuffle(images_copy)
        
        # Calculate split point
        train_size = int(len(images_copy) * self.train_ratio)
        
        train_images = images_copy[:train_size]
        test_images = images_copy[train_size:]
        
        return train_images, test_images
    
    def organize_dataset_structure(self):
        """
        Organize dataset in DeepfakeBench structure:
        datasets/HiDF/
        ├── frames/
        │   ├── train/
        │   │   ├── HiDF-fake/
        │   │   │   └── video_xxx/
        │   │   │       └── 00000.png
        │   │   └── HiDF-real/
        │   │       └── video_xxx/
        │   │           └── 00000.png
        │   └── test/
        │       ├── HiDF-fake/
        │       └── HiDF-real/
        """
        print("\n📁 Organizing dataset structure...")
        
        # Create directory structure
        frames_dir = self.output_base / "frames"
        
        # Get all images
        fake_images, real_images = self.get_image_files()
        
        # Split into train/test
        fake_train, fake_test = self.split_dataset(fake_images)
        real_train, real_test = self.split_dataset(real_images)
        
        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(fake_train)} fake, {len(real_train)} real")
        print(f"  Test:  {len(fake_test)} fake, {len(real_test)} real")
        
        # Copy files to new structure
        dataset_info = {
            "train": {
                "HiDF-fake": fake_train,
                "HiDF-real": real_train
            },
            "test": {
                "HiDF-fake": fake_test,
                "HiDF-real": real_test
            }
        }
        
        file_mappings = {}
        
        for split in ["train", "test"]:
            for label in ["HiDF-fake", "HiDF-real"]:
                images = dataset_info[split][label]
                
                print(f"\n📋 Processing {split}/{label}: {len(images)} images")
                
                for idx, img_path in enumerate(tqdm(images, desc=f"{split}/{label}")):
                    # Create video folder (each image as single-frame video)
                    video_name = f"video_{idx:06d}"
                    video_dir = frames_dir / split / label / video_name
                    video_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy image as frame 00000
                    dest_path = video_dir / f"00000{img_path.suffix}"
                    
                    # Option 1: Copy files (safer, uses more space)
                    shutil.copy2(img_path, dest_path)
                    
                    # Option 2: Create symbolic links (saves space, requires permission)
                    # dest_path.symlink_to(img_path.absolute())
                    
                    # Store mapping for JSON creation
                    key = f"{split}_{label}_{video_name}"
                    file_mappings[key] = {
                        "original": str(img_path),
                        "new": str(dest_path),
                        "video_name": video_name,
                        "label": label,
                        "split": split
                    }
        
        return file_mappings
    
    def create_dataset_json(self, file_mappings: Dict):
        """Create JSON configuration file for DeepfakeBench"""
        print("\n📝 Creating dataset JSON configuration...")
        
        # Initialize dataset structure
        dataset_json = {
            "HiDF": {
                "HiDF-fake": {
                    "train": {},
                    "test": {}
                },
                "HiDF-real": {
                    "train": {},
                    "test": {}
                }
            }
        }
        
        # Populate JSON structure
        for key, info in file_mappings.items():
            split = info["split"]
            label = info["label"]
            video_name = info["video_name"]
            
            dataset_json["HiDF"][label][split][video_name] = {
                "label": label,
                "frames": [info["new"]]
            }
        
        # Save JSON file
        json_path = Path("./preprocessing/dataset_json/HiDF.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print(f"✅ Dataset JSON saved to {json_path}")
        return json_path
    
    def update_config_files(self):
        """Update configuration files with HiDF dataset information"""
        print("\n⚙️ Updating configuration files...")
        
        # Update train_config.yaml
        train_config_path = Path("./training/config/train_config.yaml")
        if train_config_path.exists():
            with open(train_config_path, 'r') as f:
                train_config = yaml.safe_load(f)
            
            # Add HiDF labels
            if 'label_dict' not in train_config:
                train_config['label_dict'] = {}
            
            train_config['label_dict']['HiDF-fake'] = 1
            train_config['label_dict']['HiDF-real'] = 0
            
            # Add dataset root if needed
            train_config['dataset_root_rgb'] = str(self.output_base.parent)
            
            with open(train_config_path, 'w') as f:
                yaml.dump(train_config, f, default_flow_style=False)
            
            print(f"✅ Updated {train_config_path}")
        
        # Update test_config.yaml
        test_config_path = Path("./training/config/test_config.yaml")
        if test_config_path.exists():
            with open(test_config_path, 'r') as f:
                test_config = yaml.safe_load(f)
            
            if 'label_dict' not in test_config:
                test_config['label_dict'] = {}
            
            test_config['label_dict']['HiDF-fake'] = 1
            test_config['label_dict']['HiDF-real'] = 0
            
            test_config['dataset_root_rgb'] = str(self.output_base.parent)
            
            with open(test_config_path, 'w') as f:
                yaml.dump(test_config, f, default_flow_style=False)
            
            print(f"✅ Updated {test_config_path}")
    
    def create_training_config(self):
        """Create specific training configuration for HiDF"""
        print("\n📋 Creating HiDF training configuration...")
        
        config = {
            # Model settings
            'model_name': 'effort',
            'backbone_name': 'vit',
            'pretrained': 'no need',
            
            # Dataset settings
            'all_dataset': ['HiDF'],
            'train_dataset': ['HiDF'],
            'test_dataset': ['HiDF'],
            
            # Training parameters
            'nEpochs': 10,
            'start_epoch': 0,
            'train_batchSize': 32,
            'test_batchSize': 32,
            'workers': 8,
            'frame_num': {'train': 1, 'test': 1},  # Single frame per "video"
            'resolution': 224,
            'with_mask': False,
            'with_landmark': False,
            
            # Data augmentation
            'use_data_augmentation': True,
            'data_aug': {
                'flip_prob': 0.5,
                'rotate_prob': 0.5,
                'rotate_limit': [-10, 10],
                'blur_prob': 0.5,
                'blur_limit': [3, 7],
                'brightness_prob': 0.5,
                'brightness_limit': [-0.1, 0.1],
                'contrast_limit': [-0.1, 0.1],
                'quality_lower': 40,
                'quality_upper': 100
            },
            
            # Normalization (CLIP)
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
            
            # Optimizer
            'optimizer': {
                'type': 'adam',
                'adam': {
                    'lr': 0.0002,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8,
                    'weight_decay': 0.0005,
                    'amsgrad': False
                }
            },
            
            # Learning rate scheduler
            'lr_scheduler': 'step',
            'lr_step': 3,
            'lr_gamma': 0.5,
            
            # Logging
            'save_epoch': 2,
            'rec_iter': 100,
            'log_dir': './logs/hidf_training',
            'save_ckpt': True,
            'save_feat': False,
            
            # Loss and metrics
            'loss_func': 'cross_entropy',
            'metric_scoring': 'auc',
            
            # System
            'ngpu': 1,
            'cuda': True,
            'cudnn': True,
            'manualSeed': 42
        }
        
        # Save configuration
        config_path = Path("./training/config/detector/effort_hidf.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Training configuration saved to {config_path}")
        return config_path
    
    def create_training_scripts(self):
        """Create training and testing scripts"""
        print("\n📜 Creating training scripts...")
        
        # Training script
        train_script = """#!/bin/bash
# HiDF Training Script

echo "🚀 Starting HiDF training..."

# Training from scratch
python training/train.py \\
    --detector_path ./training/config/detector/effort_hidf.yaml \\
    --train_dataset HiDF \\
    --test_dataset HiDF

# For fine-tuning from pretrained weights, uncomment below:
# python training/train.py \\
#     --detector_path ./training/config/detector/effort_hidf.yaml \\
#     --train_dataset HiDF \\
#     --test_dataset HiDF \\
#     --resume ./weights/effort_ckpt.pth \\
#     --finetune

echo "✅ Training completed!"
"""
        
        with open("train_hidf.sh", "w") as f:
            f.write(train_script)
        os.chmod("train_hidf.sh", 0o755)
        
        # Testing script
        test_script = """#!/bin/bash
# HiDF Testing Script

echo "🔍 Starting HiDF testing..."

# Test the trained model
python training/test.py \\
    --detector_path ./training/config/detector/effort_hidf.yaml \\
    --test_dataset HiDF \\
    --weights_path ./logs/hidf_training/test/HiDF/ckpt_best.pth

echo "✅ Testing completed!"
"""
        
        with open("test_hidf.sh", "w") as f:
            f.write(test_script)
        os.chmod("test_hidf.sh", 0o755)
        
        print("✅ Created train_hidf.sh and test_hidf.sh")
    
    def prepare_all(self):
        """Execute all preparation steps"""
        print("\n" + "="*60)
        print("🎯 HiDF Dataset Preparation for DeepfakeBench")
        print("="*60)
        
        # Step 1: Organize dataset structure
        file_mappings = self.organize_dataset_structure()
        
        # Step 2: Create JSON configuration
        self.create_dataset_json(file_mappings)
        
        # Step 3: Update config files
        self.update_config_files()
        
        # Step 4: Create training configuration
        self.create_training_config()
        
        # Step 5: Create training scripts
        self.create_training_scripts()
        
        print("\n" + "="*60)
        print("✅ HiDF Dataset Preparation Complete!")
        print("="*60)
        print("\n📚 Next Steps:")
        print("1. Start training:    ./train_hidf.sh")
        print("2. Monitor progress:  tensorboard --logdir ./logs/hidf_training")
        print("3. Test the model:    ./test_hidf.sh")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HiDF dataset for DeepfakeBench training"
    )
    parser.add_argument(
        "--dataset_path", 
        required=True,
        help="Path to HiDF dataset (containing Fake-img and Real-img)"
    )
    parser.add_argument(
        "--output_path",
        default="./datasets/HiDF",
        help="Output path for organized dataset (default: ./datasets/HiDF)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symbolic links instead of copying files (saves space)"
    )
    
    args = parser.parse_args()
    
    # Create preparer and execute
    preparer = HiDFDatasetPreparer(
        dataset_path=args.dataset_path,
        output_base=args.output_path,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    preparer.prepare_all()


if __name__ == "__main__":
    main()