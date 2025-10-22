"""
HiDF 데이터셋 설정 및 학습 실행을 위한 통합 스크립트
"""
import os
import sys
import json
import yaml
from pathlib import Path
from glob import glob
import subprocess

# ===== 설정 부분 - 여기를 수정하세요 =====
HIDF_ROOT = "/dev/shm/dahye_tmp/datasets/HiDF/HiDF_split"  # ⚠️ HiDF_split 폴더의 절대 경로로 변경
DEEPFAKEBENCH_ROOT = "/data/users/dahye/workspace/iclr/baseline/effort/Effort-AIGI-Detection/DeepfakeBench"
# ==========================================

def create_hidf_json():
    """Step 1: HiDF JSON 파일 생성"""
    print("\n" + "="*60)
    print("Step 1: Creating HiDF dataset JSON file...")
    print("="*60)
    
    json_dir = os.path.join(DEEPFAKEBENCH_ROOT, "preprocessing", "dataset_json")
    os.makedirs(json_dir, exist_ok=True)
    
    dataset_info = {
        "HiDF": {
            "HiDF_Real": {"train": {}, "test": {}},
            "HiDF_Fake": {"train": {}, "test": {}}
        }
    }
    
    # Process training data
    for split in ["Train", "Test"]:
        for class_name, ext in [("Real-img", "*.png"), ("Fake-img", "*.jpg")]:
            path = os.path.join(HIDF_ROOT, split, class_name)
            images = glob(os.path.join(path, ext))
            
            label = "HiDF_Real" if "Real" in class_name else "HiDF_Fake"
            split_lower = split.lower()
            
            print(f"Found {len(images)} images in {path}")
            
            for idx, img_path in enumerate(images):
                video_name = f"{label.lower()}_{split_lower}_{idx:06d}"
                dataset_info["HiDF"][label][split_lower][video_name] = {
                    "label": label,
                    "frames": [img_path]
                }
    
    # Save JSON
    output_path = os.path.join(json_dir, "HiDF.json")
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ JSON file created: {output_path}")
    return output_path

def update_config_files():
    """Step 2: Config 파일 업데이트"""
    print("\n" + "="*60)
    print("Step 2: Updating config files...")
    print("="*60)
    
    for config_name in ["train_config.yaml", "test_config.yaml"]:
        config_path = os.path.join(DEEPFAKEBENCH_ROOT, "training/config", config_name)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update paths and labels
        config['dataset_json_folder'] = os.path.join(DEEPFAKEBENCH_ROOT, "preprocessing/dataset_json")
        config['label_dict']['HiDF_Real'] = 0
        config['label_dict']['HiDF_Fake'] = 1
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Updated: {config_path}")

def create_detector_config():
    """Step 3: Detector config 파일 생성"""
    print("\n" + "="*60)
    print("Step 3: Creating detector config...")
    print("="*60)
    
    detector_config = {
        'log_dir': './logs/hidf_effort',
        'pretrained': 'no need',
        'model_name': 'effort',
        'backbone_name': 'vit',
        'backbone_config': {
            'mode': 'original',
            'num_classes': 2,
            'inc': 3,
            'dropout': False
        },
        'all_dataset': ['HiDF'],
        'train_dataset': ['HiDF'],
        'test_dataset': ['HiDF'],
        'compression': 'c23',
        'train_batchSize': 16,  # 메모리에 따라 조정
        'test_batchSize': 16,
        'workers': 4,
        'frame_num': {'train': 1, 'test': 1},
        'resolution': 224,
        'with_mask': False,
        'with_landmark': False,
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
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711],
        'optimizer': {
            'type': 'adam',
            'adam': {
                'lr': 0.0001,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'weight_decay': 0.0005,
                'amsgrad': False
            }
        },
        'lr_scheduler': None,
        'nEpochs': 10,
        'start_epoch': 0,
        'save_epoch': 1,
        'rec_iter': 100,
        'logdir': './logs',
        'manualSeed': 1024,
        'save_ckpt': True,
        'save_feat': False,
        'loss_func': 'cross_entropy',
        'losstype': None,
        'metric_scoring': 'auc',
        'ngpu': 1,
        'cuda': True,
        'cudnn': True,
        'save_avg': False
    }
    
    detector_path = os.path.join(DEEPFAKEBENCH_ROOT, "training/config/detector/effort_hidf.yaml")
    with open(detector_path, 'w') as f:
        yaml.dump(detector_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Detector config created: {detector_path}")
    return detector_path

def verify_setup():
    """Step 4: 설정 검증"""
    print("\n" + "="*60)
    print("Step 4: Verifying setup...")
    print("="*60)
    
    # Check JSON file
    json_path = os.path.join(DEEPFAKEBENCH_ROOT, "preprocessing/dataset_json/HiDF.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        train_real = len(data['HiDF']['HiDF_Real']['train'])
        train_fake = len(data['HiDF']['HiDF_Fake']['train'])
        test_real = len(data['HiDF']['HiDF_Real']['test'])
        test_fake = len(data['HiDF']['HiDF_Fake']['test'])
        
        print(f"✓ JSON file exists")
        print(f"  - Train: {train_real} real, {train_fake} fake")
        print(f"  - Test: {test_real} real, {test_fake} fake")
    else:
        print(f"✗ JSON file not found: {json_path}")
        return False
    
    # Check config files
    for config_name in ["train_config.yaml", "test_config.yaml"]:
        config_path = os.path.join(DEEPFAKEBENCH_ROOT, "training/config", config_name)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'HiDF_Real' in config['label_dict'] and 'HiDF_Fake' in config['label_dict']:
            print(f"✓ {config_name}: HiDF labels found")
        else:
            print(f"✗ {config_name}: HiDF labels missing")
            return False
    
    return True

def run_training():
    """Step 5: 학습 실행"""
    print("\n" + "="*60)
    print("Step 5: Starting training...")
    print("="*60)
    
    os.chdir(DEEPFAKEBENCH_ROOT)
    
    cmd = [
        "python", "training/train.py",
        "--detector_path", "./training/config/detector/effort_hidf.yaml",
        "--train_dataset", "HiDF",
        "--test_dataset", "HiDF"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("\nTraining output:")
    print("-" * 40)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    
    return True

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("HiDF Dataset Setup and Training Script")
    print("="*60)
    
    # Check paths
    if not os.path.exists(HIDF_ROOT):
        print(f"❌ Error: HiDF dataset not found at: {HIDF_ROOT}")
        print("Please update HIDF_ROOT variable in this script!")
        return
    
    if not os.path.exists(DEEPFAKEBENCH_ROOT):
        print(f"❌ Error: DeepfakeBench not found at: {DEEPFAKEBENCH_ROOT}")
        return
    
    # Run setup steps
    try:
        create_hidf_json()
        update_config_files()
        detector_path = create_detector_config()
        
        if verify_setup():
            print("\n" + "="*60)
            print("✅ Setup completed successfully!")
            print("="*60)
            
            response = input("\nDo you want to start training now? (y/n): ")
            if response.lower() == 'y':
                run_training()
            else:
                print("\nTo run training later, use:")
                print(f"cd {DEEPFAKEBENCH_ROOT}")
                print(f"python training/train.py --detector_path ./training/config/detector/effort_hidf.yaml --train_dataset HiDF --test_dataset HiDF")
        else:
            print("\n❌ Setup verification failed. Please check the errors above.")
    
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()