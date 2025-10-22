"""
train_config.yaml과 test_config.yaml의 경로와 레이블 업데이트 스크립트
"""
import yaml
import os

def update_config_files(deepfakebench_root):
    """
    Config 파일들을 업데이트하여 HiDF 데이터셋 지원 추가
    """
    
    # Config 파일 경로
    train_config_path = os.path.join(deepfakebench_root, "training/config/train_config.yaml")
    test_config_path = os.path.join(deepfakebench_root, "training/config/test_config.yaml")
    
    # 1. train_config.yaml 업데이트
    print("Updating train_config.yaml...")
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # dataset_json_folder 경로를 프로젝트 내부 경로로 변경
    train_config['dataset_json_folder'] = os.path.join(deepfakebench_root, "preprocessing/dataset_json")
    
    # HiDF 레이블 추가
    train_config['label_dict']['HiDF_Real'] = 0
    train_config['label_dict']['HiDF_Fake'] = 1
    
    # 저장
    with open(train_config_path, 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated: {train_config_path}")
    
    # 2. test_config.yaml 업데이트
    print("Updating test_config.yaml...")
    with open(test_config_path, 'r') as f:
        test_config = yaml.safe_load(f)
    
    # dataset_json_folder 경로를 프로젝트 내부 경로로 변경
    test_config['dataset_json_folder'] = os.path.join(deepfakebench_root, "preprocessing/dataset_json")
    
    # HiDF 레이블 추가
    test_config['label_dict']['HiDF_Real'] = 0
    test_config['label_dict']['HiDF_Fake'] = 1
    
    # 저장
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated: {test_config_path}")
    
    print("\nConfig files updated successfully!")
    print(f"dataset_json_folder set to: {os.path.join(deepfakebench_root, 'preprocessing/dataset_json')}")
    print("HiDF labels added to label_dict")

if __name__ == "__main__":
    # DeepfakeBench 프로젝트 루트 경로
    DEEPFAKEBENCH_ROOT = "/data/users/dahye/workspace/iclr/baseline/effort/Effort-AIGI-Detection/DeepfakeBench"
    
    update_config_files(DEEPFAKEBENCH_ROOT)