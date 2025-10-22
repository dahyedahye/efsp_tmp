"""
HiDF 데이터셋을 위한 DeepfakeBench JSON 파일 생성 스크립트
경로 문제 수정 버전
"""
import json
import os
from pathlib import Path
from glob import glob

def create_hidf_json(hidf_root, deepfakebench_root):
    """
    HiDF 데이터셋을 위한 JSON 파일 생성
    
    Args:
        hidf_root: HiDF_split 폴더의 절대 경로
        deepfakebench_root: DeepfakeBench 프로젝트 루트 경로
    """
    
    # JSON 파일이 저장될 디렉토리 생성
    json_dir = os.path.join(deepfakebench_root, "preprocessing", "dataset_json")
    os.makedirs(json_dir, exist_ok=True)
    
    dataset_info = {
        "HiDF": {
            "HiDF_Real": {
                "train": {},
                "test": {}
            },
            "HiDF_Fake": {
                "train": {},
                "test": {}
            }
        }
    }
    
    # Train 데이터 처리
    train_real_path = os.path.join(hidf_root, "Train", "Real-img")
    train_fake_path = os.path.join(hidf_root, "Train", "Fake-img")
    
    print(f"Looking for train real images in: {train_real_path}")
    print(f"Looking for train fake images in: {train_fake_path}")
    
    # Real train 이미지 처리
    real_images = glob(os.path.join(train_real_path, "*.png"))
    print(f"Found {len(real_images)} real training images")
    
    for idx, img_path in enumerate(real_images):
        video_name = f"real_train_{idx:06d}"
        # 절대 경로로 저장
        dataset_info["HiDF"]["HiDF_Real"]["train"][video_name] = {
            "label": "HiDF_Real",
            "frames": [img_path]  # 전체 절대 경로
        }
    
    # Fake train 이미지 처리
    fake_images = glob(os.path.join(train_fake_path, "*.jpg"))
    print(f"Found {len(fake_images)} fake training images")
    
    for idx, img_path in enumerate(fake_images):
        video_name = f"fake_train_{idx:06d}"
        dataset_info["HiDF"]["HiDF_Fake"]["train"][video_name] = {
            "label": "HiDF_Fake",
            "frames": [img_path]
        }
    
    # Test 데이터 처리
    test_real_path = os.path.join(hidf_root, "Test", "Real-img")
    test_fake_path = os.path.join(hidf_root, "Test", "Fake-img")
    
    print(f"Looking for test real images in: {test_real_path}")
    print(f"Looking for test fake images in: {test_fake_path}")
    
    # Real test 이미지 처리
    real_test_images = glob(os.path.join(test_real_path, "*.png"))
    print(f"Found {len(real_test_images)} real test images")
    
    for idx, img_path in enumerate(real_test_images):
        video_name = f"real_test_{idx:06d}"
        dataset_info["HiDF"]["HiDF_Real"]["test"][video_name] = {
            "label": "HiDF_Real",
            "frames": [img_path]
        }
    
    # Fake test 이미지 처리
    fake_test_images = glob(os.path.join(test_fake_path, "*.jpg"))
    print(f"Found {len(fake_test_images)} fake test images")
    
    for idx, img_path in enumerate(fake_test_images):
        video_name = f"fake_test_{idx:06d}"
        dataset_info["HiDF"]["HiDF_Fake"]["test"][video_name] = {
            "label": "HiDF_Fake",
            "frames": [img_path]
        }
    
    # JSON 파일 저장
    output_path = os.path.join(json_dir, "HiDF.json")
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 통계 출력
    print("=" * 50)
    print("HiDF Dataset JSON Creation Complete!")
    print("=" * 50)
    print(f"Train Real Images: {len(real_images)}")
    print(f"Train Fake Images: {len(fake_images)}")
    print(f"Test Real Images: {len(real_test_images)}")
    print(f"Test Fake Images: {len(fake_test_images)}")
    print(f"JSON saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # 경로를 실제 환경에 맞게 수정하세요
    HIDF_ROOT = "/dev/shm/dahye_tmp/datasets/HiDF/HiDF_split"  # HiDF_split 폴더의 절대 경로
    DEEPFAKEBENCH_ROOT = "/data/users/dahye/workspace/iclr/baseline/effort/Effort-AIGI-Detection/DeepfakeBench"  # DeepfakeBench 프로젝트 루트
    
    # JSON 파일 생성
    json_path = create_hidf_json(HIDF_ROOT, DEEPFAKEBENCH_ROOT)
    print(f"\nJSON file created at: {json_path}")
    print("\nNext steps:")
    print("1. Update train_config.yaml and test_config.yaml with HiDF labels")
    print("2. Update the dataset_json_folder path in config files")
    print("3. Run the training script")