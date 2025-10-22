"""
Test 전용 데이터셋을 위한 DeepfakeBench JSON 파일 생성 스크립트
사용자 제공 경로 구조(real/ / fake/)에 맞게 수정됨
"""
import json
import os
from glob import glob

def create_testset_json(testset_root, deepfakebench_root, dataset_name):
    """
    Test 전용 데이터셋을 위한 JSON 파일 생성
    
    Args:
        testset_root (str): 'real', 'fake' 폴더가 있는 테스트셋 루트의 절대 경로
        deepfakebench_root (str): DeepfakeBench 프로젝트 루트 경로
        dataset_name (str): JSON 파일과 내부 키에서 사용할 데이터셋 이름
    """
    
    # JSON 파일이 저장될 디렉토리 생성
    json_dir = os.path.join(deepfakebench_root, "preprocessing", "dataset_json")
    os.makedirs(json_dir, exist_ok=True)
    
    # JSON 구조 초기화 (DeepfakeBench 형식에 맞게 train은 비워둠)
    dataset_info = {
        dataset_name: {
            f"{dataset_name}_Real": {
                "train": {},
                "test": {}
            },
            f"{dataset_name}_Fake": {
                "train": {},
                "test": {}
            }
        }
    }
    
    # --- Test 데이터 처리 ---
    # 사용자가 제공한 'tree' 구조에 맞게 경로 설정
    test_real_path = os.path.join(testset_root, "real")
    test_fake_path = os.path.join(testset_root, "fake")
    
    print(f"Looking for test real images in: {test_real_path}")
    print(f"Looking for test fake images in: {test_fake_path}")
    
    # Real test 이미지 처리 (확장자 .png)
    real_test_images = glob(os.path.join(test_real_path, "*.png"))
    print(f"Found {len(real_test_images)} real test images")
    
    for idx, img_path in enumerate(real_test_images):
        video_name = f"real_test_{idx:06d}"
        dataset_info[dataset_name][f"{dataset_name}_Real"]["test"][video_name] = {
            "label": f"{dataset_name}_Real",
            "frames": [img_path]  # 전체 절대 경로
        }
    
    # Fake test 이미지 처리 (확장자 .png)
    fake_test_images = glob(os.path.join(test_fake_path, "*.png"))
    print(f"Found {len(fake_test_images)} fake test images")
    
    for idx, img_path in enumerate(fake_test_images):
        video_name = f"fake_test_{idx:06d}"
        dataset_info[dataset_name][f"{dataset_name}_Fake"]["test"][video_name] = {
            "label": f"{dataset_name}_Fake",
            "frames": [img_path]
        }
    
    # JSON 파일 저장
    output_path = os.path.join(json_dir, f"{dataset_name}.json")
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 통계 출력
    print("=" * 50)
    print(f"{dataset_name} Test Dataset JSON Creation Complete!")
    print("=" * 50)
    print(f"Test Real Images: {len(real_test_images)}")
    print(f"Test Fake Images: {len(fake_test_images)}")
    print(f"JSON saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # --- 경로를 실제 환경에 맞게 수정하세요 ---
    
    # 1. Testset 루트 경로 (사용자가 제공한 경로)
    # 'real'과 'fake' 폴더가 있는 상위 디렉토리입니다.
    TESTSET_ROOT = "/data/dataset/AI-Hub_6000_retinaface_processed"
    
    # 2. DeepfakeBench 프로젝트 루트
    DEEPFAKEBENCH_ROOT = "/data/users/dahye/workspace/iclr/baseline/effort/Effort-AIGI-Detection/DeepfakeBench"
    
    # 3. 생성할 JSON 파일의 이름 (데이터셋 이름)
    # 예: "TestDataset_1_DeepFakeCeleb"
    DATASET_NAME = "AI-Hub_6000_retinaface_processed"
    
    # ----------------------------------------
    
    # JSON 파일 생성
    json_path = create_testset_json(TESTSET_ROOT, DEEPFAKEBENCH_ROOT, DATASET_NAME)
    
    print(f"\nJSON file created at: {json_path}")
    print("\nNext steps:")
    print(f"1. Update test_config.yaml with {DATASET_NAME} labels (e.g., {DATASET_NAME}_Real, {DATASET_NAME}_Fake)")
    print("2. Update the dataset_json_folder path in config files")
    print("3. Run the testing script")