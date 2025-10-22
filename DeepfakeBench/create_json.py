import os
import json
from glob import glob

# --- 사용자 설정 ---
USER_CONFIG = {
    # 'images'와 'with_postprocessing'이 있는 상위 폴더 경로
    "DATASET_ROOT": "/datasets/LDMFakeDetect/dataset", 
    # 'images' 또는 'with_postprocessing' 중 선택
    "SUB_FOLDER": "with_postprocessing", 
    # JSON 파일을 저장할 위치
    "JSON_SAVE_FOLDER": "preprocessing/dataset_json/LDMFakeDetect_postproc"
}
# --------------------

def create_all_jsons(root_path, sub_folder, json_save_path):
    base_path = os.path.join(root_path, sub_folder)
    if not os.path.isdir(base_path):
        print(f"오류: '{base_path}' 경로를 찾을 수 없습니다.")
        return

    # '0_real' 폴더를 기준으로 fake 폴더 목록 찾기
    real_folder = "0_real"
    all_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    fake_folders = [d for d in all_folders if d != real_folder]

    print(f"'{base_path}' 에서 다음 Fake 폴더들을 발견했습니다: {fake_folders}")

    for fake_folder in fake_folders:
        generate_json_for_one(base_path, real_folder, fake_folder, json_save_path)

def generate_json_for_one(base_path, real_folder_name, fake_folder_name, json_folder):
    print(f"-> '{fake_folder_name}' 데이터셋 JSON 생성 중...")
    dataset_structure = {fake_folder_name: {}}

    # Real 이미지 처리
    real_label = f"{fake_folder_name}_Real"
    dataset_structure[fake_folder_name][real_label] = {"test": {}}
    real_images = glob(os.path.join(base_path, real_folder_name, '*.png')) + \
                  glob(os.path.join(base_path, real_folder_name, '*.jpg'))
    for i, img_path in enumerate(real_images):
        video_name = f"real_{i:05d}"
        dataset_structure[fake_folder_name][real_label]["test"][video_name] = {
            "label": real_label, "frames": [img_path]
        }

    # Fake 이미지 처리
    fake_label = f"{fake_folder_name}_Fake"
    dataset_structure[fake_folder_name][fake_label] = {"test": {}}
    fake_images = glob(os.path.join(base_path, fake_folder_name, '*.png')) + \
                  glob(os.path.join(base_path, fake_folder_name, '*.jpg'))
    for i, img_path in enumerate(fake_images):
        video_name = f"fake_{i:05d}"
        dataset_structure[fake_folder_name][fake_label]["test"][video_name] = {
            "label": fake_label, "frames": [img_path]
        }

    # JSON 파일 저장
    output_path = os.path.join(json_folder, f"{fake_folder_name}.json")
    with open(output_path, 'w') as f:
        json.dump(dataset_structure, f, indent=4)
    print(f"   '{output_path}'에 저장 완료.")

if __name__ == '__main__':
    os.makedirs(USER_CONFIG["JSON_SAVE_FOLDER"], exist_ok=True)
    create_all_jsons(
        USER_CONFIG["DATASET_ROOT"],
        USER_CONFIG["SUB_FOLDER"],
        USER_CONFIG["JSON_SAVE_FOLDER"]
    )