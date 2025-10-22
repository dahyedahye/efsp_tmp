#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# --- 1. 'images' 폴더 테스트 ---
echo "======================================================"
echo "    Starting test on: images" Chameleon
echo "======================================================"

CUDA_VISIBLE_DEVICES=5 python3 training/demo_251014_test.py \
--detector_config training/config/detector/effort.yaml \
--weights /workspace/Effort-AIGI-Detection/UniversalFakeDetect_Benchmark/checkpoints/effort_chameleon.pth \
--dataset_root /datasets/LDMFakeDetect/dataset \
--sub_folder images \
--result_folder /outputs/all_images_demo_chameleon_251014_LDMFakeDetect

# --- 2. 'with_postprocessing' 폴더 테스트 ---
echo "======================================================"
echo "    Starting test on: with_postprocessing" Chameleon
echo "======================================================"

CUDA_VISIBLE_DEVICES=5 python3 training/demo_251014_test.py \
--detector_config training/config/detector/effort.yaml \
--weights /workspace/Effort-AIGI-Detection/UniversalFakeDetect_Benchmark/checkpoints/effort_chameleon.pth \
--dataset_root /datasets/LDMFakeDetect/dataset \
--sub_folder with_postprocessing \
--result_folder /outputs/all_postproc_demo_chameleon_251014_LDMFakeDetect

echo "======================================================"
echo "    All tests completed successfully!"
echo "======================================================"