import numpy as np

# 저장된 .npz 파일 로드
npz_data = np.load("/home/lee2/chloeka_proj/ViT/ViT-pytorch-main_with-visualize/output/imagenet1k_seed365_ft1_has05_3_checkpoint.npz")

# 저장된 키 목록 확인
#print("Keys in .npz file:", npz_data.files)
print("Available keys in pretrained weights:", list(npz_data.keys()))
