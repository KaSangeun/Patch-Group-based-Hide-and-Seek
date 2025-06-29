# Patch Group-based Hide-and-Seek for Corruption Robustness of Vision Transformer

## Duration
2025.01 ~ 2025.06

## Abstract
Various data augmentation techniques have been proposed in Convolutional Neural Networks (CNNs) to improve generalization ability. However, it remains unclear whether these techniques are equally beneficial in Vision Transformer (ViT). In this paper, we 1) investigate whether three augmentation methods that have proven effective in CNNs−Random Erasing, Random Erasing in the Frequency domain (REF), and Hide-and-Seek−can also lead to meaningful performance improvements in ViT, and 2) propose a novel augmentation method, Patch Group-based Hide-and-Seek, which aligns with the architectural characteristics of ViT and enhances robustness against corrupted input images. The main idea is to leverage the patch-based image processing of ViT by grouping spatially adjacent nxn patches within the image and randomly hiding each group. Experimental results using the ViT-B/16 model on CIFAR-10 and CIFAR-100 demonstrate that applying 3x3 patch grouping reduces corruption error by up to 0.335% and 0.22%, respectively, compared to the baseline. These findings highlight the effectiveness of patch-based data augmentation strategies in ViT and their contribution to improving robustness against corrupted datasets. 


## Main Idea



## Run
### 1. Download Pre-trained Model ([Official Checkpoiont](https://console.cloud.google.com/storage/browser/vit_models?pli=1&inv=1&invt=Ab1bZw))
- ImageNet21k pre-trained models
  - ViT-B_16 

Download the .npz checkpoint file and place it in the checkpoint folder.

### 2. Train Model
```
python3 train.py --name cifar_10 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/imagenet21k_ViT-B_16.npz --fp16_opt_level O2
```

### 3. Test Model 
```
python3 test2.py --name cifar_10 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir output/cifar_10_checkpoint.bin --output_dir output
```


## Outputs
### Journal 
Published in KMMS(Journal of Korea Multimedia Society), June 2025 issue.


## Based on
This project is based on [jeonsworld / ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch), which is licensed under the [MIT License](https://github.com/KaSangeun/Patch-Group-based-Hide-and-Seek/blob/main/README.md).
