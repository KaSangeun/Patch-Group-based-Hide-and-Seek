# Patch Group-based-Hide-and-Seek
Patch Group-based Hide-and-Seek for Corruption Robustness of Vision Transformer

## Duration


## Abstract
Various data augmentation techniques have been proposed in Convolutional Neural Networks (CNNs) to improve generalization ability. However, it remains unclear whether these techniques are equally beneficial in Vision Transformer (ViT). In this paper, we 1) investigate whether three augmentation methods that have proven effective in CNNs−Random Erasing, Random Erasing in the Frequency domain (REF), and Hide-and-Seek−can also lead to meaningful performance improvements in ViT, and 2) propose a novel augmentation method, Patch Group-based Hide-and-Seek, which aligns with the architectural characteristics of ViT and enhances robustness against corrupted input images. The main idea is to leverage the patch-based image processing of ViT by grouping spatially adjacent nxn patches within the image and randomly hiding each group. Experimental results using the ViT-B/16 model on CIFAR-10 and CIFAR-100 demonstrate that applying 3x3 patch grouping reduces corruption error by up to 0.335% and 0.22%, respectively, compared to the baseline. These findings highlight the effectiveness of patch-based data augmentation strategies in ViT and their contribution to improving robustness against corrupted datasets. 


## Main Idea



## Run


## Outputs
### Journal 
Published in KMMS(Journal of Korea Multimedia Society), June 2025 issue.


## Based on
This project is based on [jeonsworld / ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch), which is licensed under the [MIT License]().
