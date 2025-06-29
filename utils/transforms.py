from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

class HideAndSeek(object):
    """
    Summary:
        Hide-and-seek augmentaion
    
    """
    def __init__(self, probability = 0.5,grid_ratio=0.25,patch_probabilty=0.5, mean=[0.4914, 0.4822, 0.4465]): # grid_ratio-patch size 
        self.probability = probability
        self.grid_ratio = grid_ratio
        self.patch_prob = patch_probabilty
        self.mean = torch.tensor(mean).reshape(-1,1,1)

    def __call__(self,img:torch.Tensor):
        if random.uniform(0,1)>self.probability:
            return img
        img= img.squeeze()
        c,h,w=torch.tensor(img.shape,dtype=torch.int)
        # h_grid_step, w_grid_step 모두 반올림 되는 것 같음 
        h_grid_step = torch.round(h*self.grid_ratio).int() # grid_ratio-patch size 
        w_grid_step = torch.round(w*self.grid_ratio).int() # grid_ratio-patch size 

        #print("### grid size ### ", h_grid_step, w_grid_step)

        for y in range(0,h,h_grid_step):
            for x in range(0,w,w_grid_step):
                y_end = min(h, y+h_grid_step)  
                x_end = min(w, x+w_grid_step)
                if(random.uniform(0,1) >self.patch_prob):
                    continue
                else:
                    img[:,y:y_end,x:x_end]= self.mean
                
        return img

# original RE code
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img
    
class PatchRandomErasing:
    """
    Applies Random Erasing on individual patches of an image.
    Designed specifically for Vision Transformer (ViT) models that process
    images divided into fixed-size patches.
    -------------------------------------------------------------------------------------
    probability: Probability of applying random erasing to a patch.
    sl: Minimum erasing area ratio within a patch.
    sh: Maximum erasing area ratio within a patch.
    r1: Minimum aspect ratio of the erasing region within a patch.
    mean: Erasing value for normalization.
    patch_size: Size of each patch (e.g., 16 for 16x16 patches in ViT).
    -------------------------------------------------------------------------------------
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465], patch_size=16):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.patch_size = patch_size

    def __call__(self, img):
        #print(f"RandomErasing called for image with size: {img.size()}")
        # Get image dimensions
        c, h, w = img.size()
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        # Compute number of patches in both dimensions
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        # Iterate over patches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Apply Random Erasing with given probability
                if random.uniform(0, 1) > self.probability:
                    continue

                # Compute patch coordinates
                x_start = i * self.patch_size
                y_start = j * self.patch_size
                x_end = x_start + self.patch_size
                y_end = y_start + self.patch_size

                # Extract patch
                patch = img[:, x_start:x_end, y_start:y_end]
                patch_area = self.patch_size * self.patch_size

                # Attempt erasing within the patch
                for attempt in range(100):
                    target_area = random.uniform(self.sl, self.sh) * patch_area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
                    w_erase = int(round(math.sqrt(target_area / aspect_ratio)))

                    if h_erase < self.patch_size and w_erase < self.patch_size:
                        x_erase = random.randint(0, self.patch_size - h_erase)
                        y_erase = random.randint(0, self.patch_size - w_erase)
                        
                        # Apply erasing
                        for k in range(c):
                            patch[k, x_erase:x_erase + h_erase, y_erase:y_erase + w_erase] = self.mean[k]
                        break

                # Replace the patch back into the image
                img[:, x_start:x_end, y_start:y_end] = patch

        return img

# fixed RE code 
"""class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        # img가 3차원일 경우, 4차원으로 차원 확장
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [768, n, n] -> [1, 768, n, n]
    
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            # 패치 그룹 크기 계산
            area = img.size(2) * img.size(3)  # H, W는 n에 해당

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size(3) and h < img.size(2):  # H, W 범위 내에서 erasing 영역 설정
                x1 = random.randint(0, img.size(2) - h)
                y1 = random.randint(0, img.size(3) - w)

                # 각 채널에 대해 erasing 처리
                img[:, :, x1:x1+h, y1:y1+w] = self.mean[0]  # 첫 번째 채널
                img[:, :, x1:x1+h, y1:y1+w] = self.mean[1]  # 두 번째 채널
                img[:, :, x1:x1+h, y1:y1+w] = self.mean[2]  # 세 번째 채널

                return img

        return img"""

'''### MREF ###
class BlockFrequencyRandomErasing(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        ####nxn
        n = 4                   # change here according to the conditions

        a = [int(i * 32 / n) for i in range(n)]
        b= a
        c = a[1:]
        d = a[1:]
        c.append(32)
        d.append(32)

        image_list = []
        for i in range(n):
            for j in range(n):
                imgcrop = x.crop((a[i], b[j], c[i], d[j]))
                image_list.append(imgcrop)
        ####

        result_image_list = []
        ##
        for img in image_list:
            img = np.array(img).astype(np.uint8)
            fft_1 = np.fft.fftshift(np.fft.fftn(img))

            # 랜덤 영역 뽑기
            x_min = np.random.randint(0, 16//n)        # 16
            x_max = np.random.randint(x_min+1, 32//n)        # 32
            y_min = np.random.randint(0, 16//n)        # 16
            y_max = np.random.randint(y_min+1, 32//n)        # 32
            # RE
            fft_1[x_min:x_max, y_min:y_max] = 0

            img = np.fft.ifftn(np.fft.ifftshift(fft_1))
            result_image_list.append(img)

        ####nxn
        row = []
        for ii in range(n):
            r = np.concatenate(result_image_list[ii * n : (ii + 1) * n], axis=0)
            row.append(r)
        new_image = np.concatenate(row, axis=1)
        ####
        x = np.clip(new_image, 0, 255).astype(np.uint8)
        x = Image.fromarray(x)
        # x.show()
        return x'''


### REF ###
class FrequencyRandomErasing(object):
    def __init__(self, probability=0.5, visualize=False):
        self.probability = probability
        self.visualize = visualize

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        x = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        x_min = np.random.randint(0, 16)
        x_max = np.random.randint(x_min, 32)
        y_min = np.random.randint(0, 16)
        y_max = np.random.randint(y_min, 32)
        fft_1[x_min:x_max, y_min:y_max] = 0

        x = np.fft.ifftn(np.fft.ifftshift(fft_1))
        x = x.astype(np.uint8)
        '''
        x = np.abs(x).astype(np.uint8)  # np.abs() 추가하여 복소수 문제 방지
        if self.visualize:
            plt.figure(figsize=(4, 4))
            plt.imshow(x)
            plt.title("Image after FrequencyRandomErasing")
            plt.axis('off')
            plt.show()
        '''
        x = Image.fromarray(x)
        return x

        return x
    

### REF by grid ###
class GridFrequencyRandomErasing(object):
    def __init__(self, probability=0.5, grid_size=16, erase_size=16, grid_p=0.5):
        self.probability = probability
        self.grid_size = grid_size  # 패치 크기 (ViT에서 사용)
        self.erase_size = erase_size
        self.grid_p = grid_p

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        x = np.array(x).astype(np.uint8)
        #print("### x shape: ", x.shape)  # [3, 384, 384]
        h, w = x.shape[1], x.shape[2]  # (height, width)

        for i in range(0, h, self.erase_size):
            for j in range(0, w, self.erase_size):

                if random.uniform(0, 1) > self.grid_p:
                    continue

                # 패치 영역 좌표 설정 (이미지 경계를 넘지 않도록 조정)
                x_min, x_max = i, min(i + self.erase_size, h)
                y_min, y_max = j, min(j + self.erase_size, w)

                patch = x[:, x_min:x_max, y_min:y_max]

                #print(f"Patch shape before FFT: {patch.shape}")  # [3, 96, 96]

                # 푸리에 변환
                fft_patch = np.fft.fftshift(np.fft.fftn(patch))

                # erase_size별 주파수 제거
                patch_h, patch_w = fft_patch.shape[1], fft_patch.shape[2]
                #print(f"FFT Patch shape: {fft_patch.shape}")  # [3, 96, 96]

                erase_h_min = random.randint(0, patch_h // 4)
                erase_h_max = random.randint(erase_h_min, patch_h // 2)
                erase_w_min = random.randint(0, patch_w // 4)
                erase_w_max = random.randint(erase_w_min, patch_w // 2)

                fft_patch[:, erase_h_min:erase_h_max, erase_w_min:erase_w_max] = 0

                # 역푸리에 변환
                patch_restored = np.fft.ifftn(np.fft.ifftshift(fft_patch))
                patch_restored = np.abs(patch_restored).astype(np.uint8)

                #print(f"Patch Restored shape: {patch_restored.shape}") # [3, 96, 96]

                # 원본 이미지에 수정된 패치 적용
                x[:, x_min:x_max, y_min:y_max] = patch_restored

        #x = Image.fromarray(x)
        #x = Image.fromarray(x.transpose(1,2,0))
        x = torch.tensor(x).float()
        #print("After x size: ", x.size()) # [3, 384, 384]
        return x

