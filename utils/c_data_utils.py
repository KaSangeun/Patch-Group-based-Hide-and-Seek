import logging
import torch
import os
import numpy as np


from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, SubsetRandomSampler, random_split

from utils.transforms import RandomErasing, FrequencyRandomErasing, PatchRandomErasing, HideAndSeek, GridFrequencyRandomErasing

logger = logging.getLogger(__name__)

class CIFARC(datasets.CIFAR10):
    def __init__(
            self,
            root,
            key = 'zoom_blur',
            transform = None,
            target_transform = None,
    ):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        data_path = os.path.join(root, key+'.npy')
        labels_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(labels_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    outkeys = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        #FrequencyRandomErasing(probability=args.p2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #PatchRandomErasing(pc, sh = args.sh, r1 = args.r1, patch_size=16, ),
        #RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, ),
        HideAndSeek(probability = args.p3, grid_ratio= args.grid_ratio, patch_probabilty= args.patch_p),
        #GridFrequencyRandomErasing(probability = args.p3, erase_size=16, grid_p=args.grid_p), 
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        validset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_test)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None
        droot = '/home/lee2/chloeka_proj/ViT/ViT-pytorch-main/data/CIFAR-10-C'
        outloaders = dict()
        for key in outkeys:
            outset = CIFARC(root=droot, key=key, transform=transform_test)
            outloader = DataLoader(outset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            outloaders[key] = outloader

    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        validset = datasets.CIFAR100(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_test)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
        droot = '/home/lee2/chloeka_proj/ViT/ViT-pytorch-main/data/CIFAR-100-C'
        outloaders = dict()
        for key in outkeys:
            outset = CIFARC(root=droot, key=key, transform=transform_test)
            outloader = DataLoader(outset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            outloaders[key] = outloader

    elif args.dataset == 'imagenet1k':
        trainset = datasets.ImageFolder(root="./data/imagenet1k/train",
                                     transform=transform_train)
        validset = datasets.ImageFolder(root="./data/imagenet1k/train",
                                    transform=transform_test)
        testset = datasets.ImageFolder(root="./data/imagenet1k/val",
                                    transform=transform_test)
        outloaders = {}

    else:
        raise ValueError("Dataset not supported")

    if args.local_rank == 0:
        torch.distributed.barrier()

    '''outkeys = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']
    
    outloaders = dict()
    for key in outkeys:
        outset = CIFARC(root=droot, key=key, transform=transform_test)
        outloader = DataLoader(outset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
        outloaders[key] = outloader'''

    # Train/Validation Split
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.seed(args.seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    valid_loader = DataLoader(validset,
                              sampler=valid_sampler,
                              batch_size=args.eval_batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    if testset is not None:
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                                sampler=test_sampler,
                                batch_size=args.eval_batch_size,
                                num_workers=4,
                                pin_memory=True) if testset is not None else None
        
    #print('train size: ', len(train_sampler))
    #print('valid size: ', len(valid_sampler))

    return train_loader, valid_loader, test_loader, outloaders 