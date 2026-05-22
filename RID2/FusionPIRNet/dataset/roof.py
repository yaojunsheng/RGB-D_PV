import os
import torch
import fnmatch
import numpy as np
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
#用sam掩码实现

class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label_seg6, label_seg9, sam_masks, sam_edges):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        
        # 图像插值
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        
        # 标签插值 - 保持整数类型
        label_seg6_ = F.interpolate(label_seg6[None, None, i:i + h, j:j + w].float(), size=(height, width), mode='nearest').squeeze(0).squeeze(0).long()
        label_seg9_ = F.interpolate(label_seg9[None, None, i:i + h, j:j + w].float(), size=(height, width), mode='nearest').squeeze(0).squeeze(0).long()
        
        # SAM masks和edges插值
        sam_masks_ = F.interpolate(sam_masks[None,:, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        sam_edges_ = F.interpolate(sam_edges[None,:, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        
        _sc = sc
        _h, _w, _i, _j = h, w, i, j

        return img_, label_seg6_, label_seg9_, sam_masks_, sam_edges_, torch.tensor([_sc, _h, _w, _i, _j, height, width])


class Roof(Dataset):
    """
    Roof dataset for two semantic segmentation tasks
    """
    def __init__(self, root, train=True, index=None):
        self.train = train
        self.root = os.path.expanduser(root)
        
        # 类别定义
        self.label_classes_segments_6 = ['background', 'N', 'E', 'S', 'W', 'flat']  # 值0-5
        self.original_superstructures_classes = ['background', 'pvmodule', 'dormer', 'window', 'balcony', 'other']  # 改为6类
        
        # 读取数据文件列表：找不到split文件直接报错
        if train:
            split_file = os.path.join(root, 'roof', 'train.txt')
        else:
            split_file = os.path.join(root, 'roof', 'val.txt')
            
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}\nPlease check the file path or prepare the split file first.")
        
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        
        self.data_len = len(self.file_list)
        
        # 路径定义
        self.image_dir = os.path.join(root, 'roof', 'VOCdevkit', 'VOC2010', 'JPEGImages')
        self.seg6_dir = os.path.join(root, 'roof', 'seg6')
        self.seg9_dir = os.path.join(root, 'roof', 'seg9')

    def __getitem__(self, index):
        file_name = self.file_list[index]
        
        # 加载图像：找不到图像文件直接报错
        image_path = os.path.join(self.image_dir, f'{file_name}.jpg')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}\nPlease check the image path or prepare the image file first.")
        
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        
        # 加载seg6标签：找不到标签文件直接报错
        seg6_path = os.path.join(self.seg6_dir, f'{file_name}.png')
        if not os.path.exists(seg6_path):
            raise FileNotFoundError(f"Seg6 label file not found: {seg6_path}\nPlease check the seg6 label path or prepare the label file first.")
        
        label_seg6 = Image.open(seg6_path)
        label_seg6 = torch.from_numpy(np.array(label_seg6, dtype=np.int64))
        
        # 加载seg9标签：找不到标签文件直接报错
        seg9_path = os.path.join(self.seg9_dir, f'{file_name}.png')
        if not os.path.exists(seg9_path):
            raise FileNotFoundError(f"Seg9 label file not found: {seg9_path}\nPlease check the seg9 label path or prepare the label file first.")
        
        label_seg9 = Image.open(seg9_path)
        label_seg9 = torch.from_numpy(np.array(label_seg9, dtype=np.int64))
        
        # 训练模式和测试模式返回不同的值（保持原逻辑）
        if self.train:
            return image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), label_seg9.type(torch.LongTensor), index
        else:
            return image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), label_seg9.type(torch.LongTensor)
    
    def __len__(self):
        return self.data_len


class Roof_crop(Dataset):
    """
    Roof dataset with data augmentation for training
    """
    def __init__(self, root, train=True, index=None, augmentation=False, aug_twice=False, aug_extra=False, flip=False, sam_edge=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.aug_twice = aug_twice
        self.aug_extra = aug_extra
        self.flip = flip
        self.sam_edge = sam_edge
        
        # 类别定义
        self.label_classes_segments_6 = ['background', 'N', 'E', 'S', 'W', 'flat']
        self.original_superstructures_classes = ['background', 'pvmodule', 'dormer', 'window', 'balcony', 'other']  # 改为6类
        
        self.extra_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((512, 512)),#原始是288*384
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # 读取数据文件列表：找不到split文件直接报错
        if train:
            split_file = os.path.join(root, 'roof', 'train.txt')
        else:
            split_file = os.path.join(root, 'roof', 'val.txt')
            
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}\nPlease check the file path or prepare the split file first.")
        
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        
        self.data_len = len(self.file_list)
        
        # 路径定义
        self.image_dir = os.path.join(root, 'roof', 'VOCdevkit', 'VOC2010', 'JPEGImages')
        self.seg6_dir = os.path.join(root, 'roof', 'seg6')
        self.seg9_dir = os.path.join(root, 'roof', 'seg9')
        self.sam_mask_dir = os.path.join(root, 'roof', 'sam_GRAY')
        self.sam_edge_dir = os.path.join(root, 'roof', 'sam_edge')

    def _load_sam_data(self, file_name, h, w):
        """加载SAM数据：找不到文件直接报错"""
        if self.sam_edge:
            # 加载SAM mask：找不到文件直接报错
            sam_mask_path = os.path.join(self.sam_mask_dir, f'{file_name}.png')
            if not os.path.exists(sam_mask_path):
                raise FileNotFoundError(f"SAM mask file not found: {sam_mask_path}\nPlease check the SAM mask path or prepare the file first.")
            
            sam_masks = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
            if sam_masks is None:
                raise ValueError(f"SAM mask file is corrupted: {sam_mask_path}\nPlease check the file integrity.")
            
            sam_masks = torch.from_numpy(sam_masks).unsqueeze(0).float()
            
            # 加载SAM edge：找不到文件直接报错
            sam_edge_path = os.path.join(self.sam_edge_dir, f'{file_name}.png')
            if not os.path.exists(sam_edge_path):
                raise FileNotFoundError(f"SAM edge file not found: {sam_edge_path}\nPlease check the SAM edge path or prepare the file first.")
            
            sam_edges = cv2.imread(sam_edge_path, cv2.IMREAD_GRAYSCALE)
            if sam_edges is None:
                raise ValueError(f"SAM edge file is corrupted: {sam_edge_path}\nPlease check the file integrity.")
            
            sam_edges = torch.from_numpy(sam_edges / 255.0).unsqueeze(0).float()
        else:
            # sam_edge=False时保持原逻辑（返回全零张量）
            sam_masks = torch.zeros((1, h, w)).float()
            sam_edges = torch.zeros((1, h, w)).float()
        
        return sam_masks, sam_edges

    def __getitem__(self, index):
        file_name = self.file_list[index]
        
        # 加载图像：找不到文件直接报错
        image_path = os.path.join(self.image_dir, f'{file_name}.jpg')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}\nPlease check the image path or prepare the image file first.")
        
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        
        # 加载seg6标签：找不到文件直接报错
        seg6_path = os.path.join(self.seg6_dir, f'{file_name}.png')
        if not os.path.exists(seg6_path):
            raise FileNotFoundError(f"Seg6 label file not found: {seg6_path}\nPlease check the seg6 label path or prepare the label file first.")
        
        label_seg6 = Image.open(seg6_path)
        label_seg6 = torch.from_numpy(np.array(label_seg6, dtype=np.int64))
        
        # 加载seg9标签：找不到文件直接报错
        seg9_path = os.path.join(self.seg9_dir, f'{file_name}.png')
        if not os.path.exists(seg9_path):
            raise FileNotFoundError(f"Seg9 label file not found: {seg9_path}\nPlease check the seg9 label path or prepare the label file first.")
        
        label_seg9 = Image.open(seg9_path)
        label_seg9 = torch.from_numpy(np.array(label_seg9, dtype=np.int64))
        
        # 加载SAM数据（找不到文件会在_load_sam_data中报错）
        h, w = image.shape[-2:]
        sam_masks, sam_edges = self._load_sam_data(file_name, h, w)
        
        # ===== 保持原数据增强逻辑不变 =====
        if self.augmentation and self.aug_twice:
            # 第一次翻转
            if self.flip and torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label_seg6 = torch.flip(label_seg6, dims=[1])
                label_seg9 = torch.flip(label_seg9, dims=[1])
                sam_masks = torch.flip(sam_masks, dims=[2])
                sam_edges = torch.flip(sam_edges, dims=[2])
            
            # 第一次增强
            image, label_seg6, label_seg9, sam_masks, sam_edges, _ = RandomScaleCrop()(image, label_seg6, label_seg9, sam_masks, sam_edges)
            
            # 准备第二次增强的数据
            if self.flip and torch.rand(1) < 0.5:
                image1 = torch.flip(image, dims=[2])
                label_seg6_1 = torch.flip(label_seg6, dims=[1])
                label_seg9_1 = torch.flip(label_seg9, dims=[1])
                sam_masks1 = torch.flip(sam_masks, dims=[2])
                sam_edges1 = torch.flip(sam_edges, dims=[2])
            else:
                image1 = image.clone()
                label_seg6_1 = label_seg6.clone()
                label_seg9_1 = label_seg9.clone()
                sam_masks1 = sam_masks.clone()
                sam_edges1 = sam_edges.clone()
            
            # 第二次增强
            image1, label_seg6_1, label_seg9_1, sam_masks1, sam_edges1, trans_params = RandomScaleCrop()(image1, label_seg6_1, label_seg9_1, sam_masks1, sam_edges1)
            
            # 保持原返回值格式
            return (image.type(torch.FloatTensor),           # train_data
                    label_seg6.type(torch.LongTensor),       # train_label_seg6
                    label_seg9.type(torch.LongTensor),       # train_label_seg9
                    sam_masks.type(torch.FloatTensor),       # train_sam
                    sam_edges.type(torch.FloatTensor),       # train_edge
                    index,                                   # image_index
                    image1.type(torch.FloatTensor),          # train_data1
                    label_seg6_1.type(torch.LongTensor),     # train_label_seg6_1
                    label_seg9_1.type(torch.LongTensor),     # train_label_seg9_1
                    sam_masks1.type(torch.FloatTensor),      # train_sam1
                    sam_edges1.type(torch.FloatTensor),      # train_edge1
                    trans_params)                            # trans_params
        
        elif self.augmentation and self.aug_extra:
            image_extra = self.extra_aug(image)
            
            if self.flip and torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label_seg6 = torch.flip(label_seg6, dims=[1])
                label_seg9 = torch.flip(label_seg9, dims=[1])
            
            image, label_seg6, label_seg9, sam_masks, sam_edges, _ = RandomScaleCrop()(image, label_seg6, label_seg9, sam_masks, sam_edges)
            
            if self.flip and torch.rand(1) < 0.5:
                image1 = torch.flip(image, dims=[2])
                label_seg6_1 = torch.flip(label_seg6, dims=[1])
                label_seg9_1 = torch.flip(label_seg9, dims=[1])
                sam_masks1 = torch.flip(sam_masks, dims=[2])
                sam_edges1 = torch.flip(sam_edges, dims=[2])
                flip = 1
            else:
                image1 = image.clone()
                label_seg6_1 = label_seg6.clone()
                label_seg9_1 = label_seg9.clone()
                sam_masks1 = sam_masks.clone()
                sam_edges1 = sam_edges.clone()
                flip = 0
            
            image1, label_seg6_1, label_seg9_1, sam_masks1, sam_edges1, trans_params = RandomScaleCrop()(image1, label_seg6_1, label_seg9_1, sam_masks1, sam_edges1)
            
            return (image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), label_seg9.type(torch.LongTensor), index, 
                    image1.type(torch.FloatTensor), label_seg6_1.type(torch.LongTensor), label_seg9_1.type(torch.LongTensor), 
                    trans_params, flip, image_extra)
        
        elif self.augmentation and not self.aug_twice:
            image, label_seg6, label_seg9, sam_masks, sam_edges, _ = RandomScaleCrop()(image, label_seg6, label_seg9, sam_masks, sam_edges)
            if self.flip and torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label_seg6 = torch.flip(label_seg6, dims=[1])
                label_seg9 = torch.flip(label_seg9, dims=[1])
            return image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), label_seg9.type(torch.LongTensor), index
        
        # 默认模式返回值（保持原逻辑）
        if self.train:
            return (image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), label_seg9.type(torch.LongTensor), 
                    sam_masks.type(torch.FloatTensor), sam_edges.type(torch.FloatTensor), index)
        else:
            return image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), label_seg9.type(torch.LongTensor)
    
    def __len__(self):
        return self.data_len