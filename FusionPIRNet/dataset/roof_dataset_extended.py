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


# ==================== 原始高度数据归一化配置 ====================
# 基于原始高度数据统计（未处理背景）

HEIGHT_MEAN = 3.108809
HEIGHT_STD = 3.943665
HEIGHT_MIN = -4.591034
HEIGHT_MAX = 52.247986
HEIGHT_P1 = -0.151031
HEIGHT_P5 = -0.026001
HEIGHT_P95 = 10.182983
#HEIGHT_P99 = 17.951996
HEIGHT_P99 = 10.5#根据周围推测
HEIGHT_P25 = 0.207977
HEIGHT_P75 = 5.231995
HEIGHT_IQR = 5.024017
HEIGHT_IQR_LOWER = -7.328049
HEIGHT_IQR_UPPER = 12.768021

#HEIGHT_MEAN = 1.447297
#HEIGHT_STD = 2.877818
#HEIGHT_MIN = -1.693024
#HEIGHT_MAX = 52.247986
#HEIGHT_P1 = 0.000000
#HEIGHT_P5 = 0.000000
#HEIGHT_P95 = 8.038971
#HEIGHT_P99 = 10.259979
#HEIGHT_P25 = 0.000000
#HEIGHT_P75 = 0.000000
#HEIGHT_IQR = 0.000000
#HEIGHT_IQR_LOWER = 0.000000
#HEIGHT_IQR_UPPER = 0.000000

def normalize_height(height_array, method='roof_aware'):
    """
    针对原始屋顶高度数据的归一化方法
    
    Args:
        height_array: numpy array of height values (original raw data)
        method: normalization method
            - 'roof_aware': 专为屋顶设计，负值裁剪为0，正值0-1归一化 (推荐)
            - 'standard': P5-P95裁剪归一化
            - 'robust': P1-P99裁剪归一化  
            - 'minmax': Min-Max归一化
            - 'zscore': Z-score标准化
            - 'none': 不归一化
    
    Returns:
        Normalized height array
    """
    if method == 'roof_aware':  # 推荐：专为原始屋顶数据设计
        # 将负值（可能是噪声或地面以下）裁剪为0
        # 使用P99作为上限进行归一化，覆盖99%的有效高度数据
        clipped = np.clip(height_array, 0, HEIGHT_P99)
        normalized = clipped / HEIGHT_P99
        return normalized
    
    elif method == 'standard':  # P5-P95裁剪
        clipped = np.clip(height_array, HEIGHT_P5, HEIGHT_P95)
        return (clipped - HEIGHT_P5) / (HEIGHT_P95 - HEIGHT_P5)
    
    elif method == 'robust':  # P1-P99裁剪
        clipped = np.clip(height_array, HEIGHT_P1, HEIGHT_P99)
        return (clipped - HEIGHT_P1) / (HEIGHT_P99 - HEIGHT_P1)
    
    elif method == 'minmax':
        return (height_array - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN)
    
    elif method == 'zscore':
        return (height_array - HEIGHT_MEAN) / HEIGHT_STD
    
    elif method == 'none':
        return height_array
    
    else:
        raise ValueError(f'Unknown normalization method: {method}. '
                        f'Available: roof_aware, standard, robust, minmax, zscore, none')


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    扩展到支持height任务，参照原版depth处理方式
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label_seg6, label_seg9, height, sam_masks, sam_edges):
        height_img, width_img = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height_img / sc), int(width_img / sc)
        i = random.randint(0, height_img - h)
        j = random.randint(0, width_img - w)
        
        # 图像插值
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height_img, width_img), mode='bilinear', align_corners=True).squeeze(0)
        
        # 标签插值 - 保持整数类型
        label_seg6_ = F.interpolate(label_seg6[None, None, i:i + h, j:j + w].float(), size=(height_img, width_img), mode='nearest').squeeze(0).squeeze(0).long()
        label_seg9_ = F.interpolate(label_seg9[None, None, i:i + h, j:j + w].float(), size=(height_img, width_img), mode='nearest').squeeze(0).squeeze(0).long()
        
        # 高度数据插值 - 参照原版depth处理方式
        height_ = F.interpolate(height[None, :, i:i + h, j:j + w], size=(height_img, width_img), mode='bilinear', align_corners=True).squeeze(0)
        
        # SAM masks和edges插值
        sam_masks_ = F.interpolate(sam_masks[None,:, i:i + h, j:j + w], size=(height_img, width_img), mode='nearest').squeeze(0)
        sam_edges_ = F.interpolate(sam_edges[None,:, i:i + h, j:j + w], size=(height_img, width_img), mode='nearest').squeeze(0)
        
        _sc = sc
        _h, _w, _i, _j = h, w, i, j

        return img_, label_seg6_, label_seg9_, height_, sam_masks_, sam_edges_, torch.tensor([_sc, _h, _w, _i, _j, height_img, width_img])


class RoofExtended(Dataset):
    """
    简洁的三任务Roof dataset：基于两任务版本，参照原版添加height
    """
    def __init__(self, root, train=True, index=None, height_norm_method='roof_aware'):
        self.train = train
        self.root = os.path.expanduser(root)
        self.height_norm_method = height_norm_method
        
        # 保持两任务版本的类别定义不变
        self.label_classes_segments_6 = ['background', 'N', 'E', 'S', 'W', 'flat']
        self.original_superstructures_classes = ['background', 'pvmodule', 'dormer', 'window', 'ladder',
                                                 'chimney', 'shadow', 'tree', 'unknown']
        
        # 文件加载逻辑 - 不创建虚拟文件列表
        if train:
            split_file = os.path.join(root, 'roof', 'train.txt')
        else:
            split_file = os.path.join(root, 'roof', 'val.txt')
            
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        self.data_len = len(self.file_list)
        
        # 路径定义 - 改为原始高度数据路径
        self.image_dir = os.path.join(root, 'roof', 'VOCdevkit', 'VOC2010', 'JPEGImages')
        self.seg6_dir = os.path.join(root, 'roof', 'seg6')
        self.seg9_dir = os.path.join(root, 'roof', 'seg9')
        self.height_dir = os.path.join(root, 'roof', 'seg_height')  # 注意是seg_height
        
        print(f"RoofExtended dataset initialized with ORIGINAL height data")
        print(f"Height normalization method: {self.height_norm_method}")

    def __getitem__(self, index):
        file_name = self.file_list[index]
        
        # 图像加载 - 找不到就报错
        image_path = os.path.join(self.image_dir, f'{file_name}.jpg')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        
        # seg6标签加载 - 找不到就报错
        seg6_path = os.path.join(self.seg6_dir, f'{file_name}.png')
        if not os.path.exists(seg6_path):
            raise FileNotFoundError(f"Seg6 label file not found: {seg6_path}")
        
        label_seg6 = Image.open(seg6_path)
        label_seg6 = torch.from_numpy(np.array(label_seg6, dtype=np.int64))
        
        # seg9标签加载 - 找不到就报错
        seg9_path = os.path.join(self.seg9_dir, f'{file_name}.png')
        if not os.path.exists(seg9_path):
            raise FileNotFoundError(f"Seg9 label file not found: {seg9_path}")
        
        label_seg9 = Image.open(seg9_path)
        label_seg9 = torch.from_numpy(np.array(label_seg9, dtype=np.int64))
        
        # 原始height数据加载 - 找不到就报错
        height_path = os.path.join(self.height_dir, f'{file_name}.tif')
        if not os.path.exists(height_path):
            raise FileNotFoundError(f"Height file not found: {height_path}")
        
        try:
            height_img = Image.open(height_path)
            height_array = np.array(height_img, dtype=np.float32)
            # 如果是多通道，取第一通道
            if len(height_array.shape) == 3:
                height_array = height_array[:, :, 0]
            
            # 使用配置的归一化方法处理原始高度数据
            height_array = normalize_height(height_array, method=self.height_norm_method)
            
            height = torch.from_numpy(height_array).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Error loading height TIF {height_path}: {e}")
        
        # 返回数据
        if self.train:
            return (image.type(torch.FloatTensor), 
                   label_seg6.type(torch.LongTensor), 
                   label_seg9.type(torch.LongTensor), 
                   height.type(torch.FloatTensor),
                   index)
        else:
            return (image.type(torch.FloatTensor), 
                   label_seg6.type(torch.LongTensor), 
                   label_seg9.type(torch.LongTensor),
                   height.type(torch.FloatTensor))
    
    def __len__(self):
        return self.data_len


class RoofCropExtended(Dataset):
    """
    基于两任务版本，简洁地扩展到三任务的数据增强版本
    """
    def __init__(self, root, train=True, index=None, augmentation=False, aug_twice=False, 
                 aug_extra=False, flip=False, sam_edge=False, height_norm_method='roof_aware'):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.aug_twice = aug_twice
        self.aug_extra = aug_extra
        self.flip = flip
        self.sam_edge = sam_edge
        self.height_norm_method = height_norm_method
        
        # 完全保持两任务版本的类别定义
        self.label_classes_segments_6 = ['background', 'N', 'E', 'S', 'W', 'flat']
        self.original_superstructures_classes = ['background', 'pvmodule', 'dormer', 'window', 'ladder',
                                                 'chimney', 'shadow', 'tree', 'unknown']
        
        # 完全保持两任务版本的增强设置
        self.extra_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((512, 512)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # 文件列表加载 - 不创建虚拟文件列表
        if train:
            split_file = os.path.join(root, 'roof', 'train.txt')
        else:
            split_file = os.path.join(root, 'roof', 'val.txt')
            
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        self.data_len = len(self.file_list)
        
        # 路径定义 - 改为原始高度数据路径
        self.image_dir = os.path.join(root, 'roof', 'VOCdevkit', 'VOC2010', 'JPEGImages')
        self.seg6_dir = os.path.join(root, 'roof', 'seg6')
        self.seg9_dir = os.path.join(root, 'roof', 'seg9')
        self.height_dir = os.path.join(root, 'roof', 'seg_height')  # 原始高度数据路径
        self.sam_mask_dir = os.path.join(root, 'roof', 'sam_GRAY')
        self.sam_edge_dir = os.path.join(root, 'roof', 'sam_edge')
        
        print(f"RoofCropExtended dataset initialized with ORIGINAL height data")
        print(f"Height normalization method: {self.height_norm_method}")

    def _load_sam_data(self, file_name, h, w):
        """加载SAM数据 - 找不到就报错，不创造虚拟数据"""
        if self.sam_edge:
            # 加载真实的SAM数据，找不到就报错
            sam_mask_path = os.path.join(self.sam_mask_dir, f'{file_name}.png')
            sam_edge_path = os.path.join(self.sam_edge_dir, f'{file_name}.png')
            
            if not os.path.exists(sam_mask_path):
                raise FileNotFoundError(f"SAM mask file not found: {sam_mask_path}")
            
            if not os.path.exists(sam_edge_path):
                raise FileNotFoundError(f"SAM edge file not found: {sam_edge_path}")
            
            # 加载SAM masks
            sam_masks = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
            if sam_masks is None:
                raise RuntimeError(f"Failed to load SAM mask file: {sam_mask_path}")
            sam_masks = torch.from_numpy(sam_masks).unsqueeze(0).float()
            
            # 加载SAM edges
            sam_edges = cv2.imread(sam_edge_path, cv2.IMREAD_GRAYSCALE)
            if sam_edges is None:
                raise RuntimeError(f"Failed to load SAM edge file: {sam_edge_path}")
            sam_edges = torch.from_numpy(sam_edges / 255.0).unsqueeze(0).float()
        else:
            # 如果不需要sam_edge，返回零张量（这是合理的默认行为）
            sam_masks = torch.zeros((1, h, w)).float()
            sam_edges = torch.zeros((1, h, w)).float()
        
        return sam_masks, sam_edges

    def _load_height_data(self, file_name, h, w):
        """加载和归一化原始height数据 - 找不到就报错"""
        height_path = os.path.join(self.height_dir, f'{file_name}.tif')
        if not os.path.exists(height_path):
            raise FileNotFoundError(f"Height file not found: {height_path}")
        
        try:
            height_img = Image.open(height_path)
            height_array = np.array(height_img, dtype=np.float32)
            # 如果是多通道，取第一通道
            if len(height_array.shape) == 3:
                height_array = height_array[:, :, 0]
            
            # 使用配置的归一化方法处理原始高度数据
            height_array = normalize_height(height_array, method=self.height_norm_method)
            
            height = torch.from_numpy(height_array).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Error loading height TIF {height_path}: {e}")
        
        return height

    def __getitem__(self, index):
        file_name = self.file_list[index]
        
        # 图像加载 - 找不到就报错
        image_path = os.path.join(self.image_dir, f'{file_name}.jpg')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        
        # 标签加载 - 找不到就报错
        seg6_path = os.path.join(self.seg6_dir, f'{file_name}.png')
        if not os.path.exists(seg6_path):
            raise FileNotFoundError(f"Seg6 label file not found: {seg6_path}")
        
        label_seg6 = Image.open(seg6_path)
        label_seg6 = torch.from_numpy(np.array(label_seg6, dtype=np.int64))
        
        seg9_path = os.path.join(self.seg9_dir, f'{file_name}.png')
        if not os.path.exists(seg9_path):
            raise FileNotFoundError(f"Seg9 label file not found: {seg9_path}")
        
        label_seg9 = Image.open(seg9_path)
        label_seg9 = torch.from_numpy(np.array(label_seg9, dtype=np.int64))
        
        # 使用统一的height加载方法
        h, w = image.shape[-2:]
        height = self._load_height_data(file_name, h, w)
        
        # SAM数据加载
        sam_masks, sam_edges = self._load_sam_data(file_name, h, w)
        
        # 完全保持两任务版本的数据增强逻辑，只需扩展到height
        if self.augmentation and self.aug_twice:
            # 第一次翻转
            if self.flip and torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label_seg6 = torch.flip(label_seg6, dims=[1])
                label_seg9 = torch.flip(label_seg9, dims=[1])
                height = torch.flip(height, dims=[2])  # 参照原版depth处理
                sam_masks = torch.flip(sam_masks, dims=[2])
                sam_edges = torch.flip(sam_edges, dims=[2])
            
            # 第一次增强
            image, label_seg6, label_seg9, height, sam_masks, sam_edges, _ = RandomScaleCrop()(image, label_seg6, label_seg9, height, sam_masks, sam_edges)
            
            # 准备第二次增强的数据
            if self.flip and torch.rand(1) < 0.5:
                image1 = torch.flip(image, dims=[2])
                label_seg6_1 = torch.flip(label_seg6, dims=[1])
                label_seg9_1 = torch.flip(label_seg9, dims=[1])
                height1 = torch.flip(height, dims=[2])  # 参照原版depth处理
                sam_masks1 = torch.flip(sam_masks, dims=[2])
                sam_edges1 = torch.flip(sam_edges, dims=[2])
            else:
                image1 = image.clone()
                label_seg6_1 = label_seg6.clone()
                label_seg9_1 = label_seg9.clone()
                height1 = height.clone()
                sam_masks1 = sam_masks.clone()
                sam_edges1 = sam_edges.clone()
            
            # 第二次增强
            image1, label_seg6_1, label_seg9_1, height1, sam_masks1, sam_edges1, trans_params = RandomScaleCrop()(image1, label_seg6_1, label_seg9_1, height1, sam_masks1, sam_edges1)
            
            # 保持两任务版本的返回格式，只添加height数据
            return (image.type(torch.FloatTensor),           # train_data
                    label_seg6.type(torch.LongTensor),       # train_label_seg6
                    label_seg9.type(torch.LongTensor),       # train_label_seg9
                    height.type(torch.FloatTensor),          # train_height (新增)
                    sam_masks.type(torch.FloatTensor),       # train_sam
                    sam_edges.type(torch.FloatTensor),       # train_edge
                    index,                                   # image_index
                    image1.type(torch.FloatTensor),          # train_data1
                    label_seg6_1.type(torch.LongTensor),     # train_label_seg6_1
                    label_seg9_1.type(torch.LongTensor),     # train_label_seg9_1
                    height1.type(torch.FloatTensor),         # train_height1 (新增)
                    sam_masks1.type(torch.FloatTensor),      # train_sam1
                    sam_edges1.type(torch.FloatTensor),      # train_edge1
                    trans_params)                            # trans_params
        
        # aug_extra模式
        elif self.augmentation and self.aug_extra:
            image_extra = self.extra_aug(image)
            
            if self.flip and torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label_seg6 = torch.flip(label_seg6, dims=[1])
                label_seg9 = torch.flip(label_seg9, dims=[1])
                height = torch.flip(height, dims=[2])
            
            image, label_seg6, label_seg9, height, sam_masks, sam_edges, _ = RandomScaleCrop()(image, label_seg6, label_seg9, height, sam_masks, sam_edges)
            
            if self.flip and torch.rand(1) < 0.5:
                image1 = torch.flip(image, dims=[2])
                label_seg6_1 = torch.flip(label_seg6, dims=[1])
                label_seg9_1 = torch.flip(label_seg9, dims=[1])
                height1 = torch.flip(height, dims=[2])
                sam_masks1 = torch.flip(sam_masks, dims=[2])
                sam_edges1 = torch.flip(sam_edges, dims=[2])
                flip = 1
            else:
                image1 = image.clone()
                label_seg6_1 = label_seg6.clone()
                label_seg9_1 = label_seg9.clone()
                height1 = height.clone()
                sam_masks1 = sam_masks.clone()
                sam_edges1 = sam_edges.clone()
                flip = 0
            
            image1, label_seg6_1, label_seg9_1, height1, sam_masks1, sam_edges1, trans_params = RandomScaleCrop()(image1, label_seg6_1, label_seg9_1, height1, sam_masks1, sam_edges1)
            
            return (image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), 
                   label_seg9.type(torch.LongTensor), height.type(torch.FloatTensor), index, 
                   image1.type(torch.FloatTensor), label_seg6_1.type(torch.LongTensor), 
                   label_seg9_1.type(torch.LongTensor), height1.type(torch.FloatTensor),
                   trans_params, flip, image_extra)
        
        # 简单增强模式
        elif self.augmentation and not self.aug_twice:
            image, label_seg6, label_seg9, height, sam_masks, sam_edges, _ = RandomScaleCrop()(image, label_seg6, label_seg9, height, sam_masks, sam_edges)
            if self.flip and torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label_seg6 = torch.flip(label_seg6, dims=[1])
                label_seg9 = torch.flip(label_seg9, dims=[1])
                height = torch.flip(height, dims=[2])
            return (image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), 
                   label_seg9.type(torch.LongTensor), height.type(torch.FloatTensor), index)
        
        # 默认模式 - 完全保持两任务版本的逻辑
        if self.train:
            return (image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), 
                   label_seg9.type(torch.LongTensor), height.type(torch.FloatTensor),
                   sam_masks.type(torch.FloatTensor), sam_edges.type(torch.FloatTensor), index)
        else:
            return (image.type(torch.FloatTensor), label_seg6.type(torch.LongTensor), 
                   label_seg9.type(torch.LongTensor), height.type(torch.FloatTensor))
    
    def __len__(self):
        return self.data_len

# 兼容性导入，用于保持旧的训练脚本工作
Roof_crop = RoofCropExtended
Roof = RoofExtended