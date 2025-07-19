""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import argparse
import glob
import torchio as tio


class CustomDataset(Dataset):
    def __init__(self, root_dir='../ADNI', csv_path=None, augmentation=False, 
                 target_size=(64, 64, 64), resize_mode='pad'):
        """
        Args:
            target_size: 目标尺寸 (D, H, W)
            resize_mode: 'pad' - 等比例缩放+填充, 'crop' - 等比例缩放+裁剪, 'stretch' - 强制缩放
        """
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.target_size = target_size
        self.resize_mode = resize_mode
        print(f"初始化CustomDataset，目标尺寸: {self.target_size}, 缩放模式: {self.resize_mode}")
        
        if csv_path is not None:
            # 从CSV文件中读取数据
            self.samples = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        'image_path': row['image_path'],
                        'description': row['description']
                    })
        else:
            # 如果没有提供CSV，则使用原始方法查找文件
            self.file_names = glob.glob(os.path.join(
                root_dir, './**/*.nii'), recursive=True)
            self.samples = [{'image_path': path, 'description': ''} for path in self.file_names]

    def __len__(self):
        return len(self.samples)

    def proportional_resize_with_pad(self, image, target_size):
        """
        等比例缩放图像并填充到目标尺寸
        """
        original_size = np.array(image.shape)
        target_size = np.array(target_size)
        
        # 计算缩放比例（选择最小的比例以确保图像能装入目标框）
        scale_factors = target_size / original_size
        scale = np.min(scale_factors)
        
        # 计算缩放后的尺寸
        new_size = (original_size * scale).astype(int)
        
        # 缩放图像
        resized_image = resize(image, new_size, mode='constant', preserve_range=True, anti_aliasing=True)
        
        # 创建目标尺寸的零填充图像
        padded_image = np.zeros(target_size, dtype=resized_image.dtype)
        
        # 计算填充位置（居中放置）
        start_pos = (target_size - new_size) // 2
        end_pos = start_pos + new_size
        
        # 将缩放后的图像放入填充图像中
        padded_image[start_pos[0]:end_pos[0], 
                    start_pos[1]:end_pos[1], 
                    start_pos[2]:end_pos[2]] = resized_image
        
        return padded_image, scale, start_pos, new_size

    def proportional_resize_with_crop(self, image, target_size):
        """
        等比例缩放图像并裁剪到目标尺寸
        """
        original_size = np.array(image.shape)
        target_size = np.array(target_size)
        
        # 计算缩放比例（选择最大的比例以确保图像能填满目标框）
        scale_factors = target_size / original_size
        scale = np.max(scale_factors)
        
        # 计算缩放后的尺寸
        new_size = (original_size * scale).astype(int)
        
        # 缩放图像
        resized_image = resize(image, new_size, mode='constant', preserve_range=True, anti_aliasing=True)
        
        # 计算裁剪位置（居中裁剪）
        start_pos = (new_size - target_size) // 2
        end_pos = start_pos + target_size
        
        # 裁剪图像
        cropped_image = resized_image[start_pos[0]:end_pos[0], 
                                    start_pos[1]:end_pos[1], 
                                    start_pos[2]:end_pos[2]]
        
        return cropped_image, scale, start_pos, new_size

    def resize_image(self, image, target_size):
        """根据resize_mode选择合适的缩放方法"""
        if self.resize_mode == 'pad':
            resized_image, scale, pos, new_size = self.proportional_resize_with_pad(image, target_size)
            return resized_image
        elif self.resize_mode == 'crop':
            resized_image, scale, pos, new_size = self.proportional_resize_with_crop(image, target_size)
            return resized_image
        elif self.resize_mode == 'stretch':
            # 强制缩放（会变形）
            return resize(image, target_size, mode='constant', preserve_range=True)
        else:
            raise ValueError(f"Unknown resize_mode: {self.resize_mode}")

    def roi_crop(self, image):
        """ROI裁剪（可选使用）"""
        if image.ndim == 4:
            image = image[:, :, :, 0]
        elif image.ndim == 3:
            pass
        else:
            raise ValueError(f"意外的图像维度: {image.ndim}, 形状: {image.shape}")
        
        mask = image > 0
        coords = np.argwhere(mask)
        if len(coords) == 0:  # 如果没有非零值，返回原图像
            return image
        
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1
        cropped = image[x0:x1, y0:y1, z0:z1]
        
        return cropped

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample['image_path']
        description = sample['description']
        
        try:
            # 加载NIfTI文件
            img = nib.load(path)
            img_data = img.get_fdata()
            
            print(f"原始图像尺寸: {img_data.shape}")
            
            # 处理4D图像
            if img_data.ndim == 4:
                img_data = img_data[:, :, :, 0]  # 取第一个时间点或通道
            
            # 可选：ROI裁剪（去除背景）
            # img_data = self.roi_crop(img_data)
            # print(f"ROI裁剪后尺寸: {img_data.shape}")
            
            # 等比例调整到目标尺寸
            img_data = self.resize_image(img_data, self.target_size)
            print(f"缩放后尺寸: {img_data.shape}")
            
            # 数据增强
            if self.augmentation:
                random_n = torch.rand(1)
                random_i = 0.3*torch.rand(1)[0]+0.7
                if random_n[0] > 0.5:
                    img_data = np.flip(img_data, 0)
                img_data = img_data*random_i.data.cpu().numpy()
            
            # 归一化到[-1, 1]范围
            # 首先裁剪异常值
            img_min = np.percentile(img_data, 1)  # 使用1%分位数作为最小值
            img_max = np.percentile(img_data, 99)  # 使用99%分位数作为最大值
            img_data = np.clip(img_data, img_min, img_max)
            
            # 然后归一化
            if img_max > img_min:
                img_data = (img_data - img_min) / (img_max - img_min)
                img_data = img_data * 2 - 1
            else:
                img_data = np.zeros_like(img_data)
            
            # 转换为张量
            imageout = torch.from_numpy(img_data).float()
            
            if len(imageout.shape) == 3:  # 如果是3D图像，添加通道维度
                imageout = imageout.unsqueeze(0)  # 在第一个维度添加通道
            
            print(f"最终输出尺寸: {imageout.shape}")
            print(f"数值范围: {imageout.min().item():.3f} to {imageout.max().item():.3f}")
            
            return {'data': imageout, 'description': description}
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回一个默认的张量
            default_data = torch.zeros(1, *self.target_size)
            return {'data': default_data, 'description': ''}
    
    def debug_info(self, num_samples=3):
        """打印数据集的调试信息"""
        print(f"Dataset Information:")
        print(f"  Number of samples: {len(self)}")
        print(f"  Root dir: {self.root_dir}")
        print(f"  Augmentation: {self.augmentation}")
        print(f"  Target size: {self.target_size}")
        print(f"  Resize mode: {self.resize_mode}")
        
        if hasattr(self, 'samples') and self.samples:
            print("\nSample details:")
            for i in range(min(num_samples, len(self))):
                sample_info = self.samples[i]
                print(f"\nSample {i}:")
                print(f"  Path: {sample_info['image_path']}")
                print(f"  Description: {sample_info['description']}")
                try:
                    data_item = self[i]
                    print(f"  Final shape: {data_item['data'].shape}")
                    print(f"  Final type: {data_item['data'].dtype}")
                    print(f"  Final range: {data_item['data'].min().item():.3f} to {data_item['data'].max().item():.3f}")
                except Exception as e:
                    print(f"  Error processing sample: {e}")