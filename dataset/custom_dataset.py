
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


# class ADNIDataset(Dataset):
#     def __init__(self, root_dir='../ADNI', augmentation=False):
#         self.root_dir = root_dir
#         self.file_names = glob.glob(os.path.join(
#             root_dir, './**/*.nii'), recursive=True)
#         self.augmentation = augmentation

#     def __len__(self):
#         return len(self.file_names)

#     def roi_crop(self, image):
#     # 检查图像维度并适当处理
#         if image.ndim == 4:
#             # 如果是4D图像，提取第一个通道
#             image = image[:, :, :, 0]
#         elif image.ndim == 3:
#             # 如果已经是3D图像，不需要额外的操作
#             pass
#         else:
#             raise ValueError(f"意外的图像维度: {image.ndim}, 形状: {image.shape}")
        
#         # 剩余处理代码保持不变
#         mask = image > 0
#         coords = np.argwhere(mask)
#         x0, y0, z0 = coords.min(axis=0)
#         x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
#         cropped = image[x0:x1, y0:y1, z0:z1]
        
#         padded_crop = tio.CropOrPad(np.max(cropped.shape))(cropped.copy()[None])
        
#         # 确保返回的是4D格式，以符合函数的预期输出
#         if padded_crop.ndim == 3:
#             padded_crop = padded_crop[..., None]  # 添加通道维度
#         else:
#             padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))
        
#         return padded_crop

#     def __getitem__(self, index):
#         path = self.file_names[index]
#         img = nib.load(path)

#         img = np.swapaxes(img.get_data(), 1, 2)
#         img = np.flip(img, 1)
#         img = np.flip(img, 2)
#         img = self.roi_crop(image=img)
#         sp_size = 64
#         img = resize(img, (sp_size, sp_size, sp_size), mode='constant')
#         if self.augmentation:
#             random_n = torch.rand(1)
#             random_i = 0.3*torch.rand(1)[0]+0.7
#             if random_n[0] > 0.5:
#                 img = np.flip(img, 0)

#             img = img*random_i.data.cpu().numpy()

#         imageout = torch.from_numpy(img).float().view(
#             1, sp_size, sp_size, sp_size)
#         imageout = imageout*2-1

#         return {'data': imageout}


#get image and text input 
class CustomDataset(Dataset):
    def __init__(self, root_dir='../ADNI', csv_path=None, augmentation=False):
        self.root_dir = root_dir
        self.augmentation = augmentation
        
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

    def roi_crop(self, image):
        # 保持你原有的roi_crop方法不变
        if image.ndim == 4:
            image = image[:, :, :, 0]
        elif image.ndim == 3:
            pass
        else:
            raise ValueError(f"意外的图像维度: {image.ndim}, 形状: {image.shape}")
        
        mask = image > 0
        coords = np.argwhere(mask)
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1
        cropped = image[x0:x1, y0:y1, z0:z1]
        
        padded_crop = tio.CropOrPad(np.max(cropped.shape))(cropped.copy()[None])
        
        if padded_crop.ndim == 3:
            padded_crop = padded_crop[..., None]
        else:
            padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))
        
        return padded_crop

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample['image_path']
        description = sample['description']
        
        # 加载NIfTI文件
        img = nib.load(path)
        img_data = img.get_fdata()  # 使用get_fdata()替代已弃用的get_data()
        
        # 检查图像尺寸
        current_size = img_data.shape[0]  # 假设图像是立方体(相同尺寸)
        
        # 如果需要调整尺寸以适应网络(从128调整到64)
        # 如果不需要，可以移除这部分
        # sp_size = 128  # 或者根据你的网络需求设置为128
        # if current_size != sp_size:
        #     img_data = resize(img_data, (sp_size, sp_size, sp_size), mode='constant')
        
        # 数据增强(如果需要)
        # if self.augmentation:
        #     random_n = torch.rand(1)
        #     random_i = 0.3*torch.rand(1)[0]+0.7
        #     if random_n[0] > 0.5:
        #         img_data = np.flip(img_data, 0)
        #     img_data = img_data*random_i.data.cpu().numpy()
        
        # 转换为张量并规范化
        imageout = torch.from_numpy(img_data).float()  ##这里可以不规范化 选一个别的方法，因为数据已经修改好了🚧
        
        if len(imageout.shape) == 3:  # 如果是3D图像，添加通道维度
            imageout = imageout.unsqueeze(0)  # 在第一个维度添加通道
            # imageout = imageout*2-1  # 规范化到[-1,1]范围    ##已经被normalize过 🚧
        
        return {'data': imageout, 'description': description}
    
    def debug_info(self, num_samples=3):
        # """打印数据集的调试信息"""
        print(f"dataset infomation:")
        print(f" numeber of samples : {len(self)}")
        print(f"  root_dir: {self.root_dir}")
        print(f"  augment: {self.augmentation}")
        
        if hasattr(self, 'samples') and self.samples:
            print("\details:")
            for i in range(min(num_samples, len(self))):
                sample_info = self.samples[i]
                data_item = self[i]
                
                print(f"\sample {i}:")
                print(f"  dir: {sample_info['image_path']}")
                print(f"  des: {sample_info['description']}")
                print(f"  shape: {data_item['data'].shape}")
                print(f"  type: {data_item['data'].dtype}")
                print(f"  range: {data_item['data'].min().item()} to {data_item['data'].max().item()}")