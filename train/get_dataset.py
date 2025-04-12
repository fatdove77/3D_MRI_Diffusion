from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset
from torch.utils.data import WeightedRandomSampler


# def get_dataset(cfg):
#     if cfg.dataset.name == 'ADNI':
#         train_dataset = ADNIDataset(
#             root_dir=cfg.dataset.root_dir, augmentation=True)
#         val_dataset = ADNIDataset(
#             root_dir=cfg.dataset.root_dir, augmentation=True)
#         sampler = None
#         return train_dataset, val_dataset, sampler
#     raise ValueError(f'{cfg.dataset.name} Dataset is not available')



#return image and text input
def get_dataset(cfg):
    if cfg.dataset.name == 'ADNI':
        # 检查配置中是否指定了CSV路径
        csv_path = cfg.dataset.get('csv_path', None)
        
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=cfg.dataset.augmentation)
        
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=False)  # 验证集通常不需要数据增强
            
        sampler = None
        train_dataset.debug_info(5)  ###查看输入的图像的信息 
        return train_dataset, val_dataset, sampler
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')