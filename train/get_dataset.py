from dataset import MRNetDataset, BRATSDataset, DUKEDataset, LIDCDataset, DEFAULTDataset
from torch.utils.data import WeightedRandomSampler

# from dataset import ADNIDataset # change
from dataset import CustomDataset # new dataloader
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
    if cfg.dataset.name == 'DEFAULT': 
        # 检查配置中是否指定了CSV路径
        csv_path = cfg.dataset.get('csv_path', None)
        if csv_path is None:
            raise ValueError("csv_path must be specified in the config for this dataset.")
            
        train_dataset = CustomDataset( # 使用新的类名
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=cfg.dataset.get('augmentation', False))
        
        val_dataset = CustomDataset( # 使用新的类名
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=False)
            
        sampler = None
        train_dataset.debug_info(5) # 这个调试信息很有用，可以暂时保留
        return train_dataset, val_dataset, sampler
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