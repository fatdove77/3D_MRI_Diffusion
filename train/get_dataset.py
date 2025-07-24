from dataset import MRNetDataset, BRATSDataset, DUKEDataset, LIDCDataset, DEFAULTDataset
from torch.utils.data import WeightedRandomSampler

# from dataset import ADNIDataset # change
from dataset import CustomDataset # new dataloader

#return image and text input
def get_dataset(cfg):
    if cfg.dataset.name == 'DEFAULT': 
        # 检查配置中是否指定了CSV路径
        csv_path = cfg.dataset.get('csv_path', None)
        if csv_path is None:
            raise ValueError("csv_path must be specified in the config for this dataset.")
        
        # 获取目标尺寸配置，默认为(64, 64, 64)
        target_size = cfg.dataset.get('target_size', (64, 64, 64))
        # 获取缩放模式，默认为'pad'
        resize_mode = cfg.dataset.get('resize_mode', 'pad')
        
        train_dataset = CustomDataset(
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=cfg.dataset.get('augmentation', False),
            target_size=target_size,
            resize_mode=resize_mode)
        
        val_dataset = CustomDataset(
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=False,
            target_size=target_size,
            resize_mode=resize_mode)
            
        sampler = None
        train_dataset.debug_info(3)
        return train_dataset, val_dataset, sampler
    
    if cfg.dataset.name == 'ADNI':
        # 检查配置中是否指定了CSV路径
        csv_path = cfg.dataset.get('csv_path', None)
        target_size = cfg.dataset.get('target_size', (64, 64, 64))
        resize_mode = cfg.dataset.get('resize_mode', 'pad')
        
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=cfg.dataset.augmentation,
            target_size=target_size,
            resize_mode=resize_mode)
        
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, 
            csv_path=csv_path,
            augmentation=False,
            target_size=target_size,
            resize_mode=resize_mode)
            
        sampler = None
        train_dataset.debug_info(3)
        return train_dataset, val_dataset, sampler
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')