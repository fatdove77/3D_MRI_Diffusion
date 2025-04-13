from re import I
from ddpm import Unet3D, GaussianDiffusion, Trainer
from dataset import MRNetDataset, BRATSDataset
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from train.get_dataset import get_dataset
import torch
import os
from ddpm.unet import UNet
from vq_gan_3d.model.vqgan import VQGAN


# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)
    
    # 获取数据集
    train_dataset, *_ = get_dataset(cfg)
    sample_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    sample_batch = next(iter(sample_loader))
    sample_data = sample_batch['data'].cuda()
    
    print(f"原始数据形状: {sample_data.shape}")
    
    # 加载VQGAN并获取编码后的形状
    vqgan = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt).cuda()
    vqgan.eval()
    
    with torch.no_grad():
        encoded = vqgan.encode(sample_data, quantize=False, include_embeddings=True)
        print(f"VQGAN编码后形状: {encoded.shape}")
    
    # 动态设置尺寸参数
    _, channels, depth, height, width = encoded.shape
    
    # 计算合适的 groups 值
    original_groups = 8  # 默认组数
    groups = min(channels, original_groups)
    while channels % groups != 0 and groups > 1:
        groups -= 1
        
    print(f"使用参数: channels={channels}, groups={groups}")
    
    # 确保参数不为None
    with open_dict(cfg):
        if cfg.model.diffusion_img_size is None:
            cfg.model.diffusion_img_size = max(height, width)
            print(f"自动设置 diffusion_img_size = {cfg.model.diffusion_img_size}")
            
        if cfg.model.diffusion_depth_size is None:
            cfg.model.diffusion_depth_size = depth
            print(f"自动设置 diffusion_depth_size = {cfg.model.diffusion_depth_size}")
            
        if cfg.model.diffusion_num_channels is None:
            cfg.model.diffusion_num_channels = channels
            print(f"自动设置 diffusion_num_channels = {cfg.model.diffusion_num_channels}")
    
    # 现在创建模型，使用确保不为None的参数
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            resnet_groups=groups,  # 使用计算出的组数
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()
    

    train_dataset, *_ = get_dataset(cfg)

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        # logger=cfg.model.logger
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    trainer.train()


if __name__ == '__main__':
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
