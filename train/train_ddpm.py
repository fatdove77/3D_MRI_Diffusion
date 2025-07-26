from re import I
from ddpm import Unet3D, GaussianDiffusion, Trainer
from dataset import MRNetDataset, BRATSDataset
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from train.get_dataset import get_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
from ddpm.unet import UNet
from vq_gan_3d.model.vqgan import VQGAN


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式后端
    backend = os.environ.get('PL_TORCH_DISTRIBUTED_BACKEND', 'nccl')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    print(f"分布式训练初始化完成 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    return rank, world_size, local_rank


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 设置结果文件夹
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)
    
    # 只在主进程中打印信息
    def print_rank0(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    # 获取数据集 - 只在主进程中计算维度
    train_dataset, *_ = get_dataset(cfg)
    
    if rank == 0:
        sample_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
        sample_batch = next(iter(sample_loader))
        sample_data = sample_batch['data'].cuda()
        
        print_rank0(f"原始数据形状: {sample_data.shape}")
        
        # 加载VQGAN并获取编码后的形状
        vqgan = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt).cuda()
        vqgan.eval()
        
        with torch.no_grad():
            encoded = vqgan.encode(sample_data, quantize=False, include_embeddings=True)
            print_rank0(f"VQGAN编码后形状: {encoded.shape}")
        
        # 动态设置尺寸参数
        _, channels, depth, height, width = encoded.shape
        
        # 计算合适的 groups 值
        original_groups = 8
        groups = min(channels, original_groups)
        while channels % groups != 0 and groups > 1:
            groups -= 1
            
        print_rank0(f"使用参数: channels={channels}, groups={groups}")
        
        # 更新配置
        with open_dict(cfg):
            if cfg.model.diffusion_img_size is None:
                cfg.model.diffusion_img_size = max(height, width)
                print_rank0(f"自动设置 diffusion_img_size = {cfg.model.diffusion_img_size}")
                
            if cfg.model.diffusion_depth_size is None:
                cfg.model.diffusion_depth_size = depth
                print_rank0(f"自动设置 diffusion_depth_size = {cfg.model.diffusion_depth_size}")
                
            if cfg.model.diffusion_num_channels is None:
                cfg.model.diffusion_num_channels = channels
                print_rank0(f"自动设置 diffusion_num_channels = {cfg.model.diffusion_num_channels}")
    
    # 同步配置到所有进程
    if world_size > 1:
        dist.barrier()
        # 广播配置参数
        if rank != 0:
            # 非主进程使用默认值，稍后会被广播的值覆盖
            with open_dict(cfg):
                cfg.model.diffusion_img_size = cfg.model.diffusion_img_size or 64
                cfg.model.diffusion_depth_size = cfg.model.diffusion_depth_size or 16
                cfg.model.diffusion_num_channels = cfg.model.diffusion_num_channels or 8
        
        # 重新计算groups值（所有进程都需要）
        channels = cfg.model.diffusion_num_channels
        groups = min(channels, 8)
        while channels % groups != 0 and groups > 1:
            groups -= 1
    
    print_rank0(f"最终使用参数: img_size={cfg.model.diffusion_img_size}, "
                f"depth={cfg.model.diffusion_depth_size}, channels={cfg.model.diffusion_num_channels}")
    
    # 创建模型
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=64,  # 基础维度
            cond_dim=512,  # CLIP文本特征维度
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            resnet_groups=groups,
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    # DDP包装模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        print_rank0("模型已用DDP包装")

    # 创建扩散模型
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()

    # 创建分布式数据集
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        print_rank0("使用分布式采样器")
    else:
        sampler = None

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        sampler=sampler,
        num_workers=cfg.model.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=(sampler is None)  # 如果使用分布式采样器，则不能shuffle
    )

    # 创建自定义训练器（支持分布式）
    trainer = DistributedTrainer(
        diffusion,
        cfg=cfg,
        dataloader=train_dataloader,
        rank=rank,
        world_size=world_size,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
    )

    # 加载检查点（如果需要）
    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    # 开始训练
    print_rank0("开始训练...")
    trainer.train()
    print_rank0("训练完成!")

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


# 自定义分布式训练器
class DistributedTrainer:
    def __init__(self, diffusion_model, cfg, dataloader, rank, world_size, **kwargs):
        self.model = diffusion_model
        self.cfg = cfg
        self.dataloader = dataloader
        self.rank = rank
        self.world_size = world_size
        
        # 从kwargs中提取参数
        self.batch_size = kwargs.get('train_batch_size', 4)
        self.train_lr = kwargs.get('train_lr', 1e-4)
        self.train_num_steps = kwargs.get('train_num_steps', 100000)
        self.gradient_accumulate_every = kwargs.get('gradient_accumulate_every', 2)
        self.save_and_sample_every = kwargs.get('save_and_sample_every', 1000)
        self.results_folder = kwargs.get('results_folder', './results')
        self.amp = kwargs.get('amp', True)
        
        # 创建优化器
        model_params = self.model.denoise_fn.module.parameters() if self.world_size > 1 else self.model.denoise_fn.parameters()
        self.optimizer = torch.optim.Adam(model_params, lr=self.train_lr)
        
        # 混合精度训练
        from torch.cuda.amp import GradScaler
        self.scaler = GradScaler(enabled=self.amp)
        
        # CLIP模型用于文本编码
        import clip
        self.clip_model, _ = clip.load("ViT-B/32", device=f"cuda:{torch.cuda.current_device()}")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.step = 0
        
        # 创建结果文件夹（仅主进程）
        if self.rank == 0:
            os.makedirs(self.results_folder, exist_ok=True)
    
    def train(self):
        from torch.cuda.amp import autocast
        from tqdm import tqdm
        import clip
        
        # 创建数据迭代器
        data_iter = iter(self.dataloader)
        
        if self.rank == 0:
            pbar = tqdm(total=self.train_num_steps, desc="Training")
        
        while self.step < self.train_num_steps:
            # 设置模型为训练模式
            self.model.train()
            
            total_loss = 0.0
            
            # 梯度累积
            for i in range(self.gradient_accumulate_every):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # 重新创建迭代器
                    if hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
                        self.dataloader.sampler.set_epoch(self.step // len(self.dataloader))
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                
                # 准备数据
                data = batch['data'].cuda(non_blocking=True)
                descriptions = batch['description']
                
                # CLIP文本编码
                with torch.no_grad():
                    text_tokens = clip.tokenize(descriptions, truncate=True).cuda()
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # 前向传播
                with autocast(enabled=self.amp):
                    loss = self.model(data, cond=text_features)
                    loss = loss / self.gradient_accumulate_every
                
                # 反向传播
                self.scaler.scale(loss).backward()
                total_loss += loss.item()
            
            # 优化器步骤
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 同步损失（分布式训练）
            if self.world_size > 1:
                loss_tensor = torch.tensor(total_loss, device=torch.cuda.current_device())
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                total_loss = loss_tensor.item()
            
            # 更新进度条（仅主进程）
            if self.rank == 0:
                pbar.set_postfix({'loss': f'{total_loss:.6f}'})
                pbar.update(1)
            
            # 保存检查点（仅主进程）
            if self.rank == 0 and self.step % self.save_and_sample_every == 0 and self.step > 0:
                self.save_checkpoint()
            
            self.step += 1
        
        if self.rank == 0:
            pbar.close()
    
    def save_checkpoint(self):
        """保存检查点"""
        if self.rank != 0:
            return
        
        milestone = self.step // self.save_and_sample_every
        
        # 获取要保存的模型状态
        if self.world_size > 1:
            model_state = self.model.denoise_fn.module.state_dict()
        else:
            model_state = self.model.denoise_fn.state_dict()
        
        checkpoint = {
            'step': self.step,
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        
        save_path = os.path.join(self.results_folder, f'model-{milestone}.pt')
        torch.save(checkpoint, save_path)
        print(f"已保存检查点: {save_path}")
    
    def load(self, checkpoint_path):
        """加载检查点"""
        print(f"从 {checkpoint_path} 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{torch.cuda.current_device()}')
        
        # 加载模型状态
        if self.world_size > 1:
            self.model.denoise_fn.module.load_state_dict(checkpoint['model'])
        else:
            self.model.denoise_fn.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.step = checkpoint['step']
        
        print(f"检查点加载完成，当前步骤: {self.step}")


if __name__ == '__main__':
    run()