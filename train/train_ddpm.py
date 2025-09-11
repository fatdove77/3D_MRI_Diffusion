from ddpm import Unet3D, GaussianDiffusion, Trainer
import hydra
from omegaconf import DictConfig, open_dict
from train.get_dataset import get_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
from vq_gan_3d.model.vqgan import VQGAN
from transformers import AutoTokenizer, AutoModel


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
    
    backend = os.environ.get('PL_TORCH_DISTRIBUTED_BACKEND', 'nccl')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    print(f"分布式训练初始化完成 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    return rank, world_size, local_rank


class BioBERTEncoder:
    """BioBERT文本编码器"""
    def __init__(self, model_name='dmis-lab/biobert-v1.1', device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # 冻结BioBERT参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_text(self, texts, max_length=256):
        """
        编码文本为768维特征向量
        Args:
            texts: 文本列表
            max_length: 最大序列长度（医学文本可能较长）
        Returns:
            text_features: [batch_size, 768]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # 获取BioBERT输出
        outputs = self.model(**inputs)
        
        # 使用[CLS] token的表示作为句子表示
        # hidden_states shape: [batch_size, seq_len, 768]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        return cls_embeddings


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 设置结果文件夹
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)
    
    def print_rank0(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    # 获取数据集
    train_dataset, *_ = get_dataset(cfg)
    
    # 动态获取VQGAN编码后的维度（仅主进程）
    if rank == 0:
        sample_loader = DataLoader(train_dataset, batch_size=1)
        sample_batch = next(iter(sample_loader))
        sample_data = sample_batch['data'].cuda()
        
        print_rank0(f"原始数据形状: {sample_data.shape}")
        
        # 加载VQGAN并获取编码后的形状
        vqgan = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt).cuda()
        vqgan.eval()
        
        with torch.no_grad():
            encoded = vqgan.encode(sample_data, quantize=False, include_embeddings=True)
            print_rank0(f"VQGAN编码后形状: {encoded.shape}")
        
        _, channels, depth, height, width = encoded.shape
        
        # 计算合适的 groups 值
        groups = min(channels, 8)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        
        print_rank0(f"使用参数: channels={channels}, groups={groups}")
        
        # 更新配置
        with open_dict(cfg):
            if cfg.model.diffusion_img_size is None:
                cfg.model.diffusion_img_size = max(height, width)
            if cfg.model.diffusion_depth_size is None:
                cfg.model.diffusion_depth_size = depth
            if cfg.model.diffusion_num_channels is None:
                cfg.model.diffusion_num_channels = channels
    
    # 同步配置到所有进程
    if world_size > 1:
        dist.barrier()
        if rank != 0:
            with open_dict(cfg):
                cfg.model.diffusion_img_size = cfg.model.diffusion_img_size or 64
                cfg.model.diffusion_depth_size = cfg.model.diffusion_depth_size or 16
                cfg.model.diffusion_num_channels = cfg.model.diffusion_num_channels or 8
        
        channels = cfg.model.diffusion_num_channels
        groups = min(channels, 8)
        while channels % groups != 0 and groups > 1:
            groups -= 1
    
    print_rank0(f"最终参数: img_size={cfg.model.diffusion_img_size}, "
                f"depth={cfg.model.diffusion_depth_size}, channels={cfg.model.diffusion_num_channels}")
    
    # 创建模型 - 注意这里条件维度改为768（BioBERT的输出维度）
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=64,
            cond_dim=768,  # BioBERT特征维度
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            resnet_groups=groups,
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
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

    # 创建数据加载器
    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                     rank=rank, shuffle=True)
    else:
        sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        sampler=sampler,
        num_workers=cfg.model.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=(sampler is None)
    )

    # 创建训练器
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

    # 加载检查点
    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    # 开始训练
    print_rank0("开始训练...")
    trainer.train()
    print_rank0("训练完成!")

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


class DistributedTrainer:
    """分布式训练器"""
    def __init__(self, diffusion_model, cfg, dataloader, rank, world_size, **kwargs):
        self.model = diffusion_model
        self.cfg = cfg
        self.dataloader = dataloader
        self.rank = rank
        self.world_size = world_size
        
        # 训练参数
        self.batch_size = kwargs.get('train_batch_size', 4)
        self.train_lr = kwargs.get('train_lr', 1e-4)
        self.train_num_steps = kwargs.get('train_num_steps', 100000)
        self.gradient_accumulate_every = kwargs.get('gradient_accumulate_every', 2)
        self.save_and_sample_every = kwargs.get('save_and_sample_every', 1000)
        self.results_folder = kwargs.get('results_folder', './results')
        self.amp = kwargs.get('amp', True)
        
        # 创建优化器
        model_params = (self.model.denoise_fn.module.parameters() 
                       if self.world_size > 1 
                       else self.model.denoise_fn.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=self.train_lr)
        
        # 混合精度训练
        from torch.cuda.amp import GradScaler
        self.scaler = GradScaler(enabled=self.amp)
        
        # BioBERT文本编码器
        self.text_encoder = BioBERTEncoder(
            model_name=cfg.model.get('biobert_model', 'dmis-lab/biobert-v1.1'),
            device=f"cuda:{torch.cuda.current_device()}"
        )
        
        self.step = 0
        
        # 创建结果文件夹（仅主进程）
        if self.rank == 0:
            os.makedirs(self.results_folder, exist_ok=True)
    
    def train(self):
        """训练循环"""
        from torch.cuda.amp import autocast
        from tqdm import tqdm
        
        data_iter = iter(self.dataloader)
        
        if self.rank == 0:
            pbar = tqdm(total=self.train_num_steps, desc="Training")
        
        while self.step < self.train_num_steps:
            self.model.train()
            total_loss = 0.0
            
            # 梯度累积
            for i in range(self.gradient_accumulate_every):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
                        self.dataloader.sampler.set_epoch(self.step // len(self.dataloader))
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                
                # 准备数据
                data = batch['data'].cuda(non_blocking=True)
                descriptions = batch['description']
                
                # BioBERT文本编码
                text_features = self.text_encoder.encode_text(descriptions)
                
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
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = loss_tensor.item() / self.world_size
            
            # 更新进度条（仅主进程）
            if self.rank == 0:
                pbar.set_postfix({'loss': f'{total_loss:.6f}'})
                pbar.update(1)
            
            # 保存检查点和采样（仅主进程）
            if self.rank == 0 and self.step % self.save_and_sample_every == 0 and self.step > 0:
                self.save_checkpoint()
                self.sample_and_save()
            
            self.step += 1
        
        if self.rank == 0:
            pbar.close()
    
    def sample_and_save(self):
        """生成样本并保存"""
        if self.rank != 0:
            return
        
        self.model.eval()
        with torch.no_grad():
            # 从数据集中获取真实的文本描述
            sample_batch = next(iter(self.dataloader))
            sample_descriptions = sample_batch['description'][:self.cfg.model.num_sample_rows**2]
            
            # 编码文本
            text_features = self.text_encoder.encode_text(sample_descriptions)
            
            # 生成样本
            samples = self.model.sample(
                cond=text_features,
                batch_size=len(sample_descriptions)
            )
            
            # 保存样本
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 如果是3D数据，选择中间切片
            if len(samples.shape) == 5:  # [B, C, D, H, W]
                B, C, D, H, W = samples.shape
                # 选择中间切片
                middle_slice = D // 2
                samples_2d = samples[:, :, middle_slice, :, :]
            else:
                samples_2d = samples
            
            # 创建图像网格
            milestone = self.step // self.save_and_sample_every
            save_path = os.path.join(self.results_folder, f'samples_step_{self.step}.png')
            
            # 反归一化如果需要
            samples_2d = (samples_2d + 1) * 0.5  # 从[-1,1]到[0,1]
            samples_2d = torch.clamp(samples_2d, 0, 1)
            
            # 绘制图像网格
            fig, axes = plt.subplots(
                self.cfg.model.num_sample_rows, 
                self.cfg.model.num_sample_rows, 
                figsize=(15, 15)
            )
            
            for i in range(min(len(samples_2d), self.cfg.model.num_sample_rows**2)):
                row = i // self.cfg.model.num_sample_rows
                col = i % self.cfg.model.num_sample_rows
                
                if self.cfg.model.num_sample_rows == 1:
                    ax = axes
                else:
                    ax = axes[row, col]
                
                # 显示图像
                img = samples_2d[i, 0].cpu().numpy()  # 取第一个通道
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                # 添加文本描述作为标题（截断以适应显示）
                title = sample_descriptions[i][:50] + '...' if len(sample_descriptions[i]) > 50 else sample_descriptions[i]
                ax.set_title(title, fontsize=8, wrap=True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"保存了 {len(samples_2d)} 个样本到 {save_path}")
    
    def save_checkpoint(self):
        """保存检查点"""
        if self.rank != 0:
            return
        
        milestone = self.step // self.save_and_sample_every
        
        model_state = (self.model.denoise_fn.module.state_dict() 
                      if self.world_size > 1 
                      else self.model.denoise_fn.state_dict())
        
        checkpoint = {
            'step': self.step,
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'cfg': self.cfg
        }
        
        save_path = os.path.join(self.results_folder, f'model-{milestone}.pt')
        torch.save(checkpoint, save_path)
        print(f"已保存检查点: {save_path}")
    
    def load(self, checkpoint_path):
        """加载检查点"""
        print(f"从 {checkpoint_path} 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{torch.cuda.current_device()}')
        
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