"Adapted from https://github.com/SongweiGe/TATS"

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ddpm.diffusion import default
from vq_gan_3d.model import VQGAN 
from train.callbacks import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict
import torch

# 修复PyTorch 2.6的weights_only问题
import torch.serialization
torch.serialization.add_safe_globals([DictConfig])

torch.backends.cuda.matmul.allow_tf32 = False


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # 添加安全的全局变量
    import torch.serialization
    torch.serialization.add_safe_globals([DictConfig])
    
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(cfg)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    # 修复checkpoint加载逻辑
    resume_from_checkpoint = None
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    
    if os.path.exists(base_dir):
        try:
            log_folder = ckpt_file = ''
            version_id_used = -1
            
            # 找到最新的version文件夹
            for folder in os.listdir(base_dir):
                if folder.startswith('version_'):
                    try:
                        version_id = int(folder.split('_')[1])
                        if version_id > version_id_used:
                            version_id_used = version_id
                            log_folder = folder
                    except (ValueError, IndexError):
                        continue
            
            if log_folder and version_id_used >= 0:
                ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
                
                # 检查checkpoints文件夹是否存在
                if os.path.exists(ckpt_folder):
                    # 查找checkpoint文件
                    for fn in os.listdir(ckpt_folder):
                        if fn == 'latest_checkpoint.ckpt':
                            ckpt_file = 'latest_checkpoint_prev.ckpt'
                            old_path = os.path.join(ckpt_folder, fn)
                            new_path = os.path.join(ckpt_folder, ckpt_file)
                            os.rename(old_path, new_path)
                            break
                        elif fn.endswith('.ckpt'):
                            # 如果没有latest_checkpoint.ckpt，使用最新的ckpt文件
                            ckpt_file = fn
                    
                    if ckpt_file:
                        resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
                        print(f'将从最近的checkpoint继续训练: {resume_from_checkpoint}')
                else:
                    print(f'Checkpoints文件夹不存在: {ckpt_folder}')
                    print('将从头开始训练')
        except Exception as e:
            print(f'加载checkpoint时出错: {e}')
            print('将从头开始训练')

    # 更新配置
    with open_dict(cfg):
        cfg.model.resume_from_checkpoint = resume_from_checkpoint

    # 配置分布式训练
    strategy = None
    if cfg.model.gpus > 1:
        # 更稳定的DDP配置
        strategy = 'ddp_find_unused_parameters_false'
        

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.model.gpus,
        strategy=strategy,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint, 
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        sync_batchnorm=True,
        enable_progress_bar=True,
        logger=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    run()