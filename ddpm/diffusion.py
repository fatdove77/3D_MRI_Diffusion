"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import copy
import torch
from einops import repeat

# import torch
import clip
from PIL import Image

from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import Dataset, DataLoader
from vq_gan_3d.model.vqgan import VQGAN

import matplotlib.pyplot as plt

# helpers functions


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # 如果维度是奇数，需要填充
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant', value=0)
            
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
    
    
    
    
    

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        
        # 确保groups是dim_out的因子
        effective_groups = groups
        while dim_out % effective_groups != 0 and effective_groups > 1:
            effective_groups -= 1
        
        self.norm = nn.GroupNorm(effective_groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # 确保groups是dim_out的因子
        effective_groups = groups
        while dim_out % effective_groups != 0 and effective_groups > 1:
            effective_groups -= 1
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=effective_groups)
        self.block2 = Block(dim_out, dim_out, groups=effective_groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        x: query张量，期望形状 [batch, sequence_length, query_dim]
        context: key/value张量，期望形状 [batch, context_length, context_dim] 
        """
        # 确保输入是3维的
        original_shape = x.shape
        if len(original_shape) != 3:
            raise ValueError(f"Expected 3D input for CrossAttention, got shape {original_shape}")
            
        h = self.heads
        batch_size, seq_len, query_dim = x.shape

        q = self.to_q(x)  # [batch, seq_len, inner_dim]
        context = default(context, x)
        k = self.to_k(context)  # [batch, context_len, inner_dim]
        v = self.to_v(context)  # [batch, context_len, inner_dim]

        # 重塑为多头注意力格式
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h) 
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        # 计算注意力分数
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # 应用softmax得到注意力权重
        attn = sim.softmax(dim=-1)

        # 应用注意力权重到value
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim=512, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        
        # 文本特征投影到图像特征空间
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.LayerNorm(dim)
        ) if context_dim != dim else nn.Identity()
        
        # 交叉注意力
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,  # 投影后都是dim维度
            heads=heads,
            dim_head=dim_head
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, context=None):
        """
        x: 图像特征 [b, c, d, h, w] 
        context: 文本特征 [b, seq_len, context_dim] 或 [b, context_dim]
        """
        if context is None:
            return x
            
        # 保存原始形状
        b, c, d, h, w = x.shape
        
        # 1. 将3D图像特征展平为序列
        x_flat = rearrange(x, 'b c d h w -> b (d h w) c')
        
        # 2. 处理文本特征维度
        if len(context.shape) == 2:  # [b, context_dim]
            context = context.unsqueeze(1)  # [b, 1, context_dim]
        
        # 3. 投影文本特征到图像特征维度
        context_proj = self.context_proj(context)  # [b, seq_len, dim]
        
        # 4. 应用层归一化
        x_norm = self.norm(x_flat)
        
        # 5. 交叉注意力
        attn_out = self.cross_attn(x_norm, context_proj)  # [b, d*h*w, dim]
        
        # 6. 残差连接
        x_out = x_flat + attn_out
        
        # 7. 恢复3D形状
        x_out = rearrange(x_out, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        
        return x_out

class AdaptiveCrossAttentionBlock(nn.Module):
    """
    修复后的自适应交叉注意力块
    """
    def __init__(self, dim, context_dim=512, heads=8, dim_head=None, chunk_size=None):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.chunk_size = chunk_size
        
        # 自动计算合适的头维度
        if dim_head is None:
            dim_head = max(32, dim // heads)
            
        # 确保heads能整除dim
        while dim % heads != 0 and heads > 1:
            heads -= 1
            
        self.heads = heads
        self.dim_head = dim_head
        
        # 上下文投影 - 支持任意输入维度
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        # 位置编码 - 可学习的3D位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # 交叉注意力
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head
        )
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        # 层归一化
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def _chunk_forward(self, x_flat, context_proj):
        """分块处理大尺寸图像"""
        if self.chunk_size is None or x_flat.shape[1] <= self.chunk_size:
            return self.cross_attn(x_flat, context_proj)
        
        # 分块处理
        chunks = []
        for i in range(0, x_flat.shape[1], self.chunk_size):
            chunk = x_flat[:, i:i+self.chunk_size]  # [b, chunk_size, dim]
            chunk_out = self.cross_attn(chunk, context_proj)
            chunks.append(chunk_out)
        
        return torch.cat(chunks, dim=1)
        
    def forward(self, x, context=None):
        """
        x: [b, c, d, h, w] - 3D图像特征，尺寸可变
        context: [b, seq_len, context_dim] 或 [b, context_dim] - 文本特征
        """
        if context is None:
            return x
            
        # 保存原始形状并展平
        orig_shape = x.shape
        b, c, d, h, w = orig_shape
        x_flat = rearrange(x, 'b c d h w -> b (d h w) c')
        # 关键修复：正确展平3D特征
        
        
        # 处理上下文维度
        if len(context.shape) == 2:  # [b, context_dim]
            context = context.unsqueeze(1)  # [b, 1, context_dim]
        elif len(context.shape) == 3:  # [b, seq_len, context_dim] 
            pass  # 已经是正确格式
        else:
            raise ValueError(f"Unexpected context shape: {context.shape}")
            
        # 投影上下文到图像特征空间
        context_proj = self.context_proj(context)  # [b, seq_len, dim]
        
        # 添加位置编码（可选）
        seq_len = x_flat.shape[1]
        if seq_len > 1 and hasattr(self, 'pos_encoding'):
            pos_enc = self.pos_encoding.expand(b, seq_len, -1)
            x_flat = x_flat + pos_enc
        
        # 第一个残差块：交叉注意力
        x_norm1 = self.norm1(x_flat)  # 保持3D
        attn_out = self._chunk_forward(x_norm1, context_proj)  # [b, d*h*w, dim]
        x_flat = x_flat + attn_out

        # 第二个残差块：前馈网络
        x_norm2 = self.norm2(x_flat)  # 保持3D
        ff_out = self.ff(x_norm2)
        x_flat = x_flat + ff_out
        
        # 恢复原始3D形状
        x_out = rearrange(x_flat, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        
        return x_out

# 用于检测和适配维度的工具函数
# def get_adaptive_cross_attention(dim, context_dim, image_size_3d=None):
#     """
#     根据图像尺寸自动选择合适的交叉注意力配置
#     """
#     if image_size_3d is None:
#         # 默认配置
#         return AdaptiveCrossAttentionBlock(dim, context_dim)
    
#     d, h, w = image_size_3d
#     total_spatial = d * h * w
    
#     # 根据空间尺寸选择配置
#     if total_spatial > 64 * 64 * 64:  # 大尺寸
#         chunk_size = 4096  # 分块处理
#         heads = min(8, dim // 64)
#     elif total_spatial > 32 * 32 * 32:  # 中等尺寸
#         chunk_size = 8192
#         heads = min(16, dim // 32)
#     else:  # 小尺寸
#         chunk_size = None
#         heads = min(32, dim // 16)
    
#     return AdaptiveCrossAttentionBlock(
#         dim=dim,
#         context_dim=context_dim,
#         heads=heads,
#         chunk_size=chunk_size
#     )


def get_adaptive_cross_attention(dim, context_dim, image_size_3d=None):
    return CrossAttentionBlock(dim, context_dim)


# 修改 Unet3D 中的时间嵌入部分
class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=512,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    ):
        super().__init__()
        self.channels = channels
        self.has_cond = cond_dim is not None
        
        # 文本条件处理
        if self.has_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim * 4)
            )
            self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim))
        
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): 
            return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
                dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 修复时间嵌入维度匹配
        time_dim = dim * 4
        
        # 使用固定的时间嵌入维度
        time_emb_dim = dim  # 基础时间嵌入维度
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),  # 输出 time_emb_dim 维度
            nn.Linear(time_emb_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 条件维度计算
        if self.has_cond:
            cond_dim_final = time_dim + time_dim  # time_dim + processed_cond_dim
        else:
            cond_dim_final = time_dim

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim_final)

        # modules for all layers
        # modules for all layers - ENCODER
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),  # block1
                block_klass_cond(dim_out, dim_out), # block2
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(), # spatial_attn
                Residual(PreNorm(dim_out, temporal_attn(dim_out))), # temporal_attn
                get_adaptive_cross_attention(dim_out, cond_dim) if self.has_cond else nn.Identity(), # cross_attn
                Downsample(dim_out) if not is_last else nn.Identity() # downsample
            ]))

        # 中间层交叉注意力
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        # 添加中间层的交叉注意力
        self.mid_cross_attn = get_adaptive_cross_attention(mid_dim, cond_dim) if self.has_cond else nn.Identity()

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # modules for all layers - DECODER  
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in), # block1 (注意这里输入维度*2，因为有skip connection)
                block_klass_cond(dim_in, dim_in),      # block2
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(), # spatial_attn
                Residual(PreNorm(dim_in, temporal_attn(dim_in))), # temporal_attn
                get_adaptive_cross_attention(dim_in, cond_dim) if self.has_cond else nn.Identity(), # cross_attn
                Upsample(dim_in) if not is_last else nn.Identity() # upsample
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
# gaussian diffusion trainer class
    def forward(
    self,
    x,
    time,
    cond=None,
    null_cond_prob=0.,
    focus_present_mask=None,
    prob_focus_present=0.
):
    # """
    # x: [b, c, d, h, w] - 3D图像特征
    # time: [b] - 时间步
    # cond: [b, context_dim] 或 [b, seq_len, context_dim] - 文本条件
    # """
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        
        batch, device = x.shape[0], x.device
        
        # 焦点掩码处理
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        # 初始卷积
        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        # 时间嵌入处理
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        # 文本条件处理
        cond_emb = None
        if self.has_cond and exists(cond):
            # 处理classifier free guidance
            if null_cond_prob > 0:
                # 创建null条件掩码
                null_mask = prob_mask_like((batch,), null_cond_prob, device=device)
                # 扩展null_cond_emb以匹配cond的形状
                if len(cond.shape) == 2:  # [b, context_dim]
                    null_cond = self.null_cond_emb.expand(batch, -1)
                else:  # [b, seq_len, context_dim]
                    null_cond = self.null_cond_emb.expand(batch, cond.shape[1], -1)
                
                # 根据掩码选择条件
                cond = torch.where(
                    null_mask.unsqueeze(-1) if len(cond.shape) == 2 else null_mask.unsqueeze(-1).unsqueeze(-1),
                    null_cond,
                    cond
                )
            
            # 投影文本条件
            if len(cond.shape) == 2:  # [b, context_dim]
                cond_emb = self.cond_proj(cond)  # [b, time_dim]
                # 与时间嵌入拼接
                t = torch.cat((t, cond_emb), dim=-1) if exists(t) else cond_emb
            else:  # [b, seq_len, context_dim] - 保持原始形状用于交叉注意力
                # 对于交叉注意力，我们保持cond的原始形状
                # 但仍需要创建一个全局条件嵌入与时间嵌入拼接
                cond_global = cond.mean(dim=1)  # [b, context_dim] - 平均池化作为全局特征
                cond_emb_global = self.cond_proj(cond_global)  # [b, time_dim]
                t = torch.cat((t, cond_emb_global), dim=-1) if exists(t) else cond_emb_global
        else:
            # 如果没有条件，cond设为None
            cond = None

        # Encoder阶段
        h = []
        
        for idx, (block1, block2, spatial_attn, temporal_attn, cross_attn, downsample) in enumerate(self.downs):
            # ResNet blocks with time/condition embedding
            x = block1(x, t)
            x = block2(x, t)
            
            # Spatial attention
            x = spatial_attn(x)
            
            # Temporal attention with positional bias
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                            focus_present_mask=focus_present_mask)
            
            # Cross attention with text condition
            x = cross_attn(x, context=cond)
            
            h.append(x)
            x = downsample(x)

        # Middle blocks
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, 
                                focus_present_mask=focus_present_mask)
        
        # Middle cross attention
        x = self.mid_cross_attn(x, context=cond)
        
        x = self.mid_block2(x, t)

        # Decoder阶段
        for idx, (block1, block2, spatial_attn, temporal_attn, cross_attn, upsample) in enumerate(self.ups):
            # Skip connection
            x = torch.cat((x, h.pop()), dim=1)
            
            # ResNet blocks with time/condition embedding
            x = block1(x, t)
            x = block2(x, t)
            
            # Spatial attention
            x = spatial_attn(x)
            
            # Temporal attention with positional bias
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                            focus_present_mask=focus_present_mask)
            
            # Cross attention with text condition
            x = cross_attn(x, context=cond)
            
            x = upsample(x)

        # Final output
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        """
        支持classifier free guidance的前向传播
        """
        if cond_scale == 1 or not self.has_cond:
            return self.forward(*args, null_cond_prob=0., **kwargs)

        # 有条件的预测
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        
        # 无条件的预测
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        
        # 应用guidance scale
        return null_logits + (logits - null_logits) * cond_scale

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        # 检查denoise_fn是否支持条件
        self.has_cond = hasattr(denoise_fn, 'has_cond') and denoise_fn.has_cond

        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
            # 冻结VQGAN参数
            for param in self.vqgan.parameters():
                param.requires_grad = False
        else:
            self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function
        def register_buffer(name, val): 
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', 
                       torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * 
                       torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * 
                       torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        # 使用条件缩放的前向传播
        if self.has_cond and cond is not None and hasattr(self.denoise_fn, 'forward_with_cond_scale'):
            predicted_noise = self.denoise_fn.forward_with_cond_scale(
                x, t, cond=cond, cond_scale=cond_scale
            )
        else:
            predicted_noise = self.denoise_fn(x, t, cond=cond)

        x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        
        # 初始化噪声
        img = torch.randn(shape, device=device)
        
        if return_intermediates:
            intermediates = []

        # 逐步去噪
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(
                img, 
                torch.full((b,), i, device=device, dtype=torch.long), 
                cond=cond, 
                cond_scale=cond_scale
            )
            
            if return_intermediates:
                intermediates.append(img.clone())

        if return_intermediates:
            return img, intermediates
        return img

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16, return_intermediates=False):
        device = next(self.denoise_fn.parameters()).device

        # 处理条件
        if cond is not None:
            if is_list_str(cond):
                # 如果是字符串列表，使用BERT编码（向后兼容）
                cond = bert_embed(tokenize(cond)).to(device)
            elif torch.is_tensor(cond):
                # 如果已经是张量(如CLIP编码的特征)，确保在正确的设备上
                cond = cond.to(device)
            batch_size = cond.shape[0]

        # 获取采样形状
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        
        sample_shape = (batch_size, channels, num_frames, image_size, image_size)
        
        # 采样
        sample = self.p_sample_loop(
            sample_shape, 
            cond=cond, 
            cond_scale=cond_scale,
            return_intermediates=return_intermediates
        )

        # 如果使用VQGAN，进行解码
        if isinstance(self.vqgan, VQGAN):
            if return_intermediates:
                final_sample, intermediates = sample
                # 对最终样本进行VQGAN解码
                final_sample = self._vqgan_decode(final_sample)
                return final_sample, intermediates
            else:
                sample = self._vqgan_decode(sample)

        return sample

    def _vqgan_decode(self, sample):
        """VQGAN解码辅助函数"""
        # 反归一化
        sample = ((sample + 1.0) / 2.0) * (
            self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min()
        ) + self.vqgan.codebook.embeddings.min()
        
        # VQGAN解码
        sample = self.vqgan.decode(sample, quantize=True)
        return sample

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 添加噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 处理文本条件
        if cond is not None:
            if is_list_str(cond):
                # BERT编码（向后兼容）
                cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
                cond = cond.to(device)
            elif torch.is_tensor(cond):
                # 已经是张量，确保在正确设备上
                cond = cond.to(device)
                
        # 预测噪声
        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        # 计算损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        return loss

    def forward(self, x, *args, **kwargs):
        """
        训练时的前向传播
        """
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                # VQGAN编码
                x = self.vqgan.encode(x, quantize=False, include_embeddings=True)
                
                # 归一化处理
                x = ((x - self.vqgan.codebook.embeddings.min()) /
                    (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            x = normalize_img(x)

        # 获取批次大小和设备
        b, c, f, h, w = x.shape
        device = x.device
        
        # 随机时间步
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # 计算损失
        return self.p_losses(x, t, *args, **kwargs)
# trainer class


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=16,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)

# trainer class

#--------------training-----------
class Trainer(object):
    def __init__(
        self,
        diffusion_model,   
        cfg,
        folder=None,
        dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
    ):
        super().__init__()
        
        # import clip model 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        #freeze the parms
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        
        self.model = diffusion_model  ##instance GaussianDiffusion class
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.cfg = cfg
        if dataset:
            self.ds = dataset
        else:
            assert folder is not None, 'Provide a folder path to the dataset'
            self.ds = Dataset(folder, image_size,
                              channels=channels, num_frames=num_frames)
        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(
            self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])



    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                data = batch['data'].cuda()
                description = batch['description']
                print("trainer data ✅✅✅✅✅:", data)
                print("trainer description ✅✅✅✅✅:", description)
                
                # use clip to embedd 
                with torch.no_grad():  # 不需要计算梯度
                    text_tokens = clip.tokenize(description).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # 可选：归一化特征向量 🚧 
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                with autocast(enabled=self.amp):
                    loss = self.model(
                        data,
                        cond=text_features,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            # ... 梯度裁剪和优化器步骤 ...
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 优化器步骤
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema_model.eval()
                
                # 为采样准备一些描述
                # 获取新的批次以获取描述
                sample_batch = next(self.dl)
                sample_descriptions = sample_batch['description'][:self.num_sample_rows**2]
                
                with torch.no_grad():
                    # 使用CLIP编码描述
                    text_tokens = clip.tokenize(sample_descriptions).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    
                    milestone = self.step // self.save_and_sample_every
                    num_samples = min(len(sample_descriptions), self.num_sample_rows**2)
                    
                    # 生成样本
                    all_videos_list = []
                    for i in range(num_samples):
                        sample = self.ema_model.sample(cond=text_features[i:i+1], batch_size=1)
                        all_videos_list.append(sample)
                    
                    # 如果样本数量不足，生成额外样本
                    extra_needed = self.num_sample_rows**2 - len(all_videos_list)
                    if extra_needed > 0:
                        extra_samples = self.ema_model.sample(batch_size=extra_needed)
                        all_videos_list.append(extra_samples)
                    
                    all_videos_list = torch.cat(all_videos_list, dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                # ... 其余的可视化和保存代码不变 ...
                one_gif = rearrange(
                    all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}

                # Selects one random 2D image from each 3D Image
                B, C, D, H, W = all_videos_list.shape
                frame_idx = torch.randint(0, D, [B]).cuda()
                frame_idx_selected = frame_idx.reshape(
                    -1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                frames = torch.gather(
                    all_videos_list, 2, frame_idx_selected).squeeze(2)

                path = str(self.results_folder /
                        f'sample-{milestone}.jpg')
                plt.figure(figsize=(50, 50))
                cols = 5
                for num, frame in enumerate(frames.cpu()):
                    plt.subplot(
                        math.ceil(len(frames) / cols), cols, num + 1)
                    plt.axis('off')
                    plt.imshow(frame[0], cmap='gray')
                    plt.savefig(path)

                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')
