import torch
from torch import nn
import numpy as np
import math
from pytorch_wavelets import DWTForward, DWTInverse
from .vq import VectorQuantize

class Waveletify(nn.Module):
    def __init__(self):
        super().__init__()
        self.xfm = DWTForward(J=1, mode='reflect', wave='db1')
        
    def forward(self, x):
        x = self.xfm(x)
        x = torch.cat([x[0], x[1][0][:, :, 0], x[1][0][:, :, 1], x[1][0][:, :, 2]], dim=1)
        return x
    
class Unwaveletify(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifm = DWTInverse(mode='reflect', wave='db1')
        
    def forward(self, x):
        ll, lh, hl, hh = x.chunk(4, dim=1)
        x = self.ifm((ll, [torch.stack([lh, hl, hh], dim=2)]))
        return x
    
class ModulatedLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, channels_first=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))
        self.channels_first = channels_first

    def forward(self, x, w=None):
        x = x.permute(0, 2, 3, 1) if self.channels_first else x
        if w is None:
            x = self.ln(x)
        else:
            x = self.gamma * w * self.ln(x) + self.beta * w
        x = x.permute(0, 3, 1, 2) if self.channels_first else x
        return x

class ResBlock(nn.Module):
    def __init__(self, c, c_hidden, c_cond=0, c_skip=0, scaler=None, layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, kernel_size=3, groups=c)
        )
        self.ln = ModulatedLayerNorm(c, channels_first=False)
        self.channelwise = nn.Sequential(
            nn.Linear(c+c_skip, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c),  requires_grad=True) if layer_scale_init_value > 0 else None
        self.scaler = scaler
        if c_cond > 0:
            self.cond_mapper = nn.Linear(c_cond, c)

    def forward(self, x, s=None, skip=None):
        res = x
        x = self.depthwise(x)
        if s is not None:
            s = self.cond_mapper(s.permute(0, 2, 3, 1))
            if s.size(1) == s.size(2) == 1:
                s = s.expand(-1, x.size(2), x.size(3), -1)
        x = self.ln(x.permute(0, 2, 3, 1), s)
        if skip is not None:
            x = torch.cat([x, skip.permute(0, 2, 3, 1)], dim=-1)
        x = self.channelwise(x)
        x = self.gamma * x if self.gamma is not None else x
        x = res + x.permute(0, 3, 1, 2)
        if self.scaler is not None:
            x = self.scaler(x)
        return x
    
class VQModule(nn.Module):
    def __init__(self, c_hidden, k):
        super().__init__()
        self.vquantizer = VectorQuantize(c_hidden, k=k, ema_loss=True)
        self.register_buffer('q_step_counter', torch.tensor(0))
    
    def forward(self, x, dim=-1):
        qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)        
        return qe, commit_loss, indices
    
class VQModel(nn.Module):
    def __init__(self, levels=3, bottleneck_blocks=32, c_hidden=480, c_latent=4, codebook_size=8192): # 3 levels = f8 (because of wavelets)
        super().__init__()
        c_levels = [c_hidden//(2**i) for i in reversed(range(levels))]
        
        # Encoder blocks
        self.in_block = nn.Conv2d(3*4, c_levels[0], kernel_size=1)
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i-1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = ResBlock(c_levels[i], c_levels[i]*4)
            block.channelwise[-1].weight.data *= np.sqrt(1 / levels)
            down_blocks.append(block)
        self.down_blocks = nn.Sequential(*down_blocks)
        self.latent_mapper = nn.Sequential(
            nn.Conv2d(c_levels[-1], c_latent, kernel_size=1),
            nn.BatchNorm2d(c_latent)
        )
        self.vqmodule = VQModule(c_latent, k=codebook_size)
        
        # Decoder blocks
        self.latent_unmapper = nn.Conv2d(c_latent, c_levels[-1], kernel_size=1)
        self.up_blocks = nn.ModuleList()
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels-1-i], c_levels[levels-1-i]*4)
                block.channelwise[-1].weight.data *= np.sqrt(1 / (levels+bottleneck_blocks))
                self.up_blocks.append(block)
            if i < levels-1:
                self.up_blocks.append(nn.ConvTranspose2d(c_levels[levels-1-i], c_levels[levels-2-i], kernel_size=4, stride=2, padding=1))
        self.out_block = nn.Conv2d(c_levels[0], 3*4, kernel_size=1)

        self.waveletify = Waveletify()
        self.unwaveletify = Unwaveletify()
        
    def encode(self, x):
        x = self.waveletify(x)
        x = self.in_block(x)
        x = self.down_blocks(x)
        x = self.latent_mapper(x)
        qe, commit_loss, indices = self.vqmodule(x, dim=1)
        return (x, qe), commit_loss, indices
    
    def decode(self, x):
        x = self.latent_unmapper(x)
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                x = block(x)
            else:
                x = block(x)
        x = self.out_block(x)
        x = self.unwaveletify(x)
        return x
        
    def forward(self, x, vq_mode=None):
        (_, qe), commit_loss, _ = self.encode(x)
        x = self.decode(qe)
        return x, commit_loss  

class PriorModel(nn.Module):
    def __init__(self, clip_r=1024, c_hidden=1280, c_r=64, num_blocks=48):
        super().__init__()
        self.c_r = c_r
        self.blocks = nn.ModuleList([
            nn.Linear(clip_r, c_hidden),
        ])
        for i in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(clip_r+c_hidden+c_r, c_hidden*4, bias=False),
                    nn.LayerNorm(c_hidden*4),
                    nn.GELU(),
                    nn.Linear(c_hidden*4, c_hidden, bias=False),
                    nn.LayerNorm(c_hidden),
                )
            )
            self.blocks[-1][0].weight.data *= np.sqrt(1 / num_blocks)
        self.blocks.append(nn.Linear(c_hidden, clip_r))
        
    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
        
    def forward(self, x, c, r):
        r = self.gen_r_embedding(r)
        x_prev = None
        for i, block in enumerate(self.blocks):
            if 0 < i < len(self.blocks)-1:
                x = torch.cat([x, c, r], dim=1)
            x = block(x) 
            if x_prev is not None and i < len(self.blocks)-1:
                x = x + x_prev
            x_prev = x
        return x

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1-beta)

class DiffusionModel(nn.Module):
    def __init__(self, c_hidden=1280, c_r=64, c_embd=1024, down_levels=[4, 12, 16], up_levels=[16, 12, 4]):
        super().__init__()
        self.c_r = c_r
        c_levels = [c_hidden//(2**i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Conv2d(4, c_levels[0], kernel_size=1)
        
        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv2d(c_levels[i-1], c_levels[i], kernel_size=4, stride=2, padding=1))
            for _ in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i]*4, c_r+c_embd)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels)-1-i], c_levels[len(c_levels)-1-i]*4, c_r+c_embd, c_levels[len(c_levels)-1-i] if (j == 0 and i > 0) else 0)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels)-1:
                blocks.append(nn.ConvTranspose2d(c_levels[len(c_levels)-1-i], c_levels[len(c_levels)-2-i], kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList(blocks))
            
        self.clf = nn.Conv2d(c_levels[0], 4, kernel_size=1)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    
    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    x = block(x, s)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    if i > 0 and j == 0:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                else:
                    x = block(x)
        return x

    def forward(self, x, c, r): # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x)
        s = torch.cat([c, r_embed], dim=1)[:, :, None, None]
        level_outputs = self._down_encode_(x, s)
        x = self._up_decode(level_outputs, s)
        x = self.clf(x)
        return x

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1-beta)

# ----
    
def to_latent(x, vqmodel):
    return vqmodel.encode(x)[0][1].contiguous()

def from_latent(x, vqmodel):
    return vqmodel.decode(x)

def sample(diffuzz, model, c, t_start=1.0, t_end=0.0, timesteps=20, size=(64, 64), x_init=None, mode='ddpm', inpaint_mask=None, x_orig=None, cfg=3.0, device='cpu'):
    r_range = torch.linspace(t_start, t_end, timesteps+1)[:, None].expand(-1, c.size(0)).to(device)
    x = torch.randn(c.size(0), 4, *size, device=device) if x_init is None else x_init.clone()
    preds = []
    for i in range(0, timesteps):
        if inpaint_mask is not None and x_orig is not None:
            x_renoised, _ = diffuzz.diffuse(x_orig, r_range[i])
            x = x * inpaint_mask + x_renoised * (1-inpaint_mask)
        pred_noise = model(x, c, r_range[i])
        if cfg is not None:
            pred_noise_unconditional = model(x, torch.zeros_like(c), r_range[i])
            pred_noise = torch.lerp(pred_noise_unconditional, pred_noise, cfg)
        x = diffuzz.undiffuse(x, r_range[i], r_range[i+1], pred_noise, mode=mode)
        preds.append(x)
    return preds

def sample_prior(diffuzz, model, c, t_start=1.0, t_end=0.0, timesteps=10, mode='ddpm', cfg=3.0, device='cpu'):
    r_range = torch.linspace(t_start, t_end, timesteps+1)[:, None].expand(-1, c.size(0)).to(device)
    x = torch.randn(c.size(0), 1024, device=device)
    preds = []
    for i in range(0, timesteps):
        pred_noise = model(x, c, r_range[i])
        if cfg is not None:
            pred_noise_unconditional = model(x, torch.zeros_like(c), r_range[i])
            pred_noise = torch.lerp(pred_noise_unconditional, pred_noise, cfg)
        x = diffuzz.undiffuse(x, r_range[i], r_range[i+1], pred_noise, mode=mode)
        preds.append(x)
    return preds