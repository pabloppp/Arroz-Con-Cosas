import torch
from torch import nn
import numpy as np
import math
    
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

class Attention2D(nn.Module):
    def __init__(self, c, nhead=8):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.attn = torch.nn.MultiheadAttention(c, nhead, bias=True, batch_first=True)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1) # Bx4xHxW -> Bx(HxW)x4
        norm_x = self.ln(x) 
        x = x + self.attn(norm_x, norm_x, norm_x, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class ResBlock(nn.Module):
    def __init__(self, c, c_hidden, c_cond=0, c_skip=0, scaler=None, layer_scale_init_value=1e-6, use_attention=False):
        super().__init__()
        if use_attention:
            self.depthwise = Attention2D(c)
        else:
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

class DiffusionModel(nn.Module):
    def __init__(self, c_hidden=1280, c_r=64, c_embd=1024, down_levels=[1, 2, 8, 32], up_levels=[32, 8, 2, 1], down_attn=[[], [0], range(4, 8, 2), range(16, 32, 2)], up_attn=[range(1, 16, 2), range(1, 4, 2), [1], []]):
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
            for j in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i]*4, c_r+c_embd, use_attention=j in down_attn[i])
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.ConvTranspose2d(c_levels[len(c_levels)-i], c_levels[len(c_levels)-1-i], kernel_size=4, stride=2, padding=1))
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels)-1-i], c_levels[len(c_levels)-1-i]*4, c_r+c_embd, c_levels[len(c_levels)-1-i] if (j == 0 and i > 0) else 0, use_attention=j in up_attn[i])
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            self.up_blocks.append(nn.ModuleList(blocks))
            
        self.clf = nn.ModuleList([nn.Conv2d(c_levels[i], 4, kernel_size=1) for i in reversed(range(len(up_levels)))])

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
        decoded_outputs = []
        for i, (blocks, clf) in enumerate(zip(self.up_blocks, self.clf)):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    if i > 0 and j == 1:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                else:
                    x = block(x)
            decoded_outputs.append(clf(x))
        return decoded_outputs

    def forward(self, x, r, c): # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x)
        s = torch.cat([c, r_embed], dim=1)[:, :, None, None]
        level_outputs = self._down_encode_(x, s)
        decoded_outputs = self._up_decode(level_outputs, s)
        x = torch.stack([nn.functional.interpolate(x, size=decoded_outputs[-1].shape[-2:], mode='bilinear') for x in decoded_outputs], dim=1).sum(dim=1)
        return x

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1-beta)
# ----
            
def to_latent(x, vqmodel):
    return vqmodel.encode(x)[0][1].contiguous()

def from_latent(x, vqmodel):
    return vqmodel.decode(x)