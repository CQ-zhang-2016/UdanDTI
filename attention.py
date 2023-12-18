import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        ]))

    def forward(self, x, cls_):
        x = torch.cat((cls_, x), dim=1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, vd_dim, vp_dim, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            PreNorm(vp_dim, Attention(vp_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(vd_dim, Attention(vd_dim, heads = heads, dim_head = dim_head, dropout = dropout))
        ]))

    def forward(self, vd_tokens, vp_tokens, vd_cls, vp_cls):
        

        for vd_attend_vp, vp_attend_vd in self.layers:
            vd_cls1 = vd_attend_vp(vp_cls, context = vd_tokens, kv_include_self = True)
            vp_cls1 = vp_attend_vd(vd_cls, context = vp_tokens, kv_include_self = True)

        return vd_cls1, vp_cls1

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        vd_dim,
        vp_dim,
        vd_enc_params,
        vp_enc_params,
        cross_attn_heads,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = vd_dim, dropout = dropout, **vp_enc_params),
                Transformer(dim = vp_dim, dropout = dropout, **vd_enc_params),
                CrossTransformer(vd_dim = vd_dim, vp_dim = vp_dim, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, v_p, v_d, cls_tokens):
        for vp_enc, vd_enc, cross_attend in self.layers:
            vd_tokens, vp_tokens = vd_enc(v_d, cls_tokens), vp_enc(v_p, cls_tokens)
            (vd_cls, vd_tokens), (vp_cls, vp_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (vd_tokens, vp_tokens))
            cls_tokens = 0.2 * cls_tokens + 0.4 * vd_cls + 0.4 * vp_cls
            
            vd_cls1, vp_cls1 = cross_attend(vd_tokens, vp_tokens, vd_cls, vp_cls)
            cls_tokens = 0.2 * cls_tokens + 0.4 * vd_cls1 + 0.4 * vp_cls1

        return cls_tokens



class aggregation(nn.Module):
    def __init__(self, depth):
        super().__init__()

        mlp_hidden_dim = 32

        vd_dim = 128
        vd_enc_heads = 2
        vd_enc_mlp_dim = 512
        vd_enc_dim_head = 32

        vp_dim = 128
        vp_enc_heads = 4
        vp_enc_mlp_dim = 512
        vp_enc_dim_head = 32

        cross_attn_heads = 4
        cross_attn_dim_head = 64
        dropout = 0.1

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            vd_dim = vd_dim,
            vp_dim = vp_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            vd_enc_params = dict(
                heads = vd_enc_heads,
                mlp_dim = vd_enc_mlp_dim,
                dim_head = vd_enc_dim_head
            ),
            vp_enc_params = dict(
                heads = vp_enc_heads,
                mlp_dim = vp_enc_mlp_dim,
                dim_head = vp_enc_dim_head
            ),
            dropout = dropout
        )
        
        
    def forward(self, v_p, v_d, cls_tokens):
        cls_tokens = self.multi_scale_encoder(v_p, v_d, cls_tokens)
        return cls_tokens.squeeze(1)
