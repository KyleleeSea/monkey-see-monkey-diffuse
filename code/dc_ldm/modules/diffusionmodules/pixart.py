"""
PixArt-alpha backbone for Latent Diffusion Models.

Replaces UNetModel/DiT as the denoising backbone in the MinD-Vis pipeline.
Adapts the PixArt-alpha architecture (adaLN-single + cross-attention) for
fMRI-conditioned image generation.

Implements four block variants for comparison:
  - adaLN-single:         Faithful PixArt (shared t_block + per-block scale_shift_table + cross-attn)
  - adaLN-single-nocross: adaLN-single without cross-attention (pooled fMRI in timestep)
  - adaLN-zero:           Per-block adaLN-Zero modulation + cross-attention (DiT-PixArt hybrid)
  - full:                 adaLN-single + cross-attn + pooled fMRI injected into timestep

Reference: "Fast Training of Diffusion Transformers with Decomposed Generation"
           (Chen et al., 2023 — PixArt-alpha)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dc_ldm.modules.diffusionmodules.util import timestep_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift.  shift/scale have unsqueezed seq dim."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    """PixArt-style modulation where shift/scale already have the seq dim."""
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Patch Embedding & Unpatchify (reused from DiT, kept local for clarity)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D image to patch embedding via Conv2d."""

    def __init__(self, img_size=32, patch_size=2, in_channels=4, embed_dim=1152):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


def unpatchify(x, patch_size, channels, img_size):
    """Convert (B, N, patch_size**2 * C) back to (B, C, H, W)."""
    h = w = img_size // patch_size
    x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                  h=h, w=w, p1=patch_size, p2=patch_size, c=channels)
    return x


# ---------------------------------------------------------------------------
# Timestep Embedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP embedding."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_emb = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb)


# ---------------------------------------------------------------------------
# fMRI Context Pooler (for variants that inject pooled fMRI into timestep)
# ---------------------------------------------------------------------------

class ContextPooler(nn.Module):
    """Pool variable-length fMRI context into a single conditioning vector."""

    def __init__(self, context_dim, hidden_size, global_pool=False):
        super().__init__()
        self.global_pool = global_pool
        if global_pool:
            self.proj = nn.Linear(context_dim, hidden_size)
        else:
            self.pool = nn.Sequential(
                nn.Conv1d(77, 77 // 2, 1, bias=True),
                nn.Conv1d(77 // 2, 1, 1, bias=True),
            )
            self.proj = nn.Linear(context_dim, hidden_size)

    def forward(self, context):
        if self.global_pool:
            x = context.squeeze(1)
        else:
            x = self.pool(context).squeeze(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Cross-Attention (xformers-free, uses PyTorch SDPA)
# ---------------------------------------------------------------------------

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: query from image tokens, key/value from conditioning.

    Uses F.scaled_dot_product_attention (PyTorch 2.0+) for efficiency.
    """

    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        dropout_p = self.attn_drop if self.training else 0.0
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Self-Attention (standard PyTorch, no xformers dependency)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Standard multi-head self-attention using F.scaled_dot_product_attention."""

    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dropout_p = self.attn_drop if self.training else 0.0
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# PixArt Block Variants
# ---------------------------------------------------------------------------

class PixArtBlock_adaLN_single(nn.Module):
    """Faithful PixArt block: adaLN-single modulation + cross-attention.

    Uses a shared t_block output (6*D) combined with a per-block learnable
    scale_shift_table to produce 6 modulation vectors:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    Cross-attention to fMRI conditioning tokens in every block.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio))
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        """
        x: (B, N, D) image tokens
        y: (B, S, D) fMRI conditioning tokens (already projected to hidden_size)
        t: (B, 6*D) shared timestep modulation from t_block
        """
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # Self-attention with adaLN-single modulation
        h = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h)
        x = x + gate_msa * h.reshape(B, N, C)

        # Cross-attention to fMRI tokens
        x = x + self.cross_attn(x, y, mask)

        # MLP with adaLN-single modulation
        h = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)

        return x


class PixArtBlock_adaLN_single_nocross(nn.Module):
    """PixArt block with adaLN-single but NO cross-attention.

    fMRI conditioning is handled externally by pooling into the timestep
    embedding. This tests whether adaLN-single alone suffices for fMRI.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio))
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        h = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h)
        x = x + gate_msa * h.reshape(B, N, C)

        h = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)

        return x


class PixArtBlock_adaLN_zero(nn.Module):
    """Hybrid block: DiT-style per-block adaLN-Zero + PixArt cross-attention.

    Each block has its own modulation MLP (adaLN-Zero with gating),
    plus cross-attention to fMRI tokens for rich conditioning.
    """

    def __init__(self, hidden_size, num_heads, context_dim=512,
                 mlp_ratio=4.0, drop_path=0., **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio))
        # Per-block adaLN-Zero: 6 params (shift/scale/gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, y, t, mask=None, **kwargs):
        """
        t: (B, D) conditioning vector (timestep + optionally pooled fMRI)
        y: (B, S, D) fMRI conditioning tokens
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(t).chunk(6, dim=-1)

        # Self-attention with adaLN-Zero gating
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.attn(h)
        x = x + gate1.unsqueeze(1) * h

        # Cross-attention to fMRI tokens
        x = x + self.cross_attn(self.norm_cross(x), y, mask)

        # MLP with adaLN-Zero gating
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)

        return x


# Alias for the "full" variant — same block structure as adaLN-single,
# the difference is handled at the model level (pooled fMRI added to timestep)
PixArtBlock_full = PixArtBlock_adaLN_single


# ---------------------------------------------------------------------------
# Final Layers
# ---------------------------------------------------------------------------

class T2IFinalLayer(nn.Module):
    """PixArt-style final layer with per-layer scale_shift_table."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, t):
        """t: (B, D) raw timestep embedding (NOT the 6D t_block output)."""
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AdaLNFinalLayer(nn.Module):
    """DiT-style final layer with adaLN modulation (for adaLN-zero variant)."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# 2D Sin-Cos Positional Embedding
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sin-cos positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: int, spatial grid size (grid_size x grid_size patches).

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) numpy array.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# ---------------------------------------------------------------------------
# Block Registry
# ---------------------------------------------------------------------------

PIXART_BLOCK_REGISTRY = {
    'adaLN-single': PixArtBlock_adaLN_single,
    'adaLN-single-nocross': PixArtBlock_adaLN_single_nocross,
    'adaLN-zero': PixArtBlock_adaLN_zero,
    'full': PixArtBlock_full,
}

# Block types that use adaLN-single (shared t_block + scale_shift_table)
ADALN_SINGLE_TYPES = {'adaLN-single', 'adaLN-single-nocross', 'full'}

# Block types that use per-block adaLN (conditioning vector directly)
ADALN_PERBLOCK_TYPES = {'adaLN-zero'}


# ---------------------------------------------------------------------------
# Main PixArt Model for fMRI
# ---------------------------------------------------------------------------

class PixArtForFMRI(nn.Module):
    """PixArt-alpha backbone adapted for fMRI-conditioned latent diffusion.

    Drop-in replacement for UNetModel / DiT. Accepts the same forward signature:
        forward(x, timesteps, context=None, y=None, **kwargs)

    where context is (B, seq_len, context_dim) fMRI conditioning from cond_stage_model.

    Args:
        image_size: Spatial size of the latent (default 32).
        patch_size: Patch size for patchification (default 2).
        in_channels: Number of input latent channels (default 4).
        out_channels: Number of output channels (default 4).
        hidden_size: Transformer hidden dimension (default 1152).
        depth: Number of transformer blocks (default 28).
        num_heads: Number of attention heads (default 16).
        mlp_ratio: FFN hidden dim multiplier (default 4.0).
        context_dim: Dimension of fMRI conditioning embeddings (default 512).
        block_type: One of 'adaLN-single', 'adaLN-single-nocross', 'adaLN-zero', 'full'.
        use_time_cond: Whether to add pooled fMRI to timestep embedding.
        global_pool: Whether fMRI encoder uses global pooling (1 token vs 77).
        dropout: Dropout rate (default 0.0).
        use_checkpoint: Whether to use gradient checkpointing (default False).
    """

    def __init__(
        self,
        image_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        context_dim=512,
        block_type='adaLN-single',
        use_time_cond=True,
        global_pool=False,
        dropout=0.0,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.block_type = block_type
        self.use_time_cond = use_time_cond
        self.global_pool = global_pool
        self.use_checkpoint = use_checkpoint

        # --- Patch embedding ---
        self.x_embedder = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches

        # --- Fixed sin-cos positional embedding ---
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        # --- Timestep embedding ---
        self.t_embedder = TimestepEmbedder(hidden_size)

        # --- Shared t_block for adaLN-single variants ---
        if block_type in ADALN_SINGLE_TYPES:
            self.t_block = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )
        else:
            self.t_block = None

        # --- fMRI context projection (context_dim -> hidden_size for cross-attn) ---
        if block_type != 'adaLN-single-nocross':
            self.context_proj = nn.Linear(context_dim, hidden_size)
        else:
            self.context_proj = None

        # --- fMRI pooler for timestep injection ---
        # For nocross: must pool fMRI into timestep since there's no cross-attn
        # For full: additionally inject pooled fMRI on top of adaLN-single
        # For adaLN-zero: pool for the per-block conditioning vector
        if block_type == 'adaLN-single-nocross' or block_type == 'full':
            self.context_pooler = ContextPooler(context_dim, hidden_size, global_pool)
        elif block_type == 'adaLN-zero':
            self.context_pooler = ContextPooler(context_dim, hidden_size, global_pool) \
                if use_time_cond else None
        else:
            self.context_pooler = None

        # --- Transformer blocks ---
        BlockClass = PIXART_BLOCK_REGISTRY[block_type]
        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        if block_type == 'adaLN-zero':
            block_kwargs['context_dim'] = context_dim

        self.blocks = nn.ModuleList([
            BlockClass(**block_kwargs) for _ in range(depth)
        ])

        # --- Final layer ---
        if block_type in ADALN_SINGLE_TYPES:
            self.final_layer = T2IFinalLayer(hidden_size, patch_size, out_channels)
        else:
            self.final_layer = AdaLNFinalLayer(hidden_size, patch_size, out_channels)

        # --- Initialize weights ---
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Fixed sin-cos positional embedding
        grid_size = self.image_size // self.patch_size
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.zeros_(self.x_embedder.proj.bias)

        # Timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # t_block
        if self.t_block is not None:
            nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Context projection
        if self.context_proj is not None:
            nn.init.normal_(self.context_proj.weight, std=0.02)

        # Zero-initialize cross-attention output projections for stable init
        for block in self.blocks:
            if hasattr(block, 'cross_attn'):
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-initialize final layer output
        if hasattr(self.final_layer, 'linear'):
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Forward pass matching DiffusionWrapper interface.

        Args:
            x: (B, C, H, W) noisy latent images.
            timesteps: (B,) diffusion timestep indices.
            context: (B, seq_len, context_dim) fMRI conditioning embeddings.
            y: unused (class labels for class-conditional models).

        Returns:
            (B, C, H, W) predicted noise.
        """
        # 1. Patchify + positional embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = x + self.pos_embed

        # 2. Timestep embedding
        t = self.t_embedder(timesteps)  # (B, D)

        # 3. Prepare conditioning based on block type
        if self.block_type in ADALN_SINGLE_TYPES:
            # --- adaLN-single path ---
            # Optionally inject pooled fMRI into timestep before t_block
            if self.context_pooler is not None and context is not None:
                t = t + self.context_pooler(context)

            t0 = self.t_block(t)  # (B, 6*D) — shared modulation for all blocks

            # Project fMRI context for cross-attention
            if self.context_proj is not None and context is not None:
                y_cond = self.context_proj(context)  # (B, S, D)
            else:
                y_cond = None

            # Apply blocks
            if self.use_checkpoint:
                for block in self.blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, y_cond, t0, use_reentrant=False)
            else:
                for block in self.blocks:
                    x = block(x, y_cond, t0)

            # Final layer uses raw timestep embedding
            x = self.final_layer(x, t)

        elif self.block_type == 'adaLN-zero':
            # --- Per-block adaLN-Zero path ---
            c = t
            if self.context_pooler is not None and context is not None:
                c = c + self.context_pooler(context)

            # Project fMRI context for cross-attention
            if self.context_proj is not None and context is not None:
                y_cond = self.context_proj(context)
            else:
                y_cond = None

            if self.use_checkpoint:
                for block in self.blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, y_cond, c, use_reentrant=False)
            else:
                for block in self.blocks:
                    x = block(x, y_cond, c)

            x = self.final_layer(x, c)

        # 4. Unpatchify
        x = unpatchify(x, self.patch_size, self.out_channels, self.image_size)
        return x
