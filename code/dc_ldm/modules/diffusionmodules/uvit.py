"""
U-ViT backbone for Latent Diffusion Models, adapted for fMRI conditioning.

Based on U-ViT from Bao et al. (https://github.com/baofff/u-vit).
U-ViT uses a ViT-based U-Net architecture with long skip connections
between encoder and decoder transformer blocks. Conditioning (timestep,
class labels / fMRI) is injected as prepended tokens in the sequence.

This file adapts U-ViT to accept fMRI context embeddings instead of
integer class labels, following the token-based conditioning paradigm.

Reference: "All are Worth Words: A ViT Backbone for Diffusion Models"
           (Bao et al., CVPR 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from dc_ldm.modules.diffusionmodules.util import timestep_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unpatchify(x, channels):
    """Convert (B, N, patch_dim) back to (B, C, H, W) using einops.

    Assumes square images and square patches. N = (H/p)^2, patch_dim = p^2 * C.
    """
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                  h=h, w=w, p1=patch_size, p2=patch_size, c=channels)
    return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D image to patch embedding via Conv2d."""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with support for multiple backends.

    Uses native scaled_dot_product_attention (flash/memory-efficient when
    available) for best performance.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's native scaled_dot_product_attention
        # (automatically selects flash / memory-efficient / math backend)
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block (with optional skip connection)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """U-ViT transformer block.

    For encoder (in_blocks): standard pre-norm transformer block.
    For decoder (out_blocks): includes skip_linear to merge encoder activations.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, skip, use_reentrant=False)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# fMRI Context Pooler (reused from DiT, pools context to single token)
# ---------------------------------------------------------------------------

class ContextPooler(nn.Module):
    """Pool variable-length fMRI context into a single conditioning vector.

    For global_pool=True input is (B, 1, context_dim).
    For global_pool=False input is (B, 77, context_dim).
    Output is always (B, context_dim).
    """

    def __init__(self, context_dim, global_pool=False):
        super().__init__()
        self.global_pool = global_pool
        if not global_pool:
            self.pool = nn.Sequential(
                nn.Conv1d(77, 77 // 2, 1, bias=True),
                nn.Conv1d(77 // 2, 1, 1, bias=True),
            )

    def forward(self, context):
        if self.global_pool:
            return context.squeeze(1)  # (B, context_dim)
        else:
            return self.pool(context).squeeze(1)  # (B, context_dim)


# ---------------------------------------------------------------------------
# U-ViT with fMRI Conditioning
# ---------------------------------------------------------------------------

class UViT_fMRI(nn.Module):
    """U-ViT backbone adapted for fMRI-conditioned latent diffusion.

    Drop-in replacement for UNetModel / DiT. Accepts the same forward signature:
        forward(x, timesteps, context=None, y=None, **kwargs)

    fMRI conditioning is injected as a single pooled token prepended to the
    sequence (replacing the class label token from the original U-ViT).
    This preserves the pretrained positional embedding structure exactly:
        pos_embed shape = (1, 2 + num_patches, embed_dim)
        where 2 = time token + fMRI token (was: time token + class label token)

    Args:
        img_size: Spatial size of latent (default 32 for 256x256 images).
        patch_size: Patch size for patchification (default 2).
        in_chans: Number of input latent channels (default 4).
        embed_dim: Transformer hidden dimension (default 1152).
        depth: Total number of transformer blocks (default 28).
                Encoder gets depth//2, decoder gets depth//2, plus 1 mid block.
        num_heads: Number of attention heads (default 16).
        mlp_ratio: FFN hidden dim multiplier (default 4).
        qkv_bias: Whether to use bias in QKV projection (default False).
        mlp_time_embed: Whether to use MLP for time embedding (default False).
        use_checkpoint: Gradient checkpointing (default True).
        conv: Use Conv2d refinement layer at output (default False).
        skip: Use skip connections in decoder blocks (default True).
        context_dim: Dimension of fMRI conditioning embeddings (default 512).
        global_pool: Whether fMRI encoder uses global pooling (default False).
        # use_time_cond and fmri_seq_len accepted for config compat but unused
    """

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        mlp_time_embed=False,
        use_checkpoint=False,
        conv=False,
        skip=True,
        context_dim=512,
        global_pool=False,
        # Accepted for config compatibility, unused:
        use_time_cond=True,
        fmri_seq_len=77,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        # Time embedding: sinusoidal -> optional MLP
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        # fMRI context pooler + projector (replaces label_emb)
        # Pools fMRI context (B, 77, 512) -> (B, 512) then projects to embed_dim
        self.context_pool = ContextPooler(context_dim, global_pool=global_pool)
        self.context_proj = nn.Linear(context_dim, embed_dim)

        # extras = 2: time token + fMRI token (matches pretrained pos_embed shape)
        self.extras = 2

        # Positional embedding: covers extras + num_patches tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.extras + num_patches, embed_dim))

        # Encoder blocks (depth // 2)
        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        # Mid block
        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
            use_checkpoint=use_checkpoint)

        # Decoder blocks (depth // 2) with skip connections
        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        # Output head
        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = (nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1)
                            if conv else nn.Identity())

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Forward pass matching UNetModel / DiT interface.

        Args:
            x: (B, C, H, W) noisy latent images.
            timesteps: (B,) diffusion timestep indices.
            context: (B, seq_len, context_dim) fMRI conditioning embeddings.
            y: unused (class labels).

        Returns:
            (B, C, H, W) predicted noise (same shape as input).
        """
        # 1. Patchify
        x = self.patch_embed(x)  # (B, N, D)
        B, L, D = x.shape

        # 2. Prepare conditioning tokens
        # Time token
        time_token = self.time_embed(
            timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)  # (B, 1, D)

        # fMRI token: pool context -> project to embed_dim
        if context is not None:
            fmri_pooled = self.context_pool(context)       # (B, context_dim)
            fmri_token = self.context_proj(fmri_pooled)    # (B, embed_dim)
            fmri_token = fmri_token.unsqueeze(dim=1)       # (B, 1, D)
        else:
            fmri_token = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)

        # 3. Concatenate: [time_token, fmri_token, patch_tokens] + pos_embed
        # Original U-ViT order: [time_token, label_token, patch_tokens]
        # We replace label_token with fmri_token, preserving extras=2
        x = torch.cat((time_token, fmri_token, x), dim=1)
        x = x + self.pos_embed  # (B, extras + N, D)

        # 4. Encoder: in_blocks with skip collection
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        # 5. Mid block
        x = self.mid_block(x)

        # 6. Decoder: out_blocks with skip connections
        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        # 7. Output head
        x = self.norm(x)
        x = self.decoder_pred(x)

        # 8. Strip conditioning tokens, keep only patch tokens
        x = x[:, self.extras:, :]

        # 9. Unpatchify back to image space
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)

        return x
