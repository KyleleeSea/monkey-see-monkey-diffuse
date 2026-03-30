"""
Diffusion Transformer (DiT) adapted for fMRI conditioning.

Based on: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
Reference: https://github.com/facebookresearch/DiT

Modifications from original DiT:
- Removed class-label embedding (LabelEmbedder)
- Added cross-attention layers in each DiTBlock for fMRI conditioning
- Added optional time_embed_condition for fMRI→timestep fusion
- Forward signature: forward(x, t, context=None) to match UNet interface
"""

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """2D image to patch embedding via Conv2d."""

    def __init__(self, input_size=32, patch_size=2, in_channels=4, embed_dim=1152):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=16, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention for fMRI conditioning."""

    def __init__(self, dim, context_dim, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        # x: (B, N, D) - queries from DiT features
        # context: (B, S, context_dim) - keys/values from fMRI embeddings
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning and cross-attention.

    Architecture per block:
    1. adaLN-Zero modulated self-attention (pretrained from DiT)
    2. Cross-attention with fMRI context (new, trainable)
    3. adaLN-Zero modulated FFN (pretrained from DiT)
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, context_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # Cross-attention for fMRI conditioning (new layers, not in pretrained DiT)
        if context_dim is not None:
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, context_dim, num_heads)

    def forward(self, x, c, context=None):
        # c: (B, D) conditioning vector from timestep (+fMRI) for adaLN
        # context: (B, S, context_dim) fMRI sequence for cross-attention
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention with adaLN-Zero
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # Cross-attention with fMRI context
        if context is not None and hasattr(self, 'cross_attn'):
            x = x + self.cross_attn(self.norm_cross(x), context)

        # FFN with adaLN-Zero
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer of DiT: adaLN + linear projection to output patches."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first for consistency with DiT
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class DiT_fMRI(nn.Module):
    """
    Diffusion Transformer adapted for fMRI-conditioned image generation.

    Replaces the UNet backbone in the LDM pipeline. Uses:
    - adaLN-Zero for timestep conditioning (pretrained from DiT-XL/2)
    - Cross-attention for fMRI sequence conditioning (new layers)
    - Optional fMRI→timestep fusion via time_embed_condition

    Forward signature matches UNet: forward(x, timesteps, context=None)
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        context_dim=1152,
        use_time_cond=True,
        global_pool=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.context_dim = context_dim
        self.use_time_cond = use_time_cond
        self.global_pool = global_pool

        # Patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Fixed sinusoidal positional embedding
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(num_patches ** 0.5))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0),
                                      requires_grad=False)

        # Optional: fMRI → timestep embedding fusion
        if use_time_cond:
            if not global_pool:
                self.time_embed_condition = nn.Sequential(
                    nn.Linear(context_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            else:
                self.time_embed_condition = nn.Sequential(
                    nn.Linear(context_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )

        # DiT blocks with cross-attention
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, context_dim=context_dim)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for new (non-pretrained) layers."""
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch embedding like nn.Linear (per ViT)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize time_embed_condition if present
        if hasattr(self, 'time_embed_condition'):
            for layer in self.time_embed_condition:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.constant_(layer.bias, 0)

        # Initialize cross-attention layers with small weights
        for block in self.blocks:
            if hasattr(block, 'cross_attn'):
                for p in block.cross_attn.parameters():
                    nn.init.normal_(p, std=0.02)
                for p in block.norm_cross.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    def unpatchify(self, x):
        """Convert patch tokens back to spatial image.
        x: (B, num_patches, patch_size**2 * out_channels)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x, timesteps, context=None, **kwargs):
        """
        Forward pass compatible with the DiffusionWrapper interface.

        Args:
            x: (B, C, H, W) noisy latent input
            timesteps: (B,) integer timesteps
            context: (B, S, context_dim) fMRI conditioning sequence
        Returns:
            (B, out_C, H, W) noise prediction (and optionally variance)
        """
        # Patchify and add positional embedding
        x = self.x_embedder(x) + self.pos_embed  # (B, num_patches, D)

        # Timestep embedding
        t = self.t_embedder(timesteps)  # (B, D)

        # Optional: fuse pooled fMRI into timestep embedding
        if self.use_time_cond and hasattr(self, 'time_embed_condition') and context is not None:
            if self.global_pool:
                fmri_pool = context.mean(dim=1)  # (B, context_dim)
            else:
                fmri_pool = context.mean(dim=1)  # (B, context_dim)
            t = t + self.time_embed_condition(fmri_pool)

        # Process through DiT blocks
        for block in self.blocks:
            x = block(x, t, context)

        # Final layer
        x = self.final_layer(x, t)  # (B, num_patches, patch_size**2 * out_channels)

        # Unpatchify to spatial
        x = self.unpatchify(x)  # (B, out_channels, H, W)
        return x

    def load_pretrained_dit(self, ckpt_path):
        """Load pretrained DiT-XL/2 weights, skipping class-label embedding."""
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Remove class-label embedding keys (y_embedder) — we use fMRI conditioning instead
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("y_embedder")]
        for k in keys_to_remove:
            del state_dict[k]

        # Load with strict=False: new cross_attn/norm_cross/time_embed_condition layers will be missing
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"DiT pretrained weights loaded. Missing keys (new layers): {len(missing)}")
        print(f"  Missing: {[k for k in missing[:10]]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        return missing, unexpected
