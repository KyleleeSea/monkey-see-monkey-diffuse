"""
Diffusion Transformer (DiT) backbone for Latent Diffusion Models.

Replaces UNetModel as the denoising backbone in the MinD-Vis pipeline.
Implements three conditioning block variants:
  - adaLN: Adaptive LayerNorm with scale+shift modulation
  - adaLN-Zero: adaLN with zero-initialized gating (Peebles & Xie, 2023)
  - cross-attn: Cross-attention to fMRI context tokens

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""

import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dc_ldm.modules.diffusionmodules.util import timestep_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Patch Embedding & Unpatchify
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D image to patch embedding via Conv2d."""

    def __init__(self, img_size=64, patch_size=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
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
# Timestep MLP
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP embedding."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_emb = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb)


# ---------------------------------------------------------------------------
# fMRI Context Pooler (for adaLN / adaLN-Zero conditioning vector)
# ---------------------------------------------------------------------------

class ContextPooler(nn.Module):
    """Pool variable-length fMRI context into a single conditioning vector.

    For global_pool=True input is (B, 1, context_dim).
    For global_pool=False input is (B, 77, context_dim).
    Output is always (B, hidden_size).
    """

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
        # context: (B, seq_len, context_dim)
        if self.global_pool:
            x = context.squeeze(1)  # (B, context_dim)
        else:
            x = self.pool(context).squeeze(1)  # (B, context_dim)
        return self.proj(x)  # (B, hidden_size)


# ---------------------------------------------------------------------------
# DiT Block Variants
# ---------------------------------------------------------------------------

class DiTBlock_adaLN(nn.Module):
    """DiT block with Adaptive LayerNorm conditioning.

    Timestep + pooled fMRI produce scale/shift parameters for LayerNorm
    before self-attention and feed-forward layers.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        # Modulation MLP: conditioning vector -> 4 modulation params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size),
        )

    def forward(self, x, c, context=None):
        # x: (B, N, D), c: (B, D) conditioning vector
        shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=-1)
        # Self-attention with modulated norm
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN with modulated norm
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + self.ffn(h)
        return x


class DiTBlock_adaLN_Zero(nn.Module):
    """DiT block with adaLN-Zero conditioning (from the DiT paper).

    Same as adaLN but adds gating parameters initialized to zero,
    so each block starts as an identity function.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        # 6 outputs: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-initialize the modulation MLP final layer
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, context=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        # Self-attention with gating
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * h
        # FFN with gating
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.ffn(h)
        return x


class DiTBlock_CrossAttn(nn.Module):
    """DiT block with cross-attention to fMRI context.

    Uses adaLN for timestep conditioning (scale/shift on norms)
    and a separate cross-attention layer for fMRI context tokens.
    This preserves the full sequence structure of fMRI embeddings.
    """

    def __init__(self, hidden_size, num_heads, context_dim=512,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads,
                                               dropout=dropout, batch_first=True)
        # Cross-attention to fMRI context
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_q = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_k = nn.Linear(context_dim, hidden_size)
        self.cross_attn_v = nn.Linear(context_dim, hidden_size)
        self.cross_attn_out = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # FFN
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        # adaLN modulation from timestep: 6 params (shift/scale for 3 norms)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def _cross_attention(self, x, context):
        """Manual cross-attention supporting different input/context dims."""
        B, N, D = x.shape
        h = self.cross_attn_heads
        d = self.head_dim

        q = rearrange(self.cross_attn_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.cross_attn_k(context), 'b s (h d) -> b h s d', h=h)
        v = rearrange(self.cross_attn_v(context), 'b s (h d) -> b h s d', h=h)

        scale = d ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.cross_attn_out(out)

    def forward(self, x, c, context=None):
        # c: (B, D) timestep conditioning, context: (B, seq, context_dim) fMRI
        shift1, scale1, shift2, scale2, shift3, scale3 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        # Self-attention
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.self_attn(h, h, h)
        x = x + h
        # Cross-attention to fMRI context
        h = modulate(self.norm2(x), shift2, scale2)
        h = self._cross_attention(h, context)
        x = x + h
        # FFN
        h = modulate(self.norm3(x), shift3, scale3)
        x = x + self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# Final Layer
# ---------------------------------------------------------------------------

class DiTFinalLayer(nn.Module):
    """Final adaLN-modulated layer that projects to patch predictions."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Zero-initialize final projection
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Main DiT Model
# ---------------------------------------------------------------------------

BLOCK_REGISTRY = {
    'adaLN': DiTBlock_adaLN,
    'adaLN-Zero': DiTBlock_adaLN_Zero,
    'cross-attn': DiTBlock_CrossAttn,
}


class DiT(nn.Module):
    """Diffusion Transformer for latent diffusion.

    Drop-in replacement for UNetModel. Accepts the same forward signature:
        forward(x, timesteps, context=None, y=None, **kwargs)

    Args:
        image_size: Spatial size of the latent (default 64).
        patch_size: Patch size for patchification (default 2).
        in_channels: Number of input latent channels (default 3).
        out_channels: Number of output channels (default 3).
        hidden_size: Transformer hidden dimension (default 768).
        depth: Number of transformer blocks (default 12).
        num_heads: Number of attention heads (default 12).
        mlp_ratio: FFN hidden dim multiplier (default 4.0).
        context_dim: Dimension of fMRI conditioning embeddings (default 512).
        block_type: One of 'adaLN', 'adaLN-Zero', 'cross-attn'.
        use_time_cond: Whether to add pooled fMRI to timestep conditioning.
        global_pool: Whether fMRI encoder uses global pooling (1 token vs 77).
        dropout: Dropout rate (default 0.0).
        use_checkpoint: Whether to use gradient checkpointing (default False).
    """

    def __init__(
        self,
        image_size=64,
        patch_size=2,
        in_channels=3,
        out_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        context_dim=512,
        block_type='adaLN-Zero',
        use_time_cond=True,
        global_pool=False,
        dropout=0.0,
        use_checkpoint=False,
        # Unused UNet params accepted for config compatibility
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

        # Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # fMRI context pooler for adaLN conditioning
        # For cross-attn block: only timestep goes into the conditioning vector
        # For adaLN / adaLN-Zero: timestep + pooled fMRI go into conditioning
        if use_time_cond and block_type != 'cross-attn':
            self.context_pooler = ContextPooler(context_dim, hidden_size, global_pool)
        else:
            self.context_pooler = None

        # For cross-attn block type, also pool fMRI into timestep if use_time_cond
        if use_time_cond and block_type == 'cross-attn':
            self.context_time_pooler = ContextPooler(context_dim, hidden_size, global_pool)
        else:
            self.context_time_pooler = None

        # Transformer blocks
        BlockClass = BLOCK_REGISTRY[block_type]
        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        if block_type == 'cross-attn':
            block_kwargs['context_dim'] = context_dim

        self.blocks = nn.ModuleList([
            BlockClass(**block_kwargs) for _ in range(depth)
        ])

        # Final layer
        self.final_layer = DiTFinalLayer(hidden_size, patch_size, out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding like a linear layer
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Initialize timestep embedding MLP
        for module in self.t_embedder.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

        # Initialize transformer blocks
        for block in self.blocks:
            # Initialize attention output projections
            if hasattr(block, 'attn'):
                nn.init.xavier_uniform_(block.attn.in_proj_weight)
                nn.init.zeros_(block.attn.in_proj_bias)
                nn.init.xavier_uniform_(block.attn.out_proj.weight)
                nn.init.zeros_(block.attn.out_proj.bias)
            if hasattr(block, 'self_attn'):
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                nn.init.zeros_(block.self_attn.in_proj_bias)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                nn.init.zeros_(block.self_attn.out_proj.bias)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Forward pass matching UNetModel interface.

        Args:
            x: (B, C, H, W) noisy latent images.
            timesteps: (B,) diffusion timestep indices.
            context: (B, seq_len, context_dim) fMRI conditioning embeddings.
            y: unused (class labels for class-conditional models).

        Returns:
            (B, C, H, W) predicted noise.
        """
        # 1. Patchify + positional embedding
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed

        # 2. Compute conditioning vector
        t = self.t_embedder(timesteps)  # (B, D)
        c = t
        if self.context_pooler is not None and context is not None:
            c = c + self.context_pooler(context)
        if self.context_time_pooler is not None and context is not None:
            c = c + self.context_time_pooler(context)

        # 3. Apply transformer blocks
        if self.use_checkpoint:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, context, use_reentrant=False
                )
        else:
            for block in self.blocks:
                x = block(x, c, context)

        # 4. Final layer + unpatchify
        x = self.final_layer(x, c)  # (B, N, patch_size**2 * C)
        x = unpatchify(x, self.patch_size, self.out_channels, self.image_size)

        return x
