# CLAUDE.md — PixArt Backbone Integration Plan

## Project Goal

Replace the UNet denoising backbone in the MinD-Vis (Chen et al.) SC-MBM + DC-LDM pipeline with a **PixArt-alpha** transformer backbone for generating images from fMRI brain scans. Experiment with multiple PixArt block variants and compare their performance against the existing UNet and DiT baselines.

---

## Current Codebase Architecture

### Two-Stage Pipeline

The pipeline has two main stages:

**Stage A — Sparse-Coded Masked Brain Modeling (SC-MBM)**
- `stageA1_mbm_pretrain.py` — Pretrain a Masked Autoencoder on fMRI voxels (HCP + Kamitani data)
- `stageA2_mbm_finetune.py` — Finetune the MAE on subject-specific fMRI data
- Model: `sc_mbm/mae_for_fmri.py` → `MAEforFMRI` (1D patch embedding + ViT encoder/decoder)
- Output: A pretrained fMRI encoder checkpoint (`fmri_encoder.pth`)

**Stage B — Double-Conditioned Latent Diffusion Model (DC-LDM)**
- `stageB_ldm_finetune.py` — Finetune a pretrained LDM conditioned on fMRI embeddings
- Orchestrator: `dc_ldm/ldm_for_fmri.py` → `fLDM` class
- Core model: `dc_ldm/models/diffusion/ddpm.py` → `LatentDiffusion` (extends `DDPM`, a PyTorch Lightning module)
- Denoising backbone: `DiffusionWrapper` wraps either a `UNetModel` or a `DiT`
- VAE: `dc_ldm/models/autoencoder.py` → `VQModelInterface` (original) or `AutoencoderKL` (for SD-VAE)

### Key Data Flow in Stage B

```
fMRI voxels → cond_stage_model (fMRI encoder + linear projections) → conditioning embeddings
                                                                          ↓
Image → first_stage_model (VQ-VAE / AutoencoderKL) → latent z → noise + denoise via DiffusionWrapper
                                                                          ↑
                                                              timestep embedding + conditioning
```

The `DiffusionWrapper` (ddpm.py:1475) routes conditioning to the backbone:
- `conditioning_key='crossattn'` → calls `diffusion_model(x, t, context=cc)`
- The backbone must accept `forward(x, timesteps, context=None, **kwargs)` where `x` is `(B, C, H, W)` noisy latent, `timesteps` is `(B,)`, and `context` is `(B, seq_len, cond_dim)`.

### Existing Backbone Options

1. **UNet** (`dc_ldm/modules/diffusionmodules/openaimodel.py` → `UNetModel`)
   - Config: `pretrains/ldm/label2img/config.yaml`
   - Latent space: 3ch, 64×64 (VQ-VAE, `VQModelInterface`)
   - `context_dim=512`, image_size=64, model_channels=192

2. **DiT** (`dc_ldm/modules/diffusionmodules/dit.py` → `DiT`)
   - Config: `pretrains/ldm/label2img/config_dit_adaln_zero.yaml`
   - Latent space: 4ch, 32×32 (SD-VAE, `AutoencoderKL`, `scale_factor=0.18215`)
   - Three block variants already implemented: `adaLN`, `adaLN-Zero`, `cross-attn`
   - `context_dim=512`, hidden_size=1152, depth=28, num_heads=16

### Configuration System

- `config.py` → `Config_Generative_Model` holds runtime config; `self.backbone` selects the backbone
- `BACKBONE_CONFIG_MAP` in `ldm_for_fmri.py` maps backbone name → YAML config filename
- YAML configs live in `pretrains/ldm/label2img/` and specify `unet_config.target` (the backbone class)
- CLI arg `--backbone` in `stageB_ldm_finetune.py` accepts `'unet'`, `'dit-adaln'`, `'dit-adaln-zero'`, `'dit-crossattn'`

### fMRI Conditioning

`cond_stage_model` (ldm_for_fmri.py:13) wraps the pretrained MAE encoder:
- When `global_pool=True`: output is `(B, 1, cond_dim)` — single vector
- When `global_pool=False`: output is `(B, 77, cond_dim)` — sequence of tokens
- `cond_dim` is set to match the backbone's `context_dim` (currently 512 for UNet, 512 for DiT)

### Evaluation

`eval_metrics.py` computes: MSE, PCC, SSIM, LPIPS (PSM), top-1 class accuracy (50-way)
`gen_eval.py` is likely a standalone evaluation script.

---

## PixArt-alpha Architecture (Reference in `PixArt-alpha/`)

### Key Components

- **`PixArt`** (`diffusion/model/nets/PixArt.py`) — Main model class
  - `PixArt_XL_2`: depth=28, hidden_size=1152, patch_size=2, num_heads=16
  - Uses `adaLN-single` conditioning: a shared `t_block` produces 6·D modulation params from timestep, applied per-block via `scale_shift_table`
  - Cross-attention to text/caption via `MultiHeadCrossAttention` in each block
  - `CaptionEmbedder` projects text tokens (from T5, 4096-dim) → hidden_size
  - `T2IFinalLayer` with its own `scale_shift_table`

- **`PixArtBlock`** (`diffusion/model/nets/PixArt_blocks.py`) — Core block
  - Self-attention (via `WindowAttention` with xformers) + adaLN-single modulation
  - Cross-attention to conditioning tokens (`MultiHeadCrossAttention` with xformers)
  - MLP with adaLN-single modulation
  - `scale_shift_table`: learnable `(6, D)` parameter, added to timestep embedding, then chunked into 6 modulation vectors (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

- **Forward signature**: `forward(x, timestep, y, mask=None)` — different from UNet/DiT!
  - `x`: `(B, C, H, W)` noisy latent
  - `timestep`: `(B,)` diffusion timesteps
  - `y`: `(B, 1, L, D)` conditioning tokens (text/caption)
  - `mask`: optional attention mask for variable-length conditioning

### Key Differences from Current DiT

| Feature | Current DiT | PixArt-alpha |
|---|---|---|
| Conditioning approach | adaLN / adaLN-Zero / cross-attn (separate block types) | adaLN-single + cross-attn (combined in every block) |
| Modulation source | Timestep MLP + pooled fMRI | Shared `t_block` MLP + per-block `scale_shift_table` |
| Cross-attention | Only in cross-attn block variant | In every block |
| Attention implementation | `nn.MultiheadAttention` | xformers memory-efficient attention |
| Caption embedding | Not applicable (fMRI encoder used) | `CaptionEmbedder` (MLP projection of T5 tokens) |
| Forward signature | `(x, timesteps, context)` | `(x, timestep, y, mask)` |

---

## Implementation Plan

### Phase 1: Create PixArt Backbone Module

**File: `code/dc_ldm/modules/diffusionmodules/pixart.py`**

Create a `PixArtForFMRI` class that adapts the PixArt architecture to work as a drop-in replacement inside `DiffusionWrapper`. This means it must implement:

```python
def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
    # x: (B, C, H, W) noisy latents
    # timesteps: (B,) diffusion timestep
    # context: (B, seq_len, context_dim) fMRI conditioning
    # Returns: (B, C, H, W) predicted noise
```

**What to port from PixArt-alpha:**
1. `PixArtBlock` — the core transformer block with adaLN-single + cross-attention
2. `T2IFinalLayer` — final layer with scale_shift_table
3. `TimestepEmbedder` — can reuse from existing dit.py or PixArt_blocks.py
4. `PatchEmbed` — 2D patch embedding (same concept as DiT)
5. Sincos 2D positional embeddings

**What to adapt for fMRI:**
- Replace `CaptionEmbedder` with fMRI context handling:
  - When `global_pool=False`: fMRI conditioning arrives as `(B, 77, context_dim)` — project to `(B, 77, hidden_size)` for cross-attention
  - When `global_pool=True`: fMRI conditioning arrives as `(B, 1, context_dim)` — project to `(B, 1, hidden_size)` for cross-attention
- The `MultiHeadCrossAttention` in PixArt uses xformers; provide a fallback using standard PyTorch attention for environments without xformers
- `y` in original PixArt is `(B, 1, L, D)` — our fMRI context is `(B, L, D)`, so reshape accordingly

### Phase 2: Implement Multiple Block Variants for Comparison

Create multiple PixArt-style block variants to experiment with:

#### Block Variant 1: `pixart-adaLN-single` (Faithful PixArt)
- adaLN-single conditioning (shared `t_block` + per-block `scale_shift_table`)
- Cross-attention to fMRI tokens in every block
- This is the standard PixArt approach

#### Block Variant 2: `pixart-adaLN-single-nocross` (PixArt without cross-attn)
- adaLN-single conditioning only
- fMRI conditioning is pooled and added to the timestep embedding (no cross-attention)
- Tests whether the adaLN-single approach alone is effective for fMRI

#### Block Variant 3: `pixart-adaLN-zero` (PixArt with DiT-style adaLN-Zero)
- Replace adaLN-single with per-block adaLN-Zero modulation (each block gets its own modulation MLP)
- Keep cross-attention to fMRI tokens
- Hybrid of DiT adaLN-Zero + PixArt cross-attention

#### Block Variant 4: `pixart-full` (PixArt with both adaLN-single and fMRI injection)
- adaLN-single from timestep
- Cross-attention to fMRI tokens
- Additionally inject pooled fMRI into the timestep embedding (like `use_time_cond` in DiT)
- Tests whether double-conditioning improves PixArt

All variants should be selectable via a `block_type` parameter in the `PixArtForFMRI` constructor.

### Phase 3: YAML Configs

Create config files in `pretrains/ldm/label2img/`:

**`config_pixart_adaln_single.yaml`**
```yaml
model:
  target: dc_ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    image_size: 32
    channels: 4
    scale_factor: 0.18215
    conditioning_key: crossattn
    unet_config:
      target: dc_ldm.modules.diffusionmodules.pixart.PixArtForFMRI
      params:
        image_size: 32
        patch_size: 2
        in_channels: 4
        out_channels: 4
        hidden_size: 1152
        depth: 28
        num_heads: 16
        mlp_ratio: 4.0
        context_dim: 512
        block_type: adaLN-single
        use_time_cond: true
        global_pool: false
    first_stage_config:
      target: dc_ldm.models.autoencoder.AutoencoderKL
      # ... (same as config_dit_adaln_zero.yaml)
    cond_stage_config:
      # ... (same as config_dit_adaln_zero.yaml)
```

Create analogous configs for each block variant.

### Phase 4: Integration Points

#### 4a. Update `BACKBONE_CONFIG_MAP` in `ldm_for_fmri.py`

```python
BACKBONE_CONFIG_MAP = {
    'unet': 'config.yaml',
    'dit-adaln': 'config_dit_adaln.yaml',
    'dit-adaln-zero': 'config_dit_adaln_zero.yaml',
    'dit-crossattn': 'config_dit_crossattn.yaml',
    'pixart-adaln-single': 'config_pixart_adaln_single.yaml',
    'pixart-adaln-single-nocross': 'config_pixart_adaln_single_nocross.yaml',
    'pixart-adaln-zero': 'config_pixart_adaln_zero.yaml',
    'pixart-full': 'config_pixart_full.yaml',
}
```

#### 4b. Update `_load_dit_pretrained` or add `_load_pixart_pretrained` in `ldm_for_fmri.py`

For PixArt pretrained weights:
- SD-VAE loading is the same as DiT (already handled)
- PixArt pretrained weights need a key mapping function similar to `_map_dit_pretrained_keys`
- Official PixArt-alpha checkpoints have keys like: `x_embedder.*`, `t_embedder.*`, `y_embedder.*`, `blocks.N.*`, `final_layer.*`, `t_block.*`
- Map these to our `PixArtForFMRI` naming, skipping `y_embedder` (we use fMRI, not T5 captions)

#### 4c. Update `fLDM.__init__` in `ldm_for_fmri.py`

Add PixArt backbone detection:
```python
self.is_dit = backbone.startswith('dit')
self.is_pixart = backbone.startswith('pixart')
```

In the weight loading section:
```python
if self.is_pixart:
    _load_pixart_pretrained(model, pretrain_root, block_type)
elif self.is_dit:
    _load_dit_pretrained(model, pretrain_root, block_type)
else:
    # Load original UNet checkpoint
```

#### 4d. Update CLI args in `stageB_ldm_finetune.py`

Extend the `--backbone` choices:
```python
parser.add_argument('--backbone', type=str,
    choices=['unet', 'dit-adaln', 'dit-adaln-zero', 'dit-crossattn',
             'pixart-adaln-single', 'pixart-adaln-single-nocross',
             'pixart-adaln-zero', 'pixart-full'],
    help='Denoising backbone')
```

### Phase 5: Pretrained Weights Setup

#### Required checkpoints in `pretrains/ldm/label2img/`:

1. **`sd_vae.ckpt`** — SD-VAE weights (already needed for DiT, supports both diffusers and LDM format via auto-conversion in `ldm_for_fmri.py`)
2. **`pixart_xl_2.pth`** — PixArt-alpha pretrained weights (download from PixArt-alpha releases)
   - Official checkpoints: https://huggingface.co/PixArt-alpha/PixArt-alpha
   - These are T5-conditioned, so `y_embedder` weights will be skipped

#### Weight download script

Add to `scripts/download_pixart_weights.sh`:
```bash
# Download PixArt-alpha XL/2 256px checkpoint
# Download SD-VAE (stabilityai/sd-vae-ft-ema)
```

### Phase 6: Handling the xformers Dependency

PixArt's `MultiHeadCrossAttention` and `WindowAttention` use `xformers.ops.memory_efficient_attention`. Options:

1. **Preferred**: Write a `MultiHeadCrossAttention` that uses standard PyTorch `F.scaled_dot_product_attention` (available in PyTorch 2.0+) as fallback
2. **Alternative**: Add `xformers` to `env.yaml` dependencies
3. **Hybrid**: Try xformers first, fall back to PyTorch SDPA

The self-attention in `PixArtBlock` uses `WindowAttention` which inherits from timm's `Attention` — this also uses xformers in the original code but can be replaced with standard attention.

### Phase 7: Testing & Validation

#### Smoke test (no training):
```bash
# Test that PixArt backbone initializes and forward-passes correctly
python -c "
from dc_ldm.modules.diffusionmodules.pixart import PixArtForFMRI
import torch
model = PixArtForFMRI(image_size=32, in_channels=4, out_channels=4, context_dim=512)
x = torch.randn(2, 4, 32, 32)
t = torch.randint(0, 1000, (2,))
ctx = torch.randn(2, 77, 512)
out = model(x, timesteps=t, context=ctx)
print(out.shape)  # should be (2, 4, 32, 32)
"
```

#### Training run:
```bash
python code/stageB_ldm_finetune.py \
    --backbone pixart-adaln-single \
    --dataset GOD \
    --num_epoch 500
```

### Phase 8: Experiment Matrix

Run all backbone variants on the same dataset/subject and compare:

| Backbone | Block Type | Params | MSE ↓ | PCC ↑ | SSIM ↑ | LPIPS ↓ | Top-1 Acc ↑ |
|---|---|---|---|---|---|---|---|
| UNet | — | ~274M | baseline | | | | |
| DiT adaLN-Zero | adaLN-Zero | ~675M | | | | | |
| DiT cross-attn | cross-attn | ~675M | | | | | |
| PixArt adaLN-single | adaLN-single + cross-attn | ~611M | | | | | |
| PixArt adaLN-single-nocross | adaLN-single only | ~500M | | | | | |
| PixArt adaLN-zero | adaLN-Zero + cross-attn | ~700M | | | | | |
| PixArt full | adaLN-single + cross-attn + time-cond | ~615M | | | | | |

---

## Files to Create

| File | Purpose |
|---|---|
| `code/dc_ldm/modules/diffusionmodules/pixart.py` | PixArt backbone module with all block variants |
| `pretrains/ldm/label2img/config_pixart_adaln_single.yaml` | Config for adaLN-single variant |
| `pretrains/ldm/label2img/config_pixart_adaln_single_nocross.yaml` | Config for no-cross-attn variant |
| `pretrains/ldm/label2img/config_pixart_adaln_zero.yaml` | Config for adaLN-zero + cross-attn variant |
| `pretrains/ldm/label2img/config_pixart_full.yaml` | Config for full double-conditioning variant |
| `scripts/download_pixart_weights.sh` | Download PixArt pretrained weights |

## Files to Modify

| File | Changes |
|---|---|
| `code/dc_ldm/ldm_for_fmri.py` | Add PixArt entries to `BACKBONE_CONFIG_MAP`, add `_load_pixart_pretrained()`, update `fLDM.__init__` |
| `code/stageB_ldm_finetune.py` | Add PixArt backbone choices to `--backbone` argument |
| `code/config.py` | No changes needed (backbone is already configurable via `self.backbone`) |

---

## Technical Considerations

### 1. Latent Space Compatibility
PixArt-alpha was designed for SD-VAE latent space (4ch, 32×32 at 256px input, `scale_factor=0.18215`). Use `AutoencoderKL` as the first stage, matching the DiT configs. The original UNet uses VQ-VAE (3ch, 64×64) — these are incompatible, so PixArt will use the SD-VAE pipeline only.

### 2. Context Dimension Mismatch
PixArt-alpha was trained with T5-XXL text embeddings (4096-dim). Our fMRI conditioning is 512-dim (from `cond_stage_model`). Options:
- **Option A (recommended)**: Use a linear projection layer inside `PixArtForFMRI` to project fMRI embeddings from 512 → hidden_size for cross-attention. This matches how `cond_stage_model` already handles dimension mapping.
- **Option B**: Change `cond_dim` in `cond_stage_model` to 4096 to match T5 dimension, but this wastes parameters.

### 3. Sequence Length
- fMRI conditioning: 77 tokens (when `global_pool=False`) or 1 token (when `global_pool=True`)
- T5 text embeddings: up to 120 tokens
- The cross-attention mechanism handles variable sequence lengths natively.

### 4. Pred Sigma
PixArt predicts both noise and variance (`pred_sigma=True`, `out_channels = in_channels * 2 = 8`). The MinD-Vis pipeline uses `parameterization="eps"` with fixed variance. Set `pred_sigma=False` in `PixArtForFMRI` so `out_channels = in_channels = 4`.

### 5. Gradient Checkpointing
PixArt supports gradient checkpointing via `auto_grad_checkpoint`. Enable this for large models to fit in GPU memory, controlled by a config flag.

### 6. Weight Initialization
When loading pretrained PixArt weights:
- `x_embedder`, `t_embedder`, `t_block`, `blocks.N.{norm1,attn,norm2,mlp,scale_shift_table}` can be loaded
- `y_embedder` (T5 caption embedder) must be **skipped** — we replace this with fMRI projection
- `blocks.N.cross_attn` — shape mismatch if `context_dim` differs (4096 vs 512); must be **skipped** or re-initialized
- `final_layer` — may have shape mismatch if `out_channels` differs; skip if needed

### 7. Training Strategy
Following the existing pipeline:
1. Freeze `first_stage_model` (VAE)
2. Train `cond_stage_model` (fMRI encoder projections) + denoising backbone
3. The `train_cond_stage_only` flag in stage one only trains the conditional encoder
4. Then unfreeze the full model for joint finetuning

For PixArt, consider:
- **Frozen PixArt + train conditioning only**: Good for initial experiments, faster
- **LoRA on PixArt + train conditioning**: Memory-efficient finetuning
- **Full finetuning**: Best quality but expensive; the existing pipeline does this for UNet

---

## Recommended Execution Order

1. Implement `pixart.py` with the `pixart-adaln-single` variant first (faithful PixArt reproduction)
2. Write the YAML config and integrate into `ldm_for_fmri.py`
3. Smoke-test forward pass
4. Download SD-VAE weights and run a short training
5. Add remaining block variants one at a time
6. Run full experiment matrix on GOD dataset (subject sbj_3)
7. Compare metrics and analyze results
