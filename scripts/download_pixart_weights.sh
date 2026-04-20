#!/bin/bash
# Download pretrained weights for PixArt-alpha backbone
# Run from the repository root directory

set -euo pipefail

PRETRAIN_DIR="pretrains/ldm/label2img"
mkdir -p "$PRETRAIN_DIR"

echo "=== PixArt-alpha Pretrained Weights Download ==="
echo ""

# --- 1. SD-VAE (stabilityai/sd-vae-ft-ema) ---
# Required for all PixArt and DiT backbones
VAE_PATH="$PRETRAIN_DIR/sd_vae.ckpt"
if [ -f "$VAE_PATH" ]; then
    echo "[OK] SD-VAE already exists at $VAE_PATH"
else
    echo "[DOWNLOAD] Downloading SD-VAE (stabilityai/sd-vae-ft-ema)..."
    echo "  You can download from: https://huggingface.co/stabilityai/sd-vae-ft-ema"
    echo ""
    echo "  Option A — Using huggingface-cli:"
    echo "    pip install huggingface_hub"
    echo "    python -c \""
    echo "from huggingface_hub import hf_hub_download"
    echo "hf_hub_download('stabilityai/sd-vae-ft-ema', 'diffusion_pytorch_model.safetensors',"
    echo "                local_dir='$PRETRAIN_DIR', local_dir_use_symlinks=False)"
    echo "\""
    echo "    # Then rename: mv $PRETRAIN_DIR/diffusion_pytorch_model.safetensors $VAE_PATH"
    echo ""
    echo "  Option B — Using the conversion script (for diffusers format):"
    echo "    python scripts/convert_vae_diffusers_to_ldm.py \\"
    echo "      --input_path <path_to_diffusers_vae_dir> \\"
    echo "      --output_path $VAE_PATH"
    echo ""
    echo "  Note: The code auto-detects diffusers vs LDM format, so either works."
    echo ""
fi

# --- 2. PixArt-alpha XL/2 256px checkpoint ---
PIXART_PATH="$PRETRAIN_DIR/pixart_xl_2.pth"
if [ -f "$PIXART_PATH" ]; then
    echo "[OK] PixArt-alpha XL/2 already exists at $PIXART_PATH"
else
    echo "[DOWNLOAD] Downloading PixArt-alpha XL/2 checkpoint..."
    echo "  You can download from: https://huggingface.co/PixArt-alpha/PixArt-alpha"
    echo ""
    echo "  Using huggingface-cli:"
    echo "    pip install huggingface_hub"
    echo "    python -c \""
    echo "from huggingface_hub import hf_hub_download"
    echo "hf_hub_download('PixArt-alpha/PixArt-XL-2-256x256', 'transformer/diffusion_pytorch_model.safetensors',"
    echo "                local_dir='$PRETRAIN_DIR', local_dir_use_symlinks=False)"
    echo "\""
    echo "    # Then convert safetensors -> pt and rename to $PIXART_PATH"
    echo ""
    echo "  Alternatively, use the PixArt-alpha training checkpoint if available."
    echo ""
fi

echo ""
echo "=== Summary ==="
echo "SD-VAE:    $VAE_PATH  ($([ -f "$VAE_PATH" ] && echo 'EXISTS' || echo 'MISSING'))"
echo "PixArt:    $PIXART_PATH  ($([ -f "$PIXART_PATH" ] && echo 'EXISTS' || echo 'MISSING'))"
echo ""
echo "After downloading, you can train with:"
echo "  python code/stageB_ldm_finetune.py --backbone pixart-adaln-single --dataset GOD"
