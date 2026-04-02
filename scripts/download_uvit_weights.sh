#!/bin/bash
# Download pretrained U-ViT-H/2 ImageNet 256x256 and SD-VAE weights.
#
# The U-ViT-H/2 checkpoint is hosted on Google Drive.
# The SD-VAE is the same one used by DiT (reused if already downloaded).
#
# Usage:
#   bash scripts/download_uvit_weights.sh [PRETRAIN_DIR]
#
# Default PRETRAIN_DIR: pretrains/ldm/label2img

set -e

PRETRAIN_DIR="${1:-pretrains/ldm/label2img}"
mkdir -p "$PRETRAIN_DIR"

echo "=== Downloading pretrained weights to $PRETRAIN_DIR ==="

# 1. Download U-ViT-H/2 ImageNet 256x256 checkpoint
UVIT_PATH="$PRETRAIN_DIR/uvit_h2_imagenet256.pth"
# Google Drive file ID from the U-ViT repo README
GDRIVE_ID="13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u"

if [ -f "$UVIT_PATH" ]; then
    echo "U-ViT-H/2 checkpoint already exists at $UVIT_PATH, skipping."
else
    echo "Downloading U-ViT-H/2 ImageNet 256x256 checkpoint from Google Drive..."
    # Try gdown first (pip install gdown)
    if command -v gdown &> /dev/null; then
        gdown "$GDRIVE_ID" -O "$UVIT_PATH"
    else
        echo "gdown not found. Install it with: pip install gdown"
        echo "Then re-run this script."
        echo ""
        echo "Alternatively, download manually from:"
        echo "  https://drive.google.com/file/d/${GDRIVE_ID}/view?usp=share_link"
        echo "and save to: $UVIT_PATH"
        exit 1
    fi
    echo "U-ViT-H/2 downloaded to $UVIT_PATH"
fi

# 2. Download SD-VAE (same as DiT pipeline, skip if already present)
SD_VAE_PATH="$PRETRAIN_DIR/sd_vae.ckpt"
if [ -f "$SD_VAE_PATH" ]; then
    echo "SD-VAE checkpoint already exists at $SD_VAE_PATH, skipping."
else
    echo "Downloading SD-VAE (sd-vae-ft-mse) and extracting state_dict..."
    python -c "
import torch
try:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
    sd = vae.state_dict()
    torch.save(sd, '$SD_VAE_PATH')
    print('SD-VAE saved (diffusers format) to $SD_VAE_PATH')
except ImportError:
    print('diffusers not installed. Trying direct download...')
    import urllib.request
    url = 'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors'
    print(f'Downloading from {url}...')
    urllib.request.urlretrieve(url, '$SD_VAE_PATH.safetensors')
    print('Downloaded .safetensors file. Converting...')
    from safetensors.torch import load_file
    sd = load_file('$SD_VAE_PATH.safetensors')
    torch.save(sd, '$SD_VAE_PATH')
    import os
    os.remove('$SD_VAE_PATH.safetensors')
    print('SD-VAE saved to $SD_VAE_PATH')
"
fi

echo ""
echo "=== Download complete ==="
echo "U-ViT-H/2 checkpoint: $UVIT_PATH"
echo "SD-VAE checkpoint:     $SD_VAE_PATH"
echo ""
echo "Usage:"
echo "  python code/stageB_ldm_finetune.py --dataset GOD --backbone uvit"
