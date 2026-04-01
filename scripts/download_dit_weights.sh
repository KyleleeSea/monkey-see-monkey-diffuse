#!/bin/bash
# Download pretrained DiT-XL/2-256 and SD-VAE weights for the DiT pipeline.
#
# Usage:
#   bash scripts/download_dit_weights.sh [PRETRAIN_DIR]
#
# Default PRETRAIN_DIR: pretrains/ldm/label2img

set -e

PRETRAIN_DIR="${1:-pretrains/ldm/label2img}"
mkdir -p "$PRETRAIN_DIR"

echo "=== Downloading pretrained weights to $PRETRAIN_DIR ==="

# 1. Download DiT-XL/2-256 pretrained checkpoint (~2.4 GB)
DIT_PATH="$PRETRAIN_DIR/dit_xl_2.pt"
if [ -f "$DIT_PATH" ]; then
    echo "DiT-XL/2 checkpoint already exists at $DIT_PATH, skipping."
else
    echo "Downloading DiT-XL/2-256 checkpoint..."
    # Official DiT weights hosted on Meta's dl.fbaipublicfiles.com
    wget -O "$DIT_PATH" \
        "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
    echo "DiT-XL/2 downloaded to $DIT_PATH"
fi

# 2. Download SD-VAE (sd-vae-ft-mse) and extract state_dict
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
    # Convert diffusers format to LDM-compatible state_dict
    # diffusers keys need to be mapped to the ldm Encoder/Decoder naming
    sd = vae.state_dict()
    torch.save(sd, '$SD_VAE_PATH')
    print('SD-VAE saved (diffusers format) to $SD_VAE_PATH')
    print('NOTE: You may need to run scripts/convert_vae_diffusers_to_ldm.py to convert keys.')
except ImportError:
    print('diffusers not installed. Trying direct download...')
    import urllib.request, json
    # Download from HuggingFace Hub directly
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
    print('NOTE: You may need to run scripts/convert_vae_diffusers_to_ldm.py to convert keys.')
"
fi

echo ""
echo "=== Download complete ==="
echo "DiT checkpoint:  $DIT_PATH"
echo "SD-VAE checkpoint: $SD_VAE_PATH"
echo ""
echo "IMPORTANT: The SD-VAE checkpoint may be in diffusers format."
echo "If you get key mismatches when loading, run:"
echo "  python scripts/convert_vae_diffusers_to_ldm.py --input $SD_VAE_PATH --output $SD_VAE_PATH"
