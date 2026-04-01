#!/usr/bin/env python
"""Convert SD-VAE weights from diffusers format to LDM (CompVis) format.

The diffusers library uses different key naming than the LDM Encoder/Decoder
classes in dc_ldm/modules/diffusionmodules/model.py. This script converts
a diffusers-format state_dict so it can be loaded by AutoencoderKL in our
pipeline.

Usage:
    python scripts/convert_vae_diffusers_to_ldm.py \
        --input pretrains/ldm/label2img/sd_vae.ckpt \
        --output pretrains/ldm/label2img/sd_vae.ckpt

If --output is the same as --input, the file is overwritten in place.
"""

import argparse
import re
import torch


# Mapping from diffusers key patterns to LDM key patterns.
# Applied in order; first match wins.
DIFFUSERS_TO_LDM_PATTERNS = [
    # Encoder
    (r'^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.', r'encoder.down.\1.block.\2.'),
    (r'^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.', r'encoder.down.\1.downsample.conv.'),
    (r'^encoder\.mid_block\.resnets\.0\.', r'encoder.mid.block_1.'),
    (r'^encoder\.mid_block\.resnets\.1\.', r'encoder.mid.block_2.'),
    (r'^encoder\.mid_block\.attentions\.0\.', r'encoder.mid.attn_1.'),
    (r'^encoder\.conv_in\.', r'encoder.conv_in.'),
    (r'^encoder\.conv_out\.', r'encoder.conv_out.'),
    (r'^encoder\.conv_norm_out\.', r'encoder.norm_out.'),
    # Decoder
    (r'^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.', None),  # special handling below
    (r'^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.', None),  # special handling below
    (r'^decoder\.mid_block\.resnets\.0\.', r'decoder.mid.block_1.'),
    (r'^decoder\.mid_block\.resnets\.1\.', r'decoder.mid.block_2.'),
    (r'^decoder\.mid_block\.attentions\.0\.', r'decoder.mid.attn_1.'),
    (r'^decoder\.conv_in\.', r'decoder.conv_in.'),
    (r'^decoder\.conv_out\.', r'decoder.conv_out.'),
    (r'^decoder\.conv_norm_out\.', r'decoder.norm_out.'),
    # Quant conv
    (r'^quant_conv\.', r'quant_conv.'),
    (r'^post_quant_conv\.', r'post_quant_conv.'),
]

# ResNet block internal key mappings (diffusers -> LDM)
RESNET_KEY_MAP = {
    'norm1': 'norm1',
    'conv1': 'conv1',
    'norm2': 'norm2',
    'conv2': 'conv2',
    'time_emb_proj': 'temb_proj',
    'conv_shortcut': 'nin_shortcut',
}

# Attention block key mappings (diffusers -> LDM)
ATTN_KEY_MAP = {
    'group_norm': 'norm',
    'to_q': 'q',
    'to_k': 'k',
    'to_v': 'v',
    'to_out.0': 'proj_out',
}


def convert_resnet_key(diffusers_suffix):
    """Convert the suffix after the resnet block prefix."""
    for diff_name, ldm_name in RESNET_KEY_MAP.items():
        if diffusers_suffix.startswith(diff_name):
            return diffusers_suffix.replace(diff_name, ldm_name, 1)
    return diffusers_suffix


def convert_attn_key(diffusers_suffix):
    """Convert the suffix after the attention block prefix."""
    for diff_name, ldm_name in ATTN_KEY_MAP.items():
        if diffusers_suffix.startswith(diff_name):
            return diffusers_suffix.replace(diff_name, ldm_name, 1)
    return diffusers_suffix


def convert_key(key, num_up_blocks=4):
    """Convert a single diffusers key to LDM format."""
    # Encoder resnets
    m = re.match(r'^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', key)
    if m:
        level, block, suffix = m.groups()
        return f'encoder.down.{level}.block.{block}.{convert_resnet_key(suffix)}'

    # Encoder downsamplers
    m = re.match(r'^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.*)', key)
    if m:
        level, suffix = m.groups()
        return f'encoder.down.{level}.downsample.conv.{suffix}'

    # Encoder mid resnets
    m = re.match(r'^encoder\.mid_block\.resnets\.(\d+)\.(.*)', key)
    if m:
        idx, suffix = m.groups()
        block_name = 'block_1' if idx == '0' else 'block_2'
        return f'encoder.mid.{block_name}.{convert_resnet_key(suffix)}'

    # Encoder mid attention
    m = re.match(r'^encoder\.mid_block\.attentions\.0\.(.*)', key)
    if m:
        suffix = m.group(1)
        return f'encoder.mid.attn_1.{convert_attn_key(suffix)}'

    # Decoder up blocks: diffusers up_blocks are in reverse order vs LDM
    # diffusers up_blocks.0 = lowest res, LDM up.{N-1} = lowest res
    m = re.match(r'^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', key)
    if m:
        level, block, suffix = m.groups()
        ldm_level = num_up_blocks - 1 - int(level)
        return f'decoder.up.{ldm_level}.block.{block}.{convert_resnet_key(suffix)}'

    m = re.match(r'^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.*)', key)
    if m:
        level, suffix = m.groups()
        ldm_level = num_up_blocks - 1 - int(level)
        return f'decoder.up.{ldm_level}.upsample.conv.{suffix}'

    # Decoder mid resnets
    m = re.match(r'^decoder\.mid_block\.resnets\.(\d+)\.(.*)', key)
    if m:
        idx, suffix = m.groups()
        block_name = 'block_1' if idx == '0' else 'block_2'
        return f'decoder.mid.{block_name}.{convert_resnet_key(suffix)}'

    # Decoder mid attention
    m = re.match(r'^decoder\.mid_block\.attentions\.0\.(.*)', key)
    if m:
        suffix = m.group(1)
        return f'decoder.mid.attn_1.{convert_attn_key(suffix)}'

    # Encoder/Decoder conv_in, conv_out, norm_out
    m = re.match(r'^(encoder|decoder)\.conv_norm_out\.(.*)', key)
    if m:
        module, suffix = m.groups()
        return f'{module}.norm_out.{suffix}'

    # Direct pass-through keys (conv_in, conv_out, quant_conv, post_quant_conv)
    return key


def convert_state_dict(diffusers_sd, num_up_blocks=4):
    """Convert an entire diffusers VAE state_dict to LDM format."""
    ldm_sd = {}
    for key, value in diffusers_sd.items():
        new_key = convert_key(key, num_up_blocks)
        ldm_sd[new_key] = value
    return ldm_sd


def check_if_already_ldm_format(sd):
    """Heuristic: check if keys already use LDM naming."""
    ldm_indicators = ['encoder.down.0.block.0', 'decoder.up.0.block.0']
    diffusers_indicators = ['encoder.down_blocks.0.resnets.0', 'decoder.up_blocks.0.resnets.0']
    for ind in ldm_indicators:
        if any(k.startswith(ind) for k in sd.keys()):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Convert SD-VAE from diffusers to LDM format')
    parser.add_argument('--input', required=True, help='Path to diffusers-format VAE checkpoint')
    parser.add_argument('--output', required=True, help='Path to save LDM-format VAE checkpoint')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sd = torch.load(args.input, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    if check_if_already_ldm_format(sd):
        print("Checkpoint appears to already be in LDM format. No conversion needed.")
        return

    print(f"Converting {len(sd)} keys from diffusers to LDM format...")
    ldm_sd = convert_state_dict(sd)

    print(f"Saving to {args.output}...")
    torch.save(ldm_sd, args.output)
    print(f"Done. Converted {len(ldm_sd)} keys.")


if __name__ == '__main__':
    main()
