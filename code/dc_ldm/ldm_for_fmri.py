import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_fmri import fmri_encoder

def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels, cond_dim=1280, global_pool=True):
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out

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


def _map_dit_pretrained_keys(pretrained_sd, block_type='adaLN-Zero'):
    """Map official DiT checkpoint keys to our DiT module's naming convention.

    Official DiT (Peebles & Xie) key structure:
        x_embedder.proj.{weight,bias}      -> patch_embed.proj.{weight,bias}
        t_embedder.mlp.{0,2}.{weight,bias} -> t_embedder.mlp.{0,2}.{weight,bias}  (same)
        y_embedder.*                        -> SKIPPED (class-conditional, we use fMRI)
        pos_embed                           -> pos_embed (same)
        blocks.N.attn.qkv.{weight,bias}    -> blocks.N.attn.in_proj_{weight,bias}
        blocks.N.attn.proj.{weight,bias}    -> blocks.N.attn.out_proj.{weight,bias}
        blocks.N.mlp.fc1.{weight,bias}      -> blocks.N.ffn.0.{weight,bias}
        blocks.N.mlp.fc2.{weight,bias}      -> blocks.N.ffn.3.{weight,bias}
        blocks.N.adaLN_modulation.*         -> blocks.N.adaLN_modulation.* (same)
        final_layer.*                       -> SKIPPED (out_channels mismatch: 8 vs 4)

    For cross-attn blocks, self_attn is used instead of attn:
        blocks.N.attn.qkv -> blocks.N.self_attn.in_proj_{weight,bias}
        blocks.N.attn.proj -> blocks.N.self_attn.out_proj.{weight,bias}
    """
    mapped = {}
    skipped = []

    for k, v in pretrained_sd.items():
        new_key = None

        # Skip class-conditional embedding and final layer
        if k.startswith('y_embedder.') or k.startswith('final_layer.'):
            skipped.append(k)
            continue

        # Patch embedding
        if k.startswith('x_embedder.'):
            new_key = k.replace('x_embedder.', 'patch_embed.')

        # Timestep embedding (already matches)
        elif k.startswith('t_embedder.'):
            new_key = k

        # Positional embedding (already matches)
        elif k == 'pos_embed':
            new_key = k

        # Transformer blocks
        elif k.startswith('blocks.'):
            parts = k.split('.', 2)  # ['blocks', 'N', 'rest']
            block_idx = parts[1]
            rest = parts[2]

            # Attention: qkv -> in_proj, proj -> out_proj
            if rest.startswith('attn.qkv.'):
                param = rest.split('.')[-1]  # weight or bias
                attn_name = 'self_attn' if block_type == 'cross-attn' else 'attn'
                new_key = f'blocks.{block_idx}.{attn_name}.in_proj_{param}'
            elif rest.startswith('attn.proj.'):
                param_path = rest.replace('attn.proj.', '')
                attn_name = 'self_attn' if block_type == 'cross-attn' else 'attn'
                new_key = f'blocks.{block_idx}.{attn_name}.out_proj.{param_path}'

            # MLP: fc1 -> ffn.0, fc2 -> ffn.3
            elif rest.startswith('mlp.fc1.'):
                param_path = rest.replace('mlp.fc1.', '')
                new_key = f'blocks.{block_idx}.ffn.0.{param_path}'
            elif rest.startswith('mlp.fc2.'):
                param_path = rest.replace('mlp.fc2.', '')
                new_key = f'blocks.{block_idx}.ffn.3.{param_path}'

            # adaLN modulation (already matches)
            elif rest.startswith('adaLN_modulation.'):
                new_key = k
            else:
                skipped.append(k)
                continue
        else:
            skipped.append(k)
            continue

        if new_key is not None:
            mapped[new_key] = v

    return mapped, skipped


def _is_diffusers_vae_format(sd):
    """Check if VAE state_dict uses diffusers naming (e.g. encoder.down_blocks)."""
    return any(k.startswith('encoder.down_blocks.') or k.startswith('decoder.up_blocks.')
               for k in sd.keys())


def _convert_vae_diffusers_to_ldm(sd):
    """Inline conversion of diffusers VAE keys to LDM format."""
    import sys, importlib
    # Import the converter from scripts/
    script_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
    sys.path.insert(0, os.path.abspath(script_dir))
    from convert_vae_diffusers_to_ldm import convert_state_dict
    sys.path.pop(0)
    return convert_state_dict(sd)


def _load_dit_pretrained(model, pretrain_root, block_type):
    """Load pretrained SD-VAE and DiT weights into a LatentDiffusion model.

    Expects:
        pretrain_root/sd_vae.ckpt   - SD-VAE state_dict (diffusers or LDM format)
        pretrain_root/dit_xl_2.pt   - Official DiT-XL/2 state_dict (EMA weights)
    """
    # 1. Load SD-VAE weights into first_stage_model
    vae_path = os.path.join(pretrain_root, 'sd_vae.ckpt')
    if os.path.exists(vae_path):
        vae_sd = torch.load(vae_path, map_location='cpu')
        # Handle different checkpoint formats
        if isinstance(vae_sd, dict) and 'state_dict' in vae_sd:
            vae_sd = vae_sd['state_dict']
        # Auto-convert from diffusers format if needed
        if _is_diffusers_vae_format(vae_sd):
            print("SD-VAE: detected diffusers format, converting to LDM format...")
            vae_sd = _convert_vae_diffusers_to_ldm(vae_sd)
        m, u = model.first_stage_model.load_state_dict(vae_sd, strict=False)
        print(f"SD-VAE: loaded from {vae_path} ({len(m)} missing, {len(u)} unexpected)")
    else:
        print(f"WARNING: SD-VAE checkpoint not found at {vae_path}. "
              f"VAE will be randomly initialized.")

    # 2. Load pretrained DiT weights into the diffusion model
    dit_path = os.path.join(pretrain_root, 'dit_xl_2.pt')
    if os.path.exists(dit_path):
        dit_sd = torch.load(dit_path, map_location='cpu')
        # Official pretrained checkpoints are raw state_dicts.
        # Training checkpoints have 'ema' key.
        if isinstance(dit_sd, dict) and 'ema' in dit_sd:
            dit_sd = dit_sd['ema']

        mapped_sd, skipped = _map_dit_pretrained_keys(dit_sd, block_type)

        # Filter out keys with shape mismatches (e.g. adaLN has 4*D modulation
        # but pretrained adaLN-Zero has 6*D). This avoids RuntimeError on load.
        target_sd = model.model.diffusion_model.state_dict()
        shape_mismatched = []
        for k in list(mapped_sd.keys()):
            if k in target_sd and mapped_sd[k].shape != target_sd[k].shape:
                shape_mismatched.append(k)
                del mapped_sd[k]

        # Load into diffusion_model (inside DiffusionWrapper)
        m, u = model.model.diffusion_model.load_state_dict(mapped_sd, strict=False)
        n_loaded = len(mapped_sd) - len(u)
        print(f"DiT pretrained: loaded {n_loaded}/{len(mapped_sd)} mapped weights, "
              f"{len(m)} missing (new params), {len(skipped)} skipped (y_embedder/final_layer)")
        if shape_mismatched:
            print(f"  {len(shape_mismatched)} keys skipped due to shape mismatch "
                  f"(e.g. adaLN modulation size differs from pretrained adaLN-Zero)")
    else:
        print(f"WARNING: DiT checkpoint not found at {dit_path}. "
              f"DiT will be randomly initialized.")


def _map_pixart_pretrained_keys(pretrained_sd):
    """Map official PixArt-alpha checkpoint keys to our PixArtForFMRI naming.

    Official PixArt-alpha key structure:
        x_embedder.*            -> x_embedder.*  (same)
        t_embedder.*            -> t_embedder.*  (same)
        t_block.*               -> t_block.*     (same)
        y_embedder.*            -> SKIPPED (T5 caption embedder, replaced by fMRI projection)
        pos_embed               -> pos_embed     (same)
        blocks.N.attn.*         -> blocks.N.attn.*  (self-attention, needs qkv remapping)
        blocks.N.cross_attn.*   -> SKIPPED (shape mismatch: 4096 vs 512 context_dim)
        blocks.N.norm1/norm2.*  -> blocks.N.norm1/norm2.* (same)
        blocks.N.mlp.*          -> blocks.N.mlp.* (needs fc1/fc2 remapping)
        blocks.N.scale_shift_table -> blocks.N.scale_shift_table (same)
        final_layer.*           -> final_layer.* (same if out_channels match)
    """
    mapped = {}
    skipped = []

    for k, v in pretrained_sd.items():
        # Skip caption embedder (T5-conditioned, we use fMRI)
        if k.startswith('y_embedder.'):
            skipped.append(k)
            continue

        # Skip cross-attention (shape mismatch: 4096 vs 512 context_dim)
        if '.cross_attn.' in k:
            skipped.append(k)
            continue

        # Self-attention: PixArt uses timm-style qkv, our SelfAttention also uses qkv
        # PixArt: blocks.N.attn.qkv.{weight,bias} -> blocks.N.attn.qkv.{weight,bias}
        # PixArt: blocks.N.attn.proj.{weight,bias} -> blocks.N.attn.proj.{weight,bias}
        # These match our SelfAttention naming directly.

        # MLP: PixArt uses timm Mlp with fc1/fc2, our Mlp also has fc1/fc2
        # These match directly.

        # Everything else maps as-is
        mapped[k] = v

    return mapped, skipped


def _load_pixart_pretrained(model, pretrain_root, block_type):
    """Load pretrained SD-VAE and PixArt-alpha weights into a LatentDiffusion model.

    Expects:
        pretrain_root/sd_vae.ckpt       - SD-VAE state_dict (diffusers or LDM format)
        pretrain_root/pixart_xl_2.pth   - Official PixArt-alpha XL/2 state_dict
    """
    # 1. Load SD-VAE weights into first_stage_model
    vae_path = os.path.join(pretrain_root, 'sd_vae.ckpt')
    if os.path.exists(vae_path):
        vae_sd = torch.load(vae_path, map_location='cpu')
        if isinstance(vae_sd, dict) and 'state_dict' in vae_sd:
            vae_sd = vae_sd['state_dict']
        if _is_diffusers_vae_format(vae_sd):
            print("SD-VAE: detected diffusers format, converting to LDM format...")
            vae_sd = _convert_vae_diffusers_to_ldm(vae_sd)
        m, u = model.first_stage_model.load_state_dict(vae_sd, strict=False)
        print(f"SD-VAE: loaded from {vae_path} ({len(m)} missing, {len(u)} unexpected)")
    else:
        print(f"WARNING: SD-VAE checkpoint not found at {vae_path}. "
              f"VAE will be randomly initialized.")

    # 2. Load pretrained PixArt-alpha weights into the diffusion model
    pixart_path = os.path.join(pretrain_root, 'pixart-xl-2-256x256', 'PixArt-XL-2-256x256.pth')
    if os.path.exists(pixart_path):
        pixart_sd = torch.load(pixart_path, map_location='cpu')
        if isinstance(pixart_sd, dict) and 'state_dict' in pixart_sd:
            pixart_sd = pixart_sd['state_dict']

        mapped_sd, skipped = _map_pixart_pretrained_keys(pixart_sd)

        # Filter out keys with shape mismatches
        target_sd = model.model.diffusion_model.state_dict()
        shape_mismatched = []
        for k in list(mapped_sd.keys()):
            if k in target_sd and mapped_sd[k].shape != target_sd[k].shape:
                shape_mismatched.append(k)
                del mapped_sd[k]

        m, u = model.model.diffusion_model.load_state_dict(mapped_sd, strict=False)
        n_loaded = len(mapped_sd) - len(u)
        print(f"PixArt pretrained: loaded {n_loaded}/{len(mapped_sd)} mapped weights, "
              f"{len(m)} missing (new params), {len(skipped)} skipped (y_embedder/cross_attn)")
        if shape_mismatched:
            print(f"  {len(shape_mismatched)} keys skipped due to shape mismatch")
    else:
        print(f"WARNING: PixArt checkpoint not found at {pixart_path}. "
              f"PixArt will be randomly initialized.")


class fLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True,
                 backbone='unet'):
        self.backbone = backbone
        self.is_dit = backbone.startswith('dit')
        self.is_pixart = backbone.startswith('pixart')

        config_filename = BACKBONE_CONFIG_MAP.get(backbone, 'config.yaml')
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.config_path = os.path.join(pretrain_root, config_filename)
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)

        if self.is_pixart:
            # Load SD-VAE + pretrained PixArt weights separately
            block_type = config.model.params.unet_config.params.block_type
            _load_pixart_pretrained(model, pretrain_root, block_type)
        elif self.is_dit:
            # Load SD-VAE + pretrained DiT weights separately
            block_type = config.model.params.unet_config.params.block_type
            _load_dit_pretrained(model, pretrain_root, block_type)
        else:
            # Load original UNet + VQ-VAE checkpoint
            pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
            m, u = model.load_state_dict(pl_sd, strict=False)

        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()
            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['fmri']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


