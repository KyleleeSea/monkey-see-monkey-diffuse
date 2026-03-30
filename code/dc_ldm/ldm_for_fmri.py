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


class SDVAEWrapper(nn.Module):
    """Wrapper around Stable Diffusion's VAE to match the interface expected by LatentDiffusion."""

    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def encode(self, x):
        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.sample()
        return z  # Returns tensor; get_first_stage_encoding handles scale_factor

    def decode(self, z):
        with torch.no_grad():
            return self.vae.decode(z).sample


class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels, cond_dim=1152, global_pool=True):
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

class fLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/dit',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True):
        self.config_path = os.path.join(pretrain_root, 'config.yaml')
        config = OmegaConf.load(self.config_path)

        # Pass use_time_cond and global_pool to DiT config
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        # Instantiate model (LatentDiffusion -> DiffusionWrapper -> DiT_fMRI + SDVAEWrapper)
        model = instantiate_from_config(config.model)

        # Load pretrained DiT-XL/2-256 weights into the diffusion backbone
        dit_ckpt_path = os.path.join(pretrain_root, 'DiT-XL-2-256x256.pt')
        if os.path.exists(dit_ckpt_path):
            print(f'Loading pretrained DiT weights from {dit_ckpt_path}')
            model.model.diffusion_model.load_pretrained_dit(dit_ckpt_path)
        else:
            print(f'WARNING: DiT pretrained weights not found at {dit_ckpt_path}')
            print('Download from: https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt')
            print('Continuing with randomly initialized DiT weights.')

        # Replace cond_stage_model with fMRI encoder
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size

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
        self.model.run_full_validation_threshold = 0.15

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
            # SD VAE uses 8x downsampling
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 8, HW[1] // 8)

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
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
