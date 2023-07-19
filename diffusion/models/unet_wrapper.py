import torch
import torch.nn as nn
import json
from diffusion.models.multi_text_encoder import MultiTextEncoder
from typing import Dict, Tuple
from typing import Optional
from .unet.unet_2d_condition import UNet2DConditionModel
from .unet.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,
)


class UNetWrapper(nn.Module):

    def __init__(
            self,
            model_name: str,
            unet_model_config_path: Optional[str],
            mtext_encoder: MultiTextEncoder,
            gather_order: Tuple = ('clip_G', 'clip_B', 't5_L'),
    ):

        super().__init__()

        if unet_model_config_path is not None:
            unet_config = json.load(open(unet_model_config_path))
            # unet_config['transformer_layers_per_block'] = unet_config.pop('transformer_depth')
            unet = UNet2DConditionModel.from_config(unet_config)
        else:
            unet = UNet2DConditionModel.from_pretrained(model_name,
                                                        subfolder='unet')

        feature_dim = unet.config.encoder_hid_dim or unet.config.cross_attention_dim  # type: ignore
        self.unet = unet

        projs = {}
        for name, encoder in mtext_encoder.encoders.items():
            # multiple projection layers
            encoder_dim = encoder.config.hidden_size
            if not encoder_dim == feature_dim:
                proj = nn.Linear(encoder_dim, feature_dim)
            else:
                proj = nn.Identity()
            projs[name] = proj
            print(f'Encoder-{name} Adapter {encoder_dim=} -> {feature_dim=}')

        self.projs = nn.ModuleDict(projs)

        self.gather_order = gather_order
        self.proj_keys = list(projs.keys())
        assert set(self.gather_order) == set(self.proj_keys)

    def forward(
        self,
        noised_latents,
        timesteps,
        embeds_unaligned: Dict[str, torch.Tensor],
        conditioning=None,
    ):

        if conditioning is None:
            conditioning = self.forward_projs(embeds_unaligned)
        ret = self.unet(noised_latents, timesteps, conditioning)
        return ret

    def forward_projs(self, embeds_unaligned: Dict[str, torch.Tensor]):

        # forward multiple projection layers
        embeds = {
            name: proj(embeds_unaligned[name])
            for name, proj in self.projs.items()
        }
        encoder_hidden_states = [embeds[k] for k in self.gather_order]
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
        return encoder_hidden_states

    def forward_adapter(self, embeds_unaligned):

        # just for consistency
        return self.forward_projs(embeds_unaligned)
