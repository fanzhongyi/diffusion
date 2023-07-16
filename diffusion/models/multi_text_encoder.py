# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Multiple Text Encoder."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, T5EncoderModel


class MultiTextEncoder(nn.Module):
    """MultiTextEncoder ComposerModel.

    Args:
        text_encoders (torch.nn.Module): HuggingFace CLIP or LLM text enoders.
        e.g. {
            "clip-G": ...,
            "t5-L": ...,
            "clip-B": ...,
        }
        encode_latents_in_fp16 (bool): whether to encode latents in fp16.
            Default: `False`.
    """

    def __init__(
            self,
            text_encoders: Dict[str, str],
            feature_dim: int = 4096,
            encode_latents_in_fp16: bool = False,
            gather_order: Tuple = ('clip_G', 'clip_B', 't5_L'),
    ):
        super().__init__()

        self.feature_dim = feature_dim
        encoders = {}
        projs = {}

        # initlization
        latent_type = torch.float16 if encode_latents_in_fp16 else torch.float
        for name, path in text_encoders.items():

            # multiple encoders
            if 'clip' in name.lower():
                encoder = CLIPTextModel.from_pretrained(
                    path, subfolder='text_encoder', torch_dtype=latent_type)
            elif 't5' in name.lower():
                encoder = T5EncoderModel.from_pretrained(
                    path, subfolder='text_encoder', torch_dtype=latent_type)
            else:
                assert 't5' in name.lower() or 'clip' in name.lower()
                encoder = None  # makes pyright happy

            encoders[name] = encoder

            # multiple projection layers
            encoder_dim = encoder.config.hidden_size  # XXX: where should I read the hidden_size?
            if not encoders[name].config.hidden_size == feature_dim:
                proj = nn.Linear(encoder_dim, feature_dim)
            else:
                proj = nn.Identity()

            projs[name] = proj

        self.encoders = nn.ModuleDict(encoders)
        self.projs = nn.ModuleDict(projs)

        # TODO: remove gather order
        self.gather_order = gather_order
        self.encoder_keys = list(encoders.keys())
        assert set(self.gather_order) == set(self.encoder_keys)

    def forward(
        self,
        input_clip,
        input_t5,
        mask_clip=None,
        mask_t5=None,
    ):
        # forward multiple encoders
        with torch.no_grad():
            embeds_unaligned = {}
            for name, encoder in self.encoders.items():
                if 'clip' in name.lower():
                    embeds_unaligned[name] = encoder(
                        input_ids=input_clip, attention_mask=mask_clip)[0]
                if 't5' in name.lower():
                    embeds_unaligned[name] = encoder(input_ids=input_t5,
                                                     attention_mask=mask_t5)[0]
        # forward multiple projection layers
        embeds = {
            name: proj(embeds_unaligned[name])
            for name, proj in self.projs.items()
        }
        encoder_hidden_states = [embeds[k] for k in self.gather_order]
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
        return encoder_hidden_states
