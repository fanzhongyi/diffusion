# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Multiple Text Encoder."""

from typing import Dict

import torch
import torch.nn as nn
from transformers import CLIPTextModel, T5EncoderModel


class MultiTextEncoder(nn.Module):
    """MultiTextEncoder torch Model.

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
        encode_latents_in_fp16: bool = False,
    ):
        super().__init__()

        assert not encode_latents_in_fp16, 'Current MultiTextEncoder only support fp32'

        # initlization
        latent_type = torch.float16 if encode_latents_in_fp16 else torch.float
        for name, path in text_encoders.items():
            print(f'Start Init text_encoder: {name}')

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

            self._modules[name] = encoder

    @torch.no_grad()
    def forward_encoders(
        self,
        input_clip,
        input_t5,
        mask_clip=None,
        mask_t5=None,
    ):
        self.eval()
        # forward multiple encoders
        embeds_unaligned = {}
        for name, encoder in self.named_children():
            if 'clip' in name.lower():
                embeds_unaligned[name] = encoder(input_ids=input_clip,
                                                 attention_mask=mask_clip)[0]
            if 't5' in name.lower():
                embeds_unaligned[name] = encoder(input_ids=input_t5,
                                                 attention_mask=mask_t5)[0]

        return embeds_unaligned

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(
        self,
        input_clip,
        input_t5,
        mask_clip=None,
        mask_t5=None,
    ):
        embeds_unaligned = self.forward_encoders(
            input_clip=input_clip,
            input_t5=input_t5,
            mask_clip=mask_clip,
            mask_t5=mask_t5,
        )
        return embeds_unaligned
