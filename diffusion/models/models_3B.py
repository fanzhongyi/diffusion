# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Constructors for diffusion models."""

import json
from typing import Dict, List, Optional

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

from diffusion.datasets.multi_tokenizer import MultiTokenizer
from diffusion.models.multi_text_encoder import MultiTextEncoder
from diffusion.models.stable_diffusion_3B import StableDiffusion3B

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False


def stable_diffusion_3B(
    model_name: str,
    text_encoders: Dict[str, str],
    tokenizer: MultiTokenizer,
    unet_model_config_path: Optional[str] = None,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
):
    """Stable diffusion 3B model training setup.

    Requires batches of matched images and CLIP&T5 text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str, optional): Name of the model to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        text_encoders (Optional[str, Dict]): names list of multiple text encoder.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int, optional): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        encode_latents_in_fp16 (bool, optional): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool, optional): Whether to use FSDP. Defaults to True.
    """
    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [
            MeanSquaredError(),
            FrechetInceptionDistance(normalize=True)
        ]
    if val_guidance_scales is None:
        val_guidance_scales = [1.0, 3.0, 7.0]
    if loss_bins is None:
        loss_bins = [(0, 1)]
    # Fix a bug where CLIPScore requires grad
    for metric in val_metrics:
        if isinstance(metric, CLIPScore):
            metric.requires_grad_(False)

    if unet_model_config_path is not None:
        unet_config = json.load(open(unet_model_config_path))
        unet = UNet2DConditionModel.from_config(unet_config)
    else:
        unet = UNet2DConditionModel.from_pretrained(model_name,
                                                    subfolder='unet')

    mtext_encoder = MultiTextEncoder(
        text_encoders=text_encoders,
        feature_dim=unet.config['encoder_hid_dim'],
        encode_latents_in_fp16=encode_latents_in_fp16,
    )

    if encode_latents_in_fp16:
        vae = AutoencoderKL.from_pretrained(model_name,
                                            subfolder='vae',
                                            torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')

    noise_scheduler = DDPMScheduler.from_pretrained(model_name,
                                                    subfolder='scheduler')
    inference_noise_scheduler = DDIMScheduler.from_pretrained(
        model_name, subfolder='scheduler')

    model = StableDiffusion3B(
        unet=unet,
        vae=vae,
        mtext_encoder=mtext_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
    return model
