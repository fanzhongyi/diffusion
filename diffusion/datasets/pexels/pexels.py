# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Pexels dataset."""

import json
import os
import random
from typing import Callable, List, Optional, Sequence, Union

import torch
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.pexels.transforms import LargestCenterSquare

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class PexelsDataset(Dataset):

    def __init__(
        self,
        transform,
        tokenizer,
        img_dataset_prefix,
        json_list,
        caption_drop_prob,
        client,
    ):

        self.transform = transform
        self.tokenizer = tokenizer

        self.img_dataset_prefix = img_dataset_prefix
        self.caption_drop_prob = caption_drop_prob

        self.client = client

        self.json_list = json.load(open(json_list))
        self.files = []
        for js in self.json_list:
            dic = json.load(open(js))
            for value in dic.values():
                sample = {
                    'json': value,
                    'img': value['img_params']['local_paths'][0],
                }
                self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = self.files[index]

        # process img
        img_paths = sample['json']['img_params']['local_paths']
        img_path = random.choice(img_paths)

        local_img_path = os.path.join(self.img_dataset_prefix, img_path)
        if os.path.exists(local_img_path):
            img = Image.open(local_img_path).convert("RGB")
        else:
            if 's3' not in sample['json']['img_params']:
                raise KeyError(f"s3 not in the json {sample['json']}")

            img_path = os.path.join(sample['json']['img_params']['s3'],
                                    img_path)
            # TODO: need backoff due to network requests.
            stream = self.client.get(img_path, enable_stream=True)
            img = Image.open(stream)
            if img.mode != 'RGB':
                img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # process txt
        try:
            prompts = sample['json']['img_params']['text_prompts']
        except:
            prompts = sample['json']['user_input']['text_prompts']

        prompt = random.choice(prompts) if isinstance(
            prompts, list) and len(prompts) > 0 else ''

        tags = sample["json"]["img_params"]["tags"]
        if len(tags) > 5:
            tags = random.sample(tags, 5)

        caption = ",".join([prompt, *tags, 'raw data'])

        if torch.rand(1) < self.caption_drop_prob:
            caption = ''

        tokenized_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )['input_ids']
        tokenized_caption = torch.tensor(tokenized_caption)
        out = {'image': img, 'captions': tokenized_caption}
        return out


def build_pexels_dataloader(
    img_dataset_prefix: str,
    json_list: str,
    batch_size: int,
    petrel_conf_path: str,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_samples: Optional[int] = None,
    drop_last: bool = True,
    **dataloader_kwargs,
):
    """Builds a Pexels dataloader.

    Args:
        batch_size (int): The batch size to use.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (int): The size to resize the image to. Default: ``256``.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """

    # Create a client for s3 remote access
    client = Client(petrel_conf_path)

    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path,
                                              subfolder='tokenizer')

    center_square_crop = LargestCenterSquare(resize_size)
    # Normalize from 0 to 1 to -1 to 1
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose(
        [center_square_crop,
         transforms.ToTensor(), normalize])

    dataset = PexelsDataset(
        transform=transform,
        tokenizer=tokenizer,
        img_dataset_prefix=img_dataset_prefix,
        json_list=json_list,
        caption_drop_prob=caption_drop_prob,
        client=client,
    )
    # Create a subset of the dataset
    if num_samples is not None:
        dataset = Subset(dataset, range(num_samples))  # type: ignore

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
