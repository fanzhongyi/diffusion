# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Img dataset for individual image files, i.e. Pexels """

# TODO:
# 1. reimplement the dataset with Datapipe for better controller
# 2. replace json with orjson for speedup
# 3. add filter strategy
# 4. optimizing

import json
import os
import random
from typing import Optional

import backoff
import torch
from composer.utils.dist import get_sampler
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.pexels.transforms import LargestCenterSquare

# import orjson

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
        filter_fn=None,
    ):

        self.transform = transform
        self.tokenizer = tokenizer

        self.img_dataset_prefix = img_dataset_prefix
        self.caption_drop_prob = caption_drop_prob

        self.client = client
        self.filter_fn = filter_fn

        self.files = []
        for js in json.load(open(json_list)):
            sublist = json.load(open(js)).values()
            # once read in, start filtering, reduce load
            if filter_fn is not None:
                sublist = list(filter(filter_fn, sublist))
            self.files.extend(sublist)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = self.files[index]

        # process img
        img_paths = sample['img_params']['local_paths']
        img_path = random.choice(img_paths)

        img_path_local = os.path.join(self.img_dataset_prefix, img_path)
        if os.path.exists(img_path_local):
            img = Image.open(img_path_local)
        else:
            assert 's3' in sample['img_params']
            img_s3_path = os.path.join(sample['img_params']['s3'], img_path)
            try:
                img = self.load_image(img_s3_path)
            except Exception as e:
                print(f"Cannot load {img_s3_path=}. Ignore {e}. Return None.")
                return None

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # process txt
        try:
            prompts = sample['img_params']['text_prompts']
        except:
            prompts = sample['user_input']['text_prompts']

        prompt = random.choice(prompts) if isinstance(
            prompts, list) and len(prompts) > 0 else ''

        # NOTE: pexels datasets need to add labels as auxiliary inputs
        tags = sample.get("img_params", {}).get("tags", [])
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

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=10)
    def load_image(self, img_path):
        stream = self.client.get(img_path, enable_stream=True)
        img = Image.open(stream)
        return img


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
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
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
        sampler=get_sampler(dataset),
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
