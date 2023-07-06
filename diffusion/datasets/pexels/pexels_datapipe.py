# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Pexels Datapipe & dataloader2"""
'''
TODO:
2. replace json with orjson for speedup
4. optimizing
'''

import json
import os
import random
from functools import partial
from pprint import pprint
from typing import Optional

from pprint import pprint
import backoff
import torch
import torch.distributed as dist
from petrel_client.client import Client
from PIL import Image
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import (DataLoader2, DistributedReadingService,
                                   MultiProcessingReadingService,
                                   SequentialReadingService)
from torchdata.dataloader2.adapter import CacheTimeout, Shuffle
from torchdata.datapipes.iter import Batcher, Collator, FileLister, FileOpener, IterableWrapper, Mapper, ShardingFilter
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.pexels.transforms import LargestCenterSquare

from .utils import filter_fn

# import orjson

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


def load_image(img_params, transform, data_path, client):

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=10)
    def load_image_with_backoff(img_path):
        stream = client.get(img_path, enable_stream=True)
        img = Image.open(stream)
        return img

    img_paths = img_params['local_paths']
    img_path = random.choice(img_paths)

    img_path_local = os.path.join(data_path, img_path)
    if os.path.exists(img_path_local):
        img = Image.open(img_path_local)
    else:
        assert 's3' in img_params
        img_s3_path = os.path.join(img_params['s3'], img_path)
        try:
            img = load_image_with_backoff(img_s3_path)
        except Exception as e:
            print(f"Cannot load {img_s3_path=}. Ignore {e}. Return None.")
            return None

    if img.mode != 'RGB':
        img = img.convert('RGB')

    if transform is not None:
        img = transform(img)

    return img


def load_caption(sample, caption_drop_prob, tokenizer):

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

    if torch.rand(1) < caption_drop_prob:
        caption = ''

    tokenized_caption = tokenizer(
        caption,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    )['input_ids']
    tokenized_caption = torch.tensor(tokenized_caption)

    sample['captions'] = tokenized_caption
    return sample


def select(sample):
    return {
        'image': sample['image'],
        'captions': sample['captions'],
    }


def ImgDatapipe(
    data_path,
    json_list,
    img_transform=None,
    caption_drop_prob=0.,
    tokenizer=None,
    client=None,
    filter_strategy=None,
):

    files = []
    for js in json.load(open(json_list)):
        files.extend(json.load(open(js)).values())

    dp = IterableWrapper(files)

    if filter_strategy is not None:
        filter_strategy = json.load(
            open(filter_strategy)) if filter_strategy else None
        dp = dp.filter(partial(filter_fn, filter_strategy=filter_strategy))

    dp = dp.shuffle(buffer_size=10000)

    if dist.is_initialized():
        dp = dp.sharding_filter(SHARDING_PRIORITIES.MULTIPROCESSING)
        # NOTE: This will mark the pipeline above as non-replicable
        # dp = dp.sharding_round_robin_dispatch(
        #     SHARDING_PRIORITIES.MULTIPROCESSING)
        dp.apply_sharding(dist.get_world_size(), dist.get_rank())

    dp = dp.map(partial(load_image,
                        data_path=data_path,
                        transform=img_transform,
                        client=client),
                input_col='img_params',
                output_col='image')

    if tokenizer is not None:
        dp = dp.map(
            partial(load_caption,
                    caption_drop_prob=caption_drop_prob,
                    tokenizer=tokenizer))

    dp = dp.map(select)
    return dp


def build_pexels_dataloader(
    data_path: str,
    json_list: str,
    batch_size: int = 4,
    petrel_conf: str = '',
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    filter_strategy: Optional[str] = None,
    caption_drop_prob: float = 0.0,
    filter_strategy: str = '',
    resize_size: int = 256,
    num_workers: int = 4,
    prefetch_count: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """ Building a image dataset (Pexels/ Midjourney).

    Args:
        data_path (str): the Pexels / Midjourney image data path.
        json_list (str): the json list file path.
        batch_size (int): The batch size to use.
        petrel_conf (str): petrel_oss conf for s3 file loading.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        filter_strategy (str): the filter strategy json path.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (int): The size to resize the image to. Default: ``256``.
        num_workers (Optional[int]): the workers number to load data. Default: ``None`` uses all available samples.
        prefetch_count (int): the prefetch_count in data loading graph.
        shuffle (bool): Shuffle or not.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
    """

    # Create a client for s3 remote access
    client = Client(petrel_conf)

    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path,
                                              subfolder='tokenizer')

    center_square_crop = LargestCenterSquare(resize_size)
    # Normalize from 0 to 1 to -1 to 1
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose(
        [center_square_crop,
         transforms.ToTensor(), normalize])

    dp = ImgDatapipe(
        data_path=data_path,
        json_list=json_list,
        img_transform=transform,
        caption_drop_prob=caption_drop_prob,
        tokenizer=tokenizer,
        client=client,
        filter_strategy=filter_strategy,
    )

    dp = dp.batch(batch_size=batch_size, drop_last=drop_last)

    dp = dp.collate()

    if dist.is_initialized():
        dp = dp.fullsync()

    dp = dp.prefetch(prefetch_count)

    mp_rs = MultiProcessingReadingService(num_workers=num_workers)
    if dist.is_initialized():
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
    else:
        rs = mp_rs

    dl = DataLoader2(
        dp,
        datapipe_adapter_fn=[
            Shuffle(shuffle),
            # CacheTimeout(600),
        ],
        reading_service=rs,
    )
    dl.batch_size = batch_size

    # for obj in dl:
    #     __import__('ipdb').set_trace()
    #     pprint(obj)
    #     pprint(type(obj))

    return dl


if __name__ == '__main__':
    ImgDatapipe(data_path='/home/babyfan/')
