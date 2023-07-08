# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Webdataset Datapipe & dataloader2"""

import json
import os
from functools import partial
from pprint import pprint
from typing import Optional

import backoff
import torch
import torch.distributed as dist
from petrel_client.client import Client
from PIL import Image
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import (DataLoader2, DistributedReadingService, MultiProcessingReadingService,
                                   SequentialReadingService)
from torchdata.dataloader2.adapter import CacheTimeout, Shuffle
from torchdata.datapipes.iter import Batcher, Collator, FileLister, FileOpener, IterableWrapper, ShardingFilter
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.wds.transforms import LargestCenterSquare

from .utils import filter_fn

# import orjson

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


def load_shards(file_path: str, template: str = '{}'):
    if os.path.isdir(file_path):
        return [
            os.path.join(path, file_name)
            for path, _, file_list in os.walk(file_path)
            for file_name in file_list
            if file_name.endswith('.tar')
        ]
    elif os.path.isfile(file_path) and file_path.endswith('.json'):
        file_path = json.load(open(file_path))
        assert isinstance(file_path, list)
        return [template.format(ele) for ele in file_path]
    else:
        return None


def load_caption(caption, tokenizer, caption_drop_prob=0.):

    if torch.rand(1) < caption_drop_prob:
        caption = ''

    tokenized_caption = tokenizer(
        caption,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    )['input_ids']
    tokenized_caption = torch.tensor(tokenized_caption)
    return tokenized_caption


def rename_key(sample):
    for ext in ['.jpg', '.png', '.jpeg', '.webp']:
        if ext in sample:
            return {
                # '__key__': sample['__key__'],
                'image': sample[ext],
                'text': sample['.txt'],
                'json': sample['.json'],
            }
    else:
        print(sample.keys())
        return None


def filter_no_caption_or_no_image(sample):
    has_caption = ('.txt' in sample)
    has_image = ('.png' in sample or '.jpg' in sample or '.jpeg' in sample or
                 '.webp' in sample)
    has_json = ('.json' in sample)
    # print(f'{len(sample.keys())}, {sample.keys()}')
    if has_caption and has_image and not has_json:
        print(f'{sample=} no json loaded, ignore')
    return has_caption and has_image and has_json


def decode(item):
    key, value = item
    if key.endswith('.json'):
        return key, json.loads(value.read())
    if key.endswith('.jpg') or key.endswith('.png') or key.endswith(
            '.jpeg') or key.endswith('webp'):
        return key, Image.open(value).convert('RGB')
    if key.endswith('.txt'):
        return key, value.read().decode('utf-8')


def select(sample):
    return {
        'image': sample['image'],
        'captions': sample['captions'],
    }


def WdsDatapipe(
    data_path='/mnt/sdd6/fanzhongyi1/datasets/laion/',
    img_transform=None,
    caption_drop_prob=0.,
    tokenizer=None,
    client=None,
    filter_strategy=None,
):

    input_shards = load_shards(data_path, template='{}')
    # print(input_shards[:10], len(input_shards))
    dp = IterableWrapper(input_shards)
    # dp = FileLister(data_path, '*.new.tar')

    dp = FileOpener(dp, mode='b')
    dp = dp.load_from_tar()
    dp = dp.map(decode)
    dp = dp.webdataset()
    dp = dp.filter(filter_no_caption_or_no_image)

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

    dp = dp.map(rename_key)

    if img_transform is not None:
        dp = dp.map(img_transform, input_col='image', output_col='image')

    if tokenizer is not None:
        dp = dp.map(partial(load_caption,
                            caption_drop_prob=caption_drop_prob,
                            tokenizer=tokenizer),
                    input_col='text',
                    output_col='captions')

    dp = dp.map(select)
    return dp


def build_wds_dataloader(
    data_path: str,
    batch_size: int = 4,
    petrel_conf: str = '',
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    filter_strategy: Optional[str] = None,
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_workers: int = 4,
    prefetch_count: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """ Building a tar format dataset (Webdataset).

    Args:
        data_path (str): a json path contain the webdataset tar format data path.
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

    dp = WdsDatapipe(
        data_path=data_path,
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
    WdsDatapipe(data_path='/home/babyfan/datasets/lain2b/')
