# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Webdataset Datapipe & dataloader2"""

import json
import os
from functools import partial
from typing import Optional

import backoff
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import (DataLoader2, DistributedReadingService,
                                   MultiProcessingReadingService,
                                   SequentialReadingService)
from torchdata.dataloader2.adapter import Shuffle
from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchvision import transforms

from diffusion.datasets.multi_tokenizer import MultiTokenizer
from diffusion.datasets.wds.transforms import LargestCenterSquare

from .utils import filter_fn

# import orjson

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        return []


def load_caption(
    sample,
    caption_drop_prob,
    tokenizer: MultiTokenizer,
):
    # TODO: separate drop probability
    tokenized_caption = tokenizer.tokenize_with_drop(
        sample['text'],
        clip_drop_prob=caption_drop_prob,
        t5_drop_prob=caption_drop_prob,
        uni_drop_prob=0.,
    )
    sample['captions'] = tokenized_caption['input_clip']
    sample['captions_t5'] = tokenized_caption['input_t5']
    sample['mask_t5'] = tokenized_caption['mask_t5']
    return sample


def rename_key(sample):
    for ext in ['.jpg', '.png', '.jpeg', '.webp']:
        if ext in sample:
            return {
                '__key__': sample['__key__'],
                'image': sample[ext],
                'text': sample['.txt'],
                'json': sample['.json'],
            }
    else:
        print(sample.keys())
        return None


def filter_no_caption_or_no_image(sample):
    has_caption = ('text' in sample)
    has_image = ('image' in sample)
    has_json = ('json' in sample)
    # print(f'{len(sample.keys())}, {sample.keys()}')
    if has_caption and has_image and not has_json:
        print(f'{sample=} no json loaded, ignore')
    return has_caption and has_image and has_json


def decode(sample):
    sample_decoded = {'__key__': sample['__key__']}

    try:
        sample_decoded['json'] = json.loads(sample['json'].read())
    except Exception:
        print(f'json parse error, {sample=}')

    try:
        sample_decoded['text'] = sample['text'].read().decode('utf-8')
    except Exception:
        print(f'text parse error, {sample=}')

    try:
        sample_decoded['image'] = Image.open(sample['image']).convert('RGB')
    except Exception:
        print(f'image parse error, {sample=}')

    return sample_decoded


def WdsDatapipe(
    data_path='/mnt/sdd6/fanzhongyi1/datasets/laion/',
    img_transform=None,
    caption_drop_prob=0.,
    tokenizer=None,
    client=None,
    filter_strategy=None,
    shuffle_buffer_size=5000,
):

    input_shards = load_shards(data_path, template='{}')
    dp = IterableWrapper(input_shards)

    # input_jsons = [
    #     tar_name.replace('/LAION5B-clean/', '/LAION5B-json/LAION5B-clean/').replace("datasets", "datasets-json/datasets").replace('.new.tar', '.json')
    #     for tar_name in input_shards
    # ]

    dp = dp.shuffle(buffer_size=shuffle_buffer_size)

    dp = FileOpener(dp, mode='b')
    dp = dp.load_from_tar()
    dp = dp.webdataset()

    dp = dp.map(rename_key)
    dp = dp.map(decode)
    dp = dp.filter(filter_no_caption_or_no_image)

    if filter_strategy is not None:
        filter_strategy = json.load(
            open(filter_strategy)) if filter_strategy else None
        dp = dp.filter(partial(filter_fn, filter_strategy=filter_strategy))

    dp = dp.shuffle(buffer_size=shuffle_buffer_size)

    if dist.is_initialized():
        dp = dp.sharding_filter(SHARDING_PRIORITIES.MULTIPROCESSING)
        dp.apply_sharding(dist.get_world_size(), dist.get_rank())
        # NOTE: This will mark the pipeline above as non-replicable
        # dp = dp.sharding_round_robin_dispatch(
        #     SHARDING_PRIORITIES.MULTIPROCESSING)

    if img_transform is not None:
        dp = dp.map(img_transform, input_col='image', output_col='image')

    if tokenizer is not None:
        dp = dp.map(
            partial(load_caption,
                    caption_drop_prob=caption_drop_prob,
                    tokenizer=tokenizer))

    dp = dp.slice(['image', 'captions', 'captions_t5', 'mask_t5'])
    return dp


def build_wds_dataloader(
    data_path: str,
    tokenizer: MultiTokenizer,
    batch_size: int = 4,
    petrel_conf: str = '',
    filter_strategy: Optional[str] = None,
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_workers: int = 4,
    prefetch_count: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = -1,
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
        seed (int): the random seed for multiple data source mixture.
    """

    # Create a client for s3 remote access
    from petrel_client.client import Client
    client = Client(petrel_conf)

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

    dp = dp.pin_memory()

    mp_rs = MultiProcessingReadingService(num_workers=num_workers)
    if dist.is_initialized():
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
    else:
        rs = mp_rs

    dl = DataLoader2(
        dp,
        datapipe_adapter_fn=[Shuffle(shuffle)],
        reading_service=rs,
    )
    dl.batch_size = batch_size
    dl.seed(seed)

    # for obj in dl:
    #     __import__('ipdb').set_trace()
    #     __import__('pprint').pprint(obj)
    #     __import__('pprint').pprint(type(obj))

    return dl


if __name__ == '__main__':
    WdsDatapipe(
        data_path='/mnt/CV_550w/LAION5B-clean/afs-laion2b.json',
        tokenizer=MultiTokenizer(
            {
                'clip_G':
                    '/mnt/CV_teamz/pretrained/CLIP-ViT-bigG-14-laion2B-39B-b160k/',
                'clip_B':
                    '/mnt/CV_teamz/pretrained/stable-diffusion-v1-4/',
                't5_L':
                    '/mnt/CV_teamz/pretrained/flan-t5-xxl/',
            },
            fast_version=True,
        ),
        filter_strategy='/mnt/CV_teamz/users/qiming/dataset/filter_strategy/stage2/afs_wbs_filter_strategy.json',
    )
