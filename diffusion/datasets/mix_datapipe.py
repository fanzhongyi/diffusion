# Copyright 2022 babyfan authors
# SPDX-License-Identifier: Apache-2.0
'''Unified mix datapipe for multiple data source.'''

import json
import os
from functools import partial
from pprint import pprint

import hydra
import torch.distributed as dist
from petrel_client.client import Client
from torchdata.dataloader2 import (DataLoader2, DistributedReadingService, MultiProcessingReadingService,
                                   SequentialReadingService)
from torchdata.dataloader2.adapter import CacheTimeout, Shuffle
from torchdata.datapipes.iter import (Batcher, Collator, FileLister, FileOpener, IterableWrapper, SampleMultiplexer,
                                      ShardingFilter)
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.wds.transforms import LargestCenterSquare


def build_mix_dataloader(
    datapipes,
    batch_size: int = 4,
    petrel_conf: str = '',
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    filter_strategy: str = '',
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_workers: int = 4,
    prefetch_count: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: bool = -1,
):
    """ Building a tar format dataset (Webdataset).

    Args:
        datapipes (*): multiple datapipes.
        dataset_weight (int): sampling weights of multiple datasets.
        batch_size (int): The batch size to use.
        petrel_conf_path (str): petrel_oss conf for s3 file loading.
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
    pprint(datapipes)

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

    # mix multiple data source
    sampling_weight = {}

    weights_sum = sum([dp['weight'] for dp in datapipes.values()])
    for dp_name, dp_conf in datapipes.items():
        weight = dp_conf.pop('weight') / weights_sum

        per_dp_conf = dict(
            img_transform=transform,
            caption_drop_prob=caption_drop_prob,
            tokenizer=tokenizer,
            client=client,
            filter_strategy=filter_strategy,
        )
        per_dp_conf.update(dp_conf)
        datapipe = hydra.utils.instantiate(per_dp_conf)

        sampling_weight[datapipe] = weight
        print(f'Init data source:\t{dp_name=}\t{weight=}')

    dp = SampleMultiplexer(pipes_to_weights_dict=sampling_weight, seed=seed)

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
        datapipe_adapter_fn=[Shuffle(shuffle),
                             CacheTimeout(600)],
        reading_service=rs,
    )
    dl.batch_size = batch_size

    # for obj in dl:
    #     __import__('ipdb').set_trace()
    #     pprint(obj)
    #     pprint(type(obj))

    return dl
