# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0

# Code based on https://github.com/mlfoundations/open_clip/blob/fb72f4db1b17133befd6c67c9cf32a533b85a321/src/training/data.py
"""Laion webdataset based on open_clip repo."""

import json
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value
from typing import Callable, List, Optional, Sequence, Union

import braceexpand
# import backoff
# import orjson
import torch
import torchvision.datasets as datasets
import webdataset as wds
from composer.utils.dist import get_sampler
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, SubsetRandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import CLIPTokenizer
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

from diffusion.datasets.wds.transforms import CenterCropSDTransform, LargestCenterSquare

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class WebDataset:

    def __init__(
        self,
        path,
        tokenizer,
        num_examples_to_see,
        batch_size=256,
        workers=1,
        train=True,
        resolution=512,
        filters=None,
        **kwargs,
    ):
        self.filters = filters or {}
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.workers = workers
        self.dataset = self.get_dataset(
            path,
            tokenizer=tokenizer,
            train=train,
            num_examples_to_see=num_examples_to_see,
            filters=self.filters,
        )

        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,  # Shuffling done in the webdataset
            num_workers=workers,
            persistent_workers=True,
        )

        print(f"Unused dataset parameters for WebDataset: {kwargs}")

    def get_dataset(self, url, tokenizer, train, num_examples_to_see, filters):
        transform = CenterCropSDTransform(center_crop=True,
                                          size=self.resolution)

        pipeline = [wds.ResampledShards(url)]

        # TODO: Currently does not support validation sampling well
        # Don't split by worker and node since we're sampling with replacement
        # if train:
        #     pipeline.append(wds.shuffle(2000))

        pipeline.extend([
            tarfile_to_samples_nothrow,
        ])

        if train:
            pipeline.append(wds.shuffle(2000))

        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.select(metadata_filters(filters)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(pixel_values="jpg;png;jpeg;webp",
                       input_ids="txt",
                       text_raw="txt"),
            wds.map(filter_keys(set(["pixel_values", "input_ids",
                                     "text_raw"]))),
            wds.map_dict(
                pixel_values=transform,
                input_ids=lambda text: tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0],
                text_raw=lambda text: text,
            ),
            wds.batched(self.batch_size,
                        partial=not train,
                        collation_fn=default_collate),
        ])

        effective_batch_size = dist_utils.compute_effective_batch_size(
            self.batch_size)

        num_worker_batches = math.ceil(num_examples_to_see /
                                       (effective_batch_size * self.workers))

        # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size

        return wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)


class ClassificationWebDataset(object):

    def __init__(
        self,
        path,
        tokenizer,
        num_examples_to_see,
        class_mapping,
        batch_size=256,
        workers=1,
        train=True,
        resolution=512,
        **kwargs,
    ):
        if isinstance(class_mapping, dict):
            self.class_mapping = class_mapping
        elif isinstance(class_mapping, os.PathLike) or isinstance(
                class_mapping, str):
            self.class_mapping = json.load(Path(class_mapping).open("r"))
        else:
            raise TypeError(
                f"{type(class_mapping)} not accepted, need str or os.PathLike")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.resolution = resolution
        self.workers = workers

        self.dataset = self.get_dataset(
            path,
            tokenizer=tokenizer,
            train=train,
            num_examples_to_see=num_examples_to_see,
        )

        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,  # Shuffling done in the webdataset
            num_workers=workers,
            persistent_workers=True,
        )

        logging.info(f"Unused dataset parameters for WebDataset: {kwargs}")

    def get_dataset(self, url, tokenizer, train, num_examples_to_see):
        transform = CenterCropSDTransform(center_crop=True,
                                          size=self.resolution)

        pipeline = [wds.ResampledShards(url)]

        if train:
            pipeline.append(wds.shuffle(100))

        pipeline.extend([
            tarfile_to_samples_nothrow,
        ])

        if train:
            pipeline.append(wds.shuffle(1000))

        pipeline.extend([
            wds.select(filter_no_cls_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(
                pixel_values="jpg;png;jpeg;webp",
                input_ids="cls",
                text_raw="cls",
                class_idx="cls",
            ),
            wds.map(
                filter_keys(
                    set(["pixel_values", "input_ids", "text_raw",
                         "class_idx"]))),
            wds.map_dict(
                pixel_values=transform,
                input_ids=lambda class_idx: tokenizer(
                    self.class_mapping[str(class_idx)],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0],
                text_raw=lambda class_idx: self.class_mapping[str(class_idx)],
            ),
            wds.batched(self.batch_size,
                        partial=not train,
                        collation_fn=default_collate),
        ])

        effective_batch_size = dist_utils.compute_effective_batch_size(
            self.batch_size)

        num_worker_batches = math.ceil(num_examples_to_see /
                                       (effective_batch_size * self.workers))

        # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size

        return wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)


def build_laion_dataloader(
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

    dataset = get_wds_dataset(
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
