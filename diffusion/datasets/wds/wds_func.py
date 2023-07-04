# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0

# Code based on https://github.com/mlfoundations/open_clip/blob/fb72f4db1b17133befd6c67c9cf32a533b85a321/src/training/data.py
"""Webdataset based on open_clip repo."""
'''
TODO: Now Each Dataset have its own dataloader workers. It is necessary to organize all datasets and assign workers
'''

import ast
import json
import logging
import math
import os
import random
import re
import tarfile
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from multiprocessing import Value
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import braceexpand
import webdataset as wds
from composer.utils.dist import get_sampler
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, SubsetRandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import CLIPTokenizer
from webdataset.filters import _shuffle
from webdataset.gopen import gopen
from webdataset.handlers import reraise_exception
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

from diffusion.datasets.pexels.transforms import LargestCenterSquare

from .utils import filter_fn

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    # logging.warning(f'Handling dataset error ({repr(exn)}). Ignoring.')
    return True


class SharedEpoch:

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler,
                                                   DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum(
            [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


class detshuffle2(wds.PipelineStage):

    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class WebLoaderwithlen(wds.WebLoader):

    def __len__(self):
        return self.num_batches


def get_possible_shards(input_shards: str, template: str = 's3://{}'):
    if os.path.isdir(input_shards):
        return [
            os.path.join(path, file_name)
            for path, _, file_list in os.walk(input_shards)
            for file_name in file_list
            if file_name.endswith('.tar')
        ]
    elif os.path.isfile(input_shards) and input_shards.endswith('.json'):
        input_shards = json.load(open(input_shards))
        assert isinstance(input_shards, list)
        return [template.format(ele) for ele in input_shards]
    else:
        return None


def petrel_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    client=None,
    **kw: Dict[str, Any],
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
    for sample in data:
        assert isinstance(sample, dict) and "url" in sample
        url = sample["url"]
        global_json_url = None

        if url.startswith('s3://'):
            # Datasets on Cloud storage
            if 's3://agishared1/LAION5B' in url:
                global_json_url = url.replace(
                    's3://agishared1/',
                    's3://agishared1/LAION5B-json/').replace('.tar', '.json')
                if '.new.tar' in url:
                    global_json_url = global_json_url.replace(
                        ".new.json", ".json")
            elif "s3://agishared1/datasets/" in url:
                global_json_url = url.replace(
                    's3://agishared1/',
                    's3://agishared1/datasets-json/').replace('.tar', '.json')

            if not global_json_url or not client.contains(global_json_url):
                continue

        else:
            # Datasets on local filesystem
            global_json_url = url.replace('.tar', '.json')

            if '.new.tar' in url and 'laion5b' in url.lower():
                global_json_url = global_json_url.replace(".new.json", ".json")

            if not global_json_url or not os.path.exists(global_json_url):
                continue

        for i in range(4):
            try:

                # TODO: Need backoff
                if url.startswith('s3://'):
                    stream = client.get(url, enable_stream=True).read()
                    global_json = json.load(
                        client.get(global_json_url, enable_stream=True))
                else:
                    stream = gopen(url, **kw).read()
                    global_json = json.load(gopen(global_json_url))

                sample.update(stream=stream, global_json=global_json)
                yield sample
                break

            except Exception as exn:
                logging.info(
                    f'{url} or {global_json_url} FAILED! {"Retrying" if i<3 else "Skipped"}'
                )
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                else:
                    break


def group_by_keys_nothrow(
    data,
    keys=base_plus_ext,
    lcase=True,
    suffixes=None,
    handler=None,
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample[
                "__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                # print(f'filtering name: {current_sample["__key__"]}, sufix:{current_sample.keys()}')
                # print(f'filtering name: {current_sample["__key__"]}, sufix:{current_sample.keys()}')
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


# TODO: use args to control rename_func and select_files
def rename_func(fname, src=None, tgt=None):
    return fname[:-len(src)] + tgt if src and tgt and fname.endswith(
        src) else fname


def select_files(fname, key=None):
    return key is None or not fname.endswith(key)


def tar_file_iterator(
    fileobj: tarfile.TarFile,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    Args:
        fileobj: the tar file stream.
        skip_meta: regexp for keys that are skipped entirely. Defaults to r"__[^/]*__($|/)".
        handler: exception handler. Defaults to reraise_exception.
        select: predicate for selecting files. Defaults to None.

    Yields:
        a stream of samples.
    """
    stream = tarfile.open(fileobj=BytesIO(fileobj["stream"]), mode="r|*")
    global_json = fileobj["global_json"]
    meta_prefix = "__"
    meta_suffix = "__"
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if ("/" not in fname and fname.startswith(meta_prefix) and
                    fname.endswith(meta_suffix)):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if rename_files is not None:
                fname = rename_files(fname)
            if select_files is not None and not select_files(fname):
                continue
            data = stream.extractfile(tarinfo).read()
            if '.json' in fname and global_json is not None:
                result = dict(
                    fname=fname,
                    data=json.dumps(
                        global_json[fname.strip('.json')]).encode("utf-8"))
                logging.info(f'using global json for {fileobj["url"]}')
            else:
                result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ ",) + exn.args[1:]
            if handler(exn):
                logging.info(f'cannot read {fname=}, read next in datastream')
                continue
            else:
                break
    del stream


def tar_file_expander_oss(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.

    Args:
        data: iterator over opened tar file streams.
        handler: exception handler.
        select_files: select files from tarfiles by name (permits skipping files).

    Yields:
        a stream of samples.
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(
                    source,
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
            ):
                assert (isinstance(sample, dict) and "data" in sample and
                        "fname" in sample)
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def tarfile_to_samples_nothrow(
    src,
    handler=log_and_continue,
    client=None,
    rename_func=None,
    select_files=None,
):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = petrel_opener(src, handler=handler, client=client)
    files = tar_file_expander_oss(streams,
                                  handler=handler,
                                  rename_files=rename_func,
                                  select_files=select_files)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or
                 'webp' in sample)
    has_json = ('json' in sample)
    # print(f'{len(sample.keys())}, {sample.keys()}')
    if has_caption and has_image and not has_json:
        logging.info(f"keys: {sample.keys()}")
        logging.info('neglecting this sample because of no json loaded')
    return has_caption and has_image and has_json


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def get_wds_dataset(
    wds_data_dir,
    wds_template,
    wds_train_num_samples,
    batch_size,
    num_workers,
    seed,
    custom_filter,
    img_transform,
    tokenizer,
    floor=False,
    epoch=0,
    caption_drop_prob=0.,
    client=None,
):

    input_shards = get_possible_shards(wds_data_dir, template=wds_template)

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        num_samples = wds_train_num_samples
        if not num_samples:
            raise RuntimeError(
                'Currently, number of dataset samples must be specified for training dataset. '
                'Please specify via `--train-num-samples` if no dataset length info present.'
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    pipeline = [wds.SimpleShardList(input_shards, seed=seed)]
    # at this point we have an iterator over all the shards
    pipeline.extend([
        detshuffle2(
            bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL,
            seed=seed,
            epoch=shared_epoch,
        ),
        wds.split_by_node,
        wds.split_by_worker,
    ])  #TODO: recover

    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        partial(
            tarfile_to_samples_nothrow,
            client=client,
            rename_func=partial(rename_func, src='_en.txt', tgt='.txt'),
            select_files=partial(select_files, key='_origin.txt'),
        ),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
        wds.select(filter_no_caption_or_no_image),
        wds.select(custom_filter),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(
            image="jpg;png;jpeg;webp",
            text="txt",
            url="__url__",
            json="json",
        ),
        wds.map_dict(
            image=img_transform,
            text=lambda text: tokenizer(text if random.random() >=
                                        caption_drop_prob else ''),
            json=lambda json: json,
        ),
        wds.to_tuple("image", "text", "json"),
        wds.batched(batch_size, partial=False)
    ])

    dataset = wds.DataPipeline(*pipeline)

    assert num_shards >= num_workers * int(os.environ.get(
        'WORLD_SIZE', 1)), 'number of shards must be >= total workers'
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = batch_size * int(os.environ.get('WORLD_SIZE', 1))
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, num_workers)
    num_worker_batches = round_fn(num_batches /
                                  num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    dataset = dataset.with_epoch(
        num_worker_batches)  # each worker is iterating over this

    dataloader = WebLoaderwithlen(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    dataloader.batch_size = batch_size

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def build_wds_dataloader(
    wds_data_dir: str,
    wds_template: str,
    batch_size: int,
    filter_strategy_json: str,
    petrel_conf_path: str,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_samples: Optional[int] = None,
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

    filter_strategy = json.load(open(filter_strategy_json))

    dataset = get_wds_dataset(
        wds_data_dir=wds_data_dir,
        wds_template=wds_template,
        wds_train_num_samples=num_samples,
        batch_size=batch_size,
        num_workers=2,
        seed=0,
        custom_filter=partial(filter_fn, filter_strategy=filter_strategy),
        img_transform=transform,
        tokenizer=tokenizer,
        floor=False,
        epoch=0,
        caption_drop_prob=caption_drop_prob,
        client=client,
    )

    return dataset.dataloader

    # Create a subset of the dataset
    # if num_samples is not None:
    #     dataset = Subset(dataset, range(num_samples))  # type: ignore

    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     sampler=get_sampler(dataset),
    #     drop_last=drop_last,
    #     **dataloader_kwargs,
    # )

    # return dataloader
