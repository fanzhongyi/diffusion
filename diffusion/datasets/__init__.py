# Copyright 2022 babyfan authors
# SPDX-License-Identifier: Apache-2.0
"""
Datasets.

All 8 datasets can be divided into 2 groups, one is the crawl dataset and the other is the public webdataset.

Group 1:
    /mnt/CV_teamz/users/qiming/dataset/midjourney/meta_MJ_batch5+batch6-v4+.json \
    /mnt/CV_teamz/users/qiming/dataset/midjourney/meta_MJ_v5_origin.json \
    /mnt/CV_teamz/users/qiming/dataset/pexels/meta_0501+0601.json \

Group 2:
    /mnt/CV_teamz/open_datasets/datasets/LAION-5b/LAION-2B-en/laion2b-en.new.json \
    /mnt/CV_teamz/open_datasets/datasets/LAION-5b/LAION1B-nolang/1b-nolang-all.json \
    /mnt/CV_teamz/open_datasets/datasets/LAION-5b/LAION2B-multi/2b-multi-all.json \
    /mnt/CV_teamz/open_datasets/datasets/COYO-700M/coyo700m-all.json \
    /mnt/CV_teamz/open_datasets/datasets/CC12M/CC_all_dataset.json \

"""

from diffusion.datasets.pexels.pexels_datapipe import build_pexels_dataloader, ImgDatapipe
from diffusion.datasets.wds.wds_datapipe import build_wds_dataloader, WdsDatapipe

__all__ = [
    'build_pexels_dataloader',
    'ImgDatapipe',
    'build_wds_dataloader',
    'WdsDatapipe',
]
