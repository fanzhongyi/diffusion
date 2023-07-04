# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0
"""Transforms for the laion dataset."""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class LargestCenterSquare:
    """Center crop to the largest square of a PIL image."""

    def __init__(self, size):
        self.size = size
        self.center_crop = transforms.CenterCrop(self.size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square.
        img = self.center_crop(img)
        return img


class CenterCropSDTransform:

    def __init__(self, center_crop, size):
        self.size = size
        self.center_crop = center_crop

    def __call__(self, image):
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)
