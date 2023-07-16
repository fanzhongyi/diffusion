# Copyright 2022 babyfan
# SPDX-License-Identifier: Apache-2.0
"""Multiple Tokenizer."""

from typing import Dict

import torch
from transformers import CLIPTokenizer, CLIPTokenizerFast, T5Tokenizer, T5TokenizerFast


class MultiTokenizer:

    def __init__(
        self,
        tokenizers_path: Dict,
        clip_max_length: int = 77,
        t5_max_length: int = 77 * 3,
        fast_version: bool = False,
    ):

        if fast_version:
            clipTokenizer, t5Tokenizer = CLIPTokenizerFast, T5TokenizerFast
        else:
            clipTokenizer, t5Tokenizer = CLIPTokenizer, T5Tokenizer

        for name, path in tokenizers_path.items():
            if 'clip' in name.lower():
                self.clip_tokenizer = clipTokenizer.from_pretrained(
                    path, subfolder='tokenizer')
            elif 't5' in name.lower():
                self.t5_tokenizer = t5Tokenizer.from_pretrained(
                    path, subfolder='tokenizer')
            else:
                assert 't5' in name.lower() or 'clip' in name.lower()

        assert hasattr(self, 'clip_tokenizer') and hasattr(self, 't5_tokenizer')

        self.clip_max_length = clip_max_length or self.clip_tokenizer.model_max_length
        self.t5_max_length = t5_max_length or self.clip_max_length * 3

        # cache null vectors for faster loading
        self.clip_null_tokenized = self.tokenize_with_clip('')
        self.t5_null_tokenized = self.tokenize_with_t5('')

    def __call__(self, *args, **kwargs):
        # default to clip tokenizer for compatiblity
        return self.clip_tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        # default to clip tokenizer for compatiblity
        return self.clip_tokenizer.decode(*args, **kwargs)

    def tokenize_with_clip(self, prompt):
        return self.clip_tokenizer(prompt,
                                   padding='max_length',
                                   max_length=self.clip_max_length,
                                   truncation=True,
                                   return_tensors='pt')

    def tokenize_with_t5(self, prompt):
        return self.t5_tokenizer(prompt,
                                 padding='max_length',
                                 max_length=self.t5_max_length,
                                 truncation=True,
                                 return_tensors='pt')

    def tokenize(self, prompt, device=None) -> Dict:
        # Custom tokenize method for MultiTextEncoder.
        tokenized_clip = self.tokenize_with_clip(prompt)
        tokenized_t5 = self.tokenize_with_t5(prompt)

        input4mencoder = dict(
            input_clip=tokenized_clip.input_ids,
            input_t5=tokenized_t5.input_ids,
            mask_clip=None,  # No need mask for CLIP Encoder
            mask_t5=tokenized_t5.attention_mask,
        )

        if device is not None:
            for k, v in input4mencoder.items():
                if v is not None:
                    input4mencoder[k] = v.to(device)

        return input4mencoder

    def tokenize_with_drop(
        self,
        prompt,
        clip_drop_prob=0.,
        t5_drop_prob=0.,
        uni_drop_prob=0.,
        device=None,
    ) -> Dict:
        # Custom tokenize method for MultiTextEncoder with various caption drop.

        if uni_drop_prob > 0.:  # clip & t5 use same drop probability

            if torch.rand(1) < uni_drop_prob:
                tokenized_clip = self.clip_null_tokenized
                tokenized_t5 = self.t5_null_tokenized
            else:
                tokenized_clip = self.tokenize_with_clip(prompt)
                tokenized_t5 = self.tokenize_with_t5(prompt)

        else:  # separate drop probability

            if torch.rand(1) < clip_drop_prob:
                tokenized_clip = self.clip_null_tokenized
            else:
                tokenized_clip = self.tokenize_with_clip(prompt)

            if torch.rand(1) < t5_drop_prob:
                tokenized_t5 = self.t5_null_tokenized
            else:
                tokenized_t5 = self.tokenize_with_t5(prompt)

        input4mencoder = dict(
            input_clip=tokenized_clip.input_ids,
            input_t5=tokenized_t5.input_ids,
            mask_clip=None,  # No need mask for CLIP Encoder
            mask_t5=tokenized_t5.attention_mask,
        )

        if device is not None:
            for k, v in input4mencoder.items():
                if v is not None:
                    input4mencoder[k] = v.to(device)

        return input4mencoder
