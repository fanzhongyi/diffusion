# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion package setup."""

from setuptools import find_packages, setup

install_requires = [
    'mosaicml',
    'mosaicml-streaming',
    'hydra-core',
    'hydra-colorlog',
    'diffusers[torch]',
    'transformers[torch]',
    'sentencepiece',
    'wandb',
    'deepspeed',
    'xformers',
    'triton',
    'torchmetrics[image]',
    'torchdata',
    'torchinfo',
    'backoff',
    'orjson',
]

extras_require = {}

extras_require['dev'] = {
    'pre-commit>=2.18.1,<3',
    'pytest==7.3.0',
    'coverage[toml]==7.2.2',
    'pyarrow==11.0.0',
    'ipdb',
}

extras_require['all'] = set(dep for deps in extras_require.values() for dep in deps)

setup(
    name='diffusion',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
