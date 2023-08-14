# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Setup of RVT
Author: Ankit Goyal
"""
from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "einops",
    "pyrender",
    "tensorboardX",
    "transformers",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "tensorflow",
    "pyquaternion",
    "matplotlib",
    "clip @ git+https://github.com/openai/CLIP.git",
]

__version__ = "0.0.1"
setup(
    name="rvt",
    version=__version__,
    description="RVT",
    long_description="",
    author="Ankit Goyal",
    author_email="angoyal@nvidia.com",
    url="",
    keywords="robotics,computer vision",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=['rvt', 'mvt'],
    install_requires=requirements,
)
