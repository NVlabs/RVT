# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

PACKAGE_NAME = 'point_renderer'
DESCRIPTION = 'Fast Point Cloud Renderer'
URL = 'https://gitlab-master.nvidia.com/vblukis/point-renderer'
AUTHOR = 'Valts Blukis'
LICENSE = 'NVIDIA'
version = '0.2.0'


def get_extensions():
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    extensions = []
    sources = glob.glob('point_renderer/csrc/**/*.cpp', recursive=True)

    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1' or True:
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('point_renderer/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': ['-O3']})
        #include_dirs = get_include_dirs()
    else:
        assert(False, "CUDA is not available. Set FORCE_CUDA=1 for Docker builds")

    extensions.append(
        extension(
            name='point_renderer._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            #include_dirs=include_dirs
        )
    )

    for ext in extensions:
        ext.libraries = ['cudart_static' if x == 'cudart' else x
                         for x in ext.libraries]

    return extensions


if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        python_requires='>=3.7',

        # Package info
        packages=['point_renderer'],
        include_package_data=True,
        zip_safe=True,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        }

    )
