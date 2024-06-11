/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   bindings.cpp
 *  @author Valts Blukis, NVIDIA
 *  @brief  PyTorch bindings for pointcloud renderer
 */

#include <torch/extension.h>
#include "./render/render_pointcloud.h"

namespace rvt {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::module render = m.def_submodule("render");
    render.def("render_feature_pointcloud_to_image", &render_feature_pointcloud_to_image);
    render.def("screen_space_splatting", &screen_space_splatting);
    render.def("aa_subsample", &aa_subsample);
}

}

