/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_pointcloud.h
 *  @author Valts Blukis, NVIDIA
 *  @brief  Header for pointcloud rendering
 */

#pragma once

#include <ATen/ATen.h>


namespace rvt {

void render_feature_pointcloud_to_image(
    at::Tensor point_indices,  // Index into flattened image. -1 if out of bounds.
    at::Tensor point_depths,
    at::Tensor point_features,
    at::Tensor image_out,
    at::Tensor depth_out,
    float default_depth,
    float default_color);

void screen_space_splatting(
    int batch_size,
    at::Tensor depth_in,
    at::Tensor image_in,
    at::Tensor depth_out,
    at::Tensor image_out,
    float default_depth,
    float focal_length_px,
    float splat_radius,
    int kernel_size,
    bool orthographic
);

void aa_subsample(
  at::Tensor depth_in,
  at::Tensor image_in,
  at::Tensor depth_out,
  at::Tensor image_out,
  int aa_factor,
  float default_depth
);

} // namespace rvt