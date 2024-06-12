/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_feature_pointcloud.cu
 *  @author Valts Blukis, NVIDIA
 *  @brief  Renders a point cloud with associated feature vectors to image
 */

#include <ATen/ATen.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


namespace rvt {


// By what factor to scale depth for integer representation
__constant__ const float DEPTH_FACTOR = 1000;
__constant__ const float DEPTH_INV_FACTOR = 1 / DEPTH_FACTOR;


__global__ void render_pointcloud_to_depth_index_buffer_cuda_kernel(
    int64_t num_points,
    int64_t img_height,
    int64_t img_width,

    int64_t* point_indices,
    float* point_depths,

    uint64_t* packed_buffer
){
  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num_points) {
    int64_t pixel_index = point_indices[tidx];

    if (pixel_index >= 0 && pixel_index < img_height * img_width) {
      float point_depth = point_depths[tidx];
      uint32_t point_depth_mm = (uint32_t) (point_depth * DEPTH_FACTOR);
      uint64_t packed_depth_and_index = ((uint64_t) point_depth_mm << 32) | ((uint64_t) tidx);
      atomicMin((unsigned long long*) (packed_buffer + pixel_index), (unsigned long long)packed_depth_and_index);
    }
  }
} 


__global__ void output_render_to_feature_image_cuda_kernel(
    int64_t img_height,
    int64_t img_width,
    int64_t num_channels,
    int32_t num_points,
    float* point_features,
    uint64_t* packed_buffer,
    float* image_out,
    float* depth_out,
    float default_depth,
    float default_feature
) {
  
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < img_height * img_width) {
    uint64_t packed = packed_buffer[tidx];
    uint32_t packed_depth_mm = (uint32_t) (packed >> 32);
    // The modulo is to support batching without having to tile the features
    uint32_t packed_index = ((uint32_t) packed) % num_points;

    if (packed_depth_mm == 0xFFFFFFFF) {
      depth_out[tidx] = default_depth;
      for (int i = 0; i < num_channels; i++) {
        image_out[tidx * num_channels + i] = default_feature;
      }
    }
    else {
      depth_out[tidx] = (float) (packed_depth_mm) * DEPTH_INV_FACTOR;
      for (int i = 0; i < num_channels; i++) {
        image_out[tidx * num_channels + i] = point_features[packed_index * num_channels + i];
      }
    }
  }
}


__global__ void output_render_to_feature_image_cuda_2d_kernel(
    int64_t img_height,
    int64_t img_width,
    int64_t num_channels,
    int32_t num_points,
    float* point_features,
    uint64_t* packed_buffer,
    float* image_out,
    float* depth_out,
    float default_depth,
    float default_feature
) {
  uint pixidx = blockDim.x * blockIdx.x + threadIdx.x;
  uint cidx = blockDim.y * blockIdx.y + threadIdx.y;

  if (pixidx < img_height * img_width && cidx < num_channels) {
    uint64_t packed = packed_buffer[pixidx];
    uint32_t packed_depth_mm = (uint32_t) (packed >> 32);
    // The modulo is to support batching without having to tile the features
    uint32_t packed_index = ((uint32_t) packed) % num_points;

    if (packed_depth_mm == 0xFFFFFFFF) {
      if (cidx == 0)
        depth_out[pixidx] = default_depth;
      image_out[pixidx * num_channels + cidx] = default_feature;
    }
    else {
      if (cidx == 0)
        depth_out[pixidx] = (float) (packed_depth_mm) * DEPTH_INV_FACTOR;
      image_out[pixidx * num_channels + cidx] = point_features[packed_index * num_channels + cidx];
    }
  }
}


void render_feature_pointcloud_to_image(
    at::Tensor point_indices,  // Index into flattened image. -1 if out of bounds.
    at::Tensor point_depths,
    at::Tensor point_features,
    at::Tensor image_out,
    at::Tensor depth_out,
    float default_depth,
    float default_color) {

    int64_t num_points = point_indices.size(0);
    int32_t num_points_per_batch = point_features.size(0);
    int64_t img_height = image_out.size(0);
    int64_t img_width = image_out.size(1);
    int64_t num_channels = image_out.size(2);
    
    if (num_channels != point_features.size(1)) {
      throw std::runtime_error("Output image and point features must have the same channel dimension");
    }

    // TODO: Play with this to see if we can speed it up
    uint64_t num_threads_per_block = 1024;

    // Make sure cudaMalloc uses the correct device
    int device_index = point_indices.get_device();
    cudaSetDevice(device_index);

    // Allocate memory for storing packed depths and colors
    uint64_t* packed_depths_and_indices;
    cudaMalloc((void**) &packed_depths_and_indices, img_width*img_height*sizeof(uint64_t));
    cudaMemset(packed_depths_and_indices, 0xFFFFFFFFFFFFFFFF, img_width*img_height*sizeof(uint64_t));

    render_pointcloud_to_depth_index_buffer_cuda_kernel<<<(num_points + num_threads_per_block - 1) / num_threads_per_block, num_threads_per_block>>>(
            num_points,
            img_height,
            img_width,

            point_indices.data_ptr<int64_t>(),
            point_depths.data_ptr<float>(),

            packed_depths_and_indices);

    // With few channels, it's faster to launch a thread per pixel, in each thread looping over the channels and copying the data
    if (num_channels < 10)
    {
      output_render_to_feature_image_cuda_kernel<<<(img_height * img_width + num_threads_per_block - 1) / num_threads_per_block, num_threads_per_block>>>(
              img_height,
              img_width,
              num_channels,
              num_points_per_batch,
              point_features.data_ptr<float>(),
              packed_depths_and_indices,
              image_out.data_ptr<float>(),
              depth_out.data_ptr<float>(),
              default_depth,
              default_color);
    }
    // With more channels, it's better to launch a separate thread per pixel per channel, in each thread copying only one feature scalar
    else
    {
      output_render_to_feature_image_cuda_2d_kernel<<<dim3((img_height * img_width + num_threads_per_block - 1) / num_threads_per_block, num_channels), dim3(num_threads_per_block, 1)>>>(
            img_height,
            img_width,
            num_channels,
            num_points_per_batch,
            point_features.data_ptr<float>(),
            packed_depths_and_indices,
            image_out.data_ptr<float>(),
            depth_out.data_ptr<float>(),
            default_depth,
            default_color);
    }

    cudaFree(packed_depths_and_indices);
}


__global__ void screen_space_splatting_cuda_kernel(
    int64_t batch_size,
    int64_t img_height,
    int64_t img_width,
    int64_t num_channels,
    int k,
    float splat_radius,
    float focal_length_px,
    float* depth_in,
    float* image_in,
    float* depth_out,
    float* image_out,
    float default_depth,
    bool orthographic
) {
  uint x = blockDim.x * blockIdx.x + threadIdx.x;
  uint y = blockDim.y * blockIdx.y + threadIdx.y;
  uint c_index = y * img_width + x;
  uint batch_elem_height = img_height / batch_size;

  if (y < img_height && x < img_width) {
    float center_depth = depth_in[c_index];
    float min_depth = center_depth;
    int splat_index = c_index;

    int b_elem = y / batch_elem_height;

    // Loop over pixel's neighbourhood
    for (int dx = -k/2; dx <= k/2; dx++) {
      for (int dy = -k/2; dy <= k/2; dy++) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= img_width || nx < 0 || ny >= (b_elem + 1) * batch_elem_height || ny < b_elem * batch_elem_height) {
          continue;
        }
        // ignore the center pixel itself
        /*if (dx == 0 && dy == 0) {
          continue;
        }*/
        int n_index = ny * img_width + nx;

        // Compute neighbor's splat size in pixels
        float neighbor_depth = depth_in[n_index];

        // If neighbor is further than current center value, or is unobserved, ignore it
        if (neighbor_depth == default_depth || (neighbor_depth > min_depth && min_depth != default_depth)) {
          continue;
        }

        // Otherwise neighbor is closer to camera than center. Consider it.
        float n_splat_size_px;
        if (orthographic) {
          n_splat_size_px = focal_length_px * splat_radius;
        } else {
          n_splat_size_px = focal_length_px * splat_radius / neighbor_depth;
        }
         
        float n_dst = sqrt((float)(dx * dx + dy * dy)); 
        // If the splat is big enough to cover the center pixel, remember it
        if (n_splat_size_px > n_dst && neighbor_depth) {
          splat_index = n_index;
          min_depth = neighbor_depth;
        }
      }
    }

    // TODO: we can consider applying some blending instead of just a harsh copy
    depth_out[c_index] = depth_in[splat_index];
    for (int i = 0; i < num_channels; i++) {
      image_out[c_index * num_channels + i] = image_in[splat_index * num_channels + i];
    }
  }
}


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
){
    int64_t img_height = image_out.size(0);
    int64_t img_width = image_out.size(1);
    int64_t num_channels = image_out.size(2);

    int num_threads_per_block_x = 16;
    screen_space_splatting_cuda_kernel<<<dim3((img_width + num_threads_per_block_x - 1) / num_threads_per_block_x, 
                                              (img_height + num_threads_per_block_x - 1) / num_threads_per_block_x),
                                         dim3(num_threads_per_block_x, num_threads_per_block_x)>>>(
        batch_size,
        img_height,
        img_width,
        num_channels,
        kernel_size,
        splat_radius,
        focal_length_px,
        depth_in.data_ptr<float>(),
        image_in.data_ptr<float>(),
        depth_out.data_ptr<float>(),
        image_out.data_ptr<float>(),
        default_depth,
        orthographic
    );
}


__global__ void aa_subsampling_kernel(
    int64_t batch_size,
    int64_t img_height,
    int64_t img_width,
    int64_t num_channels,
    int aa_factor,
    float* depth_in,
    float* image_in,
    float* depth_out,
    float* image_out,
    float default_depth
) {
  uint x = blockDim.x * blockIdx.x + threadIdx.x;
  uint y = blockDim.y * blockIdx.y + threadIdx.y;
  uint b = blockDim.z * blockIdx.z + threadIdx.z;
  uint out_index = b * img_height * img_width + y * img_width + x;

  uint aa_img_width = img_width * aa_factor;
  uint aa_img_height = img_height * aa_factor;

  if (y < img_height && x < img_width && b < batch_size) {
    // Loop over pixel's corresponding patch in the input image
    for (int dx = 0; dx < aa_factor; dx++) {
      for (int dy = 0; dy < aa_factor; dy++) {
        int ox = x * aa_factor + dx;
        int oy = y * aa_factor + dy;
        if (ox >= aa_img_width|| ox < 0 || oy >= aa_img_height|| oy < 0) {
          continue;
        }
        int in_index = b * aa_img_height * aa_img_width + oy * aa_img_width + ox;

        // Average color over all pixels
        for (int i = 0; i < num_channels; i++) {
          image_out[out_index * num_channels + i] += image_in[in_index * num_channels + i];
        }
        // Take min depth across all pixels (median would be better, but that needs more memory to order the values)
        float d_in = depth_in[in_index];
        float d_out = depth_out[out_index];
        depth_out[out_index] = d_in;

        // TODO: The epsilon is hard-coded here, for some applications this might be a problem
        if (fabsf(d_in - default_depth) > 1e-6) {      //d_in != default_depth
          if (fabsf(d_out - default_depth) < 1e-6) {   //d_out == default_depth
            depth_out[out_index] = d_in;
          }
          else {
            depth_out[out_index] = min(d_out, d_in);
          }
        }
      }
    }

    for (int i = 0; i < num_channels; i++) {
      image_out[out_index * num_channels + i] = image_out[out_index * num_channels + i] / (aa_factor * aa_factor);
    }
  }
}


void aa_subsample(
  at::Tensor depth_in,
  at::Tensor image_in,
  at::Tensor depth_out,
  at::Tensor image_out,
  int aa_factor,
  float default_depth
) {
  int64_t batch_size = image_out.size(0);
  int64_t img_height = image_out.size(1);
  int64_t img_width = image_out.size(2);
  int64_t num_channels = image_out.size(3);

  int num_threads_per_block_x = 32;

  aa_subsampling_kernel<<<dim3(
                               (img_width + num_threads_per_block_x - 1) / num_threads_per_block_x, 
                               (img_height + num_threads_per_block_x - 1) / num_threads_per_block_x,
                               batch_size),
                          dim3(num_threads_per_block_x, num_threads_per_block_x, 1)>>>(
    batch_size,
    img_height,
    img_width,
    num_channels,
    aa_factor,
    depth_in.data_ptr<float>(),
    image_in.data_ptr<float>(),
    depth_out.data_ptr<float>(),
    image_out.data_ptr<float>(),
    default_depth
  );
}


} // namespace rvt
