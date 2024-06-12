# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import math
import point_renderer.ops as ops
from point_renderer.cameras import OrthographicCameras, PerspectiveCameras
import point_renderer._C.render as r

from point_renderer.profiler import PerfTimer


@torch.jit.script
def _prep_render_batch_inputs(points, features, inv_poses, intrinsics, img_h : int, img_w: int, orthographic : bool):
    batch_size = len(inv_poses)
    num_points = points.shape[0]

    # Project points from 3D world coordinates to pixels and camera coordinates
    pc_px, pc_cam = ops.project_points_3d_to_pixels(points, inv_poses, intrinsics, orthographic)

    # Convert pixel coordinates to flattened pixel coordinates, marking out-of-image points with index -1
    point_pixel_index = ops.get_batch_pixel_index(pc_px, img_h, img_w)
    imask = ops.batch_frustrum_mask(pc_px, img_h, img_w)
    point_pixel_index[~imask] = -1
    
    point_depths = pc_cam[:, :, 2]
    return point_pixel_index.reshape([batch_size * num_points]).contiguous(), point_depths.reshape([batch_size * num_points]).contiguous(), features



class PointRenderer:

    def __init__(self, device="cuda", perf_timer=False):
        self.device = device
        assert "cuda" in self.device, "Currently only a CUDA implementation is available"
        self.timer = PerfTimer(activate=perf_timer, show_memory=False, print_mode=perf_timer)

    @torch.no_grad()
    def splat_filter(self, batch_size, splat_radius, cameras, point_depths, depth_buf, image_buf, default_depth, splat_max_k):
        # This assumes same focal length for all cameras.
        if cameras.is_perspective():
            # We are assuming x and y focal lengths to be about the same.
            focal_length = cameras.intrinsics[0, 0, 0].item()
            closest_point = point_depths.min()
            biggest_splat = math.ceil((splat_radius * focal_length / closest_point))
        elif cameras.is_orthographic():
            # In orthographic cameras all points are the same size
            focal_length = cameras.intrinsics[0, 0, 0].item()
            biggest_splat = math.ceil(focal_length * splat_radius)
        else:
            raise ValueError(f"Unknown camera type: {type(cameras)}")

        if splat_max_k is None:
            splat_max_k = 7
        kernel_size = min(biggest_splat * 2 + 1, splat_max_k)

        #print(f"Splatting filter with k={kernel_size}, b={batch_size}")
        r.screen_space_splatting(
            batch_size,
            depth_buf.clone(),
            image_buf.clone(),
            depth_buf,
            image_buf,
            default_depth,
            focal_length, 
            splat_radius,
            kernel_size,
            cameras.is_orthographic()
        )


    @torch.no_grad()
    def render_batch(self, points, features, cameras, img_size, default_depth=0.0, default_color=0.0, splat_radius=None, splat_max_k=51, aa_factor=1):
        # Figure out dimensions of the problem
        img_h, img_w = img_size
        aa_factor = int(aa_factor)
        assert aa_factor >= 1, "Antialiasing factor must be greater than 1"
        img_h = img_h * aa_factor
        img_w = img_w * aa_factor
        batch_size = len(cameras)
        num_channels = features.shape[1]

        self.timer.reset()

        # Make sure inputs are on the rendering device
        cameras = cameras.to(self.device)
        features = features.to(self.device)
        points = points.to(self.device)

        # Scale the camera if we want to internally render a bigger image for fake antialiasing
        if aa_factor > 1:
            cameras.scale(aa_factor)

        # Project points to image space depending on the camera type
        # (these would be more elegant as class methods, but TorchScript doesn't support it)
        point_pixel_index, point_depths, features = _prep_render_batch_inputs(
            points, features, cameras.inv_poses, cameras.intrinsics, img_h, img_w, type(cameras) == OrthographicCameras)

        # Allocate depth and image render buffers
        # Batch size is rolled in with the height dimension
        #  a.k.a. we render a single image that contains all images vertically stacked
        depth_buf = torch.zeros((batch_size * img_h, img_w), device=self.device, dtype=torch.float32)
        assert features.dtype == torch.float32, "For now only torch.uint8 and torch.float32 colors are supported"
        image_buf = torch.zeros((batch_size * img_h, img_w, num_channels), device=self.device, dtype=torch.float32)
        
        self.timer.check("render_setup")

        # Render points to pixel buffers
        r.render_feature_pointcloud_to_image(
            point_pixel_index,
            point_depths,
            features,
            image_buf,
            depth_buf,
            default_depth,
            default_color
        )

        self.timer.check("render")
        
        # Apply screen-space splatting filter
        if splat_radius is not None:
            self.splat_filter(batch_size, splat_radius, cameras, point_depths, depth_buf, image_buf, default_depth, splat_max_k)
            self.timer.check("splatting")

        # Separate the batch dimension, so that we have a batch of images
        image_buf = image_buf.reshape((batch_size, img_h, img_w, num_channels))
        depth_buf = depth_buf.reshape((batch_size, img_h, img_w))

        if aa_factor != 1:
            # Subsample the larger render buffers to produce the output image
            img_h_out = img_h // aa_factor
            img_w_out = img_w // aa_factor
            depth_out = torch.zeros((batch_size, img_h_out, img_w_out), device=self.device, dtype=torch.float32).fill_(default_depth)
            image_out = torch.zeros((batch_size, img_h_out, img_w_out, num_channels), device=self.device, dtype=torch.float32)
            print(depth_buf.min(), depth_buf.max())

            r.aa_subsample(
                depth_buf,
                image_buf,
                depth_out,
                image_out,
                aa_factor,
                default_depth
            )
            self.timer.check("antialiasing")
            return image_out.reshape((batch_size, img_h_out, img_w_out, num_channels)), depth_out.reshape((batch_size, img_h_out, img_w_out))
        else:
            # Output render buffers as-is if we're rendering at the same resolution as the output
            self.timer.check("antialiasing")
            return image_buf, depth_buf


