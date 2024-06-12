# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn.functional as F
import math
from transforms3d import euler, quaternions, affines
import numpy as np


def transform_points_batch(pc : torch.Tensor, inv_cam_poses : torch.Tensor):
    pc_h = torch.cat([pc, torch.ones_like(pc[:, 0:1])], dim=1)
    pc_cam_h = torch.einsum("bxy,ny->bnx", inv_cam_poses, pc_h)
    pc_cam = pc_cam_h[:, :, :3]
    return pc_cam

def transform_points(pc : torch.Tensor, inv_cam_pose : torch.Tensor):
    pc_h = torch.cat([pc, torch.ones_like(pc[:, 0:1])], dim=1)
    pc_cam_h = torch.einsum("xy,ny->nx", inv_cam_pose, pc_h)
    pc_cam = pc_cam_h[:, :3]
    return pc_cam

def orthographic_camera_projection_batch(pc_cam : torch.Tensor, K : torch.Tensor):
    # For orthographic camera projection, treat all points as if they are at depth 1
    uvZ = torch.einsum("bxy,bny->bnx", K, torch.cat([pc_cam[:, :, :2], torch.ones_like(pc_cam[:, :, 2:3])], dim=2))
    return uvZ[:, :, :2]

def orthographic_camera_projection(pc_cam : torch.Tensor, K : torch.Tensor):
    # For orthographic camera projection, treat all points as if they are at depth 1
    uvZ = torch.einsum("xy,ny->nx", K, torch.cat([pc_cam[:, :2], torch.ones_like(pc_cam[:, 2:3])], dim=1))
    return uvZ[:, :2]

def perspective_camera_projection_batch(pc_cam : torch.Tensor, K : torch.Tensor):
    uvZ = torch.einsum("bxy,bny->bnx", K, pc_cam)
    uv = torch.stack([uvZ[:, :, 0] / uvZ[:, :, 2], uvZ[:, :, 1] / uvZ[:, :, 2]], dim=2)
    return uv

def perspective_camera_projection(pc_cam : torch.Tensor, K : torch.Tensor):
    uvZ = torch.einsum("xy,ny->nx", K, pc_cam)
    uv = torch.stack([uvZ[:, 0] / uvZ[:, 2], uvZ[:, 1] / uvZ[:, 2]], dim=1)
    return uv

def project_points_3d_to_pixels(pc : torch.Tensor, inv_cam_poses : torch.Tensor, intrinsics : torch.Tensor, orthographic : bool):
    """
    This combines the projection from 3D coordinates to camera coordinates using extrinsics,
    followed by projection from camera coordinates to pixel coordinates using the intrinsics.
    """
    # Project points from world to camera frame
    pc_cam = transform_points_batch(pc, inv_cam_poses)
    # Project points from camera frame to pixel space
    if orthographic:
        pc_px = orthographic_camera_projection_batch(pc_cam, intrinsics)
    else:
        pc_px = perspective_camera_projection_batch(pc_cam, intrinsics)
    return pc_px, pc_cam


def get_batch_pixel_index(pc_px : torch.Tensor, img_height : int, img_width : int):
    """
    Convert a 2D pixel coordinate from a batch of pointclouds to an index
    that indexes into a corresponding flattened batch of 2D images.
    """
    # batch_idx
    batch_idx = torch.arange(pc_px.shape[0], device=pc_px.device, dtype=torch.long)[:, None]#.repeat((1, pc_px.shape[1])).
    pix_off = 0.0
    row_idx = (pc_px[:, :, 1] + pix_off).long()
    col_idx = (pc_px[:, :, 0] + pix_off).long()
    pixel_index = batch_idx * img_height * img_width + row_idx * img_width + col_idx
    return pixel_index

def get_pixel_index(pc_px : torch.Tensor, img_width : int):
    pix_off = 0.0
    row_idx = (pc_px[:, 1] + pix_off).long()
    col_idx = (pc_px[:, 0] + pix_off).long()
    pixel_index = row_idx * img_width + col_idx
    return pixel_index.long()


def batch_frustrum_mask(pc_px : torch.Tensor, img_height : int, img_width : int):
    imask_x = torch.logical_and(pc_px[:, :, 0] >= 0, pc_px[:, :, 0] < img_width)
    imask_y = torch.logical_and(pc_px[:, :, 1] >= 0, pc_px[:, :, 1] < img_height)
    imask = torch.logical_and(imask_x, imask_y)
    return imask

def frustrum_mask(pc_px : torch.Tensor, img_height : int, img_width : int):
    imask_x = torch.logical_and(pc_px[:, 0] >= 0, pc_px[:, 0] < img_width)
    imask_y = torch.logical_and(pc_px[:, 1] >= 0, pc_px[:, 1] < img_height)
    imask = torch.logical_and(imask_x, imask_y)
    return imask

def lookat_to_cam_pose(eye, at, up=[0, 0, 1], device="cpu"):
    # This runs on CPU, moving to GPU at the end (that's faster)
    eye = torch.tensor(eye, device="cpu", dtype=torch.float32)
    at = torch.tensor(at, device="cpu", dtype=torch.float32)
    
    camera_view = F.normalize(at - eye, dim=0)
    camera_right = F.normalize(torch.cross(camera_view, torch.tensor(up, dtype=torch.float32, device="cpu"), dim=0), dim=0)
    camera_up = F.normalize(torch.cross(camera_right, camera_view, dim=0), dim=0)
    
    # rotation matrix from opencv conventions
    R = torch.stack([camera_right, -camera_up, camera_view], dim=0).T
    
    T = torch.from_numpy(affines.compose(eye, R, [1, 1, 1]))
    return T.float().to(device)

def fov_and_size_to_intrinsics(fov, img_size, device="cpu"):
    img_h, img_w = img_size
    fx = img_w / (2 * math.tan(math.radians(fov) / 2))
    fy = img_h / (2 * math.tan(math.radians(fov) / 2))
    
    intrinsics = torch.tensor([
        [fx, 0, img_h / 2],
        [0, fy, img_w / 2],
        [0, 0, 1]
    ], dtype=torch.float, device=device)
    return intrinsics

def orthographic_intrinsics_from_scales(img_sizes_w, img_size_px, device="cpu"):
    img_h, img_w = img_size_px
    fx = img_h / (img_sizes_w[:, 0])
    fy = img_w / (img_sizes_w[:, 1])

    intrinsics = torch.zeros([len(img_sizes_w), 3, 3], dtype=torch.float, device=device)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = img_h / 2
    intrinsics[:, 1, 2] = img_w / 2
    #intrinsics[:, 2, 2] = 1.0
    return intrinsics


def unravel_index(pixel_index : torch.Tensor, img_width : int):
    row_idx = pixel_index // img_width
    col_idx = pixel_index % img_width
    return torch.stack([col_idx, row_idx], dim=1)

