# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/utils.py
# utils functions for rotation augmentation
import torch
import numpy as np
from scipy.spatial.transform import Rotation


def rand_dist(size, min=-1.0, max=1.0):
    return (max - min) * torch.rand(size) + min


def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max + 1, size)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def sensitive_gimble_fix(euler):
    """
    :param euler: euler angles in degree as np.ndarray in shape either [3] or
    [b, 3]
    """
    # selecting sensitive angle
    select1 = (89 < euler[..., 1]) & (euler[..., 1] < 91)
    euler[select1, 1] = 90
    # selecting sensitive angle
    select2 = (-91 < euler[..., 1]) & (euler[..., 1] < -89)
    euler[select2, 1] = -90

    # recalulating the euler angles, see assert
    r = Rotation.from_euler("xyz", euler, degrees=True)
    euler = r.as_euler("xyz", degrees=True)

    select = select1 | select2
    assert (euler[select][..., 2] == 0).all(), euler

    return euler


def quaternion_to_discrete_euler(quaternion, resolution, gimble_fix=True):
    """
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    """
    r = Rotation.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def quaternion_to_euler(quaternion, gimble_fix=True):
    """
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    """
    r = Rotation.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    return euler


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def point_to_voxel_index(
    point: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray
):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )
    return voxel_indicy
