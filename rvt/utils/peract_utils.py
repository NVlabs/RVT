# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
from omegaconf import OmegaConf

from rvt.models.peract_official import create_agent_our
from peract_colab.arm.utils import stack_on_channel
from torch.optim.lr_scheduler import CosineAnnealingLR
from rvt.utils.lr_sched_utils import GradualWarmupScheduler

# Contants
# TODO: Unclear about the best way to handle them
CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}

DATA_FOLDER = "data"
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration
DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
# settings
NUM_LATENTS = 512  # PerceiverIO latents


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        rgb = stack_on_channel(replay_sample["%s_rgb" % n])
        pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


def get_official_peract(
    cfg_path,
    training,
    device,
    bs,
):
    """
    Creates an official peract agent
    :param cfg_path: path to the config file
    :param training: whether to build the agent in training mode
    :param device: device to build the agent on
    :param bs: batch size, does not matter when we need a model for inference.
    """
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.load(f)

    # we need to modify the batch size as in our case we specify batchsize per
    # gpu
    cfg.replay.batch_size = bs
    agent = create_agent_our(cfg)
    agent.build(training=training, device=device)

    return agent
