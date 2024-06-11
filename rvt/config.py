# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from yacs.config import CfgNode as CN

_C = CN()

_C.agent = "our"
_C.tasks = "insert_onto_square_peg,open_drawer,place_wine_at_rack_location,light_bulb_in"
_C.exp_id = "def"
_C.resume = ""
# bs per device, effective bs is scaled by num device
_C.bs = 4
_C.epochs = 20
# number of dataloader workers, >= 0
_C.num_workers = 0
# 'transition_uniform' or 'task_uniform'
_C.sample_distribution_mode = 'transition_uniform'
_C.train_iter = 16 * 10000

# arguments present in both peract and rvt
# some of them donot support every possible combination in peract
_C.peract = CN()
_C.peract.lambda_weight_l2 = 1e-6
# lr should be thought on per sample basis
# effective lr is multiplied by bs * num_devices
_C.peract.lr = 2.5e-5
_C.peract.optimizer_type = "lamb"
_C.peract.warmup_steps = 0
_C.peract.lr_cos_dec = False
_C.peract.add_rgc_loss = True
_C.peract.num_rotation_classes = 72
_C.peract.amp = False
_C.peract.bnb = False
_C.peract.transform_augmentation = True
_C.peract.transform_augmentation_xyz = [0.1, 0.1, 0.1]
_C.peract.transform_augmentation_rpy = [0.0, 0.0, 20.0]

# arguments present in only rvt and not peract
_C.rvt = CN()
_C.rvt.gt_hm_sigma = 1.5
_C.rvt.img_aug = 0.1
_C.rvt.place_with_mean = True
_C.rvt.move_pc_in_bound = True

# arguments present in peract official
_C.peract_official = CN()
_C.peract_official.cfg_path = "configs/peract_official_config.yaml"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
