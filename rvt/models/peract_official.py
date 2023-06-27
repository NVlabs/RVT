# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from rvt.libs.peract.helpers.preprocess_agent import PreprocessAgent
from rvt.libs.peract.agents.peract_bc.launch_utils import create_agent


class PreprocessAgent2(PreprocessAgent):
    def eval(self):
        self._pose_agent._qattention_agents[0]._q.eval()

    def train(self):
        self._pose_agent._qattention_agents[0]._q.train()

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self._device = self._pose_agent._qattention_agents[0]._device


def create_agent_our(cfg):
    """
    Reuses the official peract agent, but replaces PreprocessAgent2 with PreprocessAgent
    """
    agent = create_agent(cfg)
    agent = agent._pose_agent
    agent = PreprocessAgent2(agent)
    return agent
