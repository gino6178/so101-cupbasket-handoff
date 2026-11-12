"""T3 stage B: the right arm alone learns to set the cube on a bar the left arm already holds steady.

Every episode starts from a stage-A state (``stack_left_env_cfg``: left jaws on the bar, bar raised and level),
saved by ``collect.install_state_dump``. The left arm keeps its stored joint targets, so it goes on gripping the
bar; the policy acts on the right arm only. Success is the full T3 condition: the released cube resting on the
held bar for one second.
"""

from __future__ import annotations

import os

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from . import env_cfg as base
from . import replay
from . import stack_env_cfg as stack

STATES = os.environ.get("BITIDY_STACK_STATES", os.path.expanduser("~/stack_states.pt"))


def load_stack_states(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, path: str = STATES) -> None:
    replay.load_start_states(env, env_ids, path, ("bar", "cube"), len(stack.FLAGS))


@configclass
class StackRecvActionsCfg:
    right_arm, right_gripper = base._arm_actions("robot_right")


@configclass
class StackRecvEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    load = EventTerm(func=load_stack_states, mode="reset")


@configclass
class BiStackRecvEnvCfg(stack.BiStackEnvCfg):
    actions: StackRecvActionsCfg = StackRecvActionsCfg()
    events: StackRecvEventsCfg = StackRecvEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0

    def play_mode(self):
        """Evaluation also starts from the stored stage-A states."""
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
