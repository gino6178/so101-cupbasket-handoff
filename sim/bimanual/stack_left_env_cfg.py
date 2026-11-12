"""T3 stage A (one arm at a time, as in Mixline's wait training): only the left arm acts and learns to grip the bar
and hold it steady; the right arm stays still.

Success is the bar steady (gripped on its left half, raised, level) for ``HOLD_STEPS`` steps. Those states are
recorded in the replay buffer; ``collect`` saves them as the start states of stage B, where only the right arm
learns to set the cube on the held bar.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from . import env_cfg as base
from . import replay
from . import stack_env_cfg as stack

HOLD_STEPS = 20


class hold_success(ManagerTermBase):
    """Advance T3 progress, end the episode once the bar has been held steady for ``HOLD_STEPS`` steps."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        stack.progress(env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._stack_left_success = self.done

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        p = stack.progress(self._env)
        self._env._bi_final[ids] = torch.cat((p.flags[ids], self.done[ids].unsqueeze(-1)), dim=-1)
        p.reset(ids)
        self.count[ids] = 0
        self.done[ids] = False
        pending = getattr(self._env, "_bi_pending_flags", None)
        if pending is not None:
            p.flags[pending[0]] = pending[1]
            self._env._bi_pending_flags = None

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        p = stack.progress(env)
        p.update(env)
        self.count = torch.where(stack.bar_steady(env), self.count + 1, 0)
        newly = (self.count >= HOLD_STEPS) & ~self.done
        replay.buffer(env, ("bar", "cube"), len(stack.FLAGS)).record(env, newly.nonzero().squeeze(-1), p.flags)
        self.done |= newly
        return self.done.clone()


def held(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._stack_left_success.float()


def bar_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    p = stack.progress(env)
    bar_down = p.bar_up & (stack.bar_pos(env)[:, 2] < stack.BAR_REST_Z + 0.001)
    return bar_down | (stack.bar_pos(env)[:, 2] < base.TABLE_TOP - 0.05)


@configclass
class LeftActionsCfg:
    left_arm, left_gripper = base._arm_actions("robot_left")


@configclass
class LeftRewardsCfg:
    reach_left = RewTerm(func=stack.reach, weight=1.0, params={"arm": "robot_left"})
    grasp_left = RewTerm(func=stack.grasp_shaping, weight=0.5, params={"arm": "robot_left"})
    hold = RewTerm(func=stack.hold_bar, weight=1.0)
    milestones = RewTerm(func=stack.milestones, weight=10.0)
    success = RewTerm(func=held, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
    left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})


@configclass
class LeftTerminationsCfg:
    success = DoneTerm(func=hold_success)
    dropped = DoneTerm(func=bar_dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiStackLeftEnvCfg(stack.BiStackEnvCfg):
    actions: LeftActionsCfg = LeftActionsCfg()
    rewards: LeftRewardsCfg = LeftRewardsCfg()
    terminations: LeftTerminationsCfg = LeftTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
