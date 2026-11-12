"""T1 receiver: the right arm learns to take the cube from a scripted left (giver) arm and place it on the target.

Following giver/receiver handover RL (only the receiver learns; the giver holds still and lets go once the receiver
has a grip), every episode starts from a stored state in which the left arm already holds the cube in the handover
zone. These states come from the left-arm policy of the full T1 task (``collect.install_state_dump``). The policy
acts on the right arm only; the left arm keeps its stored joint targets and opens its jaws once the right jaws have
held the cube for ``RELEASE_STEPS`` steps.
"""

from __future__ import annotations

import os

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from . import env_cfg as base
from . import replay

STATES = os.environ.get("BITIDY_HANDOVER_STATES", os.path.expanduser("~/handover_states.pt"))
RELEASE_STEPS = 5
GIVE_X, GIVE_Y, GIVE_Z = (0.16, 0.22), (-0.04, 0.02), (0.10, 0.13)
"""The giver carries the cube to a random point in this box, within easy reach of the right arm, and holds it there."""


def load_handover_states(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, path: str = STATES) -> None:
    """Reset-mode event: restore every resetting environment to a stored left-arm-holds-cube state."""
    replay.load_start_states(env, env_ids, path, ("cube",), len(base.FLAGS))


class giver_release(ManagerTermBase):
    """Scripted giver. Never terminates.

    Until the receiver has a grip, the left arm carries the cube (by IK on its fingertip midpoint) to a handover point
    sampled per episode and holds it there; once the right jaws have held the cube for ``RELEASE_STEPS`` steps the
    left jaws open and the arm stays where it is.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        robot = env.scene["robot_left"]
        self.gripper = robot.find_joints("gripper")[0][0]
        from .scripted import ArmIK

        self.ik = ArmIK(env, "robot_left")
        self.goal = torch.zeros((env.num_envs, 3), device=env.device)
        self.reset(None)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.count[ids] = 0
        self.released[ids] = False
        n = self.goal[ids].shape[0]
        lo = self.goal.new_tensor((GIVE_X[0], GIVE_Y[0], GIVE_Z[0]))
        hi = self.goal.new_tensor((GIVE_X[1], GIVE_Y[1], GIVE_Z[1]))
        self.goal[ids] = lo + (hi - lo) * torch.rand((n, 3), device=self.goal.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        robot = env.scene["robot_left"]
        holding = (~self.released).nonzero().squeeze(-1)
        if len(holding):
            target = robot.data.joint_pos_target.torch.clone()
            target[:, self.ik.joints] = self.ik.joint_step(self.goal)
            robot.set_joint_position_target_index(target=target[holding], env_ids=holding)
            if os.environ.get("BITIDY_DEBUG"):
                self._t = getattr(self, "_t", 0) + 1
                if self._t % 15 == 1:
                    pad = base.pad_center(env, "robot_left")[0]
                    print(f"[giver] t={self._t} pad={pad.tolist()} goal={self.goal[0].tolist()} cube={base.cube_pos(env)[0].tolist()} "
                          f"q={robot.data.joint_pos.torch[0].tolist()} tgt={robot.data.joint_pos_target.torch[0].tolist()}", flush=True)
        self.count = torch.where(base.right_holds(env), self.count + 1, 0)
        newly = (self.count >= RELEASE_STEPS) & ~self.released
        if bool(newly.any()):
            ids = newly.nonzero().squeeze(-1)
            robot = env.scene["robot_left"]
            target = robot.data.joint_pos_target.torch[ids].clone()
            target[:, self.gripper] = base.PREGRASP_GRIPPER_POSITION
            robot.set_joint_position_target_index(target=target, env_ids=ids)
            self.released |= newly
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@configclass
class RecvActionsCfg:
    right_arm, right_gripper = base._arm_actions("robot_right")


@configclass
class RecvEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    load = EventTerm(func=load_handover_states, mode="reset")


@configclass
class RecvTerminationsCfg(base.TerminationsCfg):
    giver = DoneTerm(func=giver_release)


@configclass
class BiHandoverRecvEnvCfg(base.BiHandoverEnvCfg):
    actions: RecvActionsCfg = RecvActionsCfg()
    events: RecvEventsCfg = RecvEventsCfg()
    terminations: RecvTerminationsCfg = RecvTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0

    def play_mode(self):
        """Evaluation also starts from the stored handover states (the receiver task has no other start)."""
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
