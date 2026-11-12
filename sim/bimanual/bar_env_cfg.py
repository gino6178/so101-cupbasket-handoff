"""Bimanual SO-101 bar lift (T2): both arms grasp one end each, lift the bar level, carry it forward, set it down.

Reuses the handover scene layout: arms 30 cm apart facing +x. The 30 cm bar lies across them, each end in front of
one arm, so neither arm can lift it level alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import WORKSHOP_CONTACT_MATERIAL

from . import env_cfg as base
from . import replay
from .env_cfg import ARMS, TABLE_TOP, gripper_openness, pad_center

BAR_SIZE = (0.025, 0.30, 0.025)
BAR_REST_Z = TABLE_TOP + BAR_SIZE[2] / 2
BAR_START = (0.20, 0.0)
END_OFFSET = 0.13
"""Grasp points sit 2 cm in from each end, along the bar's long (local y) axis."""
TARGET = (0.26, 0.0)
LIFT_HEIGHT = 0.05
MAX_TILT = 0.26
"""About 15 degrees between the bar's long axis and the table."""
HOLD_RADIUS = 0.025


def bar_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["bar"].data.root_pos_w.torch - env.scene.env_origins


def bar_axis(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["bar"].data.root_quat_w.torch
    return quat_apply(quat, quat.new_tensor((0.0, 1.0, 0.0)).expand(quat.shape[0], -1))


def bar_ends(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Left (+y) and right (-y) grasp points in the env frame."""
    center, axis = bar_pos(env), bar_axis(env)
    a, b = center + END_OFFSET * axis, center - END_OFFSET * axis
    left_is_a = (a[:, 1] >= b[:, 1]).unsqueeze(-1)
    return torch.where(left_is_a, a, b), torch.where(left_is_a, b, a)


def bar_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Angle [rad] between the bar's long axis and the horizontal plane."""
    return torch.asin(bar_axis(env)[:, 2].abs().clamp(max=1.0))


def end_held(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    left_end, right_end = bar_ends(env)
    end = left_end if arm == "robot_left" else right_end
    near = torch.linalg.vector_norm(pad_center(env, arm) - end, dim=-1) < HOLD_RADIUS
    return near & (end[:, 2] > BAR_REST_Z + 0.01)


def target_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-environment target of the bar centre (env frame); the hard variant samples it at every reset."""
    target = getattr(env, "_bar_target", None)
    return target if target is not None else env.scene.env_origins.new_tensor(TARGET).expand(env.num_envs, -1)


def lift_height(env: ManagerBasedRLEnv) -> float:
    return getattr(env, "_bar_lift", LIFT_HEIGHT)


def target_distance_xy(env: ManagerBasedRLEnv, center: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(center[:, :2] - target_xy(env), dim=-1)


FLAGS = ("left_grasp", "right_grasp", "lifted", "at_target", "placed")


class BarProgress:
    """Latched per-episode milestones; ``success`` needs the bar resting level on the target for ``stable_steps``."""

    def __init__(self, num_envs: int, device, stable_steps: int = 10):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.left_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.right_count = torch.zeros_like(self.left_count)
        self.stable_count = torch.zeros_like(self.left_count)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stable_steps = stable_steps

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.left_count[ids] = 0
        self.right_count[ids] = 0
        self.stable_count[ids] = 0
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv) -> None:
        center, tilt = bar_pos(env), bar_tilt(env)
        left, right = end_held(env, "robot_left"), end_held(env, "robot_right")
        self.left_count = torch.where(left, self.left_count + 1, 0)
        self.right_count = torch.where(right, self.right_count + 1, 0)
        both = (self.left_count >= 2) & (self.right_count >= 2)
        lifted = both & (center[:, 2] > BAR_REST_Z + lift_height(env)) & (tilt < MAX_TILT)
        at_target = lifted & (target_distance_xy(env, center) < 0.03)
        speed = torch.linalg.vector_norm(env.scene["bar"].data.root_lin_vel_w.torch, dim=-1)
        resting = (target_distance_xy(env, center) < 0.03) & (center[:, 2] < BAR_REST_Z + 0.006) & (speed < 0.05)
        placed = resting & (tilt < 0.1) & self.flags[:, FLAGS.index("at_target")]
        self.flags |= torch.stack((self.left_count >= 2, self.right_count >= 2, lifted, at_target, placed), dim=-1)
        self.stable_count = torch.where(placed, self.stable_count + 1, 0)
        self.success |= self.stable_count >= self.stable_steps


def progress(env: ManagerBasedRLEnv) -> BarProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = BarProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- observations


def bar_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    bar = env.scene["bar"]
    return torch.cat((bar_pos(env), bar.data.root_quat_w.torch, bar.data.root_lin_vel_w.torch), dim=-1)


def bar_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    left_end, right_end = bar_ends(env)
    center = bar_pos(env)
    target = torch.cat((target_xy(env), torch.full_like(center[:, :1], BAR_REST_Z)), dim=-1)
    return torch.cat(
        (
            left_end - pad_center(env, "robot_left"),
            right_end - pad_center(env, "robot_right"),
            center - target,
            bar_tilt(env).unsqueeze(-1),
        ),
        dim=-1,
    )


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


class reach_end(ManagerTermBase):
    """Pay the decrease in distance from ``arm``'s pads to its end of the bar, until that end is grasped."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, arm: str, scale: float = 0.01) -> torch.Tensor:
        left_end, right_end = bar_ends(env)
        end = left_end if arm == "robot_left" else right_end
        distance = torch.linalg.vector_norm(pad_center(env, arm) - end, dim=-1)
        gain = ((self.previous - distance) / scale).clamp(-1.0, 1.0).nan_to_num(0.0)
        self.previous = distance
        p = progress(env)
        grasped = p.left_grasp if arm == "robot_left" else p.right_grasp
        return torch.where(grasped, 0.0, gain)


def grasp_end(env: ManagerBasedRLEnv, arm: str, radius: float = 0.03) -> torch.Tensor:
    """Near its end, pay ``arm`` for closing the jaws and for raising that end (up to the lift height)."""
    left_end, right_end = bar_ends(env)
    end = left_end if arm == "robot_left" else right_end
    near = torch.linalg.vector_norm(pad_center(env, arm) - end, dim=-1) < radius
    raised = ((end[:, 2] - BAR_REST_Z) / lift_height(env)).clamp(0.0, 1.0)
    closing = 1.0 - gripper_openness(env, arm)
    return torch.where(near & ~progress(env).lifted, 0.2 * closing + 0.5 * raised, 0.0)


def lift_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    """With both ends held, pay for bar height, scaled down as the bar tilts."""
    both = end_held(env, "robot_left") & end_held(env, "robot_right")
    height = ((bar_pos(env)[:, 2] - BAR_REST_Z) / lift_height(env)).clamp(0.0, 1.0)
    level = torch.exp(-bar_tilt(env) / 0.1)
    return torch.where(both & ~progress(env).at_target, height * level, 0.0)


def carry_to_target(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    """Once lifted, bring the bar over the target and then down onto it, kept level."""
    center = bar_pos(env)
    p = progress(env)
    height = torch.where(p.at_target, BAR_REST_Z, BAR_REST_Z + lift_height(env))
    goal = torch.cat((target_xy(env), height.unsqueeze(-1)), dim=-1)
    distance = torch.linalg.vector_norm(center - goal, dim=-1)
    return torch.where(p.lifted, torch.exp(-distance / std) * torch.exp(-bar_tilt(env) / 0.1), 0.0)


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 1.0, 4.0, 4.0, 6.0)

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, len(FLAGS)), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # Flags already set on the first step came from a replay reset and were earned in an earlier episode.
        flags = progress(env).flags
        reward = ((flags & ~self.paid).float() * flags.new_tensor(self.WEIGHTS, dtype=torch.float32)).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = flags.clone()
        self.started[:] = True
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).success.float()


# ---------------------------------------------------------------- terminations


class bar_success(ManagerTermBase):
    """Advance progress once per step (terminations run first) and keep each episode's final flags for evaluation."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        progress(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        p = progress(self._env)
        self._env._bi_final[ids] = torch.cat((p.flags[ids], p.success[ids].unsqueeze(-1)), dim=-1)
        p.reset(ids)
        pending = getattr(self._env, "_bi_pending_flags", None)
        if pending is not None:
            p.flags[pending[0]] = pending[1]
            self._env._bi_pending_flags = None

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        p = progress(env)
        record = FLAGS.index("lifted")
        before = p.flags[:, record].clone()
        p.update(env)
        newly = (p.flags[:, record] & ~before).nonzero().squeeze(-1)
        replay.buffer(env, ("bar",), len(FLAGS)).record(env, newly, p.flags)
        return p.success.clone()


def bar_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """After the level lift: the bar fell back away from the target, or tipped steeply."""
    center = bar_pos(env)
    p = progress(env)
    fell_back = p.lifted & (center[:, 2] < BAR_REST_Z + 0.01) & (target_distance_xy(env, center) > 0.03)
    tipped = p.lifted & (bar_tilt(env) > 2 * MAX_TILT)
    return fell_back | tipped | (center[:, 2] < TABLE_TOP - 0.05)


# ---------------------------------------------------------------- config


@configclass
class BarSceneCfg(base.BiSceneCfg):
    # At the default roll the jaws close partly along y (along the bar); an eighth turn puts them across it.
    robot_left = base._arm("robot_left", base.BASE_Y, wrist_roll_offset=math.pi / 4)
    robot_right = base._arm("robot_right", -base.BASE_Y, wrist_roll_offset=math.pi / 4)
    cube = None
    right_fixed_jaw_cube = None
    right_moving_jaw_cube = None
    target_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetPad",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.32, 0.001), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*TARGET, TABLE_TOP + 0.0006)),
    )
    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.CuboidCfg(
            size=BAR_SIZE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.06),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.65, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BAR_START, BAR_REST_Z)),
    )


@configclass
class BarPolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=bar_state)
    relative = ObsTerm(func=bar_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class BarObservationsCfg:
    policy: BarPolicyCfg = BarPolicyCfg()
    critic: BarPolicyCfg = BarPolicyCfg()


@configclass
class BarEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_bar = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.4, 0.4)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bar"),
        },
    )
    replay = EventTerm(func=replay.replay_reset, mode="reset", params={"objects": ("bar",), "num_flags": len(FLAGS)})


@configclass
class BarRewardsCfg:
    reach_left = RewTerm(func=reach_end, weight=1.0, params={"arm": "robot_left"})
    reach_right = RewTerm(func=reach_end, weight=1.0, params={"arm": "robot_right"})
    grasp_left = RewTerm(func=grasp_end, weight=0.5, params={"arm": "robot_left"})
    grasp_right = RewTerm(func=grasp_end, weight=0.5, params={"arm": "robot_right"})
    lift = RewTerm(func=lift_level, weight=0.5)
    carry = RewTerm(func=carry_to_target, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["bar_dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_right")}
    )
    left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_right")}
    )


@configclass
class BarTerminationsCfg:
    success = DoneTerm(func=bar_success)
    bar_dropped = DoneTerm(func=bar_dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiBarLiftEnvCfg(base.BiHandoverEnvCfg):
    scene: BarSceneCfg = BarSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    observations: BarObservationsCfg = BarObservationsCfg()
    events: BarEventsCfg = BarEventsCfg()
    rewards: BarRewardsCfg = BarRewardsCfg()
    terminations: BarTerminationsCfg = BarTerminationsCfg()
