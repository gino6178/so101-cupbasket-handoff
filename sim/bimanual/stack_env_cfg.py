"""Bimanual SO-101 bar + cube cooperation (T3): the left arm lifts and steadies the bar, the right arm sets the cube
on top of it.

The left arm grips the bar 6 cm left of its centre and holds it level above the table; the right arm picks the cube
from its own side and releases it on the bar 6 cm right of the centre. Success needs the released cube to rest on
the still-held bar, so neither arm can finish alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import WORKSHOP_CONTACT_MATERIAL

from . import bar_env_cfg as bar
from . import env_cfg as base
from . import replay
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, cube_pos, gripper_openness, held_by, pad_center
from .bar_env_cfg import BAR_REST_Z, BAR_SIZE, bar_axis, bar_pos, bar_tilt

BAR_START = (0.20, 0.03)
CUBE_START = (0.26, -0.17)
GRIP_OFFSET = 0.06
"""The left arm grips this far along the bar towards +y; the cube goes this far towards -y."""
GRIP_RANGE = (0.03, BAR_SIZE[1] / 2 - 0.01)
HOLD_HEIGHT = 0.005
"""The gripped end is raised just enough to pin the bar; its far end may stay on the table."""
MAX_TILT = 0.26
HOLD_RADIUS = 0.025
SETTLE_STEPS = 30
"""One second of the released cube resting on the held bar."""


def grip_point(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The point on the bar's left half nearest the left pads: any grip from 3 cm left of centre to the end counts."""
    axis = bar_axis(env)
    sign = torch.where(axis[:, 1:2] >= 0, 1.0, -1.0)
    left_dir = sign * axis
    along = ((pad_center(env, "robot_left") - bar_pos(env)) * left_dir).sum(-1, keepdim=True)
    return bar_pos(env) + along.clamp(GRIP_RANGE[0], GRIP_RANGE[1]) * left_dir


def place_point(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Where the cube centre rests when placed on the bar: 6 cm right of centre, on the top face."""
    axis = bar_axis(env)
    sign = torch.where(axis[:, 1:2] >= 0, 1.0, -1.0)
    point = bar_pos(env) - sign * GRIP_OFFSET * axis
    return point + point.new_tensor((0.0, 0.0, BAR_SIZE[2] / 2 + CUBE_SIZE / 2))


def bar_held_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    near = torch.linalg.vector_norm(pad_center(env, "robot_left") - grip_point(env), dim=-1) < HOLD_RADIUS
    return near & (bar_pos(env)[:, 2] > BAR_REST_Z + 0.004)


def bar_steady(env: ManagerBasedRLEnv) -> torch.Tensor:
    return bar_held_left(env) & (bar_pos(env)[:, 2] > BAR_REST_Z + HOLD_HEIGHT) & (bar_tilt(env) < MAX_TILT)


def cube_to_place(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(cube_pos(env) - place_point(env), dim=-1)


FLAGS = ("left_grasp", "bar_up", "right_grasp", "cube_over_bar", "placed")


class StackProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.left_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.right_count = torch.zeros_like(self.left_count)
        self.stable_count = torch.zeros_like(self.left_count)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

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
        left, right = bar_held_left(env), held_by(env, "robot_right")
        self.left_count = torch.where(left, self.left_count + 1, 0)
        self.right_count = torch.where(right, self.right_count + 1, 0)
        steady = bar_steady(env)
        distance = cube_to_place(env)
        over = right & steady & (distance < 0.04)
        released = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube_pos(env), dim=-1) > 0.03
        speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
        resting = steady & released & (distance < 0.02) & (speed < 0.05)
        self.flags |= torch.stack((self.left_count >= 2, steady, self.right_count >= 2, over, resting), dim=-1)
        self.stable_count = torch.where(resting & self.cube_over_bar, self.stable_count + 1, 0)
        self.success |= self.stable_count >= SETTLE_STEPS


def progress(env: ManagerBasedRLEnv) -> StackProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = StackProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- observations


def objects_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.cat((bar.bar_state(env), base.cube_state(env)), dim=-1)


def objects_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.cat(
        (
            grip_point(env) - pad_center(env, "robot_left"),
            cube_pos(env) - pad_center(env, "robot_right"),
            cube_pos(env) - place_point(env),
            bar_tilt(env).unsqueeze(-1),
        ),
        dim=-1,
    )


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


class reach(ManagerTermBase):
    """Pay the decrease in distance from ``arm``'s pads to its grasp target, until that arm has grasped."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, arm: str, scale: float = 0.01) -> torch.Tensor:
        target = grip_point(env) if arm == "robot_left" else cube_pos(env)
        distance = torch.linalg.vector_norm(pad_center(env, arm) - target, dim=-1)
        gain = ((self.previous - distance) / scale).clamp(-1.0, 1.0).nan_to_num(0.0)
        self.previous = distance
        p = progress(env)
        done = p.left_grasp if arm == "robot_left" else p.right_grasp
        return torch.where(done, 0.0, gain)


def grasp_shaping(env: ManagerBasedRLEnv, arm: str, radius: float = 0.03) -> torch.Tensor:
    """Near its target, pay ``arm`` for closing the jaws and for raising the object it should hold."""
    if arm == "robot_left":
        target, height, rest = grip_point(env), bar_pos(env)[:, 2], BAR_REST_Z
        active = ~progress(env).bar_up
    else:
        target, height, rest = cube_pos(env), cube_pos(env)[:, 2], CUBE_REST_Z
        active = ~progress(env).right_grasp
    near = torch.linalg.vector_norm(pad_center(env, arm) - target, dim=-1) < radius
    raised = ((height - rest) / 0.01).clamp(0.0, 1.0)
    closing = 1.0 - gripper_openness(env, arm)
    return torch.where(near & active, 0.2 * closing + 0.5 * raised, 0.0)


def hold_bar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """While the left arm holds the bar, pay for height (up to the hold height) and for keeping it level."""
    height = ((bar_pos(env)[:, 2] - BAR_REST_Z) / HOLD_HEIGHT).clamp(0.0, 1.0)
    return torch.where(bar_held_left(env), height * torch.exp(-bar_tilt(env) / 0.1), 0.0)


def carry_cube(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    """With the cube in hand, bring it 3 cm above its spot on the bar."""
    above = place_point(env) + cube_pos(env).new_tensor((0.0, 0.0, 0.03))
    distance = torch.linalg.vector_norm(cube_pos(env) - above, dim=-1)
    active = held_by(env, "robot_right") & ~progress(env).cube_over_bar
    return torch.where(active, torch.exp(-distance / std), 0.0)


def set_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Once over the bar: pay for lowering the cube onto its spot and for opening the right jaws there."""
    p = progress(env)
    on_spot = torch.exp(-cube_to_place(env) / 0.01)
    opening = gripper_openness(env, "robot_right")
    return torch.where(p.cube_over_bar & bar_steady(env), on_spot * (0.5 + 0.5 * opening), 0.0)


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 3.0, 2.0, 4.0, 6.0)

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


class stack_success(ManagerTermBase):
    """Advance progress once per step (terminations run first), record replay states, keep final flags for eval."""

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
        record = FLAGS.index("right_grasp")
        before = p.flags[:, record].clone()
        p.update(env)
        newly = (p.flags[:, record] & ~before & p.bar_up).nonzero().squeeze(-1)
        replay.buffer(env, ("bar", "cube"), len(FLAGS)).record(env, newly, p.flags)
        return p.success.clone()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The lifted bar fell back to the table, or the grasped cube fell to the table."""
    p = progress(env)
    bar_down = p.bar_up & (bar_pos(env)[:, 2] < BAR_REST_Z + 0.001)
    cube_down = p.right_grasp & (cube_pos(env)[:, 2] < CUBE_REST_Z + 0.005)
    fell = (bar_pos(env)[:, 2] < TABLE_TOP - 0.05) | (cube_pos(env)[:, 2] < TABLE_TOP - 0.05)
    return bar_down | cube_down | fell


# ---------------------------------------------------------------- config


@configclass
class StackSceneCfg(base.BiSceneCfg):
    # Only the left arm grips the bar (lying along y), so only its jaws get the eighth turn.
    robot_left = base._arm("robot_left", base.BASE_Y, wrist_roll_offset=math.pi / 4)
    target_pad = None
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.02),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CUBE_START, CUBE_REST_Z)),
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
class StackPolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=objects_state)
    relative = ObsTerm(func=objects_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class StackObservationsCfg:
    policy: StackPolicyCfg = StackPolicyCfg()
    critic: StackPolicyCfg = StackPolicyCfg()


@configclass
class StackEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_bar = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.03, 0.03), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bar"),
        },
    )
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.0), "yaw": (-0.785, 0.785)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    replay = EventTerm(
        func=replay.replay_reset, mode="reset", params={"objects": ("bar", "cube"), "num_flags": len(FLAGS)}
    )


@configclass
class StackRewardsCfg:
    reach_left = RewTerm(func=reach, weight=1.0, params={"arm": "robot_left"})
    reach_right = RewTerm(func=reach, weight=1.0, params={"arm": "robot_right"})
    grasp_left = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_left"})
    grasp_right = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_right"})
    hold = RewTerm(func=hold_bar, weight=0.3)
    carry = RewTerm(func=carry_cube, weight=0.5)
    place = RewTerm(func=set_down, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
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
class StackTerminationsCfg:
    success = DoneTerm(func=stack_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiStackEnvCfg(base.BiHandoverEnvCfg):
    scene: StackSceneCfg = StackSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    observations: StackObservationsCfg = StackObservationsCfg()
    events: StackEventsCfg = StackEventsCfg()
    rewards: StackRewardsCfg = StackRewardsCfg()
    terminations: StackTerminationsCfg = StackTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 25.0
