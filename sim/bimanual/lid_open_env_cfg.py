"""Lid open/close sub-task (LidOpen): one open-top box with a lift-off lid sits at a fixed spot between the arms.

The LEFT arm takes the lid off by its knob, lifts it level, moves it aside clear of the opening, holds it there
level for about a second, then puts it back on the box and lets go. The right arm has no actions (it holds its start
pose); the cube rests off to the right, untouched, so the scene matches the merged lid + cube task.

Geometry matches ``lid3_env_cfg`` (copied here so this module stands alone): 4 cm inside, 4 mm walls 4 cm high,
lid plate 55 x 55 x 8 mm with a 16 mm cube knob on top (knob turned
by ``KNOB_YAW`` to face the left arm), lid jittered +-3 mm / +-0.15 rad on the rim at each reset.

Success: the lid went through lifted -> aside (held level for ``HOLD_STEPS``) -> back closed on the box, the left
jaws let go and backed off, the cube was never moved, all stable for ``STABLE_STEPS``.
"""

from __future__ import annotations

import math
import os
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
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    WORKSHOP_CONTACT_MATERIAL,
    _bind_workshop_contact_material,
)

from . import env_cfg as base
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, gripper_openness, pad_center

# ---------------------------------------------------------------- constants

BOX = (0.14, 0.0)
"""Box centre in the env frame (shared sweet spot both arms reach)."""
BOX_INNER = 0.04
WALL_HEIGHT = 0.04
WALL = 0.004
LID_SIZE = (0.055, 0.055, 0.008)
KNOB_SIZE = 0.016
KNOB_OFFSET = LID_SIZE[2] / 2 + KNOB_SIZE / 2
KNOB_YAW = float(os.environ.get("BITIDY_KNOB_YAW", "-0.78"))
"""The knob is turned on the plate by -0.78 rad (-45 deg) about z, to face the left arm. The SO-101 is 5-DOF: its jaws
close along a level axis only when that axis is square to the vertical plane through the base, which over this box
(seen from the left arm) points at -0.78 rad (measured at the hover pose). A knob square to the box would need the
jaws rolled ~45 deg on the tilted approach, which tips the closing axis (and so the pinched lid) ~0.4 rad."""
LID_REST_Z = TABLE_TOP + WALL_HEIGHT + LID_SIZE[2] / 2
LID_JITTER = 0.003
LID_YAW = 0.15
CUBE_START = (0.18, -0.17)
"""The cube rests here, off to the right (the right arm's job in the merged task); this sub-task never touches it."""

KNOB_HOLD_RADIUS = 0.025
LIFT_HEIGHT = 0.02
"""Lid raised this far off the rim counts as lifted off."""
ASIDE_CLEAR = 0.065
"""Lid centre this far (xy) from the box centre: its footprint clears the 4 cm opening at any yaw."""
ASIDE_DIR = (0.0, 1.0)
"""The lid goes aside toward +y (the left arm's side), away from where the right arm works."""
LEVEL_TILT = 0.25
HOLD_STEPS = 30
"""~1 s at 30 Hz held aside and level."""
STABLE_STEPS = 10
EPISODE_S = float(os.environ.get("BITIDY_LID_OPEN_EPISODE_S", "25"))
FLAGS = ("lid_grasp", "lid_lifted", "lid_aside", "aside_hold", "lid_back")


# ---------------------------------------------------------------- lid asset: plate + knob as one rigid body


@clone
def _spawn_lid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass)
    for name, size, offset, yaw in (("plate", LID_SIZE, (0.0, 0.0, 0.0), 0.0),
                                    ("knob", (KNOB_SIZE,) * 3, (0.0, 0.0, KNOB_OFFSET), KNOB_YAW)):
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
        xform.AddRotateZOp().Set(math.degrees(yaw))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@configclass
class LidSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_lid
    mass: float = 0.02
    color: tuple[float, float, float] = (0.85, 0.25, 0.25)


# ---------------------------------------------------------------- geometry


def box_center(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene.env_origins.new_tensor(BOX).expand(env.num_envs, -1)


def lid_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["lid"].data.root_pos_w.torch - env.scene.env_origins


def lid_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["lid"].data.root_quat_w.torch


def _tilt(quat: torch.Tensor) -> torch.Tensor:
    up = quat_apply(quat, quat.new_tensor((0.0, 0.0, 1.0)).expand(quat.shape[:-1] + (3,)))
    return torch.acos(up[..., 2].clamp(-1.0, 1.0))


def lid_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _tilt(lid_quat(env))


def knob_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = lid_quat(env)
    return lid_pos(env) + quat_apply(quat, quat.new_tensor((0.0, 0.0, KNOB_OFFSET)).expand(quat.shape[0], -1))


def cube_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return base.cube_pos(env)


def lid_held(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left pads at the knob with the lid off the rim."""
    near = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) < KNOB_HOLD_RADIUS
    return near & (lid_pos(env)[:, 2] > LID_REST_Z + 0.01)


def lid_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    return lid_tilt(env) < LEVEL_TILT


def lid_lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    return lid_held(env) & lid_level(env) & (lid_pos(env)[:, 2] > LID_REST_Z + LIFT_HEIGHT)


def lid_aside(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Held, level and clear of the opening."""
    away = torch.linalg.vector_norm(lid_pos(env)[:, :2] - box_center(env), dim=-1) > ASIDE_CLEAR
    return lid_held(env) & lid_level(env) & away


def lid_closed(env: ManagerBasedRLEnv) -> torch.Tensor:
    pos = lid_pos(env)
    on_box = torch.linalg.vector_norm(pos[:, :2] - box_center(env), dim=-1) < 0.012
    return on_box & ((pos[:, 2] - LID_REST_Z).abs() < 0.006) & (lid_tilt(env) < 0.15)


def jaws_off(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) > 0.03


def cube_moved(env: ManagerBasedRLEnv) -> torch.Tensor:
    start = env.scene.env_origins.new_tensor((*CUBE_START, CUBE_REST_Z))
    return torch.linalg.vector_norm(cube_pos(env) - start, dim=-1) > 0.01


# ---------------------------------------------------------------- progress


class LidOpenProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.held_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.aside_count = torch.zeros_like(self.held_count)
        self.stable_count = torch.zeros_like(self.held_count)
        self.disturbed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.held_count[ids] = 0
        self.aside_count[ids] = 0
        self.stable_count[ids] = 0
        self.disturbed[ids] = False
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv) -> None:
        held = lid_held(env)
        self.held_count = torch.where(held, self.held_count + 1, 0)
        aside = lid_aside(env)
        self.aside_count = torch.where(aside, self.aside_count + 1, 0)
        grasp = self.held_count >= 2
        lifted = self.lid_grasp & lid_lifted(env)
        aside_now = self.lid_lifted & aside
        hold = self.lid_aside & (self.aside_count >= HOLD_STEPS)
        back = self.aside_hold & lid_closed(env)
        self.flags |= torch.stack((grasp, lifted, aside_now, hold, back), -1)
        self.disturbed |= cube_moved(env)
        done = self.lid_back & lid_closed(env) & jaws_off(env) & ~self.disturbed
        self.stable_count = torch.where(done, self.stable_count + 1, 0)
        self.success |= self.stable_count >= STABLE_STEPS


def progress(env: ManagerBasedRLEnv) -> LidOpenProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = LidOpenProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- events


def reset_objects(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: settle the lid on the box (small xy/yaw jitter), the cube at its resting spot."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    n, dev = len(ids), env.device
    zeros = torch.zeros((n, 6), device=dev)
    pose = torch.zeros((n, 7), device=dev)
    pose[:, :2] = pose.new_tensor(BOX) + (torch.rand((n, 2), device=dev) * 2 - 1) * LID_JITTER
    pose[:, 2] = LID_REST_Z
    yaw = (torch.rand(n, device=dev) * 2 - 1) * LID_YAW
    pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
    pose[:, :3] += env.scene.env_origins[ids]
    env.scene["lid"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.scene["lid"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)
    pose = torch.zeros((n, 7), device=dev)
    pose[:, :2] = pose.new_tensor(CUBE_START)
    pose[:, 2] = CUBE_REST_Z
    pose[:, 6] = 1.0
    pose[:, :3] += env.scene.env_origins[ids]
    env.scene["cube"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.scene["cube"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)


# ---------------------------------------------------------------- observations (privileged state: oracle/demo task)


def objects_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    lid = env.scene["lid"]
    return torch.cat((lid_pos(env), lid_quat(env), lid.data.root_lin_vel_w.torch, base.cube_state(env)), -1)


def objects_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    box = torch.cat((box_center(env), torch.full((env.num_envs, 1), LID_REST_Z, device=env.device)), -1)
    return torch.cat((knob_pos(env) - pad_center(env, "robot_left"), lid_pos(env) - box), -1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


def reach_knob(env: ManagerBasedRLEnv) -> torch.Tensor:
    d = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1)
    return torch.where(progress(env).lid_lifted, 1.0, 1.0 - torch.tanh(d / 0.08))


def grasp_knob(env: ManagerBasedRLEnv, radius: float = 0.03) -> torch.Tensor:
    near = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) < radius
    raised = ((lid_pos(env)[:, 2] - LID_REST_Z) / 0.04).clamp(0.0, 1.0)
    active = near & ~progress(env).lid_lifted
    return torch.where(active, 0.2 * (1.0 - gripper_openness(env, "robot_left")) + 0.5 * raised, 0.0)


def carry_aside(env: ManagerBasedRLEnv, std: float = 0.04) -> torch.Tensor:
    """While held and not yet held aside long enough, pull the lid toward the aside spot and keep it level."""
    p = progress(env)
    toward = box_center(env).new_tensor(ASIDE_DIR)
    spot = torch.cat((box_center(env) + toward * (ASIDE_CLEAR + 0.01),
                      torch.full((env.num_envs, 1), LID_REST_Z + 0.05, device=env.device)), -1)
    d = torch.linalg.vector_norm(lid_pos(env) - spot, dim=-1)
    active = lid_held(env) & p.lid_grasp & ~p.aside_hold
    return torch.where(active, torch.exp(-d / std) * torch.exp(-lid_tilt(env) / LEVEL_TILT), 0.0)


def close_lid(env: ManagerBasedRLEnv, std: float = 0.03) -> torch.Tensor:
    rest = torch.cat((box_center(env), torch.full((env.num_envs, 1), LID_REST_Z, device=env.device)), -1)
    distance = torch.linalg.vector_norm(lid_pos(env) - rest, dim=-1)
    return torch.where(progress(env).aside_hold, torch.exp(-distance / std) * torch.exp(-lid_tilt(env) / 0.2), 0.0)


def release_knob(env: ManagerBasedRLEnv) -> torch.Tensor:
    away = (torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) / 0.05).clamp(0.0, 1.0)
    return torch.where(progress(env).lid_back & lid_closed(env), away, 0.0)


def disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return cube_moved(env).float()


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 2.0, 3.0, 3.0, 6.0)

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, len(FLAGS)), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        flags = progress(env).flags
        reward = ((flags & ~self.paid).float() * flags.new_tensor(self.WEIGHTS, dtype=torch.float32)).sum(-1)
        self.paid = flags.clone()
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).success.float()


# ---------------------------------------------------------------- terminations


class lid_open_success(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        progress(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        p = progress(self._env)
        self._env._bi_final[ids] = torch.cat((p.flags[ids], p.success[ids].unsqueeze(-1)), dim=-1)
        p.reset(ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        p = progress(env)
        p.update(env)
        # Videos keep the episode running after success, so the closed box stays in view.
        if os.environ.get("BITIDY_NO_SUCCESS_END"):
            return torch.zeros_like(p.success)
        return p.success.clone()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The lid or the cube left the table."""
    return torch.minimum(lid_pos(env)[:, 2], cube_pos(env)[:, 2]) < TABLE_TOP - 0.05


# ---------------------------------------------------------------- scene


def _wall(name: str, center: tuple[float, float], size: tuple[float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(*size, WALL_HEIGHT),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*center, TABLE_TOP + WALL_HEIGHT / 2)),
    )


_H = BOX_INNER / 2 + WALL / 2
_X, _Y = BOX


@configclass
class LidOpenSceneCfg(base.BiSceneCfg):
    target_pad = None
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.02),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CUBE_START, CUBE_REST_Z)),
    )
    box_front = _wall("BoxFront", (_X + _H, _Y), (WALL, BOX_INNER + 2 * WALL))
    box_back = _wall("BoxBack", (_X - _H, _Y), (WALL, BOX_INNER + 2 * WALL))
    box_left = _wall("BoxLeft", (_X, _Y + _H), (BOX_INNER, WALL))
    box_right = _wall("BoxRight", (_X, _Y - _H), (BOX_INNER, WALL))
    lid = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lid",
        spawn=LidSpawnCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_X, _Y, LID_REST_Z)),
    )


# ---------------------------------------------------------------- config


@configclass
class LidOpenActionsCfg:
    left_arm, left_gripper = base._arm_actions("robot_left")


@configclass
class LidOpenPolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=objects_state)
    relative = ObsTerm(func=objects_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class LidOpenObservationsCfg:
    policy: LidOpenPolicyCfg = LidOpenPolicyCfg()
    critic: LidOpenPolicyCfg = LidOpenPolicyCfg()


@configclass
class LidOpenEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    objects = EventTerm(func=reset_objects, mode="reset")


@configclass
class LidOpenRewardsCfg:
    reach = RewTerm(func=reach_knob, weight=1.0)
    grasp = RewTerm(func=grasp_knob, weight=0.5)
    carry = RewTerm(func=carry_aside, weight=0.5)
    close = RewTerm(func=close_lid, weight=0.5)
    release = RewTerm(func=release_knob, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    disturbed = RewTerm(func=disturbed, weight=-1.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
    left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})


@configclass
class LidOpenTerminationsCfg:
    success = DoneTerm(func=lid_open_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LidOpenEnvCfg(base.BiHandoverEnvCfg):
    scene: LidOpenSceneCfg = LidOpenSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    actions: LidOpenActionsCfg = LidOpenActionsCfg()
    observations: LidOpenObservationsCfg = LidOpenObservationsCfg()
    events: LidOpenEventsCfg = LidOpenEventsCfg()
    rewards: LidOpenRewardsCfg = LidOpenRewardsCfg()
    terminations: LidOpenTerminationsCfg = LidOpenTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S

    def play_mode(self):
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
