"""Bimanual SO-101 lid task (T4): the left arm lifts the box lid off by its knob and holds it level, the right arm drops
the cube into the box; putting the lid back is optional.

The box is an open-top static box between the arms; its lid is a free plate with a knob, resting on the rim. There is no
hinge, so the parts can be 3D printed. Success needs the cube inside the box; the arms must take turns around the
shared opening.
"""

from __future__ import annotations

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
from . import replay
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, cube_pos, gripper_openness, held_by, pad_center

BOX_CENTER = (0.22, -0.03)
BOX_INNER = 0.08
WALL_HEIGHT = 0.05
WALL = 0.006
LID_SIZE = (0.10, 0.10, 0.008)
KNOB_SIZE = 0.02
KNOB_OFFSET = LID_SIZE[2] / 2 + KNOB_SIZE / 2
LID_REST_Z = TABLE_TOP + WALL_HEIGHT + LID_SIZE[2] / 2
CUBE_START = (0.20, -0.20)
OPEN_HEIGHT = 0.05
HOLD_RADIUS = 0.025
LEVEL_TILT = 0.35
"""About 20 degrees: the lifted lid stays roughly level instead of being flipped on its edge."""


# ---------------------------------------------------------------- lid asset: plate + knob as one rigid body


@clone
def _spawn_lid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass)
    for name, size, offset in (("plate", LID_SIZE, (0.0, 0.0, 0.0)), ("knob", (KNOB_SIZE,) * 3, (0.0, 0.0, KNOB_OFFSET))):
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.2)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@configclass
class LidSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_lid
    mass: float = 0.03


# ---------------------------------------------------------------- geometry


def lid_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["lid"].data.root_pos_w.torch - env.scene.env_origins


def knob_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["lid"].data.root_quat_w.torch
    offset = quat.new_tensor((0.0, 0.0, KNOB_OFFSET)).expand(quat.shape[0], -1)
    return lid_pos(env) + quat_apply(quat, offset)


def lid_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["lid"].data.root_quat_w.torch
    up = quat_apply(quat, quat.new_tensor((0.0, 0.0, 1.0)).expand(quat.shape[0], -1))
    return torch.acos(up[:, 2].clamp(-1.0, 1.0))


def box_center(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene.env_origins.new_tensor(BOX_CENTER)


def lid_held(env: ManagerBasedRLEnv) -> torch.Tensor:
    near = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) < HOLD_RADIUS
    return near & (lid_pos(env)[:, 2] > LID_REST_Z + 0.01)


def opening_clear(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The lid is lifted off, held roughly level high above the box or laid flat beside it.

    A lid tipped up on its edge against the rim does not count: the lid comes off whole, like a printed box lid.
    """
    lid = lid_pos(env)
    level = lid_tilt(env) < LEVEL_TILT
    aside = torch.linalg.vector_norm(lid[:, :2] - box_center(env), dim=-1) > 0.07
    return level & ((lid[:, 2] > LID_REST_Z + OPEN_HEIGHT) | aside)


def cube_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube = cube_pos(env)
    inside_xy = ((cube[:, :2] - box_center(env)).abs() < BOX_INNER / 2 - CUBE_SIZE / 4).all(-1)
    return inside_xy & (cube[:, 2] < TABLE_TOP + WALL_HEIGHT)


def lid_closed(env: ManagerBasedRLEnv) -> torch.Tensor:
    lid = lid_pos(env)
    on_box = torch.linalg.vector_norm(lid[:, :2] - box_center(env), dim=-1) < 0.015
    return on_box & ((lid[:, 2] - LID_REST_Z).abs() < 0.006) & (lid_tilt(env) < 0.15)


FLAGS = ("lid_grasp", "lid_open", "right_grasp", "cube_in", "lid_back")


class LidProgress:
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
        left, right = lid_held(env), held_by(env, "robot_right")
        self.left_count = torch.where(left, self.left_count + 1, 0)
        self.right_count = torch.where(right, self.right_count + 1, 0)
        released = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube_pos(env), dim=-1) > 0.03
        # The left arm must not be the one carrying the cube in: its pads stay clear of the cube.
        released &= torch.linalg.vector_norm(pad_center(env, "robot_left") - cube_pos(env), dim=-1) > 0.05
        # The cube only counts while the left arm holds the lid lifted off and level, not tipped up on the rim.
        holding_open = left & opening_clear(env)
        cube_in = cube_inside(env) & released & self.flags[:, FLAGS.index("right_grasp")] & holding_open
        closed = lid_closed(env) & self.flags[:, FLAGS.index("cube_in")]
        self.flags |= torch.stack(
            (self.left_count >= 2, left & opening_clear(env), self.right_count >= 2, cube_in, closed), dim=-1
        )
        # Putting the lid back is optional (paid by the shaping terms); the task is done once the cube is in.
        done = self.flags[:, FLAGS.index("cube_in")] & cube_inside(env)
        self.stable_count = torch.where(done, self.stable_count + 1, 0)
        self.success |= self.stable_count >= self.stable_steps


def progress(env: ManagerBasedRLEnv) -> LidProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = LidProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- observations


def objects_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    lid = env.scene["lid"]
    lid_state = torch.cat((lid_pos(env), lid.data.root_quat_w.torch, lid.data.root_lin_vel_w.torch), dim=-1)
    return torch.cat((lid_state, base.cube_state(env)), dim=-1)


def objects_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    box = torch.cat((box_center(env).expand(env.num_envs, -1), torch.full((env.num_envs, 1), TABLE_TOP, device=env.device)), dim=-1)
    return torch.cat(
        (
            knob_pos(env) - pad_center(env, "robot_left"),
            cube_pos(env) - pad_center(env, "robot_right"),
            cube_pos(env) - box,
            lid_pos(env) - box,
        ),
        dim=-1,
    )


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


class reach(ManagerTermBase):
    """Pay the decrease in pad distance to the knob (left) or the cube (right) until that arm has grasped."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, arm: str, scale: float = 0.01) -> torch.Tensor:
        target = knob_pos(env) if arm == "robot_left" else cube_pos(env)
        distance = torch.linalg.vector_norm(pad_center(env, arm) - target, dim=-1)
        gain = ((self.previous - distance) / scale).clamp(-1.0, 1.0).nan_to_num(0.0)
        self.previous = distance
        p = progress(env)
        done = p.lid_grasp if arm == "robot_left" else p.right_grasp
        return torch.where(done, 0.0, gain)


def grasp_shaping(env: ManagerBasedRLEnv, arm: str, radius: float = 0.03) -> torch.Tensor:
    if arm == "robot_left":
        target, height, rest, active = knob_pos(env), lid_pos(env)[:, 2], LID_REST_Z, ~progress(env).lid_open
    else:
        target, height, rest, active = cube_pos(env), cube_pos(env)[:, 2], CUBE_REST_Z, ~progress(env).right_grasp
    near = torch.linalg.vector_norm(pad_center(env, arm) - target, dim=-1) < radius
    raised = ((height - rest) / 0.04).clamp(0.0, 1.0)
    closing = 1.0 - gripper_openness(env, arm)
    return torch.where(near & active, 0.2 * closing + 0.5 * raised, 0.0)


def hold_open(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Until the cube is in, pay the left arm for holding the lid clear of the opening."""
    height = ((lid_pos(env)[:, 2] - LID_REST_Z) / OPEN_HEIGHT).clamp(0.0, 1.0)
    active = lid_held(env) & ~progress(env).cube_in
    return torch.where(active, height * torch.exp(-lid_tilt(env) / LEVEL_TILT), 0.0)


def carry_cube(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    """With the cube in hand and the opening clear, bring it above the middle of the box."""
    cube = cube_pos(env)
    above = torch.cat((box_center(env).expand(env.num_envs, -1), torch.full((env.num_envs, 1), TABLE_TOP + 0.09, device=env.device)), dim=-1)
    distance = torch.linalg.vector_norm(cube - above, dim=-1)
    active = held_by(env, "robot_right") & ~progress(env).cube_in
    return torch.where(active, torch.exp(-distance / std) * (0.3 + 0.7 * opening_clear(env).float()), 0.0)


def drop_in(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Above the box: pay for the right jaws opening so the cube drops in."""
    cube = cube_pos(env)
    over = ((cube[:, :2] - box_center(env)).abs() < BOX_INNER / 2 - 0.01).all(-1) & (cube[:, 2] > TABLE_TOP + WALL_HEIGHT)
    active = over & opening_clear(env) & ~progress(env).cube_in
    return torch.where(active, gripper_openness(env, "robot_right"), 0.0)


def close_lid(env: ManagerBasedRLEnv, std: float = 0.03) -> torch.Tensor:
    """After the cube is in, bring the lid back to its resting pose on the box."""
    lid = lid_pos(env)
    rest = torch.cat((box_center(env).expand(env.num_envs, -1), torch.full((env.num_envs, 1), LID_REST_Z, device=env.device)), dim=-1)
    distance = torch.linalg.vector_norm(lid - rest, dim=-1)
    return torch.where(progress(env).cube_in, torch.exp(-distance / std) * torch.exp(-lid_tilt(env) / 0.2), 0.0)


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 3.0, 2.0, 5.0, 6.0)

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


class lid_success(ManagerTermBase):
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
        record = FLAGS.index("lid_open")
        before = p.flags[:, record].clone()
        p.update(env)
        newly = (p.flags[:, record] & ~before).nonzero().squeeze(-1)
        replay.buffer(env, ("lid", "cube"), len(FLAGS)).record(env, newly, p.flags)
        return p.success.clone()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The cube or the lid left the table, or the grasped cube fell back to the table outside the box."""
    cube, lid = cube_pos(env), lid_pos(env)
    fell = (cube[:, 2] < TABLE_TOP - 0.05) | (lid[:, 2] < TABLE_TOP - 0.05)
    outside = ~((cube[:, :2] - box_center(env)).abs() < BOX_INNER / 2 + WALL).all(-1)
    cube_down = progress(env).right_grasp & (cube[:, 2] < CUBE_REST_Z + 0.005) & outside
    return fell | cube_down


# ---------------------------------------------------------------- config


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
_X, _Y = BOX_CENTER


@configclass
class LidSceneCfg(base.BiSceneCfg):
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
    lid = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lid",
        spawn=LidSpawnCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BOX_CENTER, LID_REST_Z)),
    )
    wall_front = _wall("WallFront", (_X + _H, _Y), (WALL, BOX_INNER + 2 * WALL))
    wall_back = _wall("WallBack", (_X - _H, _Y), (WALL, BOX_INNER + 2 * WALL))
    wall_left = _wall("WallLeft", (_X, _Y + _H), (BOX_INNER, WALL))
    wall_right = _wall("WallRight", (_X, _Y - _H), (BOX_INNER, WALL))


@configclass
class LidPolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=objects_state)
    relative = ObsTerm(func=objects_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class LidObservationsCfg:
    policy: LidPolicyCfg = LidPolicyCfg()
    critic: LidPolicyCfg = LidPolicyCfg()


@configclass
class LidEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.785, 0.785)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    reset_lid = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("lid"),
        },
    )
    replay = EventTerm(
        func=replay.replay_reset, mode="reset", params={"objects": ("lid", "cube"), "num_flags": len(FLAGS)}
    )


@configclass
class LidRewardsCfg:
    reach_left = RewTerm(func=reach, weight=1.0, params={"arm": "robot_left"})
    reach_right = RewTerm(func=reach, weight=1.0, params={"arm": "robot_right"})
    grasp_left = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_left"})
    grasp_right = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_right"})
    hold = RewTerm(func=hold_open, weight=0.3)
    carry = RewTerm(func=carry_cube, weight=0.5)
    drop = RewTerm(func=drop_in, weight=0.3)
    close = RewTerm(func=close_lid, weight=0.5)
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
class LidTerminationsCfg:
    success = DoneTerm(func=lid_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiLidEnvCfg(base.BiHandoverEnvCfg):
    scene: LidSceneCfg = LidSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    observations: LidObservationsCfg = LidObservationsCfg()
    events: LidEventsCfg = LidEventsCfg()
    rewards: LidRewardsCfg = LidRewardsCfg()
    terminations: LidTerminationsCfg = LidTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
