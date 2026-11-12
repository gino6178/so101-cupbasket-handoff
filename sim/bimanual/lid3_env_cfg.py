"""Voice-selected lidded boxes (BiLid3): three small open-top boxes with lift-off lids sit between the arms; each
episode names one of them (A, B or C). The left arm lifts that box's lid off by its knob and holds it aside, the right
arm picks the cube from its random start in front of the right arm and drops it into the box, then the left arm puts
the lid back on.

Like T4 there are no hinges, so boxes and lids can be 3D printed. The boxes are small (4 cm inside) and spaced so that
both arms reach every box (knobs 0.17-0.28 m from both bases; at 0.29 m the left arm could not reach box B's knob).
Success: the cube is in the named box, that box's lid is back on it, the left jaws have let go of the knob, and the
other two lids were never moved.
"""

from __future__ import annotations

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
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, cube_pos, gripper_openness, held_by, pad_center

# Small boxes with wide gaps (lids 1.5-3.5 cm apart), so lifting one lid does not knock the others.
BOXES = {"A": (0.10, 0.045), "B": (0.17, 0.0), "C": (0.10, -0.045)}
if os.environ.get("BITIDY_BOXES"):  # "ax,ay;bx,by;cx,cy" (layout experiments)
    BOXES = dict(zip("ABC", (tuple(map(float, b.split(","))) for b in os.environ["BITIDY_BOXES"].split(";"))))
NAMES = tuple(BOXES)
BOX_INNER = 0.04
WALL_HEIGHT = 0.04
WALL = 0.004
LID_SIZE = (0.055, 0.055, 0.008)
KNOB_SIZE = 0.016
KNOB_OFFSET = LID_SIZE[2] / 2 + KNOB_SIZE / 2
LID_REST_Z = TABLE_TOP + WALL_HEIGHT + LID_SIZE[2] / 2
CUBE_START = (0.18, -0.17)
CUBE_START_HALF = 0.02
OPEN_HEIGHT = 0.05
ASIDE = 0.045
KNOB_HOLD_RADIUS = 0.025
LEVEL_TILT = 0.35
STABLE_STEPS = 10
EPISODE_S = float(os.environ.get("BITIDY_LID3_EPISODE_S", "40"))
FLAGS = ("lid_grasp", "lid_open", "cube_grasp", "cube_in", "lid_back")


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
        cube.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@configclass
class LidSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_lid
    mass: float = 0.02
    color: tuple[float, float, float] = (0.9, 0.2, 0.2)


# ---------------------------------------------------------------- geometry (all lids, and the named one)


def _lid_names() -> tuple[str, ...]:
    return tuple(f"lid_{n.lower()}" for n in NAMES)


def box_centers(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(3, 2) box centres in the env frame."""
    return env.scene.env_origins.new_tensor([BOXES[n] for n in NAMES])


def target_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_lid3_target"):
        env._lid3_target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._lid3_target


def _pick(stacked: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select the named box's entry from a (num_envs, 3, ...) stack."""
    idx = target_index(env).view(-1, *([1] * (stacked.dim() - 1)))
    return stacked.gather(1, idx.expand(-1, 1, *stacked.shape[2:])).squeeze(1)


def all_lid_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.stack([env.scene[n].data.root_pos_w.torch for n in _lid_names()], 1) - env.scene.env_origins.unsqueeze(1)


def all_lid_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.stack([env.scene[n].data.root_quat_w.torch for n in _lid_names()], 1)


def _tilt(quat: torch.Tensor) -> torch.Tensor:
    up = quat_apply(quat, quat.new_tensor((0.0, 0.0, 1.0)).expand(quat.shape[:-1] + (3,)))
    return torch.acos(up[..., 2].clamp(-1.0, 1.0))


def lid_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _pick(all_lid_pos(env), env)


def lid_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _pick(all_lid_quat(env), env)


def lid_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _tilt(lid_quat(env))


def knob_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = lid_quat(env)
    return lid_pos(env) + quat_apply(quat, quat.new_tensor((0.0, 0.0, KNOB_OFFSET)).expand(quat.shape[0], -1))


def box_center(env: ManagerBasedRLEnv) -> torch.Tensor:
    return box_centers(env)[target_index(env)]


def lid_held(env: ManagerBasedRLEnv) -> torch.Tensor:
    near = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) < KNOB_HOLD_RADIUS
    return near & (lid_pos(env)[:, 2] > LID_REST_Z + 0.01)


def opening_clear(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The named lid is off its box: held roughly level high above it or moved aside."""
    lid = lid_pos(env)
    level = lid_tilt(env) < LEVEL_TILT
    aside = torch.linalg.vector_norm(lid[:, :2] - box_center(env), dim=-1) > ASIDE
    return level & ((lid[:, 2] > LID_REST_Z + OPEN_HEIGHT) | aside)


def cube_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube = cube_pos(env)
    inside_xy = ((cube[:, :2] - box_center(env)).abs() < BOX_INNER / 2).all(-1)
    return inside_xy & (cube[:, 2] < TABLE_TOP + WALL_HEIGHT)


def _closed(pos: torch.Tensor, quat: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    on_box = torch.linalg.vector_norm(pos[..., :2] - centers, dim=-1) < 0.012
    return on_box & ((pos[..., 2] - LID_REST_Z).abs() < 0.006) & (_tilt(quat) < 0.15)


def lid_closed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _closed(lid_pos(env), lid_quat(env), box_center(env))


def others_closed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The two lids that were not named are still on their boxes."""
    closed = _closed(all_lid_pos(env), all_lid_quat(env), box_centers(env).unsqueeze(0))
    named = torch.nn.functional.one_hot(target_index(env), len(NAMES)).bool()
    return (closed | named).all(-1)


# ---------------------------------------------------------------- progress


class Lid3Progress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.left_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.right_count = torch.zeros_like(self.left_count)
        self.stable_count = torch.zeros_like(self.left_count)
        self.disturbed = torch.zeros(num_envs, dtype=torch.bool, device=device)
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
        self.disturbed[ids] = False
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv) -> None:
        left, right = lid_held(env), held_by(env, "robot_right")
        self.left_count = torch.where(left, self.left_count + 1, 0)
        self.right_count = torch.where(right, self.right_count + 1, 0)
        cube = cube_pos(env)
        right_clear = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube, dim=-1) > 0.03
        # The left arm handles only the lid: its pads stay clear of the cube.
        left_clear = torch.linalg.vector_norm(pad_center(env, "robot_left") - cube, dim=-1) > 0.05
        cube_in = cube_inside(env) & right_clear & left_clear & self.cube_grasp & left & opening_clear(env)
        left_off = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) > 0.03
        back = lid_closed(env) & self.cube_in
        self.flags |= torch.stack((self.left_count >= 2, left & opening_clear(env), self.right_count >= 2, cube_in, back), -1)
        self.disturbed |= ~others_closed(env)
        done = self.lid_back & lid_closed(env) & cube_inside(env) & left_off & ~self.disturbed
        self.stable_count = torch.where(done, self.stable_count + 1, 0)
        self.success |= self.stable_count >= STABLE_STEPS


def progress(env: ManagerBasedRLEnv) -> Lid3Progress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = Lid3Progress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- events


def sample_task(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: name a box, settle every lid on its box (small jitter), put the cube at its random start."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    n = len(ids)
    dev = env.device
    target_index(env)[ids] = torch.randint(0, len(NAMES), (n,), device=dev)
    if os.environ.get("BITIDY_LID3_TARGET"):
        target_index(env)[ids] = NAMES.index(os.environ["BITIDY_LID3_TARGET"])
    zeros = torch.zeros((n, 6), device=dev)
    for name, center in zip(_lid_names(), box_centers(env)):
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = center + (torch.rand((n, 2), device=dev) * 2 - 1) * 0.003
        pose[:, 2] = LID_REST_Z
        yaw = (torch.rand(n, device=dev) * 2 - 1) * 0.15
        pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        pose[:, :3] += env.scene.env_origins[ids]
        env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[name].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)
    pose = torch.zeros((n, 7), device=dev)
    pose[:, :2] = pose.new_tensor(CUBE_START) + (torch.rand((n, 2), device=dev) * 2 - 1) * CUBE_START_HALF
    pose[:, 2] = CUBE_REST_Z
    yaw = (torch.rand(n, device=dev) * 2 - 1) * (torch.pi / 4)
    pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
    pose[:, :3] += env.scene.env_origins[ids]
    env.scene["cube"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.scene["cube"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)


# ---------------------------------------------------------------- observations


def objects_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Every lid pose (they must stay put), the cube state, and the named box as a one-hot."""
    lids = torch.cat((all_lid_pos(env), all_lid_quat(env)), -1).flatten(1)
    named = torch.nn.functional.one_hot(target_index(env), len(NAMES)).float()
    return torch.cat((lids, base.cube_state(env), named), -1)


def objects_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    box = torch.cat((box_center(env), torch.full((env.num_envs, 1), TABLE_TOP, device=env.device)), -1)
    return torch.cat(
        (knob_pos(env) - pad_center(env, "robot_left"), cube_pos(env) - pad_center(env, "robot_right"),
         cube_pos(env) - box, lid_pos(env) - box), -1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


def reach(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    p = progress(env)
    if arm == "robot_left":
        d, done = torch.linalg.vector_norm(pad_center(env, arm) - knob_pos(env), dim=-1), p.lid_open
    else:
        d, done = torch.linalg.vector_norm(pad_center(env, arm) - cube_pos(env), dim=-1), p.cube_grasp
    return torch.where(done, 1.0, 1.0 - torch.tanh(d / 0.08))


def grasp_shaping(env: ManagerBasedRLEnv, arm: str, radius: float = 0.03) -> torch.Tensor:
    p = progress(env)
    if arm == "robot_left":
        target, height, rest, active = knob_pos(env), lid_pos(env)[:, 2], LID_REST_Z, ~p.lid_open
    else:
        target, height, rest, active = cube_pos(env), cube_pos(env)[:, 2], CUBE_REST_Z, ~p.cube_grasp
    near = torch.linalg.vector_norm(pad_center(env, arm) - target, dim=-1) < radius
    raised = ((height - rest) / 0.04).clamp(0.0, 1.0)
    return torch.where(near & active, 0.2 * (1.0 - gripper_openness(env, arm)) + 0.5 * raised, 0.0)


def hold_open(env: ManagerBasedRLEnv) -> torch.Tensor:
    active = lid_held(env) & ~progress(env).cube_in
    return torch.where(active, opening_clear(env).float() * torch.exp(-lid_tilt(env) / LEVEL_TILT), 0.0)


def carry_cube(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    above = torch.cat((box_center(env), torch.full((env.num_envs, 1), TABLE_TOP + WALL_HEIGHT + 0.04, device=env.device)), -1)
    distance = torch.linalg.vector_norm(cube_pos(env) - above, dim=-1)
    active = held_by(env, "robot_right") & ~progress(env).cube_in
    return torch.where(active, torch.exp(-distance / std) * (0.3 + 0.7 * opening_clear(env).float()), 0.0)


def drop_in(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube = cube_pos(env)
    over = ((cube[:, :2] - box_center(env)).abs() < BOX_INNER / 2 - 0.008).all(-1) & (cube[:, 2] > TABLE_TOP + WALL_HEIGHT)
    active = over & opening_clear(env) & ~progress(env).cube_in
    return torch.where(active, gripper_openness(env, "robot_right"), 0.0)


def close_lid(env: ManagerBasedRLEnv, std: float = 0.03) -> torch.Tensor:
    rest = torch.cat((box_center(env), torch.full((env.num_envs, 1), LID_REST_Z, device=env.device)), -1)
    distance = torch.linalg.vector_norm(lid_pos(env) - rest, dim=-1)
    return torch.where(progress(env).cube_in, torch.exp(-distance / std) * torch.exp(-lid_tilt(env) / 0.2), 0.0)


def release_knob(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Lid back on: pay the left jaws for letting go and backing off."""
    away = (torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) / 0.05).clamp(0.0, 1.0)
    return torch.where(progress(env).lid_back & lid_closed(env), away, 0.0)


def disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (~others_closed(env)).float()


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 3.0, 2.0, 5.0, 6.0)

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


class lid3_success(ManagerTermBase):
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
    """The cube or a lid left the table."""
    low = torch.cat((cube_pos(env)[:, 2:3], all_lid_pos(env)[..., 2]), -1).min(-1).values
    return low < TABLE_TOP - 0.05


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
LID_COLORS = {"A": (0.85, 0.25, 0.25), "B": (0.25, 0.7, 0.3), "C": (0.25, 0.4, 0.85)}


def _box(n: str):
    """Four walls and the lid of box ``n``."""
    x, y = BOXES[n]
    return (
        _wall(f"Box{n}Front", (x + _H, y), (WALL, BOX_INNER + 2 * WALL)),
        _wall(f"Box{n}Back", (x - _H, y), (WALL, BOX_INNER + 2 * WALL)),
        _wall(f"Box{n}Left", (x, y + _H), (BOX_INNER, WALL)),
        _wall(f"Box{n}Right", (x, y - _H), (BOX_INNER, WALL)),
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Lid{n}",
            spawn=LidSpawnCfg(color=LID_COLORS[n]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, LID_REST_Z)),
        ),
    )


@configclass
class Lid3SceneCfg(base.BiSceneCfg):
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
    box_a_front, box_a_back, box_a_left, box_a_right, lid_a = _box("A")
    box_b_front, box_b_back, box_b_left, box_b_right, lid_b = _box("B")
    box_c_front, box_c_back, box_c_left, box_c_right, lid_c = _box("C")


# ---------------------------------------------------------------- config


@configclass
class Lid3PolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=objects_state)
    relative = ObsTerm(func=objects_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class Lid3ObservationsCfg:
    policy: Lid3PolicyCfg = Lid3PolicyCfg()
    critic: Lid3PolicyCfg = Lid3PolicyCfg()


@configclass
class Lid3EventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    task = EventTerm(func=sample_task, mode="reset")


@configclass
class Lid3RewardsCfg:
    reach_left = RewTerm(func=reach, weight=1.0, params={"arm": "robot_left"})
    reach_right = RewTerm(func=reach, weight=1.0, params={"arm": "robot_right"})
    grasp_left = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_left"})
    grasp_right = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_right"})
    hold = RewTerm(func=hold_open, weight=0.3)
    carry = RewTerm(func=carry_cube, weight=0.5)
    drop = RewTerm(func=drop_in, weight=0.3)
    close = RewTerm(func=close_lid, weight=0.5)
    release = RewTerm(func=release_knob, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    disturbed = RewTerm(func=disturbed, weight=-1.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_right")})
    left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_right")})


@configclass
class Lid3TerminationsCfg:
    success = DoneTerm(func=lid3_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiLid3EnvCfg(base.BiHandoverEnvCfg):
    scene: Lid3SceneCfg = Lid3SceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    observations: Lid3ObservationsCfg = Lid3ObservationsCfg()
    events: Lid3EventsCfg = Lid3EventsCfg()
    rewards: Lid3RewardsCfg = Lid3RewardsCfg()
    terminations: Lid3TerminationsCfg = Lid3TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S

    def play_mode(self):
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
