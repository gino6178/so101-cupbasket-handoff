"""Box place (right-arm half of the lidded-box task): the right SO-101 picks the cube from its random start in front of
it and drops it into one open-top box at a fixed spot between the arms, then backs its jaws off clear of the box.

The box is the same small 3D-printable box as in BiLid3 (4 cm inside, 4 cm high walls) without its lid: in the merged
task the left arm has already taken the lid away, so the space above the box stays free. Only the right arm acts; the
left arm holds its start pose.

Success: the cube was grasped, lifted, carried over the opening and released; it rests inside the box; and the right
arm's jaws are at least ``CLEAR_XY`` (horizontally) from the box centre, so a lid could go back on.
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
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import WORKSHOP_CONTACT_MATERIAL
from isaaclab_tutorial.tasks.place_vial.mdp.task import FIXED_FINGERTIP_OFFSET, MOVING_FINGERTIP_OFFSET

from . import env_cfg as base
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, cube_pos, gripper_openness, held_by, pad_center

ARM = "robot_right"
# Shared sweet spot of both arms (reach map: both within 1 cm around (0.12-0.16, 0.00) at z 0.05-0.13).
BOX = tuple(map(float, os.environ.get("BITIDY_BOX", "0.14,0.0").split(",")))
BOX_INNER = 0.04
WALL = 0.004
WALL_HEIGHT = 0.04
RIM_Z = TABLE_TOP + WALL_HEIGHT
CUBE_START = tuple(map(float, os.environ.get("BITIDY_CUBE_START", "0.18,-0.17").split(",")))
CUBE_START_HALF = 0.02
LIFT = 0.03
CLEAR_XY = 0.06
"""Jaw tips at least this far (horizontally) from the box centre count as clear of the box (lid half-diagonal ~4 cm)."""
TOP_MARGIN = 0.002
"""The cube's highest corner must be this far below the rim (flat: 25 mm under a 40 mm wall; leaning on a wall: ~31 mm)."""
STABLE_STEPS = 10
EPISODE_S = float(os.environ.get("BITIDY_BOX_EPISODE_S", "30"))
FLAGS = ("grasp", "lift", "over_box", "cube_in", "backed_off")


# ---------------------------------------------------------------- geometry


def box_center(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene.env_origins.new_tensor(BOX[:2]).expand(env.num_envs, -1)


def jaw_tips(env: ManagerBasedRLEnv, arm: str = ARM) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed and moving fingertip pad positions in the env frame."""
    robot = env.scene[arm]
    tips = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        i = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, i], robot.data.body_quat_w.torch[:, i]
        tips.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)) - env.scene.env_origins)
    return tips[0], tips[1]


def cube_over_opening(env: ManagerBasedRLEnv, margin: float = 0.0) -> torch.Tensor:
    return ((cube_pos(env)[:, :2] - box_center(env)).abs() < BOX_INNER / 2 - margin).all(-1)


def cube_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    return cube_over_opening(env) & (cube_pos(env)[:, 2] < RIM_Z)


def cube_top_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height of the cube's highest corner (it may rest tilted against an inner wall)."""
    quat = env.scene["cube"].data.root_quat_w.torch
    n = quat.shape[0]
    axes_z = torch.stack([quat_apply(quat, quat.new_tensor(a).expand(n, -1))[:, 2] for a in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))], -1)
    return cube_pos(env)[:, 2] + 0.5 * CUBE_SIZE * axes_z.abs().sum(-1)


def cube_resting_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The cube is still, its centre over the floor of the box and all of it below the rim (a lid fits on)."""
    speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
    return cube_inside(env) & (cube_top_z(env) < RIM_Z - TOP_MARGIN) & (speed < 0.05)


def jaws_clear(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Both fingertips and their midpoint are horizontally clear of the box (and its lid footprint)."""
    box = box_center(env)
    pts = (*jaw_tips(env), pad_center(env, ARM))
    return torch.stack([torch.linalg.vector_norm(p[:, :2] - box, dim=-1) > CLEAR_XY for p in pts]).all(0)


# ---------------------------------------------------------------- progress


class BoxPlaceProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable = torch.zeros_like(self.hold)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.hold[ids] = 0
        self.stable[ids] = 0
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv) -> None:
        cube = cube_pos(env)
        held = held_by(env, ARM)
        self.hold = torch.where(held, self.hold + 1, 0)
        lifted = held & (cube[:, 2] > CUBE_REST_Z + LIFT)
        over = self.lift & held & cube_over_opening(env) & (cube[:, 2] - CUBE_SIZE / 2 > RIM_Z)
        released = torch.linalg.vector_norm(pad_center(env, ARM) - cube, dim=-1) > 0.03
        resting = cube_resting_inside(env)
        cube_in = self.over_box & resting & released
        backed = self.cube_in & resting & jaws_clear(env)
        self.flags |= torch.stack((self.hold >= 2, lifted, over, cube_in, backed), -1)
        done = self.backed_off & resting & jaws_clear(env)
        self.stable = torch.where(done, self.stable + 1, 0)
        self.success |= self.stable >= STABLE_STEPS


def progress(env: ManagerBasedRLEnv) -> BoxPlaceProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = BoxPlaceProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- events


def sample_task(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: the cube at a random point (and yaw +-45 deg) of its start square."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    n = len(ids)
    dev = env.device
    pose = torch.zeros((n, 7), device=dev)
    pose[:, :2] = pose.new_tensor(CUBE_START) + (torch.rand((n, 2), device=dev) * 2 - 1) * CUBE_START_HALF
    pose[:, 2] = CUBE_REST_Z
    yaw = (torch.rand(n, device=dev) * 2 - 1) * (torch.pi / 4)
    pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
    pose[:, :3] += env.scene.env_origins[ids]
    env.scene["cube"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.scene["cube"].write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=dev), env_ids=ids)


# ---------------------------------------------------------------- observations (privileged state)


def objects_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    box = torch.cat((box_center(env), torch.full((env.num_envs, 1), TABLE_TOP, device=env.device)), -1)
    pad = pad_center(env, ARM)
    return torch.cat((cube_pos(env) - pad, cube_pos(env) - box, pad - box), -1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


def reach(env: ManagerBasedRLEnv) -> torch.Tensor:
    d = torch.linalg.vector_norm(pad_center(env, ARM) - cube_pos(env), dim=-1)
    return torch.where(progress(env).grasp, 1.0, 1.0 - torch.tanh(d / 0.08))


def grasp_shaping(env: ManagerBasedRLEnv, radius: float = 0.03) -> torch.Tensor:
    cube = cube_pos(env)
    near = torch.linalg.vector_norm(pad_center(env, ARM) - cube, dim=-1) < radius
    raised = ((cube[:, 2] - CUBE_REST_Z) / 0.04).clamp(0.0, 1.0)
    active = near & ~progress(env).cube_in
    return torch.where(active, 0.2 * (1.0 - gripper_openness(env, ARM)) + 0.5 * raised, 0.0)


def carry_cube(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    above = torch.cat((box_center(env), torch.full((env.num_envs, 1), RIM_Z + 0.03, device=env.device)), -1)
    distance = torch.linalg.vector_norm(cube_pos(env) - above, dim=-1)
    active = held_by(env, ARM) & ~progress(env).cube_in
    return torch.where(active, torch.exp(-distance / std), 0.0)


def drop_in(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube = cube_pos(env)
    over = cube_over_opening(env, 0.006) & (cube[:, 2] - CUBE_SIZE / 2 > RIM_Z)
    return torch.where(over & ~progress(env).cube_in, gripper_openness(env, ARM), 0.0)


def back_off(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube in: pay the jaws for moving away from the box."""
    d = torch.linalg.vector_norm(pad_center(env, ARM)[:, :2] - box_center(env), dim=-1)
    return torch.where(progress(env).cube_in & cube_inside(env), (d / CLEAR_XY).clamp(0.0, 1.0), 0.0)


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 2.0, 3.0, 5.0, 3.0)

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


class box_place_success(ManagerTermBase):
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
        # Videos keep the episode running after success, so the filled box stays in view.
        if os.environ.get("BITIDY_NO_SUCCESS_END"):
            return torch.zeros_like(p.success)
        return p.success.clone()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The cube left the table."""
    return cube_pos(env)[:, 2] < TABLE_TOP - 0.05


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


@configclass
class BoxPlaceSceneCfg(base.BiSceneCfg):
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
    box_front = _wall("BoxFront", (BOX[0] + _H, BOX[1]), (WALL, BOX_INNER + 2 * WALL))
    box_back = _wall("BoxBack", (BOX[0] - _H, BOX[1]), (WALL, BOX_INNER + 2 * WALL))
    box_left = _wall("BoxLeft", (BOX[0], BOX[1] + _H), (BOX_INNER, WALL))
    box_right = _wall("BoxRight", (BOX[0], BOX[1] - _H), (BOX_INNER, WALL))


# ---------------------------------------------------------------- config


@configclass
class BoxPlaceActionsCfg:
    arm_joints, gripper = base._arm_actions(ARM)


@configclass
class BoxPlacePolicyCfg(base.PolicyCfg):
    relative = ObsTerm(func=objects_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class BoxPlaceObservationsCfg:
    policy: BoxPlacePolicyCfg = BoxPlacePolicyCfg()
    critic: BoxPlacePolicyCfg = BoxPlacePolicyCfg()


@configclass
class BoxPlaceEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    task = EventTerm(func=sample_task, mode="reset")


@configclass
class BoxPlaceRewardsCfg:
    reach = RewTerm(func=reach, weight=1.0)
    grasp = RewTerm(func=grasp_shaping, weight=0.5)
    carry = RewTerm(func=carry_cube, weight=1.0)
    drop = RewTerm(func=drop_in, weight=0.3)
    back_off = RewTerm(func=back_off, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg(ARM)})
    limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg(ARM)})


@configclass
class BoxPlaceTerminationsCfg:
    success = DoneTerm(func=box_place_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BoxPlaceEnvCfg(base.BiHandoverEnvCfg):
    scene: BoxPlaceSceneCfg = BoxPlaceSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    actions: BoxPlaceActionsCfg = BoxPlaceActionsCfg()
    observations: BoxPlaceObservationsCfg = BoxPlaceObservationsCfg()
    events: BoxPlaceEventsCfg = BoxPlaceEventsCfg()
    rewards: BoxPlaceRewardsCfg = BoxPlaceRewardsCfg()
    terminations: BoxPlaceTerminationsCfg = BoxPlaceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S

    def play_mode(self):
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
