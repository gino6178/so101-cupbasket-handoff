"""Bimanual SO-101 cube handover from full state.

Two workshop-calibrated SO-101 arms stand 30 cm apart, both facing +x. The cube starts in front of the left arm;
the target pad is in front of the right arm. Success needs the right arm to have held the cube, so a left arm that
pushes the cube across the table never counts as a handover.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.tasks.place_vial import mdp

from . import replay
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    ARM_JOINTS,
    JOINTS,
    SO101_CFG,
    WORKSHOP_CONTACT_MATERIAL,
    WORKSHOP_INITIAL_JOINT_POSITION,
    PREGRASP_GRIPPER_POSITION,
    PhysicsCfg,
)
from isaaclab_tutorial.tasks.place_vial.mdp.task import FIXED_FINGERTIP_OFFSET, MOVING_FINGERTIP_OFFSET

ARMS = ("robot_left", "robot_right")
FACING_X = float(os.environ.get("BITIDY_FACING_X", "0"))
BASE_Y = float(os.environ.get("BITIDY_BASE_Y", "0.15"))  # arm bases at y = +-BASE_Y (layout experiments)
TABLE_TOP = 0.032
CUBE_SIZE = 0.025
CUBE_REST_Z = TABLE_TOP + CUBE_SIZE / 2
CUBE_START = (0.20, 0.17)
HANDOVER_POINT = (0.20, 0.0, 0.12)
"""Reference point for observations; the handover itself may happen anywhere in the zone below."""
HANDOVER_HALF_WIDTH = 0.07
HANDOVER_X = (0.10, 0.30)
HANDOVER_HEIGHT = 0.05
LEFT_CLEAR = 0.05
GRIPPER_CLOSED = math.radians(-12.0)
"""Commanded when closing: just above the lower limit, so the jaws squeeze whatever they hold."""
TARGET = (0.20, -0.17)
HOLD_RADIUS = 0.025
"""Cube centre within this distance of an arm's pad midpoint, off the table, counts as held by that arm."""
TARGET_RADIUS = 0.04


# ---------------------------------------------------------------- geometry


def pad_center(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    """Midpoint of the fixed and moving fingertip pads of ``arm`` in the env frame [m]."""
    robot = env.scene[arm]
    pads = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        body_id = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, body_id], robot.data.body_quat_w.torch[:, body_id]
        pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return 0.5 * (pads[0] + pads[1]) - env.scene.env_origins


def cube_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["cube"].data.root_pos_w.torch - env.scene.env_origins


def held_by(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    cube = cube_pos(env)
    near = torch.linalg.vector_norm(pad_center(env, arm) - cube, dim=-1) < HOLD_RADIUS
    return near & (cube[:, 2] > CUBE_REST_Z + 0.01)


RIGHT_HOLD_RADIUS = 0.04
"""During the handover the left jaws occupy the cube centre, so the right pads close on it further out."""


def right_jaw_contact(env: ManagerBasedRLEnv, threshold: float = 0.05) -> torch.Tensor:
    """Both right jaws press on the cube."""
    forces = []
    for name in ("right_fixed_jaw_cube", "right_moving_jaw_cube"):
        force = env.scene.sensors[name].data.force_matrix_w.torch
        forces.append(torch.linalg.vector_norm(force, dim=-1).reshape(env.num_envs, -1).amax(dim=-1))
    return (forces[0] > threshold) & (forces[1] > threshold)


def right_holds(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The right arm grips the lifted cube: both jaws in contact, pads within the handover radius."""
    cube = cube_pos(env)
    near = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube, dim=-1) < RIGHT_HOLD_RADIUS
    return near & right_jaw_contact(env) & (cube[:, 2] > CUBE_REST_Z + 0.01)


def left_released(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The left pads are well clear of the cube, so it cannot still be carried or pushed by the left arm."""
    return torch.linalg.vector_norm(pad_center(env, "robot_left") - cube_pos(env), dim=-1) > LEFT_CLEAR


def handover_gap(cube: torch.Tensor) -> torch.Tensor:
    """Distance from the cube to the handover zone: the middle strip between the arms, raised off the table."""
    side = (cube[:, 1].abs() - HANDOVER_HALF_WIDTH).clamp(min=0.0)
    front = (HANDOVER_X[0] - cube[:, 0]).clamp(min=0.0) + (cube[:, 0] - HANDOVER_X[1]).clamp(min=0.0)
    low = (CUBE_REST_Z + HANDOVER_HEIGHT - cube[:, 2]).clamp(min=0.0)
    return torch.sqrt(side**2 + front**2 + low**2)


def target_distance_xy(cube: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(cube[:, :2] - cube.new_tensor(TARGET), dim=-1)


# ---------------------------------------------------------------- episode progress

FLAGS = ("left_grasp", "left_lift", "at_handover", "both_hold", "right_grasp", "placed")


class HandoverProgress:
    """Latched per-episode milestones; ``success`` needs ``placed`` held still for ``stable_steps`` steps."""

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
        cube = cube_pos(env)
        left, right = held_by(env, "robot_left"), right_holds(env)
        # A real handover: the right arm alone holds the lifted cube, with the left pads clear of it.
        left_clear = left_released(env)
        self.left_count = torch.where(left, self.left_count + 1, 0)
        self.right_count = torch.where(right & left_clear, self.right_count + 1, 0)
        reached = torch.stack(
            (
                self.left_count >= 2,
                left & (cube[:, 2] > CUBE_REST_Z + 0.04),
                left & (handover_gap(cube) < 0.005),
                left & right,
                (self.right_count >= 3) & self.flags[:, FLAGS.index("left_lift")],
                torch.zeros_like(left),
            ),
            dim=-1,
        )
        self.flags |= reached
        speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
        resting = (target_distance_xy(cube) < TARGET_RADIUS) & (cube[:, 2] < CUBE_REST_Z + 0.006) & (speed < 0.05)
        placed = resting & self.right_grasp & left_clear
        self.stable_count = torch.where(placed, self.stable_count + 1, 0)
        self.flags[:, FLAGS.index("placed")] |= placed
        self.success |= self.stable_count >= self.stable_steps


def progress(env: ManagerBasedRLEnv) -> HandoverProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = HandoverProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- observations


def pad_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.cat([pad_center(env, arm) for arm in ARMS], dim=-1)


def cube_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube = env.scene["cube"]
    return torch.cat((cube_pos(env), cube.data.root_quat_w.torch, cube.data.root_lin_vel_w.torch), dim=-1)


def cube_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube = cube_pos(env)
    target = cube.new_tensor((*TARGET, CUBE_REST_Z)).expand_as(cube)
    handover = cube.new_tensor(HANDOVER_POINT).expand_as(cube)
    return torch.cat([cube - pad_center(env, arm) for arm in ARMS] + [cube - handover, cube - target], dim=-1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


class reach_progress(ManagerTermBase):
    """Pay the decrease in pad-to-cube distance of ``arm`` (in units of ``scale``) while ``phase`` is active."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, arm: str, scale: float = 0.01) -> torch.Tensor:
        distance = torch.linalg.vector_norm(pad_center(env, arm) - cube_pos(env), dim=-1)
        gain = ((self.previous - distance) / scale).clamp(-1.0, 1.0).nan_to_num(0.0)
        self.previous = distance
        p = progress(env)
        active = ~p.left_grasp if arm == "robot_left" else (p.at_handover & ~p.right_grasp)
        return torch.where(active, gain, 0.0)


def gripper_openness(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    robot = env.scene[arm]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)


def grasp_shaping(env: ManagerBasedRLEnv, arm: str, radius: float = 0.03) -> torch.Tensor:
    """Near the cube, pay for closing the jaws and for raising the cube, until this arm's grasp milestone.

    Lift is paid continuously (up to 6 cm) so the policy finds the grasp before the latched milestone exists.
    """
    cube = cube_pos(env)
    near = torch.linalg.vector_norm(pad_center(env, arm) - cube, dim=-1) < radius
    lift = ((cube[:, 2] - CUBE_REST_Z) / 0.06).clamp(0.0, 1.0)
    closing = 1.0 - gripper_openness(env, arm)
    p = progress(env)
    if arm == "robot_left":
        return torch.where(near & ~p.left_lift, 0.2 * closing + lift, 0.0)
    # The left arm already holds the cube up, so pay the right arm for actual two-jaw contact, not for height.
    near = torch.linalg.vector_norm(pad_center(env, arm) - cube, dim=-1) < RIGHT_HOLD_RADIUS
    contact = right_jaw_contact(env).float()
    return torch.where(near & p.at_handover & ~p.right_grasp, 0.2 * closing + contact, 0.0)


def carry_to_handover(env: ManagerBasedRLEnv, std: float = 0.08) -> torch.Tensor:
    cube = cube_pos(env)
    p = progress(env)
    return torch.where(held_by(env, "robot_left") & ~p.right_grasp, torch.exp(-handover_gap(cube) / std), 0.0)


def carry_to_target(env: ManagerBasedRLEnv, std: float = 0.08) -> torch.Tensor:
    """After the handover, bring the cube over the target and down onto it."""
    cube = cube_pos(env)
    above = cube.new_tensor((*TARGET, CUBE_REST_Z)).expand_as(cube)
    distance = torch.linalg.vector_norm(cube - above, dim=-1)
    return torch.where(progress(env).right_grasp, torch.exp(-distance / std), 0.0)


def left_lets_go(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Once the right arm holds the cube, pay the left pads for clearing away from it."""
    distance = torch.linalg.vector_norm(pad_center(env, "robot_left") - cube_pos(env), dim=-1)
    p = progress(env)
    return torch.where(p.right_grasp, (distance / 0.08).clamp(0.0, 1.0), 0.0)


class milestones(ManagerTermBase):
    """Pay each latched milestone once per episode, in increasing amounts along the task."""

    WEIGHTS = (1.0, 2.0, 3.0, 4.0, 4.0, 6.0)

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


class handover_success(ManagerTermBase):
    """Advance the episode progress once per step and end the episode on confirmed placement.

    Termination terms run before rewards and observations, so every other term sees this step's progress. On reset
    the final flags are copied to ``env._bi_final`` for the evaluation counter.
    """

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
        record = FLAGS.index("at_handover")
        before = p.flags[:, record].clone()
        p.update(env)
        newly = (p.flags[:, record] & ~before).nonzero().squeeze(-1)
        replay.buffer(env, ("cube",), len(FLAGS)).record(env, newly, p.flags)
        return p.success.clone()


def cube_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The lifted cube came back down on the table away from the target, or fell off the table."""
    cube = cube_pos(env)
    on_table = cube[:, 2] < CUBE_REST_Z + 0.005
    fell = cube[:, 2] < TABLE_TOP - 0.05
    return (progress(env).left_lift & on_table & (target_distance_xy(cube) > TARGET_RADIUS)) | fell


# ---------------------------------------------------------------- scene


def _arm(name: str, y: float, wrist_roll_offset: float = 0.0, x: float = -0.05, facing_back: bool = False):
    """One workshop SO-101 facing +x (or -x with ``facing_back``); ``wrist_roll_offset`` turns the jaws, e.g. to pinch
    across a bar lying along y."""
    joints = list(WORKSHOP_INITIAL_JOINT_POSITION)
    joints[JOINTS.index("wrist_roll")] += wrist_roll_offset
    return SO101_CFG.replace(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=SO101_CFG.init_state.replace(
            pos=(x, y, 0.0),
            # (x, y, z, w): +90 deg about z faces +x; facing_back is -90 deg (faces -x)
            rot=(0.0, 0.0, 0.7071068, -0.7071068) if facing_back else (0.0, 0.0, 0.7071068, 0.7071068),
            joint_pos=dict(zip(JOINTS, joints, strict=True)),
        ),
        soft_joint_pos_limit_factor=0.98,
    )


@configclass
class BiSceneCfg(InteractiveSceneCfg):
    # BITIDY_FACING_X (layout experiment): the arms face each other across the table, left at (-0.05, 0) facing +x,
    # right at (FACING_X, 0) facing -x.
    if FACING_X:
        robot_left = _arm("robot_left", 0.0)
        robot_right = _arm("robot_right", 0.0, x=FACING_X, facing_back=True)
    else:
        robot_left = _arm("robot_left", BASE_Y)
        robot_right = _arm("robot_right", -BASE_Y)

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

    desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, TABLE_TOP),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=5.0),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, TABLE_TOP / 2)),
    )

    target_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetPad",
        spawn=sim_utils.CuboidCfg(
            size=(2 * TARGET_RADIUS, 2 * TARGET_RADIUS, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*TARGET, TABLE_TOP + 0.0006)),
    )

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=500.0))

    right_fixed_jaw_cube = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot_right/gripper", filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"], history_length=2
    )
    right_moving_jaw_cube = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot_right/moving_jaw_so101_v1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=2,
    )


# ---------------------------------------------------------------- managers


def _arm_actions(arm: str):
    return (
        RelativeJointPositionActionCfg(
            asset_name=arm, joint_names=ARM_JOINTS, preserve_order=True, scale=0.033, use_zero_offset=True
        ),
        # Open/close in one step, as in the Isaac Lab lift tasks: a creeping gripper made grasps hard to discover.
        # BITIDY_GRIPPER=relative restores the old creeping gripper for policies trained with it.
        RelativeJointPositionActionCfg(asset_name=arm, joint_names=["gripper"], scale=0.02, use_zero_offset=True)
        if os.environ.get("BITIDY_GRIPPER") == "relative"
        else BinaryJointPositionActionCfg(
            asset_name=arm,
            joint_names=["gripper"],
            open_command_expr={"gripper": PREGRASP_GRIPPER_POSITION},
            close_command_expr={"gripper": GRIPPER_CLOSED},
        ),
    )


@configclass
class ActionsCfg:
    left_arm, left_gripper = _arm_actions("robot_left")
    right_arm, right_gripper = _arm_actions("robot_right")


@configclass
class PolicyCfg(ObsGroup):
    left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=JOINTS)})
    right_joint_pos = ObsTerm(
        func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=JOINTS)}
    )
    left_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=JOINTS)})
    right_joint_vel = ObsTerm(
        func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=JOINTS)}
    )
    left_joint_target = ObsTerm(
        func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=JOINTS)}
    )
    right_joint_target = ObsTerm(
        func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=JOINTS)}
    )
    previous_action = ObsTerm(func=mdp.last_action)
    pads = ObsTerm(func=pad_positions)
    cube = ObsTerm(func=cube_state)
    relative = ObsTerm(func=cube_relative)
    progress = ObsTerm(func=progress_flags)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.04, 0.04), "yaw": (-0.785, 0.785)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    replay = EventTerm(func=replay.replay_reset, mode="reset", params={"objects": ("cube",), "num_flags": len(FLAGS)})


@configclass
class RewardsCfg:
    reach_left = RewTerm(func=reach_progress, weight=1.0, params={"arm": "robot_left"})
    grasp_left = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_left"})
    carry_handover = RewTerm(func=carry_to_handover, weight=0.2)
    grasp_right = RewTerm(func=grasp_shaping, weight=0.5, params={"arm": "robot_right"})
    reach_right = RewTerm(func=reach_progress, weight=1.0, params={"arm": "robot_right"})
    carry_target = RewTerm(func=carry_to_target, weight=0.3)
    left_release = RewTerm(func=left_lets_go, weight=0.05)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["cube_dropped"]})
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
class TerminationsCfg:
    success = DoneTerm(func=handover_success)
    cube_dropped = DoneTerm(func=cube_dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiHandoverEnvCfg(ManagerBasedRLEnvCfg):
    scene: BiSceneCfg = BiSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()

    def play_mode(self):
        """Evaluate one episode per environment at the requested environment count."""
        requested = self.scene.num_envs
        super().play_mode()
        self.scene.num_envs = requested
        self.events.replay = None
