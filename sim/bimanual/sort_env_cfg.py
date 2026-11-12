"""Voice-commanded zone sorting: one SO-101 arm moves the cube between two front zones.

Three zones lie in front of the two arms: A (in front of the left arm), B (between them) and C (in front of the right
arm). Every episode samples a source and a target zone among the acting arm's two zones (left: A/B, right: B/C),
drops the cube at a random point in the source zone, and asks for it in the target zone. A to C and C to A need both
arms (handover), which is a separate skill.

Observations are limited to what the real cell measures: the acting arm's joint positions and commanded targets, the
last action, the cube position and yaw (overhead camera) and the target zone centre, so the policy transfers with a
calibrated camera and no simulator-only state. Success: the cube was lifted, then rests released inside the target
zone for ``SETTLE_STEPS`` steps.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from . import env_cfg as base

# Every zone lies 0.22-0.28 m from the arms that serve it, where the scripted oracle grasps best: beyond ~0.28 m a top
# grasp fails, and a zone straight in front of a base at 0.20 m forces a cramped wrist. Staggered layout: oracle
# success 81-97% in all four directions (in-line layout at x = 0.15: 65-94%).
ZONES = {"A": (0.20, 0.13), "B": (0.13, 0.0), "C": (0.20, -0.13)}
if os.environ.get("BITIDY_ZONES"):  # "ax,ay;bx,by;cx,cy" (layout experiments)
    ZONES = dict(zip("ABC", (tuple(map(float, z.split(","))) for z in os.environ["BITIDY_ZONES"].split(";"))))
ZONE_HALF = 0.035
SPAWN_HALF = 0.02
LIFT = 0.03
SETTLE_STEPS = 10
FLAGS = ("grasp", "lift", "over_target", "placed")
EPISODE_S = float(os.environ.get("BITIDY_SORT_EPISODE_S", "20"))
HANDOVER_EPISODE_S = float(os.environ.get("BITIDY_HANDOVER_EPISODE_S", "30"))


def _arm_list(arm) -> tuple[str, ...]:
    """Terms take one arm name or, for the two-arm handover skill, a sequence of them."""
    return (arm,) if isinstance(arm, str) else tuple(arm)


def _pad_dist(env: ManagerBasedRLEnv, arm) -> torch.Tensor:
    """Distance from the cube to the nearest of the arms' pad midpoints."""
    cube = base.cube_pos(env)
    return torch.stack([torch.linalg.vector_norm(base.pad_center(env, a) - cube, dim=-1) for a in _arm_list(arm)]).min(0).values


def _held(env: ManagerBasedRLEnv, arm) -> torch.Tensor:
    return torch.stack([base.held_by(env, a) for a in _arm_list(arm)]).any(0)


def _zones(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene.env_origins.new_tensor([ZONES[k] for k in env._sort_zone_names])


def target_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Observation shapes are probed before the first reset event sets the targets.
    if not hasattr(env, "_sort_target"):
        env._sort_target = env.scene.env_origins.new_zeros((env.num_envs, 2))
    return env._sort_target


def in_target(env: ManagerBasedRLEnv, cube: torch.Tensor) -> torch.Tensor:
    return ((cube[:, :2] - target_xy(env)).abs() < ZONE_HALF).all(-1)


def sample_task(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, zones: tuple[str, str]) -> None:
    """Reset event: pick source/target zones, place the cube at a random point (and yaw) in the source zone."""
    env._sort_zone_names = zones
    target_xy(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    n = len(ids)
    centers = _zones(env)
    src = torch.randint(0, 2, (n,), device=env.device)
    if os.environ.get("BITIDY_SORT_SRC"):  # evaluate one direction: 0 = first zone is the source
        src[:] = int(os.environ["BITIDY_SORT_SRC"])
    dst = 1 - src
    env._sort_target[ids] = centers[dst]
    pos = centers[src] + (torch.rand((n, 2), device=env.device) * 2 - 1) * SPAWN_HALF
    yaw = (torch.rand(n, device=env.device) * 2 - 1) * (torch.pi / 4)
    pose = torch.zeros((n, 7), device=env.device)
    pose[:, :2] = pos
    pose[:, 2] = base.CUBE_REST_Z
    pose[:, 5] = torch.sin(yaw / 2)
    pose[:, 6] = torch.cos(yaw / 2)
    pose[:, :3] += env.scene.env_origins[ids]
    cube = env.scene["cube"]
    cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=env.device), env_ids=ids)


# ---------------------------------------------------------------- progress


class SortProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.settle = torch.zeros_like(self.hold)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.hold[ids] = 0
        self.settle[ids] = 0
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv, arm: str) -> None:
        cube = base.cube_pos(env)
        held = _held(env, arm)
        self.hold = torch.where(held, self.hold + 1, 0)
        lifted = held & (cube[:, 2] > base.CUBE_REST_Z + LIFT)
        over = self.flags[:, 1] & held & in_target(env, cube)
        released = _pad_dist(env, arm) > 0.03
        speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
        resting = in_target(env, cube) & (cube[:, 2] < base.CUBE_REST_Z + 0.006) & (speed < 0.05)
        placed = self.flags[:, 1] & resting & released
        self.flags |= torch.stack((self.hold >= 2, lifted, over, placed), dim=-1)
        self.settle = torch.where(placed, self.settle + 1, 0)
        self.success |= self.settle >= SETTLE_STEPS


def progress(env: ManagerBasedRLEnv) -> SortProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = SortProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- observations (real-cell measurable)


def _cube_measurement(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube position and yaw folded to its 90-degree symmetry (sin/cos of 4*yaw)."""
    cube = base.cube_pos(env)
    x, y, z, w = env.scene["cube"].data.root_quat_w.torch.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.cat((cube, torch.sin(4 * yaw).unsqueeze(-1), torch.cos(4 * yaw).unsqueeze(-1)), dim=-1)


def cube_obs(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    """What the overhead camera reports: the cube pose, refreshed only while it rests on the table unoccluded.

    Above the table (in the jaws) or with the jaws over it, a top-down camera cannot measure the cube, so the last
    resting pose is held; the policy tracks a carried cube through its own joints, as on the real cell.
    """
    current = _cube_measurement(env)
    if not hasattr(env, "_cube_seen"):
        env._cube_seen = current.clone()
    cube = current[:, :3]
    resting = cube[:, 2] < base.CUBE_REST_Z + 0.004
    clear = _pad_dist(env, arm) > 0.05
    steps = getattr(env, "episode_length_buf", None)
    just_reset = torch.ones_like(clear) if steps is None else steps <= 1
    fresh = (resting & clear) | just_reset
    env._cube_seen = torch.where(fresh.unsqueeze(-1), current, env._cube_seen)
    return env._cube_seen.clone()


def target_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    seen = getattr(env, "_cube_seen", None)
    cube = base.cube_pos(env) if seen is None else seen
    return torch.cat((target_xy(env), cube[:, :2] - target_xy(env)), dim=-1)


# ---------------------------------------------------------------- rewards


def reach(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    d = _pad_dist(env, arm)
    return torch.where(progress(env).lift, 1.0, 1.0 - torch.tanh(d / 0.08))


def grasp(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    d = _pad_dist(env, arm)
    cube = base.cube_pos(env)
    closing = torch.stack([
        torch.where(torch.linalg.vector_norm(base.pad_center(env, a) - cube, dim=-1) < 0.02, 1.0 - base.gripper_openness(env, a), 0.0)
        for a in _arm_list(arm)
    ]).max(0).values
    lift = ((base.cube_pos(env)[:, 2] - base.CUBE_REST_Z) / LIFT).clamp(0.0, 1.0)
    return torch.where((d < 0.02) & ~progress(env).placed, 0.3 * closing + lift, 0.0)


def carry(env: ManagerBasedRLEnv, arm: str, std: float) -> torch.Tensor:
    cube = base.cube_pos(env)
    d = torch.linalg.vector_norm(cube[:, :2] - target_xy(env), dim=-1)
    carrying = _held(env, arm) & (cube[:, 2] > base.CUBE_REST_Z + 0.01)
    return torch.where(carrying | progress(env).placed, 1.0 - torch.tanh(d / std), 0.0)


_carry = carry  # the Rewards config below defines an attribute named ``carry``


def put_down(env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
    """Over the target: pay for lowering the cube, then for letting go."""
    cube = base.cube_pos(env)
    p = progress(env)
    inside = in_target(env, cube) & p.lift
    low = 1.0 - ((cube[:, 2] - base.CUBE_REST_Z) / 0.05).clamp(0.0, 1.0)
    away = (_pad_dist(env, arm) / 0.06).clamp(0.0, 1.0)
    return torch.where(inside, low + torch.where(low > 0.9, away, 0.0), 0.0)


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 2.0, 3.0, 5.0)

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


class sort_success(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        progress(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        p = progress(self._env)
        self._env._bi_final[ids] = torch.cat((p.flags[ids], p.success[ids].unsqueeze(-1)), dim=-1)
        p.reset(ids)

    def __call__(self, env: ManagerBasedRLEnv, arm: str) -> torch.Tensor:
        p = progress(env)
        p.update(env, arm)
        # Videos keep the episode running after success, so the placed cube stays in view (success is still recorded).
        if os.environ.get("BITIDY_NO_SUCCESS_END"):
            return torch.zeros_like(p.success)
        return p.success.clone()


def cube_off_table(env: ManagerBasedRLEnv) -> torch.Tensor:
    return base.cube_pos(env)[:, 2] < base.TABLE_TOP - 0.03


# ---------------------------------------------------------------- config builders


@configclass
class SortSceneCfg(base.BiSceneCfg):
    target_pad = None
    zone_a = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneA",
        spawn=sim_utils.CuboidCfg(size=(2 * ZONE_HALF, 2 * ZONE_HALF, 0.001),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.25))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*ZONES["A"], base.TABLE_TOP + 0.0006)),
    )
    zone_b = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneB",
        spawn=sim_utils.CuboidCfg(size=(2 * ZONE_HALF, 2 * ZONE_HALF, 0.001),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.7, 0.3))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*ZONES["B"], base.TABLE_TOP + 0.0006)),
    )
    zone_c = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneC",
        spawn=sim_utils.CuboidCfg(size=(2 * ZONE_HALF, 2 * ZONE_HALF, 0.001),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.4, 0.85))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*ZONES["C"], base.TABLE_TOP + 0.0006)),
    )


def _policy_group(arm: str):
    @configclass
    class SortPolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg(arm, joint_names=base.JOINTS)})
        joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg(arm, joint_names=base.JOINTS)})
        previous_action = ObsTerm(func=mdp.last_action)
        cube = ObsTerm(func=cube_obs, params={"arm": arm})
        target = ObsTerm(func=target_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    return SortPolicyCfg


def make_sort_cfg(arm: str, zones: tuple[str, str]):
    policy_cls = _policy_group(arm)

    @configclass
    class Actions:
        arm_joints, gripper = base._arm_actions(arm)

    @configclass
    class Observations:
        policy: policy_cls = policy_cls()
        critic: policy_cls = policy_cls()

    @configclass
    class Events:
        reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
        task = EventTerm(func=sample_task, mode="reset", params={"zones": zones})

    @configclass
    class Rewards:
        reach = RewTerm(func=reach, weight=1.0, params={"arm": arm})
        grasp = RewTerm(func=grasp, weight=1.0, params={"arm": arm})
        carry = RewTerm(func=carry, weight=3.0, params={"arm": arm, "std": 0.1})
        carry_fine = RewTerm(func=_carry, weight=2.0, params={"arm": arm, "std": 0.02})
        put_down = RewTerm(func=put_down, weight=2.0, params={"arm": arm})
        milestones = RewTerm(func=milestones, weight=10.0)
        success = RewTerm(func=success, weight=200.0)
        action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
        joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg(arm)})
        limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg(arm)})

    @configclass
    class Terminations:
        success = DoneTerm(func=sort_success, params={"arm": arm})
        off_table = DoneTerm(func=cube_off_table)
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    @configclass
    class SortEnvCfg(base.BiHandoverEnvCfg):
        scene: SortSceneCfg = SortSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
        actions: Actions = Actions()
        observations: Observations = Observations()
        events: Events = Events()
        rewards: Rewards = Rewards()
        terminations: Terminations = Terminations()

        def __post_init__(self):
            super().__post_init__()
            self.episode_length_s = EPISODE_S

        def play_mode(self):
            requested = self.scene.num_envs
            base.ManagerBasedRLEnvCfg.play_mode(self)
            self.scene.num_envs = requested

    return SortEnvCfg


def make_handover_cfg():
    """S3: A to C or C to A. The source zone is out of the other arm's reach, so the arms must hand the cube over."""
    arms = ("robot_left", "robot_right")

    @configclass
    class HandoverPolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=base.JOINTS)})
        left_joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=base.JOINTS)})
        right_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=base.JOINTS)})
        right_joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=base.JOINTS)})
        previous_action = ObsTerm(func=mdp.last_action)
        cube = ObsTerm(func=cube_obs, params={"arm": arms})
        target = ObsTerm(func=target_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Actions:
        left_arm, left_gripper = base._arm_actions("robot_left")
        right_arm, right_gripper = base._arm_actions("robot_right")

    @configclass
    class Observations:
        policy: HandoverPolicyCfg = HandoverPolicyCfg()
        critic: HandoverPolicyCfg = HandoverPolicyCfg()

    @configclass
    class Events:
        reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
        task = EventTerm(func=sample_task, mode="reset", params={"zones": ("A", "C")})

    @configclass
    class Rewards:
        reach = RewTerm(func=reach, weight=1.0, params={"arm": arms})
        grasp = RewTerm(func=grasp, weight=1.0, params={"arm": arms})
        carry = RewTerm(func=carry, weight=3.0, params={"arm": arms, "std": 0.1})
        carry_fine = RewTerm(func=_carry, weight=2.0, params={"arm": arms, "std": 0.02})
        put_down = RewTerm(func=put_down, weight=2.0, params={"arm": arms})
        milestones = RewTerm(func=milestones, weight=10.0)
        success = RewTerm(func=success, weight=200.0)
        action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
        left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
        right_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_right")})
        left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})
        right_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_right")})

    @configclass
    class Terminations:
        success = DoneTerm(func=sort_success, params={"arm": arms})
        off_table = DoneTerm(func=cube_off_table)
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    @configclass
    class SortHandoverEnvCfg(base.BiHandoverEnvCfg):
        scene: SortSceneCfg = SortSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
        actions: Actions = Actions()
        observations: Observations = Observations()
        events: Events = Events()
        rewards: Rewards = Rewards()
        terminations: Terminations = Terminations()

        def __post_init__(self):
            super().__post_init__()
            self.episode_length_s = HANDOVER_EPISODE_S

        def play_mode(self):
            requested = self.scene.num_envs
            base.ManagerBasedRLEnvCfg.play_mode(self)
            self.scene.num_envs = requested

    return SortHandoverEnvCfg


SortLeftEnvCfg = make_sort_cfg("robot_left", ("A", "B"))
SortRightEnvCfg = make_sort_cfg("robot_right", ("B", "C"))
SortHandoverEnvCfg = make_handover_cfg()
