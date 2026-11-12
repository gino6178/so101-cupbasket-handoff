"""Lidded box, both arms (LidPlace): the LEFT arm takes the lid off the box by its knob and holds it aside, level,
toward +y; the RIGHT arm picks the cube from its random start in front of it and drops it into the open box, then backs
off clear; the left arm puts the lid back on the box and lets go.

Merges the two verified halves without changing their geometry: the scene (box, lid with its turned knob, cube) is
``lid_open_env_cfg``'s, the cube start square and the "cube inside" test are ``box_place_env_cfg``'s.

Success: the lid went off and aside, the cube went in while the lid was clear of the opening (no reaching through a
half-open lid), the right jaws backed off clear of the box, the lid is back closed on the box, the left jaws let go of
the knob, and the cube rests inside, all held for ``STABLE_STEPS``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from . import box_place_env_cfg as bp
from . import env_cfg as base
from . import lid_open_env_cfg as lo
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, gripper_openness, held_by, pad_center

# Geometry shared with the two halves (one source of truth).
BOX, LID_REST_Z, KNOB_OFFSET, KNOB_YAW = lo.BOX, lo.LID_REST_Z, lo.KNOB_OFFSET, lo.KNOB_YAW
RIM_Z = bp.RIM_Z
ASIDE_CLEAR = lo.ASIDE_CLEAR
CUBE_LIFT = bp.LIFT
STABLE_STEPS = 10
EPISODE_S = float(os.environ.get("BITIDY_LID_PLACE_EPISODE_S", "40"))
FLAGS = ("lid_grasp", "lid_lifted", "lid_aside", "cube_grasp", "cube_lift", "cube_over", "cube_in", "right_clear",
         "lid_back")

box_center = lo.box_center
lid_pos, lid_quat, lid_tilt, knob_pos = lo.lid_pos, lo.lid_quat, lo.lid_tilt, lo.knob_pos
lid_held, lid_lifted, lid_aside, lid_closed = lo.lid_held, lo.lid_lifted, lo.lid_aside, lo.lid_closed
left_jaws_off = lo.jaws_off
cube_pos = base.cube_pos
cube_over_opening, cube_resting_inside = bp.cube_over_opening, bp.cube_resting_inside
right_jaws_clear = bp.jaws_clear


def lid_off_opening(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The lid's footprint is clear of the opening (wherever it is, held or not)."""
    return torch.linalg.vector_norm(lid_pos(env)[:, :2] - box_center(env), dim=-1) > ASIDE_CLEAR


# ---------------------------------------------------------------- progress


class LidPlaceProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.lid_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cube_count = torch.zeros_like(self.lid_count)
        self.stable_count = torch.zeros_like(self.lid_count)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.lid_count[ids] = 0
        self.cube_count[ids] = 0
        self.stable_count[ids] = 0
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv) -> None:
        # Lid (left arm), as in LidOpen.
        self.lid_count = torch.where(lid_held(env), self.lid_count + 1, 0)
        lid_grasp = self.lid_count >= 2
        lifted = self.lid_grasp & lid_lifted(env)
        aside = self.lid_lifted & lid_aside(env)
        # Cube (right arm), as in BoxPlace, but it only counts as over / in the box while the lid is off the opening.
        cube = cube_pos(env)
        held = held_by(env, "robot_right")
        self.cube_count = torch.where(held, self.cube_count + 1, 0)
        cube_grasp = self.cube_count >= 2
        cube_lift = self.cube_grasp & held & (cube[:, 2] > CUBE_REST_Z + CUBE_LIFT)
        lid_off = lid_off_opening(env)
        over = self.cube_lift & held & cube_over_opening(env) & (cube[:, 2] - CUBE_SIZE / 2 > RIM_Z) & lid_off
        released = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube, dim=-1) > 0.03
        resting = cube_resting_inside(env)
        cube_in = self.cube_over & resting & released & lid_off
        right_clear = self.cube_in & resting & right_jaws_clear(env)
        closed = lid_closed(env)
        back = self.lid_aside & closed
        self.flags |= torch.stack((lid_grasp, lifted, aside, cube_grasp, cube_lift, over, cube_in, right_clear, back), -1)
        done = (self.lid_back & self.right_clear & closed & resting & left_jaws_off(env) & right_jaws_clear(env))
        self.stable_count = torch.where(done, self.stable_count + 1, 0)
        self.success |= self.stable_count >= STABLE_STEPS


def progress(env: ManagerBasedRLEnv) -> LidPlaceProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = LidPlaceProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- events


def reset_objects(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: the lid settled on the box (LidOpen's jitter), the cube anywhere in BoxPlace's start square."""
    lo.reset_objects(env, env_ids)
    bp.sample_task(env, env_ids)


# ---------------------------------------------------------------- observations (privileged state: oracle/demo task)


def objects_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    lid = env.scene["lid"]
    return torch.cat((lid_pos(env), lid_quat(env), lid.data.root_lin_vel_w.torch, base.cube_state(env)), -1)


def objects_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    n = env.num_envs
    box_lid = torch.cat((box_center(env), torch.full((n, 1), LID_REST_Z, device=env.device)), -1)
    box_floor = torch.cat((box_center(env), torch.full((n, 1), TABLE_TOP, device=env.device)), -1)
    cube = cube_pos(env)
    return torch.cat((knob_pos(env) - pad_center(env, "robot_left"), lid_pos(env) - box_lid,
                      cube - pad_center(env, "robot_right"), cube - box_floor), -1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards


def reach_knob(env: ManagerBasedRLEnv) -> torch.Tensor:
    d = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1)
    return torch.where(progress(env).lid_lifted, 1.0, 1.0 - torch.tanh(d / 0.08))


def reach_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    d = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube_pos(env), dim=-1)
    return torch.where(progress(env).cube_grasp, 1.0, 1.0 - torch.tanh(d / 0.08))


def lid_aside_shaping(env: ManagerBasedRLEnv, std: float = 0.04) -> torch.Tensor:
    """Lid in hand until the cube is in: pull it toward the aside spot and keep it level."""
    p = progress(env)
    spot = torch.cat((box_center(env) + box_center(env).new_tensor(lo.ASIDE_DIR) * (ASIDE_CLEAR + 0.01),
                      torch.full((env.num_envs, 1), LID_REST_Z + 0.05, device=env.device)), -1)
    d = torch.linalg.vector_norm(lid_pos(env) - spot, dim=-1)
    active = lid_held(env) & p.lid_grasp & ~p.cube_in
    return torch.where(active, torch.exp(-d / std) * torch.exp(-lid_tilt(env) / lo.LEVEL_TILT), 0.0)


def carry_cube(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    """Cube in hand: toward the spot above the opening, once the lid is off it."""
    above = torch.cat((box_center(env), torch.full((env.num_envs, 1), RIM_Z + 0.03, device=env.device)), -1)
    d = torch.linalg.vector_norm(cube_pos(env) - above, dim=-1)
    active = held_by(env, "robot_right") & ~progress(env).cube_in & lid_off_opening(env)
    return torch.where(active, torch.exp(-d / std), 0.0)


def close_lid(env: ManagerBasedRLEnv, std: float = 0.03) -> torch.Tensor:
    rest = torch.cat((box_center(env), torch.full((env.num_envs, 1), LID_REST_Z, device=env.device)), -1)
    d = torch.linalg.vector_norm(lid_pos(env) - rest, dim=-1)
    return torch.where(progress(env).right_clear, torch.exp(-d / std) * torch.exp(-lid_tilt(env) / 0.2), 0.0)


def release_both(env: ManagerBasedRLEnv) -> torch.Tensor:
    p = progress(env)
    left = (torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1) / 0.05).clamp(0.0, 1.0)
    right = (torch.linalg.vector_norm(pad_center(env, "robot_right")[:, :2] - box_center(env), dim=-1) / bp.CLEAR_XY).clamp(0.0, 1.0)
    return torch.where(p.cube_in, 0.5 * right, 0.0) + torch.where(p.lid_back & lid_closed(env), 0.5 * left, 0.0)


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 3.0, 6.0)

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


class lid_place_success(ManagerTermBase):
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


# ---------------------------------------------------------------- config


@configclass
class LidPlaceSceneCfg(lo.LidOpenSceneCfg):
    pass


@configclass
class LidPlaceActionsCfg:
    left_arm, left_gripper = base._arm_actions("robot_left")
    right_arm, right_gripper = base._arm_actions("robot_right")


@configclass
class LidPlacePolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=objects_state)
    relative = ObsTerm(func=objects_relative)
    progress = ObsTerm(func=progress_flags)


@configclass
class LidPlaceObservationsCfg:
    policy: LidPlacePolicyCfg = LidPlacePolicyCfg()
    critic: LidPlacePolicyCfg = LidPlacePolicyCfg()


@configclass
class LidPlaceEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    objects = EventTerm(func=reset_objects, mode="reset")


@configclass
class LidPlaceRewardsCfg:
    reach_knob = RewTerm(func=reach_knob, weight=1.0)
    reach_cube = RewTerm(func=reach_cube, weight=1.0)
    lid_aside = RewTerm(func=lid_aside_shaping, weight=0.5)
    carry = RewTerm(func=carry_cube, weight=0.5)
    close = RewTerm(func=close_lid, weight=0.5)
    release = RewTerm(func=release_both, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_right")})
    left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_right")})


@configclass
class LidPlaceTerminationsCfg:
    success = DoneTerm(func=lid_place_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LidPlaceEnvCfg(base.BiHandoverEnvCfg):
    scene: LidPlaceSceneCfg = LidPlaceSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    actions: LidPlaceActionsCfg = LidPlaceActionsCfg()
    observations: LidPlaceObservationsCfg = LidPlaceObservationsCfg()
    events: LidPlaceEventsCfg = LidPlaceEventsCfg()
    rewards: LidPlaceRewardsCfg = LidPlaceRewardsCfg()
    terminations: LidPlaceTerminationsCfg = LidPlaceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S

    def play_mode(self):
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
