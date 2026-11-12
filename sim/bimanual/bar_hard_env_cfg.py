"""Harder bimanual bar lift (T2): carry the bar over a low wall to one of three targets sampled per episode.

Same arms, bar and reward terms as :mod:`.bar_env_cfg`; this variant samples the target (front, front-left,
front-right, 14-15 cm away), puts a 4 cm wall between the start and every target, and raises the level-lift
threshold to 6 cm so the bar clears the wall. The target is visible to the policy through the bar-to-target vector
and, later, is chosen by the VLA from the spoken instruction.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import WORKSHOP_CONTACT_MATERIAL

from . import bar_env_cfg as bar
from . import replay
from .env_cfg import TABLE_TOP

TARGETS = ((0.29, 0.0), (0.28, 0.06), (0.28, -0.06))
"""Front, front-left and front-right bar-centre targets, 14-15 cm from the start."""
HARD_LIFT = 0.06
HARD_START = (0.14, 0.0)
"""The bar starts closer to the arms so the wall sits in open space, ~5 cm from both the start and the targets."""
WALL_X, WALL_SIZE = 0.21, (0.02, 0.36, 0.04)


def _ids(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def sample_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Pick one of the targets for each resetting environment."""
    if not hasattr(env, "_bar_target"):
        env._bar_target = env.scene.env_origins.new_zeros((env.num_envs, 2))
        env._bar_lift = HARD_LIFT
        env._bi_extra = env._bar_target
    ids = _ids(env, env_ids)
    choice = torch.randint(0, len(TARGETS), (len(ids),), device=env.device)
    env._bar_target[ids] = env._bar_target.new_tensor(TARGETS)[choice]


def place_target_pad(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Move the green pad under the (possibly replay-restored) target."""
    ids = _ids(env, env_ids)
    pose = torch.zeros((len(ids), 7), device=env.device)
    pose[:, :2] = env._bar_target[ids]
    pose[:, 2] = TABLE_TOP + 0.0006
    pose[:, 6] = 1.0
    pose[:, :3] += env.scene.env_origins[ids]
    env.scene["target_pad"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)


@configclass
class BarHardSceneCfg(bar.BarSceneCfg):
    target_pad = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPad",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.32, 0.001),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*TARGETS[0], TABLE_TOP + 0.0006)),
    )
    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=WALL_SIZE,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.4)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(WALL_X, 0.0, TABLE_TOP + WALL_SIZE[2] / 2)),
    )


@configclass
class BarHardEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_bar = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.0), "y": (-0.04, 0.04), "yaw": (-0.3, 0.3)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bar"),
        },
    )
    target = EventTerm(func=sample_target, mode="reset")
    replay = EventTerm(
        func=replay.replay_reset, mode="reset", params={"objects": ("bar",), "num_flags": len(bar.FLAGS)}
    )
    pad = EventTerm(func=place_target_pad, mode="reset")


@configclass
class BiBarLiftHardEnvCfg(bar.BiBarLiftEnvCfg):
    scene: BarHardSceneCfg = BarHardSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    events: BarHardEventsCfg = BarHardEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.bar.init_state.pos = (*HARD_START, bar.BAR_REST_Z)
