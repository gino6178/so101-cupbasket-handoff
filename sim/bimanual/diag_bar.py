"""Print each arm's jaw-closing direction in the bar-lift scene (should be ~x, across the bar)."""
import argparse

import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, _ = parser.parse_known_args()

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_tutorial.tasks.bimanual import bar_env_cfg as bar
from isaaclab_tutorial.tasks.place_vial.mdp.task import FIXED_FINGERTIP_OFFSET, MOVING_FINGERTIP_OFFSET

import math
cfg = resolve_presets(bar.BiBarLiftEnvCfg(), ["newton_mjwarp"])
cfg.scene.num_envs = 2
ROLLS = [-0.804 + k * math.pi / 4 for k in (-3, -2, -1, 0, 1, 2, 3, 4)]
cfg.sim.device = args.device or "cuda:0"
with launch_simulation(cfg, args):
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    torch.set_printoptions(precision=3, sci_mode=False)
    for roll in ROLLS:
      for arm in ("robot_left",):
        robot = env.scene[arm]
        q = robot.data.joint_pos.torch.clone()
        q[:, robot.joint_names.index("wrist_roll")] = roll
        robot.write_joint_position_to_sim_index(position=q)
        robot.set_joint_position_target_index(target=q)
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        robot = env.scene[arm]
        pads = []
        for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
            i = robot.find_bodies(body)[0][0]
            pos, quat = robot.data.body_pos_w.torch[:, i], robot.data.body_quat_w.torch[:, i]
            pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
        axis = pads[1] - pads[0]
        print("DIAG roll", round(roll, 3), arm, "pinch axis", (axis / axis.norm(dim=-1, keepdim=True))[0], "gap", axis.norm(dim=-1)[0])
    env.close()
