"""Print the bimanual scene geometry right after reset."""
import argparse

import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, _ = parser.parse_known_args()

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_tutorial.tasks.bimanual import env_cfg as bi, lid_env_cfg as lid

cfg = resolve_presets(lid.BiLidEnvCfg(), ["newton_mjwarp"])
cfg.scene.num_envs = 2
cfg.sim.device = args.device or "cuda:0"
with launch_simulation(cfg, args):
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    zero = torch.zeros((2, env.action_manager.total_action_dim), device=env.device)
    for _ in range(30):
        env.step(zero)
    torch.set_printoptions(precision=3, sci_mode=False)
    print("DIAG cube", bi.cube_pos(env)[0], "lid", lid.lid_pos(env)[0], "rest_z", lid.LID_REST_Z, "knob", lid.knob_pos(env)[0], "tilt", lid.lid_tilt(env)[0])
    for arm in bi.ARMS:
        r = env.scene[arm]
        print("DIAG", arm, "base", (r.data.root_pos_w.torch - env.scene.env_origins)[0], "pad", bi.pad_center(env, arm)[0],
              "dist", torch.linalg.vector_norm(bi.pad_center(env, arm) - bi.cube_pos(env), dim=-1)[0])
        print("DIAG", arm, "joints", r.data.joint_pos.torch[0], "names", r.joint_names)
    env.close()
