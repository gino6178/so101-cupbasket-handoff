"""NaN guard for long RL runs: an env whose physics blows up (NaN in its observations) is reset and reported as done,
instead of RSL-RL's NaN check aborting the whole (multi-node) training job.

    --external_callback isaaclab_tutorial.tasks.bimanual.nanguard.install_residual_guarded
"""

from __future__ import annotations

import os

import torch

# BITIDY_GUARD_MAX=<v>: also treat an env as blown up when any observation exceeds |v| (physics explosion before the
# NaN; one such sample inflates the running obs normalizer). Default off.
GUARD_MAX = float(os.environ.get("BITIDY_GUARD_MAX", "0"))


def install_guard() -> None:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    inner_step = RslRlVecEnvWrapper.step
    count = {"n": 0}

    def step(self, actions):
        obs, rew, dones, extras = inner_step(self, torch.nan_to_num(actions))
        bad = torch.zeros_like(dones, dtype=torch.bool)
        for group in obs.keys():
            bad |= ~torch.isfinite(obs[group]).all(-1)
            if GUARD_MAX > 0:
                bad |= (obs[group].abs() > GUARD_MAX).any(-1)
        if bool(bad.any()):
            ids = bad.nonzero().squeeze(-1)
            env = self.unwrapped
            env._reset_idx(ids)
            for group in obs.keys():
                obs[group] = torch.nan_to_num(obs[group], nan=0.0, posinf=0.0, neginf=0.0)
                if GUARD_MAX > 0:
                    obs[group][ids] = 0.0
            # the residual wrapper reads the BC-RNN input from obs_buf
            for group in list(env.obs_buf.keys()):
                env.obs_buf[group] = torch.nan_to_num(env.obs_buf[group], nan=0.0, posinf=0.0, neginf=0.0)
            dones = dones.clone()
            dones[ids] = 1
            count["n"] += len(ids)
            print(f"BITIDY_NANGUARD reset {len(ids)} envs (total {count['n']})", flush=True)
        return obs, torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0), dones, extras

    RslRlVecEnvWrapper.step = step


def install_residual_guarded() -> list[str]:
    install_guard()
    from .residual import install_residual

    return install_residual()
