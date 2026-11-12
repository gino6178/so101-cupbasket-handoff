"""Episode accounting for ``isaaclab play``.

``--external_callback isaaclab_tutorial.tasks.bimanual.evaluation.install_episode_counter`` stops play once every
environment has finished its first episode and prints one ``BITIDY_EVAL_RESULT`` JSON line with milestone rates.
"""

from __future__ import annotations

import json
import sys
import os

import torch


def install_episode_counter() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original_step = RslRlVecEnvWrapper.step
    state = {}

    def step(self, actions):
        result = original_step(self, actions)
        env = self.unwrapped
        dones = result[2].bool()
        FLAGS = env._bi_flag_names
        if not state:
            state["done"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            state["final"] = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
            state["steps"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        state["steps"] += (~state["done"]).long()
        first = dones & ~state["done"]
        state["final"][first] = env._bi_final[first]
        state["done"] |= first
        if bool(state["done"].all()):
            rates = state["final"].float().mean(0).tolist()
            succeeded = state["final"][:, -1]
            result_json = {
                "episodes": env.num_envs,
                "success_rate": rates[-1],
                "successes": int(succeeded.sum()),
                **{f"{name}_rate": rate for name, rate in zip(FLAGS, rates[:-1], strict=True)},
                "mean_success_time_s": float(state["steps"][succeeded].float().mean() * env.step_dt)
                if bool(succeeded.any())
                else None,
            }
            print("BITIDY_EVAL_RESULT=" + json.dumps(result_json), flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
