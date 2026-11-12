"""Collect T1 handover start states for the receiver task.

``isaaclab play --task BiTidy-Handover-SO101 --checkpoint <T1 policy> --external_callback
isaaclab_tutorial.tasks.bimanual.collect.install_state_dump`` runs the full-task policy; the T1 termination term
already records the physical state whenever an episode first brings the cube into the handover zone in the left
jaws. Once ``$BITIDY_STATES_N`` states are stored (or after ``$BITIDY_STATES_STEPS`` steps) they are saved to
``$BITIDY_HANDOVER_STATES``.
"""

from __future__ import annotations

import os
import sys

import torch


def install_state_dump() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    path = os.environ.get("BITIDY_HANDOVER_STATES", os.path.expanduser("~/handover_states.pt"))
    want = int(os.environ.get("BITIDY_STATES_N", "8192"))
    max_steps = int(os.environ.get("BITIDY_STATES_STEPS", "1500"))
    original_step = RslRlVecEnvWrapper.step
    state = {"steps": 0}

    def step(self, actions):
        result = original_step(self, actions)
        env = self.unwrapped
        state["steps"] += 1
        buf = getattr(env, "_bi_replay", None)
        size = 0 if buf is None else buf.size
        if state["steps"] % 100 == 0:
            print(f"[collect] step={state['steps']} states={size}", flush=True)
        if size >= want or state["steps"] >= max_steps:
            torch.save(
                {"size": size, "joint": {a: t[:size].cpu() for a, t in buf.joint.items()},
                 "root": {o: t[:size].cpu() for o, t in buf.root.items()}, "flags": buf.flags[:size].cpu()},
                path,
            )
            print(f"BITIDY_STATES={path} n={size}", flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
