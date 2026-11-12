"""Residual RL on a frozen BC-RNN (ResiP-style): PPO learns a correction added to the imitation policy's action.

    BITIDY_BCRNN=<bcrnn.pt> [BITIDY_RES_SCALE=0.5] isaaclab train/play ... \
        --external_callback isaaclab_tutorial.tasks.bimanual.residual.install_residual

Every step the frozen BC-RNN (``bcrnn.BCRNNScript``) proposes a base action from the policy observation; the executed
action is ``base + BITIDY_RES_SCALE * residual``. The base action is appended to the "policy" and "critic" observation
groups, so the PPO networks see what they correct. The actor's output layer starts at zero, so training starts from
the BC-RNN's own success rate. Works for any task the BC-RNN was trained on (LidPlace, LidSort).

``BITIDY_DS_SNAPS=<snaps.pt>`` (LidSort, training only) adds the DemoStart reset curriculum of ``demo_start.py``.
"""

from __future__ import annotations

import os
import sys

import torch


def make_base(env):
    """Frozen base policy: the BC-RNN, or a scripted oracle given as BITIDY_RES_BASE=<module>:<Class> (same interface)."""
    spec = os.environ.get("BITIDY_RES_BASE")
    if not spec:
        from .bcrnn import BCRNNScript

        return BCRNNScript(env)
    import importlib

    mod, cls = spec.split(":")
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)(env)


def install_residual() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.models.mlp_model import MLPModel

    from .bcrnn import BCRNNScript

    scale = float(os.environ.get("BITIDY_RES_SCALE", "0.5"))
    state: dict = {}
    # Evaluation / video (play): the counter or recorder wraps the env step underneath the residual.
    if os.environ.get("BITIDY_VIDEO"):
        from . import render

        render.install_recorder()
    elif os.environ.get("BITIDY_RES_EVAL"):
        from . import evaluation

        evaluation.install_episode_counter()
    elif os.environ.get("BITIDY_DS_SNAPS"):
        # DemoStart curriculum (training only): some resets start from recorded oracle states (demo_start.py).
        from . import demo_start

        demo_start.install_curriculum()

    # Zero the actor's last layer: residual 0 at the start, exploration only from the action std.
    model_init = MLPModel.__init__

    def init(self, *args, **kwargs):
        model_init(self, *args, **kwargs)
        if self.distribution is not None:
            last = [m for m in self.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

    MLPModel.__init__ = init

    if os.environ.get("BITIDY_FREEZE_NORM"):
        # Fine-tuning (e.g. under physics randomization): keep the resumed obs normalizer fixed, so rare outliers cannot
        # rescale the actor's inputs.
        MLPModel.update_normalization = lambda self, obs: None
        print("BITIDY_FREEZE_NORM: obs normalizer frozen", flush=True)

    inner_obs, inner_step = RslRlVecEnvWrapper.get_observations, RslRlVecEnvWrapper.step

    def base_action(env):
        if "bc" not in state:
            state["bc"] = make_base(env)
            state["base"] = torch.nan_to_num(state["bc"].act()).clamp(-1.0, 1.0)
        return state["base"]

    def augment(obs, base):
        for group in ("policy", "critic"):
            if group in obs.keys():
                obs[group] = torch.cat((obs[group], base), -1)
        return obs

    def get_observations(self):
        base = base_action(self.unwrapped)
        return augment(inner_obs(self), base)

    def step(self, actions):
        env = self.unwrapped
        base = base_action(env)
        if "mask" not in state:
            # Binary grippers (one-step open/close on the sign) stay with the BC-RNN: a small residual flips them.
            state["mask"] = torch.ones(actions.shape[-1], device=actions.device)
            state["mask"][[int(i) for i in os.environ.get("BITIDY_RES_FIXED", "5,11").split(",") if i]] = 0.0
        executed = base + scale * state["mask"] * actions
        if "retreat" not in state:
            from . import retreat

            state["retreat"] = retreat.RetreatFilter(env) if retreat.enabled() else None
            state["n"] = 0
        if state["retreat"] is not None:
            # BITIDY_RETREAT: open fully, then lift straight up after a release near the goal (same as ResidualScript)
            executed = state["retreat"](executed)
            state["n"] += 1
            if state["n"] % 2000 == 0:
                print(f"BITIDY_RETREAT {state['retreat'].count}", flush=True)
        obs, rew, dones, extras = inner_step(self, executed)
        done_ids = dones.nonzero().squeeze(-1)
        if len(done_ids):
            state["bc"].reset(done_ids)
            if state["retreat"] is not None:
                state["retreat"].reset(done_ids)
        # for the next observation (reset envs already restarted); a blown-up env must not leak NaN into the obs
        state["base"] = torch.nan_to_num(state["bc"].act()).clamp(-1.0, 1.0)
        return augment(obs, state["base"]), rew, dones, extras

    RslRlVecEnvWrapper.get_observations = get_observations
    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


class ResidualScript:
    """BC-RNN + trained residual actor as one executor (scripted-oracle interface), e.g. for voice_sort chains.

    BITIDY_BCRNN=<bcrnn.pt> BITIDY_RESIDUAL=<rsl_rl model_*.pt> [BITIDY_RES_SCALE] [BITIDY_RES_FIXED]; deterministic
    (the actor mean). Mirrors the actor of agents.BiHandoverPPORunnerCfg: obs normalizer, ELU MLP.
    """

    def __init__(self, env):
        self.env = env
        self.bc = make_base(env)
        sd = torch.load(os.environ["BITIDY_RESIDUAL"], map_location=env.device, weights_only=False)["actor_state_dict"]
        self.mean, self.std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
        layers = []
        for j, i in enumerate(idx):
            w = sd[f"mlp.{i}.weight"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": sd[f"mlp.{i}.bias"]})
            layers += [lin] + ([torch.nn.ELU()] if j < len(idx) - 1 else [])
        self.mlp = torch.nn.Sequential(*layers).to(env.device).eval()
        self.scale = float(os.environ.get("BITIDY_RES_SCALE", "0.5"))
        n_act = sd[f"mlp.{idx[-1]}.weight"].shape[0]
        self.mask = torch.ones(n_act, device=env.device)
        self.mask[[int(i) for i in os.environ.get("BITIDY_RES_FIXED", "5,11").split(",") if i]] = 0.0
        from . import retreat

        self.retreat = retreat.RetreatFilter(env) if retreat.enabled() else None

    def reset(self, ids: torch.Tensor) -> None:
        self.bc.reset(ids)
        if self.retreat is not None:
            self.retreat.reset(ids)

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        base = self.bc.act()
        x = torch.cat((self.env.obs_buf["policy"], base), -1)
        res = self.mlp((x - self.mean) / (self.std + 1e-2))
        out = base + self.scale * self.mask * res
        if self.retreat is None:
            return out
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls % 300 == 0:
            print(f"BITIDY_RETREAT {self.retreat.count}", flush=True)
        return self.retreat(out)
