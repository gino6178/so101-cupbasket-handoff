"""Collect scripted-oracle demonstrations for behaviour cloning.

Run with ``isaaclab play ... --external_callback isaaclab_tutorial.tasks.bimanual.demos.install_demo_collector`` and
``BITIDY_SCRIPT=<module:Class>``; the policy's actions are replaced by the oracle. Every environment's first episode is
recorded on the GPU (policy observation before each step, oracle action); once all have ended, the successful episodes
are written to ``$BITIDY_DEMO`` (.npz: obs, act, episode index per step) and the process exits.

DAgger: with ``BITIDY_DAGGER_BETA=<b>`` each step executes the oracle action with probability b and the loaded policy's
action otherwise, the oracle still labelling every visited state, and all episodes are kept (not only successes).
``BITIDY_DAGGER_LEARNER=<module:Class>`` takes the learner's action from a script (e.g. ``bcrnn:BCRNNScript``) instead
of the loaded RSL-RL policy.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


def install_demo_collector() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from .scripted import _script_class

    inner_step = RslRlVecEnvWrapper.step
    path = os.environ.get("BITIDY_DEMO", os.path.expanduser("~/demos.npz"))
    state: dict = {}
    beta = os.environ.get("BITIDY_DAGGER_BETA")

    def step(self, actions):
        env = self.unwrapped
        if "script" not in state:
            n, horizon = env.num_envs, int(env.max_episode_length) + 2
            obs_all = self.get_observations()
            obs = obs_all["policy"]
            crit = obs_all["critic"] if "critic" in obs_all.keys() else None
            state.update(
                script=_script_class(os.environ["BITIDY_SCRIPT"])(env),
                learner=_script_class(os.environ["BITIDY_DAGGER_LEARNER"])(env) if os.environ.get("BITIDY_DAGGER_LEARNER") else None,
                obs=torch.zeros((horizon, n, obs.shape[-1]), device=env.device),
                act=torch.zeros((horizon, n, env.action_manager.total_action_dim), device=env.device),
                length=torch.zeros(n, dtype=torch.long, device=env.device),
                ended=torch.zeros(n, dtype=torch.bool, device=env.device),
                success=torch.zeros(n, dtype=torch.bool, device=env.device),
                last=obs.clone(),
                # noise-free (critic) observation alongside, same layout as the policy one when the dims agree
                obs_c=torch.zeros((horizon, n, obs.shape[-1]), device=env.device) if crit is not None and crit.shape[-1] == obs.shape[-1] else None,
                last_c=crit.clone() if crit is not None and crit.shape[-1] == obs.shape[-1] else None,
                t=0,
            )
            arms = getattr(state["script"], "arms", None)
            state["phase"] = torch.zeros((horizon, n, 2), dtype=torch.int8, device=env.device) if arms and len(arms) == 2 else None
        s = state
        scripted = s["script"].act()
        recording = ~s["ended"]
        t = s["t"]
        if s["phase"] is not None:  # oracle phase per arm at this step (for phase-weighted losses)
            s["phase"][t, recording] = torch.stack([a.p for a in s["script"].arms], -1)[recording].to(torch.int8)
        s["obs"][t, recording] = s["last"][recording]
        if s["obs_c"] is not None:
            s["obs_c"][t, recording] = s["last_c"][recording]
        # Labels are the executed oracle actions; only extreme outliers are clipped.
        s["act"][t, recording] = scripted[recording].clamp(-5.0, 5.0)
        s["length"][recording] += 1
        executed = scripted
        if s["learner"] is not None:
            actions = torch.nan_to_num(s["learner"].act(), nan=0.0, posinf=0.0, neginf=0.0)
        if s["learner"] is not None and os.environ.get("BITIDY_DAGGER_DIAG") and hasattr(s["script"], "arms"):
            _diag(s, scripted, actions, recording)
        if beta is not None:
            use_oracle = torch.rand(env.num_envs, 1, device=env.device) < float(beta)
            executed = torch.where(use_oracle, scripted, actions)
        result = inner_step(self, executed)
        obs, dones = result[0]["policy"], result[2].bool()
        s["last"] = obs.clone()
        if s["obs_c"] is not None:
            s["last_c"] = result[0]["critic"].clone()
        newly = dones & recording
        s["success"] |= newly & env._bi_final[:, -1]
        s["ended"] |= newly
        s["t"] = t + 1
        done_ids = dones.nonzero().squeeze(-1)
        if len(done_ids):
            s["script"].reset(done_ids)
            if s["learner"] is not None:
                s["learner"].reset(done_ids)
        if bool(s["ended"].all()) or s["t"] >= s["obs"].shape[0]:
            if "diag" in s:
                _diag_print(s)
            _write(path, s, keep_all=beta is not None)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def _diag(s: dict, oracle: torch.Tensor, learner: torch.Tensor, rec: torch.Tensor) -> None:
    """Learner-vs-oracle action error per arm and oracle phase (first episodes only)."""
    n_ph = 10
    if "diag" not in s:
        s["diag"] = torch.zeros((2, n_ph, 3), device=oracle.device)  # count, arm sq. error sum, gripper sign mismatches
    tr = int(os.environ.get("BITIDY_DAGGER_TRACE", "0"))
    if s["t"] < tr and s["t"] % 3 == 0:
        e = 0
        print(f"TRACE t={s['t']} pL={int(s['script'].arms[0].p[e])} orc={[round(v, 2) for v in oracle[e, :6].tolist()]} "
              f"lrn={[round(v, 2) for v in learner[e, :6].tolist()]}", flush=True)
    for k, a in enumerate(s["script"].arms):
        sl = slice(6 * k, 6 * k + 5)
        err = ((learner[:, sl] - oracle[:, sl]) ** 2).mean(-1)
        gm = ((learner[:, 6 * k + 5] > 0) != (oracle[:, 6 * k + 5] > 0)).float()
        ph = a.p.clamp(0, n_ph - 1)
        m = rec.float()
        s["diag"][k, :, 0].index_add_(0, ph, m)
        s["diag"][k, :, 1].index_add_(0, ph, err * m)
        s["diag"][k, :, 2].index_add_(0, ph, gm * m)


def _diag_print(s: dict) -> None:
    d = s["diag"].cpu()
    for k in range(2):
        rows = [f"p{i}:n={int(d[k, i, 0])},rmse={(d[k, i, 1] / d[k, i, 0].clamp(min=1)).sqrt():.3f},grip_err={d[k, i, 2] / d[k, i, 0].clamp(min=1):.3f}"
                for i in range(d.shape[1]) if d[k, i, 0] > 0]
        print(f"BITIDY_DAGGER_DIAG arm={'LR'[k]} " + " ".join(rows), flush=True)


def _write(path: str, s: dict, keep_all: bool = False) -> None:
    keep = list(range(len(s["success"]))) if keep_all else s["success"].nonzero().squeeze(-1).tolist()
    obs, obs_c, act, episode, phase = [], [], [], [], []
    dropped = 0
    for e in keep:
        n = int(s["length"][e])
        o, a = s["obs"][:n, e], s["act"][:n, e]
        oc = s["obs_c"][:n, e] if s["obs_c"] is not None else None
        # physics blow-ups (NaN / inf observations or labels) poison the normalisation and every gradient: drop the episode
        finite = bool(torch.isfinite(o).all()) and bool(torch.isfinite(a).all()) and (oc is None or bool(torch.isfinite(oc).all()))
        if not finite:
            dropped += 1
            continue
        obs.append(o.cpu())
        act.append(a.cpu())
        if oc is not None:
            obs_c.append(oc.cpu())
        if s.get("phase") is not None:
            phase.append(s["phase"][:n, e].cpu())
        episode.append(torch.full((n,), len(episode), dtype=torch.int32))
    extra = {"obs_critic": torch.cat(obs_c).numpy().astype(np.float32)} if obs_c else {}
    if phase:
        extra["phase"] = torch.cat(phase).numpy()
    np.savez(
        path,
        obs=torch.cat(obs).numpy().astype(np.float32),
        act=torch.cat(act).numpy().astype(np.float32),
        episode=torch.cat(episode).numpy(),
        **extra,
    )
    total = len(s["success"])
    wins = int(s["success"].sum())
    print(f"BITIDY_DEMOS episodes={total} successes={wins} rate={wins / total:.3f} kept={len(episode)} dropped_nonfinite={dropped} "
          f"steps={sum(len(o) for o in obs)} critic={bool(obs_c)} path={path}", flush=True)
