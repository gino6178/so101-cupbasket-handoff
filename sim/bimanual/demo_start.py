"""DemoStart-style curriculum (arXiv 2409.06613; reverse curriculum / Backplay): RL episodes start from simulator states
recorded along successful oracle demonstrations, starting near success and moving earlier as the policy masters them.

1. Record snapshots (oracle on many envs, first episode of each env; snapshots of the successful ones are kept):

    BITIDY_SCRIPT=isaaclab_tutorial.tasks.bimanual.oracle_lid_sort:LidSortScript BITIDY_DS_RECORD=~/ds_snaps.pt \
    [BITIDY_DS_EVERY=50] isaaclab play ... --external_callback isaaclab_tutorial.tasks.bimanual.demo_start.install_recorder

   A snapshot holds both arms' joint pos / vel / drive targets, every rigid object's root pose / velocity (env frame),
   the action manager's last actions, the LidSort task state (command, lid_start, cube_start, box_yaw), the progress
   milestone flags and counters, the step index t, the demo's success step t_succ and the policy observation (for the
   restore check).

2. Train with the curriculum (env-var gated; ``residual.install_residual`` installs it when BITIDY_DS_SNAPS is set):

    BITIDY_DS_SNAPS=~/ds_snaps.pt [BITIDY_DS_FRAC=0.5] [BITIDY_DS_BUCKET=250] [BITIDY_DS_THRESH=0.6]
    [BITIDY_DS_MIN_EP=256] [BITIDY_DS_TIME=remaining|<steps>]

   On every reset a fraction BITIDY_DS_FRAC of the resetting envs is put into a snapshot instead of the normal start.
   Snapshots are bucketed by the steps the demo still needed to succeed (t_succ - t) in BITIDY_DS_BUCKET-step bins;
   bucket 0 is the last stretch before success. The frontier starts at bucket 0; once >= BITIDY_DS_MIN_EP episodes
   started in the frontier bucket succeed at >= BITIDY_DS_THRESH, the frontier moves one bucket earlier. Half of the
   snapshot starts come from the frontier bucket, half uniformly from the unlocked buckets (0..frontier), so mastered
   stretches are rehearsed. Time limit: "remaining" (default) sets the episode step counter to t, i.e. the episode
   keeps the time the 75 s episode had left at that point; a number gives every snapshot start that many steps.
   The BC-RNN base's memory is reset by residual.py (the env is "done"), the milestone reward's paid flags are set
   to the restored flags (no milestone paid twice), and the "other lid / cube moved" baselines are the episode's own.

   BITIDY_DS_MODE=zvf replaces the frontier with DemoStart's zero-variance filter: every bucket's success is tracked
   online (decayed counts, BITIDY_DS_ZVF_DECAY=0.998 per reset call) and snapshot starts are drawn from the buckets
   whose recent success lies in BITIDY_DS_ZVF_BAND=0.1,0.9 (or that have too few episodes yet); always-0 / always-1
   buckets keep only weight BITIDY_DS_ZVF_FLOOR=0.03. Use BITIDY_DS_BUCKET=100 for finer buckets with zvf.

   Logs: "BITIDY_DS ..." lines (frontier, per-bucket success, normal-start success of training episodes) and
   DemoStart/* entries in the RSL-RL log.

3. Restore check / per-bucket evaluation: BITIDY_DS_EVAL=1 with BITIDY_DS_FRAC=1 puts every env (also at the first
   reset) into a snapshot; BITIDY_DS_BUCKETS=0,3 restricts to buckets. Prints the observation mismatch of the restored
   state against the recorded observation (BITIDY_DS_CHECK) and per-bucket success of the first episodes.
"""

from __future__ import annotations

import os
import sys

import torch

ROBOTS = ("robot_left", "robot_right")
OBJECTS = tuple(f"{k}{j}" for k in ("box", "lid", "cube") for j in range(3))


# ---------------------------------------------------------------- capture / restore


def capture(env, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Full task-relevant state of envs ``ids`` (positions in the env frame)."""
    from . import lid_sort_env_cfg as ls

    o = env.scene.env_origins[ids]
    s: dict[str, torch.Tensor] = {}
    for r in ROBOTS:
        d = env.scene[r].data
        s[f"{r}/joint_pos"] = d.joint_pos.torch[ids].clone()
        s[f"{r}/joint_vel"] = d.joint_vel.torch[ids].clone()
        s[f"{r}/joint_pos_target"] = d.joint_pos_target.torch[ids].clone()
        s[f"{r}/joint_vel_target"] = d.joint_vel_target.torch[ids].clone()
        pose = d.root_pose_w.torch[ids].clone()
        pose[:, :3] -= o
        s[f"{r}/root_pose"] = pose
        s[f"{r}/root_vel"] = d.root_vel_w.torch[ids].clone()
    for name in OBJECTS:
        d = env.scene[name].data
        pose = torch.cat((d.root_pos_w.torch[ids], d.root_quat_w.torch[ids]), -1)
        pose[:, :3] -= o
        s[f"{name}/root_pose"] = pose
        s[f"{name}/root_vel"] = torch.cat((d.root_lin_vel_w.torch[ids], d.root_ang_vel_w.torch[ids]), -1)
    s["act/action"] = env.action_manager.action[ids].clone()
    s["act/prev_action"] = env.action_manager.prev_action[ids].clone()
    st = ls._state(env)
    for k in ("cmd", "lid_start", "cube_start", "box_yaw"):
        s[f"ls/{k}"] = st[k][ids].clone()
    p = ls.progress(env)
    for k in ("flags", "lid_count", "cube_count", "stable_count"):
        s[f"prog/{k}"] = getattr(p, k)[ids].clone()
    s["t"] = env.episode_length_buf[ids].clone().long()
    s["obs"] = env.obs_buf["policy"][ids].clone()
    return s


def _milestone_term(env):
    rm = env.reward_manager
    try:
        cfg = rm.get_term_cfg("milestones")
    except Exception:
        return None
    return cfg.func if hasattr(cfg.func, "paid") else None


def restore(env, ids: torch.Tensor, snap: dict[str, torch.Tensor], k: torch.Tensor) -> None:
    """Put envs ``ids`` into snapshots ``k`` (indices into ``snap``); call after the env's normal reset of ``ids``."""
    from . import lid_sort_env_cfg as ls

    o = env.scene.env_origins[ids]
    state = {"articulation": {}, "rigid_object": {}}
    for r in ROBOTS:
        pose = snap[f"{r}/root_pose"][k].clone()
        pose[:, :3] += o
        state["articulation"][r] = {"root_pose": pose, "root_velocity": snap[f"{r}/root_vel"][k],
                                    "joint_position": snap[f"{r}/joint_pos"][k], "joint_velocity": snap[f"{r}/joint_vel"][k]}
    for name in OBJECTS:
        pose = snap[f"{name}/root_pose"][k].clone()
        pose[:, :3] += o
        state["rigid_object"][name] = {"root_pose": pose, "root_velocity": snap[f"{name}/root_vel"][k]}
    # Every scene entity must be present for reset_to; entities we do not track keep their current state.
    for name, art in env.scene.articulations.items():
        if name not in state["articulation"]:
            d = art.data
            state["articulation"][name] = {"root_pose": d.root_pose_w.torch[ids].clone(), "root_velocity": d.root_vel_w.torch[ids].clone(),
                                           "joint_position": d.joint_pos.torch[ids].clone(), "joint_velocity": d.joint_vel.torch[ids].clone()}
    for name, obj in env.scene.rigid_objects.items():
        if name not in state["rigid_object"]:
            d = obj.data
            state["rigid_object"][name] = {"root_pose": torch.cat((d.root_pos_w.torch[ids], d.root_quat_w.torch[ids]), -1),
                                           "root_velocity": torch.cat((d.root_lin_vel_w.torch[ids], d.root_ang_vel_w.torch[ids]), -1)}
    env.scene.reset_to(state, ids, is_relative=False)
    # reset_to sets the drive targets to the joint state; the recorded ones are what the actions had commanded.
    for r in ROBOTS:
        env.scene[r].set_joint_position_target_index(target=snap[f"{r}/joint_pos_target"][k], env_ids=ids)
        env.scene[r].set_joint_velocity_target_index(target=snap[f"{r}/joint_vel_target"][k], env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    am = env.action_manager
    am._action[ids] = snap["act/action"][k]
    am._prev_action[ids] = snap["act/prev_action"][k]
    st = ls._state(env)
    for key in ("cmd", "lid_start", "cube_start", "box_yaw"):
        v = snap[f"ls/{key}"][k]
        if key == "cmd" and v.shape[-1] < st[key].shape[-1]:  # v1 snapshot (cube, box): put-in mode
            v = torch.cat((v, torch.zeros_like(v[..., :1])), -1)
        st[key][ids] = v
    p = ls.progress(env)
    p.flags[ids] = snap["prog/flags"][k]
    p.lid_count[ids] = snap["prog/lid_count"][k]
    p.cube_count[ids] = snap["prog/cube_count"][k]
    p.stable_count[ids] = snap["prog/stable_count"][k]
    p.success[ids] = False
    p.cmd[ids] = st["cmd"][ids]
    m = _milestone_term(env)
    if m is not None:
        m.paid[ids] = snap["prog/flags"][k]


# ---------------------------------------------------------------- 1. recorder


def install_recorder() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from .scripted import _script_class

    inner_step = RslRlVecEnvWrapper.step
    path = os.path.expanduser(os.environ.get("BITIDY_DS_RECORD", "~/ds_snaps.pt"))
    every = int(os.environ.get("BITIDY_DS_EVERY", "50"))
    s: dict = {}

    def step(self, actions):
        env = self.unwrapped
        if "script" not in s:
            s.update(script=_script_class(os.environ["BITIDY_SCRIPT"])(env), chunks=[],
                     ended=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                     success=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                     t_end=torch.zeros(env.num_envs, dtype=torch.long, device=env.device), steps=0)
        recording = ~s["ended"]
        length_before = env.episode_length_buf.clone()
        result = inner_step(self, s["script"].act())
        dones = result[2].bool()
        newly = dones & recording
        s["success"] |= newly & env._bi_final[:, -1]
        s["t_end"][newly] = length_before[newly] + 1
        s["ended"] |= newly
        s["steps"] += 1
        live = recording & ~dones & (env.episode_length_buf % every == 0) & (env.episode_length_buf > 0)
        ids = live.nonzero().squeeze(-1)
        if len(ids):
            snap = capture(env, ids)
            snap["env"] = ids.clone()
            s["chunks"].append({k: v.cpu() for k, v in snap.items()})
        done_ids = dones.nonzero().squeeze(-1)
        if len(done_ids):
            s["script"].reset(done_ids)
        if s["steps"] % 250 == 0:
            print(f"BITIDY_DS_REC steps={s['steps']} ended={int(s['ended'].sum())} wins={int(s['success'].sum())}", flush=True)
        if bool(s["ended"].all()) or s["steps"] > int(env.max_episode_length) + 5:
            _write(path, s, env.num_envs)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def _write(path: str, s: dict, n_envs: int) -> None:
    out = {k: torch.cat([c[k] for c in s["chunks"]]) for k in s["chunks"][0]}
    succ, t_end = s["success"].cpu(), s["t_end"].cpu()
    keep = succ[out["env"]]
    out = {k: v[keep] for k, v in out.items()}
    out["t_succ"] = t_end[out["env"]]
    wins = int(succ.sum())
    torch.save(out, path)
    print(f"BITIDY_DS_SAVED path={path} envs={n_envs} successes={wins} rate={wins / n_envs:.3f} snapshots={len(out['t'])} "
          f"mean_t_succ={float(t_end[succ].float().mean()) if wins else 0:.0f}", flush=True)


def load(paths: str, device) -> dict[str, torch.Tensor]:
    parts = [torch.load(os.path.expanduser(p), map_location="cpu", weights_only=False) for p in paths.split(",")]
    for i, part in enumerate(parts):  # demo ids unique across files
        part["demo"] = part["env"].long() + i * 1_000_000
    return {k: torch.cat([p[k] for p in parts]).to(device) for k in parts[0]}


# ---------------------------------------------------------------- 2. curriculum


class Curriculum:
    def __init__(self, env):
        self.env = env
        dev = env.device
        self.snap = load(os.environ["BITIDY_DS_SNAPS"], dev)
        self.frac = float(os.environ.get("BITIDY_DS_FRAC", "0.5"))
        self.width = int(os.environ.get("BITIDY_DS_BUCKET", "250"))
        self.thresh = float(os.environ.get("BITIDY_DS_THRESH", "0.6"))
        self.min_ep = int(os.environ.get("BITIDY_DS_MIN_EP", "256"))
        self.time = os.environ.get("BITIDY_DS_TIME", "remaining")
        self.eval = bool(os.environ.get("BITIDY_DS_EVAL"))
        remaining = (self.snap["t_succ"] - self.snap["t"]).clamp(min=0)
        self.bucket = (remaining // self.width).long()
        self.n_buckets = int(self.bucket.max()) + 1
        self.members = [(self.bucket == b).nonzero().squeeze(-1) for b in range(self.n_buckets)]
        allowed = os.environ.get("BITIDY_DS_BUCKETS")
        self.allowed = [int(b) for b in allowed.split(",")] if allowed else None
        self.frontier = 0
        # Per-bucket counts of finished episodes since the frontier last moved (index n_buckets = normal start).
        self.n = torch.zeros(self.n_buckets + 1, dtype=torch.long, device=dev)
        self.w = torch.zeros_like(self.n)
        self.tot_n = torch.zeros_like(self.n)
        self.tot_w = torch.zeros_like(self.n)
        self.env_bucket = torch.full((env.num_envs,), self.n_buckets, dtype=torch.long, device=dev)
        # Zero-variance filtering (DemoStart's sampler): decayed per-bucket counts give an online success estimate;
        # buckets with an estimate inside [lo, hi] get weight 1, always-0 / always-1 ones only BITIDY_DS_ZVF_FLOOR (so
        # their estimate keeps updating), buckets with < BITIDY_DS_MIN_EP decayed episodes count as informative.
        self.mode = os.environ.get("BITIDY_DS_MODE", "threshold")
        self.lo, self.hi = (float(v) for v in os.environ.get("BITIDY_DS_ZVF_BAND", "0.1,0.9").split(","))
        self.floor = float(os.environ.get("BITIDY_DS_ZVF_FLOOR", "0.03"))
        self.decay = float(os.environ.get("BITIDY_DS_ZVF_DECAY", "0.998"))
        self.dn = torch.zeros(self.n_buckets + 1, device=dev)
        self.dw = torch.zeros_like(self.dn)
        self.calls = 0
        self.resets = 0
        self.checked = False
        sizes = [len(m) for m in self.members]
        print(f"BITIDY_DS loaded snapshots={len(self.bucket)} demos={len(torch.unique(self.snap['demo']))} buckets={self.n_buckets} "
              f"width={self.width} sizes={sizes} frac={self.frac} thresh={self.thresh} time={self.time}", flush=True)

    def before_reset(self, ids: torch.Tensor) -> None:
        """Account for the episodes that end now (their success is still latched in progress)."""
        from . import lid_sort_env_cfg as ls

        env = self.env
        ran = ids[env.episode_length_buf[ids] > 0]
        if not len(ran):
            return
        b = self.env_bucket[ran]
        won = ls.progress(env).success[ran].long()
        ones = torch.ones_like(b)
        for cnt, val in ((self.n, ones), (self.w, won), (self.tot_n, ones), (self.tot_w, won)):
            cnt.index_add_(0, b, val)
        self.dn.mul_(self.decay).index_add_(0, b, ones.float())
        self.dw.mul_(self.decay).index_add_(0, b, won.float())

    def zvf_weights(self) -> torch.Tensor:
        """Sampling weight per bucket (zero-variance filter)."""
        n, w = self.dn[: self.n_buckets], self.dw[: self.n_buckets]
        rate = w / n.clamp(min=1e-6)
        informative = (n < self.min_ep) | ((rate >= self.lo) & (rate <= self.hi))
        weight = torch.where(informative, 1.0, self.floor)
        sizes = torch.tensor([len(m) for m in self.members], device=n.device)
        return weight * (sizes > 0)

    def _advance(self) -> None:
        f = self.frontier
        if self.allowed is not None or self.mode == "zvf" or f >= self.n_buckets:
            return
        n, w = int(self.n[f]), int(self.w[f])
        if n >= self.min_ep and w >= self.thresh * n:
            self.frontier = f + 1
            print(f"BITIDY_DS_ADVANCE frontier {f}->{f + 1} (bucket {f}: {w}/{n}) resets={self.resets}", flush=True)
            self.n[: self.n_buckets].zero_()
            self.w[: self.n_buckets].zero_()

    def after_reset(self, ids: torch.Tensor) -> None:
        env = self.env
        self.calls += 1
        self.env_bucket[ids] = self.n_buckets
        first = self.calls == 1
        if first and not self.eval:
            return  # initial reset: RSL-RL randomises the episode counters right after it
        if self.calls % 50 == 0 or (self.eval and self.calls % 10 == 0):
            self.report()
        self._advance()
        pick = torch.rand(len(ids), device=env.device) < self.frac
        sel = ids[pick]
        if not len(sel):
            return
        m = len(sel)
        if self.allowed is not None:
            choices = torch.tensor(self.allowed, device=env.device)
            b = choices[torch.randint(len(choices), (m,), device=env.device)]
        elif self.mode == "zvf":
            b = torch.multinomial(self.zvf_weights(), m, replacement=True)
        else:
            top = min(self.frontier, self.n_buckets - 1)
            uniform = torch.randint(0, top + 1, (m,), device=env.device)
            b = torch.where(torch.rand(m, device=env.device) < 0.5, torch.full_like(uniform, top), uniform)
        # Pick a snapshot inside each chosen bucket (empty buckets fall back to the nearest later one).
        k = torch.empty(m, dtype=torch.long, device=env.device)
        for bb in torch.unique(b).tolist():
            sub = (b == bb).nonzero().squeeze(-1)
            mem = self.members[bb]
            j = bb
            while not len(mem) and j > 0:
                j -= 1
                mem = self.members[j]
            k[sub] = mem[torch.randint(len(mem), (len(sub),), device=env.device)]
        restore(env, sel, self.snap, k)
        t = self.snap["t"][k]
        if self.time != "remaining":
            t = (int(env.max_episode_length) - int(self.time)) * torch.ones_like(t)
            t = t.clamp(min=0)
        env.episode_length_buf[sel] = t.to(env.episode_length_buf.dtype)
        self.env_bucket[sel] = self.bucket[k]
        self.resets += m
        if os.environ.get("BITIDY_DS_CHECK") and not self.checked:
            self.checked = True
            obs = env.observation_manager.compute()["policy"][sel]
            diff = (obs - self.snap["obs"][k]).abs()
            worst = diff.max(0).values
            top = torch.topk(worst, 8)
            print(f"BITIDY_DS_CHECK envs={m} obs_dim={obs.shape[-1]} mean_abs={float(diff.mean()):.2e} max_abs={float(diff.max()):.2e} "
                  f"p99={float(diff.flatten().quantile(0.99)) if diff.numel() < 16_000_000 else -1:.2e} "
                  f"worst_dims={top.indices.tolist()} worst_vals={[round(v, 4) for v in top.values.tolist()]}", flush=True)

    def rates(self, n, w):
        return [round(int(a) / int(c), 3) if int(c) else None for a, c in zip(w.tolist(), n.tolist())]

    def report(self) -> dict:
        nb = self.n_buckets
        r = self.rates(self.tot_n, self.tot_w)
        info = {"frontier": self.frontier, "normal_success": r[nb], "normal_n": int(self.tot_n[nb]),
                "bucket_success": r[:nb], "bucket_n": self.tot_n[:nb].tolist(), "snap_resets": self.resets}
        if self.mode == "zvf":
            info["recent_success"] = [round(float(a / c), 3) if float(c) > 0 else None
                                      for a, c in zip(self.dw[:nb].tolist(), self.dn[:nb].tolist())]
            info["weights"] = [round(v, 2) for v in self.zvf_weights().tolist()]
        print("BITIDY_DS " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)
        return info


def install_curriculum() -> None:
    """Patch ManagerBasedRLEnv._reset_idx (call from an external callback before the env is created)."""
    from isaaclab.envs import ManagerBasedRLEnv

    if getattr(ManagerBasedRLEnv, "_ds_installed", False):
        return
    inner = ManagerBasedRLEnv._reset_idx

    def _reset_idx(self, env_ids):
        if not hasattr(self, "_ds"):
            self._ds = Curriculum(self)
        ds = self._ds
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None or isinstance(env_ids, slice)
               else torch.as_tensor(env_ids, device=self.device).long())
        ds.before_reset(ids)
        inner(self, env_ids)
        ds.after_reset(ids)
        nb = ds.n_buckets
        log = self.extras.setdefault("log", {})
        log["DemoStart/frontier"] = torch.tensor(float(ds.frontier))
        if int(ds.n[nb]):
            log["DemoStart/normal_success"] = ds.w[nb].float() / ds.n[nb].float()
        f = min(ds.frontier, nb - 1)
        if int(ds.n[f]):
            log["DemoStart/frontier_success"] = ds.w[f].float() / ds.n[f].float()

    ManagerBasedRLEnv._reset_idx = _reset_idx
    ManagerBasedRLEnv._ds_installed = True


def install_bucket_eval() -> list[str]:
    """Per-bucket success of a scripted executor (e.g. the BC-RNN) from snapshot starts: BITIDY_DS_EVAL=1, FRAC=1."""
    os.environ.setdefault("BITIDY_DS_EVAL", "1")
    os.environ.setdefault("BITIDY_DS_FRAC", "1")
    os.environ.setdefault("BITIDY_DS_CHECK", "1")
    install_curriculum()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from .scripted import _script_class

    inner_step = RslRlVecEnvWrapper.step
    s: dict = {}

    def step(self, actions):
        env = self.unwrapped
        if "script" not in s:
            s.update(script=_script_class(os.environ["BITIDY_SCRIPT"])(env),
                     done=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                     bucket=env._ds.env_bucket.clone(), won=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        result = inner_step(self, s["script"].act())
        dones = result[2].bool()
        first = dones & ~s["done"]
        s["won"] |= first & env._bi_final[:, -1]
        s["done"] |= first
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            s["script"].reset(ids)
        if bool(s["done"].all()):
            ds = env._ds
            out = []
            for b in range(ds.n_buckets + 1):
                m = s["bucket"] == b
                if int(m.sum()):
                    out.append(f"{b}:{int(s['won'][m].sum())}/{int(m.sum())}")
            print(f"BITIDY_DS_EVAL first-episode success per bucket (bucket:wins/n, {ds.n_buckets}=normal) {' '.join(out)} "
                  f"overall={float(s['won'].float().mean()):.3f}", flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
