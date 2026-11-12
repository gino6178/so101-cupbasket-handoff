"""BC-RNN (robomimic-style LSTM behaviour cloning) from oracle demonstrations (``demos.py``).

Train:
    python -m isaaclab_tutorial.tasks.bimanual.bcrnn <demos.npz[,more.npz]> <out.pt>
        [--seq 0] [--epochs 60] [--hidden 400] [--layers 2] [--lr 1e-3] [--obs_noise 0.0]

``--seq 0`` trains on whole episodes (the LSTM sees the full history, as at rollout); ``--seq L`` trains on random
L-step windows from a zero state, as robomimic does (horizon 10), and the rollout then restarts the memory every L steps.

Roll out with the scripted-oracle hook, so evaluation and video work unchanged:
    BITIDY_SCRIPT=isaaclab_tutorial.tasks.bimanual.bcrnn:BCRNNScript BITIDY_BCRNN=<out.pt> ... install_scripted
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


class BCRNN(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(obs_dim, hidden, layers, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 256), torch.nn.ELU(), torch.nn.Linear(256, act_dim))

    def forward(self, x, state=None):
        y, state = self.lstm(x, state)
        return self.head(y), state


def _episodes(paths: str, key: str = "obs"):
    """Per-episode (obs, act) arrays from one or more demo files. Episodes with any non-finite observation or action
    (physics blow-ups recorded during DAgger rollouts) are dropped: a single NaN turns the normalisation statistics and
    then every weight into NaN."""
    out = []
    for path in paths.split(","):
        data = np.load(path)
        ep = data["episode"]
        cuts = np.flatnonzero(np.diff(ep)) + 1
        obs = data[key]
        act = data["act"]
        ph = data["phase"] if "phase" in data.files else np.zeros((len(act), 2), np.int8)
        bad, n = 0, 0
        for o, a, f in zip(np.split(obs, cuts), np.split(act, cuts), np.split(ph, cuts)):
            n += 1
            if not (np.isfinite(o).all() and np.isfinite(a).all()):
                bad += 1
                continue
            out.append((o, a, f))
        print(f"BCRNN load {path} [{key}]: {n} episodes, dropped {bad} non-finite", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("demos")
    p.add_argument("out")
    p.add_argument("--seq", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=0, help="episodes (seq 0) or windows per batch")
    p.add_argument("--obs_noise", type=float, default=0.0, help="Gaussian noise on normalised observations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs_key", default="obs", help="obs (noisy policy group) or obs_critic (noise-free critic group)")
    p.add_argument("--phase_w", default="", help="loss weight per oracle phase, e.g. '1:4,2:4,3:4' (HOVER/DOWN/CLOSE)")
    p.add_argument("--drop_dims", default="", help="observation dims zeroed (after normalisation) in training and rollout, "
                   "e.g. '36-47' = the previous action: with it the net learns to copy its own last action (copycat) and "
                   "runs away in closed loop")
    p.add_argument("--max_ep", type=int, default=0, help="keep at most this many episodes (most recent files last)")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"

    eps = _episodes(args.demos, args.obs_key)
    if args.max_ep and len(eps) > args.max_ep:
        eps = eps[: args.max_ep]
    n_ep, horizon = len(eps), max(len(o) for o, _, _ in eps)
    obs_dim, act_dim = eps[0][0].shape[1], eps[0][1].shape[1]
    # Padded episode tensors on the GPU with a validity mask.
    obs = torch.zeros((n_ep, horizon, obs_dim), device=dev)
    act = torch.zeros((n_ep, horizon, act_dim), device=dev)
    length = torch.zeros(n_ep, dtype=torch.long, device=dev)
    phase = torch.zeros((n_ep, horizon, 2), dtype=torch.long, device=dev)
    for i, (o, a, f) in enumerate(eps):
        obs[i, : len(o)] = torch.from_numpy(o).to(dev)
        act[i, : len(a)] = torch.from_numpy(a).to(dev)
        phase[i, : len(f)] = torch.from_numpy(f.astype(np.int64)).to(dev)
        length[i] = len(o)
    del eps
    # per-dimension loss weights from the oracle phase of each arm (--phase_w "p:w,..."; arm dims 0-5 left, 6-11 right)
    pw = torch.ones(16, device=dev)
    for kv in filter(None, args.phase_w.split(",")):
        k, v = kv.split(":")
        pw[int(k)] = float(v)
    dim_arm = torch.tensor([0] * (act_dim // 2) + [1] * (act_dim - act_dim // 2), device=dev)
    mask = torch.arange(horizon, device=dev)[None] < length[:, None]
    mean = obs[mask].mean(0)
    std = obs[mask].std(0).clamp(min=1e-2)
    keep = _keep_mask(args.drop_dims, obs_dim, dev)
    obs = (obs - mean) / std * keep
    perm = torch.randperm(n_ep, device=dev)
    val_ids, tr_ids = perm[: max(1, n_ep // 20)], perm[max(1, n_ep // 20) :]
    print(f"BCRNN data: {n_ep} episodes, {int(length.sum())} steps, horizon {horizon}, seq {args.seq}", flush=True)

    net = BCRNN(obs_dim, act_dim, args.hidden, args.layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    batch = args.batch or (32 if args.seq == 0 else 1024)

    def batch_of(ids):
        """Inputs, targets and mask: whole episodes, or random windows of --seq steps."""
        if args.seq == 0:
            return obs[ids], act[ids], mask[ids], phase[ids]
        start = (torch.rand(len(ids), device=dev) * (length[ids] - 1).clamp(min=1)).long()
        t = start[:, None] + torch.arange(args.seq, device=dev)[None]
        tc = t.clamp(max=horizon - 1)
        return obs[ids[:, None], tc], act[ids[:, None], tc], mask[ids[:, None], tc] & (t < horizon), phase[ids[:, None], tc]

    def loss_of(ids, train: bool):
        x, y, m, ph = batch_of(ids)
        if train and args.obs_noise > 0:
            x = x + args.obs_noise * torch.randn_like(x)
        pred, _ = net(x)
        w = pw[ph.clamp(0, 15)].gather(-1, dim_arm.expand(*ph.shape[:-1], act_dim)) if args.phase_w else 1.0
        err = (w * (pred - y) ** 2).mean(-1)
        grip = ((pred[..., [5, 11]] > 0) == (y[..., [5, 11]] > 0)).float().mean(-1)
        return (err * m).sum() / m.sum(), (grip * m).sum() / m.sum()

    # Window mode: one epoch draws as many windows as there are training steps / seq.
    per_epoch = len(tr_ids) if args.seq == 0 else int(mask[tr_ids].sum()) // args.seq
    for epoch in range(args.epochs):
        net.train()
        total, nb = 0.0, 0
        for b in range(0, per_epoch, batch):
            ids = tr_ids[torch.randint(len(tr_ids), (min(batch, per_epoch - b),), device=dev)] if args.seq else tr_ids[torch.randperm(len(tr_ids), device=dev)[:batch]]
            loss, _ = loss_of(ids, True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total, nb = total + loss.item(), nb + 1
        sched.step()
        net.eval()
        with torch.no_grad():
            vl, vg = [], []
            for b in range(0, len(val_ids), 64):
                l, g = loss_of(val_ids[b : b + 64], False)
                vl.append(l.item())
                vg.append(g.item())
        print(f"epoch {epoch + 1}/{args.epochs} train {total / nb:.4f} val {np.mean(vl):.4f} gripper-sign {np.mean(vg):.3f}", flush=True)
        torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std, "args": vars(args), "obs_dim": obs_dim, "act_dim": act_dim,
                    "obs_group": "critic" if args.obs_key == "obs_critic" else "policy", "keep": keep}, args.out)
    print(f"BCRNN_SAVED {args.out}", flush=True)


def _keep_mask(spec: str, dim: int, dev) -> torch.Tensor:
    keep = torch.ones(dim, device=dev)
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        keep[int(lo) : int(hi or lo) + 1] = 0.0
    return keep


class BCRNNScript:
    """Rolls out a trained BC-RNN in place of the policy (same interface as the scripted oracles)."""

    def __init__(self, env):
        self.env = env
        ck = torch.load(os.environ["BITIDY_BCRNN"], map_location=env.device, weights_only=False)
        a = ck["args"]
        self.net = BCRNN(ck["obs_dim"], ck["act_dim"], a["hidden"], a["layers"]).to(env.device).eval()
        self.net.load_state_dict(ck["state_dict"])
        self.mean, self.std, self.seq = ck["mean"], ck["std"], a["seq"]
        self.keep = ck.get("keep", torch.ones_like(self.mean)).to(env.device)
        # observation group the net was trained on ("policy": noisy, "critic": noise-free); BITIDY_BCRNN_OBS overrides
        self.group = os.environ.get("BITIDY_BCRNN_OBS", ck.get("obs_group", "policy"))
        print(f"BCRNNScript {os.environ['BITIDY_BCRNN']} obs group {self.group} seq {self.seq}", flush=True)
        n = env.num_envs
        self.h = torch.zeros((a["layers"], n, a["hidden"]), device=env.device)
        self.c = torch.zeros_like(self.h)
        self.t = torch.zeros(n, dtype=torch.long, device=env.device)

    def reset(self, ids: torch.Tensor) -> None:
        self.h[:, ids] = 0.0
        self.c[:, ids] = 0.0
        self.t[ids] = 0

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        obs = torch.nan_to_num(self.env.obs_buf[self.group], nan=0.0, posinf=0.0, neginf=0.0)
        if self.seq:  # memory restarted every seq steps, as in training
            restart = self.t % self.seq == 0
            self.h[:, restart] = 0.0
            self.c[:, restart] = 0.0
        x = ((obs - self.mean) / self.std * self.keep)[:, None]
        y, (self.h, self.c) = self.net(x, (self.h, self.c))
        self.t += 1
        # the oracles' arm actions lie in [-1, 1] (RelativeJointPosition, scale 0.033); grippers are binary on the sign
        return y[:, 0].clamp(-1.0, 1.0)


if __name__ == "__main__":
    main()
