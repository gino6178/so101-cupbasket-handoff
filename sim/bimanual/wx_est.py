"""Learned estimator of the teacher's privileged inputs from deployable signals (``WX_EST_TRAIN=1`` / ``WX_EST=<ckpt>``).

The privileged teacher (BC-RNN + residual) reads two terms the real robot does not have: the 10 progress flags
(grasp / lift / over / placed / released per arm) and the 18-dim target-relative term (rim - pads, goal - vessel,
vessel velocity). With the coarse / fused camera estimate in their place it fails (0-8 %); with the true terms it is
back at 77-82 %. This module learns them: an LSTM (``wx_student.Student`` without images) on the deployable actor
input (``ProprioReal`` of the coarse group, incl. the rule-based ``WX_DEPLOY_FLAGS`` hints) + the last two executed
actions, trained online DAgger-style: every env runs the teacher on either the true terms (prob. beta, per episode) or
the estimator's, and the true terms label every step.

  EstTeacher   (eval / base of the deployable residual) the teacher fed the estimated terms; ``WX_EST=<ckpt>``.
Loss: BCE on the flags + MSE (x10, metres -> dm) on the relative term.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from . import wx_real as R
from .wx_student import Student, _f, _teacher, _term_slice

N_FLAG = 10


def _slices(env):
    return {g: {t: _term_slice(env, g, t) for t in ("progress", "relative")} for g in ("coarse", "critic")}


def _targets(env, sl) -> torch.Tensor:
    cr = env.obs_buf["critic"]
    p0, p1 = sl["critic"]["progress"]
    r0, r1 = sl["critic"]["relative"]
    return torch.cat((cr[:, p0: p0 + N_FLAG], cr[:, r0:r1]), -1)


class _Core:
    """Shared rollout logic: deployable input -> estimator -> teacher on the modified coarse obs -> actuator model."""

    def __init__(self, env, net: Student):
        self.env, self.net = env, net
        n, dev = env.num_envs, env.device
        self.sl = _slices(env)
        self.teacher = _teacher(env)
        self.preal = R.ProprioReal(env) if R.REAL else None
        self.areal = R.ActionReal(n, dev) if R.REAL else None
        L, H = net.cfg["layers"], net.cfg["hidden"]
        self.h = torch.zeros(L, n, H, device=dev)
        self.c = torch.zeros_like(self.h)
        self.prev_a = torch.zeros(n, 2, 12, device=dev)
        self.sticky = torch.zeros(n, N_FLAG, device=dev)

    def reset(self, ids):
        self.teacher.reset(ids)
        self.h[:, ids] = 0.0
        self.c[:, ids] = 0.0
        self.prev_a[ids] = 0.0
        self.sticky[ids] = 0.0
        if self.preal is not None:
            self.preal.reset(ids)
            self.areal.reset(ids)

    def inputs(self) -> torch.Tensor:
        c = torch.nan_to_num(self.env.obs_buf["coarse"])
        if self.preal is not None:
            c = self.preal(c)
        self.c_in = c
        return torch.cat((c, self.prev_a.flatten(1)), -1)

    @torch.no_grad()
    def estimate(self, s):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _, (h, c) = self.net.step(s, None, (self.h, self.c))
        self.h, self.c = h.float(), c.float()
        pred = pred.float()
        flags = torch.maximum(self.sticky, (pred[:, :N_FLAG] > 0).float())  # milestones are sticky
        self.sticky = flags
        return flags, pred[:, N_FLAG:] / 10.0

    @torch.no_grad()
    def act_with(self, flags, rel) -> torch.Tensor:
        ob = self.env.obs_buf
        saved = {k: ob[k] for k in ("policy", "critic")}
        c = self.c_in.clone()
        p0, _ = self.sl["coarse"]["progress"]
        r0, r1 = self.sl["coarse"]["relative"]
        c[:, p0: p0 + N_FLAG] = flags
        c[:, r0:r1] = rel
        try:
            ob["policy"] = c
            ob["critic"] = c
            a = self.teacher.act()
        finally:
            ob.update(saved)
        a = torch.nan_to_num(a)
        self.prev_a = torch.stack((a, self.prev_a[:, 0]), 1)
        return a if self.areal is None else self.areal(a)


def load_est(path, dev) -> Student:
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = Student(**ck["cfg"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    return net.eval()


class EstTeacher(_Core):
    """The privileged teacher on the coarse / fused estimate with the learned flags + relative term (``WX_EST``)."""

    def __init__(self, env):
        super().__init__(env, load_est(os.environ["WX_EST"], env.device))
        print(f"EstTeacher {os.environ['WX_EST']}", flush=True)

    @torch.no_grad()
    def act(self):
        f, r = self.estimate(self.inputs())
        tru = os.environ.get("WX_EST_TRUE", "")  # diagnostic: put true terms back, e.g. "flags", "rel", "rel:0-6"
        if tru:
            t = _targets(self.env, self.sl)
            for k in tru.split(","):
                if k == "flags":
                    f = t[:, :N_FLAG]
                elif k == "rel":
                    r = t[:, N_FLAG:]
                elif k.startswith("rel:"):
                    a, b = map(int, k[4:].split("-"))
                    r = r.clone()
                    r[:, a:b] = t[:, N_FLAG + a: N_FLAG + b]
        return self.act_with(f, r)


class EstDAgger:
    def __init__(self, env):
        self.env = env
        dev = self.dev = env.device
        n = self.n = env.num_envs
        self.T, self.C = int(_f("WX_T", 32)), int(_f("WX_BUF", 24))
        self.K, self.B = int(_f("WX_K", 24)), int(_f("WX_B", 64))
        self.iters, self.warm, self.decay = int(_f("WX_ITERS", 600)), int(_f("WX_WARM", 20)), int(_f("WX_DECAY", 200))
        self.beta_min = _f("WX_BETA_MIN", 0.0)
        self.out = os.path.expanduser(os.environ.get("WX_OUT", "~/wx/est.pt"))
        self.world, self.rank = int(_f("WX_WORLD", 1)), int(_f("WX_RANK", 0))
        if self.world > 1:
            import datetime

            import torch.distributed as dist

            dist.init_process_group("nccl", init_method=f"tcp://{os.environ['WX_MASTER']}:{os.environ.get('WX_PORT', '29531')}",
                                    rank=self.rank, world_size=self.world,
                                    timeout=datetime.timedelta(minutes=float(os.environ.get("WX_NCCL_TIMEOUT_MIN", 30))))
            self.dist = dist
        env.obs_buf  # noqa: B018
        sdim = env.obs_buf["coarse"].shape[-1] + 24
        tdim = N_FLAG + _slices(env)["critic"]["relative"][1] - _slices(env)["critic"]["relative"][0]
        self.net = Student(sdim, adim=tdim, auxdim=1, hidden=int(_f("WX_HIDDEN", 512)), layers=int(_f("WX_LAYERS", 2)),
                           use_img=False).to(dev)
        self.net.keep.fill_(1.0)
        if self.world > 1:
            for t_ in self.net.state_dict().values():
                self.dist.broadcast(t_, 0)
        self.core = _Core(env, self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=_f("WX_LR", 3e-4))
        self.lr = _f("WX_LR", 3e-4)
        T, C = self.T, self.C
        L, H = self.net.cfg["layers"], self.net.cfg["hidden"]
        self.bS = torch.zeros(C, n, T, sdim, device=dev)
        self.bY = torch.zeros(C, n, T, tdim, device=dev)
        self.bF = torch.zeros(C, n, T, dtype=torch.bool, device=dev)
        self.bV = torch.zeros(C, n, T, dtype=torch.bool, device=dev)
        self.bH = torch.zeros(C, L, n, H, device=dev)
        self.bC = torch.zeros_like(self.bH)
        self.filled, self.slot, self.t, self.it = 0, 0, 0, 0
        self.normed = False
        self.first = torch.ones(n, dtype=torch.bool, device=dev)
        self.use_t = torch.ones(n, dtype=torch.bool, device=dev)
        self.ep = {"est": [], "tea": []}
        self.tstart, self.steps = time.time(), 0
        print(f"WX_EST DAgger n={n} sdim={sdim} tdim={tdim} T={T} buf={C} iters={self.iters} out={self.out}", flush=True)

    def beta(self) -> float:
        if self.it < self.warm:
            return 1.0
        x = min(1.0, (self.it - self.warm) / max(1, self.decay - self.warm))
        return 1.0 + x * (self.beta_min - 1.0)

    def step(self, wrapper, inner_step):
        env, t, core = self.env, self.t, self.core
        s = core.inputs()
        y = torch.nan_to_num(_targets(env, core.sl))
        valid = torch.isfinite(s).all(-1)
        s = torch.nan_to_num(s)
        if not self.normed:
            mean, std = s.mean(0), s.std(0).clamp(min=1e-2)
            if self.world > 1:  # one normaliser for all ranks
                self.dist.all_reduce(mean)
                self.dist.all_reduce(std)
                mean, std = mean / self.world, std / self.world
            self.net.mean.copy_(mean)
            self.net.std.copy_(std)
            self.normed = True
        if t == 0:
            self.bH[self.slot] = core.h
            self.bC[self.slot] = core.c
        f_est, r_est = core.estimate(s)
        ut = self.use_t.unsqueeze(-1)
        flags = torch.where(ut, y[:, :N_FLAG], f_est)
        rel = torch.where(ut, y[:, N_FLAG:], r_est)
        executed = core.act_with(flags, rel)
        sl = self.slot
        self.bS[sl, :, t] = s
        self.bY[sl, :, t] = y
        self.bF[sl, :, t] = self.first & (t > 0)
        self.bV[sl, :, t] = valid
        result = inner_step(wrapper, executed)
        dones = result[2].bool()
        self.steps += 1
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            succ = env._bi_final[ids, -1]
            u = self.use_t[ids]
            self.ep["tea"] = (self.ep["tea"] + succ[u].int().tolist())[-400:]
            self.ep["est"] = (self.ep["est"] + succ[~u].int().tolist())[-400:]
            core.reset(ids)
            self.use_t[ids] = torch.rand(len(ids), device=self.dev) < self.beta()
        self.first = dones.clone()
        self.t += 1
        if self.t == self.T:
            self.t = 0
            self.end_chunk()
        return result

    def end_chunk(self):
        self.slot = (self.slot + 1) % self.C
        self.filled = min(self.filled + 1, self.C)
        frac = self.it / max(1, self.iters)
        lr = self.lr * (1.0 if frac < 0.7 else max(0.1, 1 - (frac - 0.7) / 0.3))
        for g in self.opt.param_groups:
            g["lr"] = lr
        with torch.enable_grad():
            stats = [self.train_step() for _ in range(self.K)]
        st = {k: sum(d[k] for d in stats) / len(stats) for k in stats[0]}
        self.it += 1
        if self.it % 10 == 0 or self.it == 1:
            e, te = self.ep["est"], self.ep["tea"]
            print(f"WX_EST_IT r={self.rank} it={self.it} beta={self.beta():.2f} loss={st['loss']:.4f} flag_acc={st['acc']:.4f} "
                  f"rel_cm={st['rel_cm']:.2f} est_succ={sum(e) / max(1, len(e)):.3f}(n={len(e)}) "
                  f"tea_succ={sum(te) / max(1, len(te)):.3f}(n={len(te)}) elapsed_min={(time.time() - self.tstart) / 60:.1f}", flush=True)
        if (self.it % 25 == 0 or self.it >= self.iters) and self.rank == 0:
            os.makedirs(os.path.dirname(self.out), exist_ok=True)
            torch.save({"state_dict": self.net.state_dict(), "cfg": self.net.cfg, "it": self.it}, self.out + ".tmp")
            os.replace(self.out + ".tmp", self.out)
        if self.it >= self.iters:
            print(f"WX_DONE {self.out}", flush=True)
            os._exit(0)

    def train_step(self) -> dict:
        net, dev, B = self.net, self.dev, self.B
        ci = torch.randint(self.filled, (B,), device=dev)
        ei = torch.randint(self.n, (B,), device=dev)
        s = self.bS[ci, ei].transpose(0, 1)
        y = self.bY[ci, ei].transpose(0, 1)
        f = self.bF[ci, ei].transpose(0, 1)
        v = self.bV[ci, ei].transpose(0, 1).float()
        h0 = self.bH[ci, :, ei].transpose(0, 1).contiguous()
        c0 = self.bC[ci, :, ei].transpose(0, 1).contiguous()
        net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = net.seq(s, None, f, (h0, c0))
        pred = pred.float()
        lf = F.binary_cross_entropy_with_logits(pred[..., :N_FLAG], (y[..., :N_FLAG] > 0.5).float(), reduction="none").mean(-1)
        lr_ = ((pred[..., N_FLAG:] - 10.0 * y[..., N_FLAG:]) ** 2).mean(-1)
        den = v.sum().clamp(min=1)
        loss = ((lf + lr_) * v).sum() / den
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.world > 1:
            ps = [p for p in net.parameters() if p.requires_grad]
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in ps])
            self.dist.all_reduce(flat)
            flat /= self.world
            off = 0
            for p in ps:
                k = p.numel()
                p.grad = flat[off: off + k].view_as(p).clone()
                off += k
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.opt.step()
        net.eval()
        with torch.no_grad():
            acc = ((((pred[..., :N_FLAG] > 0) == (y[..., :N_FLAG] > 0.5)).float().mean(-1)) * v).sum() / den
            rel_cm = (((pred[..., N_FLAG:] / 10.0 - y[..., N_FLAG:]).abs().mean(-1) * 100) * v).sum() / den
        return {"loss": float(loss), "acc": float(acc), "rel_cm": float(rel_cm)}
