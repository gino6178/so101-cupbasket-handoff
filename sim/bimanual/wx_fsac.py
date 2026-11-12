"""FastSAC residual RL on the frozen wrist student (multi-node, data-parallel).

Recipe after FastSAC (arXiv 2512.01996; reference implementation amazon-far/holosoma, Apache-2.0): LayerNorm + SiLU MLPs,
two distributional C51 critics averaged (no clipped double-Q), tanh policy with bounded log-std, auto-tuned entropy
(alpha_init 1e-3, target entropy 0), AdamW(wd 1e-3, betas .9/.95), fast Polyak (tau .125), several critic updates per
env step, actor every ``policy_frequency`` updates, empirical observation normalisation. Multi-node: every rank rolls out
its own envs into its own replay buffer; critic / actor / alpha gradients are all-reduced (mean) every update; rank 0
broadcasts the initial weights and alone writes checkpoints.

Residual: executed = student + WX_RRES_SCALE * residual (gripper dims per WX_RRES_FIXED) -> actuator model, exactly as
``wx_student.install_rres``. Actor input = ``rres_features`` (deployable: the student's LSTM state, action and inputs),
critic input = privileged ``critic`` obs group + the student's action. Reward = the env's task reward.

Run:  wx_run.sh fsac <tag> [WX_WORLD/WX_RANK/WX_MASTER/WX_PORT] WX_STUDENT=... <perception env> WX_FS_* ...
Eval: wx_run.sh evalc <tag> WX_EVALC_SCRIPT=wx_fsac:FSACResScript WX_FSAC=~/wx/<tag>_fs.pt [WX_FS_EVAL=mean|sample]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wx_student import RRES_FIXED, RRES_SCALE, StudentScript, _f, register, rres_features


def _i(name, default):
    return int(_f(name, default))


# ---------------------------------------------------------------- networks


def _mlp(i, dims, ln=True):
    layers = []
    for d in dims:
        layers += [nn.Linear(i, d)] + ([nn.LayerNorm(d)] if ln else []) + [nn.SiLU()]
        i = d
    return nn.Sequential(*layers), i


class Normalizer(nn.Module):
    """Running mean / var (all-reduced across ranks when ``dist`` is given); (x - mean) / (std + 1e-2)."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x, dist=None):
        x = x.float()
        n = torch.tensor(float(x.shape[0]), device=x.device)
        s, ss = x.sum(0), (x * x).sum(0)
        if dist is not None:
            for t in (n, s, ss):
                dist.all_reduce(t)
        m = s / n
        v = (ss / n - m * m).clamp(min=0)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta * delta * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x.float() - self.mean) / (self.var.sqrt() + 1e-2)


class Actor(nn.Module):
    def __init__(self, odim, adim, hidden=(512, 256, 128), bound=1.0, log_std_min=-5.0, log_std_max=0.0, std0=0.25):
        super().__init__()
        self.cfg = dict(odim=odim, adim=adim, hidden=list(hidden), bound=bound, log_std_min=log_std_min,
                        log_std_max=log_std_max, std0=std0)
        self.trunk, h = _mlp(odim, hidden)
        self.mu, self.ls = nn.Linear(h, adim), nn.Linear(h, adim)
        for m in (self.mu, self.ls):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # initial pre-tanh std = std0 (bias of the log-std head), mean 0: the policy starts as the student + noise
        u = (math.log(std0) - log_std_min) / (0.5 * (log_std_max - log_std_min)) - 1.0
        self.ls.bias.data.fill_(math.atanh(max(-0.99, min(0.99, u))))
        self.bound, self.lmin, self.lmax = bound, log_std_min, log_std_max

    def dist(self, x):
        f = self.trunk(x)
        log_std = self.lmin + 0.5 * (self.lmax - self.lmin) * (torch.tanh(self.ls(f).float()) + 1.0)
        return self.mu(f).float(), log_std

    def sample(self, x):
        mu, log_std = self.dist(x)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)  # tanh correction
        return a * self.bound, logp

    def mean_action(self, x):
        return torch.tanh(self.dist(x)[0]) * self.bound


class DistQ(nn.Module):
    def __init__(self, idim, hidden=(768, 384, 192), atoms=101, vmin=-20.0, vmax=20.0):
        super().__init__()
        self.trunk, h = _mlp(idim, hidden)
        self.out = nn.Linear(h, atoms)
        self.atoms, self.vmin, self.vmax = atoms, vmin, vmax
        self.register_buffer("support", torch.linspace(vmin, vmax, atoms))

    def forward(self, s, a):
        return self.out(self.trunk(torch.cat((s, a), -1)))

    def value(self, logits):
        return (F.softmax(logits.float(), -1) * self.support).sum(-1)

    @torch.no_grad()
    def project(self, s, a, ret, boot_disc):
        """C51 target: distribution of ret + boot_disc * Z(s, a) projected on the support (boot_disc = 0 at ends)."""
        p = F.softmax(self(s, a).float(), -1)
        dz = (self.vmax - self.vmin) / (self.atoms - 1)
        tz = (ret.unsqueeze(1) + boot_disc.unsqueeze(1) * self.support).clamp(self.vmin, self.vmax)
        b = (tz - self.vmin) / dz
        lo, up = b.floor().long(), b.ceil().long()
        lo[(up > 0) & (lo == up)] -= 1
        up[(lo < self.atoms - 1) & (lo == up)] += 1
        out = torch.zeros_like(p)
        out.scatter_add_(1, lo, p * (up.float() - b))
        out.scatter_add_(1, up, p * (b - lo.float()))
        return out


# ---------------------------------------------------------------- trainer


class FastSAC:
    def __init__(self, env):
        self.env = env
        dev = self.dev = env.device
        n = self.n = env.num_envs
        self.world, self.rank = _i("WX_WORLD", 1), _i("WX_RANK", 0)
        self.dist = None
        if self.world > 1:
            import datetime

            import torch.distributed as dist

            dist.init_process_group("nccl", init_method=f"tcp://{os.environ['WX_MASTER']}:{os.environ.get('WX_PORT', '29571')}",
                                    rank=self.rank, world_size=self.world,
                                    timeout=datetime.timedelta(minutes=_f("WX_NCCL_TIMEOUT_MIN", 30)))
            self.dist = dist
            print(f"WX_FS dist rank {self.rank}/{self.world} up", flush=True)
        with torch.inference_mode(False):
            self.s = StudentScript(env)
        self.mask = torch.ones(12, device=dev)
        self.mask[RRES_FIXED] = 0.0
        self.gamma, self.nstep = _f("WX_FS_GAMMA", 0.995), _i("WX_FS_NSTEP", 8)
        self.B = _i("WX_FS_BATCH", 2048)  # per rank
        self.updates, self.pfreq = _i("WX_FS_UPDATES", 8), _i("WX_FS_PFREQ", 4)
        self.tau = _f("WX_FS_TAU", 0.125)
        self.steps_total = _i("WX_FS_STEPS", 20000)
        self.learn_start = _i("WX_FS_START", 64)  # env steps before critic updates
        self.actor_start = _i("WX_FS_ACTOR_START", 256)  # env steps before actor updates (critic warm-up)
        self.target_ent = _f("WX_FS_TARGET_ENT", 0.0)
        self.mu_l2 = _f("WX_FS_MU_L2", 0.0)
        self.cap = _i("WX_FS_BUF", 1536)  # per-env ring length
        self.out = os.path.expanduser(os.environ.get("WX_OUT", "~/wx/fsac.pt"))
        self.base = self.s.propose()
        self.a_obs = rres_features(self.s, self.base)
        self.c_obs = torch.cat((torch.nan_to_num(env.obs_buf["critic"]), self.base), -1)
        ad, cd = self.a_obs.shape[-1], self.c_obs.shape[-1]
        self.actor = Actor(ad, 12, bound=_f("WX_FS_BOUND", 1.0), std0=_f("WX_FS_STD0", 0.25)).to(dev)
        self.q = nn.ModuleList([DistQ(cd + 12, vmin=_f("WX_FS_VMIN", -20.0), vmax=_f("WX_FS_VMAX", 20.0)) for _ in range(2)]).to(dev)
        self.qt = nn.ModuleList([DistQ(cd + 12, vmin=_f("WX_FS_VMIN", -20.0), vmax=_f("WX_FS_VMAX", 20.0)) for _ in range(2)]).to(dev)
        self.na, self.nc = Normalizer(ad).to(dev), Normalizer(cd).to(dev)
        self.log_alpha = torch.tensor([math.log(_f("WX_FS_ALPHA0", 1e-3))], device=dev, requires_grad=True)
        if self.dist is not None:
            for t in list(self.actor.state_dict().values()) + list(self.q.state_dict().values()):
                self.dist.broadcast(t, 0)
        self.qt.load_state_dict(self.q.state_dict())
        lr, wd, betas = _f("WX_FS_LR", 3e-4), _f("WX_FS_WD", 1e-3), (0.9, 0.95)
        self.opt_q = torch.optim.AdamW(self.q.parameters(), lr=lr, weight_decay=wd, betas=betas)
        self.opt_a = torch.optim.AdamW(self.actor.parameters(), lr=_f("WX_FS_LR_A", lr), weight_decay=wd, betas=betas)
        self.opt_al = torch.optim.AdamW([self.log_alpha], lr=lr, weight_decay=0.0, betas=betas)
        # replay: per-env rings, time-aligned across envs (bf16 observations)
        C = self.cap
        self.bA = torch.zeros(C, n, ad, dtype=torch.bfloat16, device=dev)
        self.bC = torch.zeros(C, n, cd, dtype=torch.bfloat16, device=dev)
        self.bU = torch.zeros(C, n, 12, device=dev)
        self.bR = torch.zeros(C, n, device=dev)
        self.bD = torch.zeros(C, n, dtype=torch.bool, device=dev)
        self.t = 0
        self.ep, self.k_upd = [], 0
        self.ret = torch.zeros(n, device=dev)
        self.ep_ret = []
        self.stats = {}
        self.t0 = time.time()
        mb = (self.bA.numel() + self.bC.numel()) * 2 / 2**30
        print(f"WX_FS n={n} adim={ad} cdim={cd} buf={C}x{n} ({mb:.1f} GB) batch={self.B}x{self.world} upd={self.updates} "
              f"pfreq={self.pfreq} gamma={self.gamma} nstep={self.nstep} steps={self.steps_total} scale={RRES_SCALE} "
              f"fixed={RRES_FIXED} out={self.out}", flush=True)

    # -- distributed helpers
    def _allreduce_grads(self, params):
        if self.dist is None:
            return
        gs = [p.grad for p in params if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in gs])
        self.dist.all_reduce(flat)
        flat /= self.world
        o = 0
        for g in gs:
            g.copy_(flat[o:o + g.numel()].view_as(g))
            o += g.numel()

    # -- replay sampling (n-step, stops at the first done)
    def _sample(self):
        C, n, N = self.cap, self.n, self.nstep
        hi = self.t - N  # latest start whose n-th successor is written
        lo = max(0, self.t - C)
        ts = torch.randint(lo, hi, (self.B,), device=self.dev)
        es = torch.randint(0, n, (self.B,), device=self.dev)
        ret = torch.zeros(self.B, device=self.dev)
        alive = torch.ones(self.B, device=self.dev)
        disc = torch.ones(self.B, device=self.dev)
        for k in range(N):
            idx = (ts + k) % C
            ret += alive * disc * self.bR[idx, es]
            disc = disc * self.gamma
            alive = alive * (~self.bD[idx, es]).float()
        nxt = (ts + N) % C
        cur = ts % C
        return (self.bA[cur, es].float(), self.bC[cur, es].float(), self.bU[cur, es], ret,
                self.bA[nxt, es].float(), self.bC[nxt, es].float(), alive * disc)

    def _update(self, do_actor):
        a_o, c_o, u, ret, a_n, c_n, boot = self._sample()
        a_o, a_n = self.na(a_o), self.na(a_n)
        c_o, c_n = self.nc(c_o), self.nc(c_n)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a2, lp2 = self.actor.sample(a_n)
            # soft target: entropy bonus folded into the return of the bootstrapped part
            r_soft = ret - boot * alpha * lp2.float()
            tgt = [qt.project(c_n, a2, r_soft, boot) for qt in self.qt]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_q = sum(-(t * F.log_softmax(q(c_o, u).float(), -1)).sum(-1).mean() for q, t in zip(self.q, tgt))
        self.opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        self._allreduce_grads(list(self.q.parameters()))
        self.opt_q.step()
        st = {"loss_q": float(loss_q)}
        if do_actor:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, lp = self.actor.sample(a_o)
                qv = torch.stack([q.value(q(c_o, a)) for q in self.q]).mean(0)
                loss_a = (alpha * lp.float() - qv).mean()
                if self.mu_l2 > 0:
                    loss_a = loss_a + self.mu_l2 * self.actor.dist(a_o)[0].pow(2).mean()
            self.opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            self._allreduce_grads(list(self.actor.parameters()))
            self.opt_a.step()
            loss_al = (-self.log_alpha.exp() * (lp.detach().float() + self.target_ent * 1.0)).mean()
            self.opt_al.zero_grad(set_to_none=True)
            loss_al.backward()
            if self.dist is not None:
                self.dist.all_reduce(self.log_alpha.grad)
                self.log_alpha.grad /= self.world
            self.opt_al.step()
            st.update(loss_a=float(loss_a), q=float(qv.mean()), ent=float(-lp.mean()), alpha=float(self.log_alpha.exp()))
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                pt.lerp_(p, self.tau)
        return st

    # -- one env step
    def step(self, wrapper, inner_step):
        env = self.env
        with torch.no_grad():
            a_o = self.na(self.a_obs)
            if self.t < self.learn_start and os.environ.get("WX_FS_WARM_ZERO", "0") == "1":
                res = torch.zeros(self.n, 12, device=self.dev)
            else:
                res, _ = self.actor.sample(a_o)
                mres = self.actor.mean_action(a_o)
                self.stats["res_abs"] = 0.95 * self.stats.get("res_abs", 0.0) + 0.05 * float(res.abs().mean())
                self.stats["mean_abs"] = 0.95 * self.stats.get("mean_abs", 0.0) + 0.05 * float(mres.abs().mean())
            res = res.float()
        executed = self.s.commanded(self.base + RRES_SCALE * self.mask * res)
        result = inner_step(wrapper, executed)
        rew, dones = result[1].float().view(-1), result[2].bool().view(-1)
        i = self.t % self.cap
        self.bA[i] = self.a_obs.to(torch.bfloat16)
        self.bC[i] = self.c_obs.to(torch.bfloat16)
        self.bU[i] = res
        self.bR[i] = torch.nan_to_num(rew)
        self.bD[i] = dones
        self.ret += torch.nan_to_num(rew)
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            self.s.reset(ids)
            self.ep = (self.ep + env._bi_final[ids, -1].int().tolist())[-2000:]
            self.ep_ret = (self.ep_ret + self.ret[ids].tolist())[-2000:]
            self.ret[ids] = 0.0
        self.base = torch.nan_to_num(self.s.propose())
        self.a_obs = torch.nan_to_num(rres_features(self.s, self.base)).clone()
        self.c_obs = torch.cat((torch.nan_to_num(env.obs_buf["critic"]), self.base), -1).clone()
        # normaliser stats from the fresh observations (all ranks)
        self.na.update(self.a_obs, self.dist)
        self.nc.update(self.c_obs, self.dist)
        self.t += 1
        if self.t > max(self.learn_start, self.nstep + 2):
            for j in range(self.updates):
                do_actor = self.t >= self.actor_start and j % self.pfreq == self.pfreq - 1
                st = self._update(do_actor)
                for k, v in st.items():
                    self.stats[k] = 0.95 * self.stats.get(k, v) + 0.05 * v
        if self.t % 100 == 0:
            succ = sum(self.ep) / max(1, len(self.ep))
            er = sum(self.ep_ret) / max(1, len(self.ep_ret))
            print(f"WX_FS r={self.rank} t={self.t} succ_last{len(self.ep)}={succ:.3f} ep_ret={er:.2f} "
                  f"{json.dumps({k: round(v, 4) for k, v in self.stats.items()})} min={(time.time() - self.t0) / 60:.1f}",
                  flush=True)
        if self.rank == 0 and self.t % _i("WX_FS_SAVE", 500) == 0:
            self.save(self.out)
            self.save(self.out.replace(".pt", f"_t{self.t}.pt"))
        if self.t >= self.steps_total:
            if self.rank == 0:
                self.save(self.out)
            print(f"WX_DONE {self.out}", flush=True)
            os._exit(0)
        return result

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "cfg": self.actor.cfg, "na": self.na.state_dict(),
                    "q": self.q.state_dict(), "nc": self.nc.state_dict(), "log_alpha": float(self.log_alpha),
                    "t": self.t, "env": {k: v for k, v in os.environ.items() if k.startswith(("WX_", "BITIDY_"))}},
                   path + ".tmp")
        os.replace(path + ".tmp", path)


def install_fsac() -> list[str]:
    """``isaaclab play`` callback (wx_run.sh fsac): the play loop's actions are replaced by student + FastSAC residual."""
    register()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    inner_step = RslRlVecEnvWrapper.step
    state: dict = {}

    def step(self, actions):
        with torch.inference_mode(False):
            if "f" not in state:
                torch.manual_seed(_i("WX_SEED", 0) + 1000 * _i("WX_RANK", 0))
                state["f"] = FastSAC(self.unwrapped)
            return state["f"].step(self, inner_step)

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


class FSACResScript(StudentScript):
    """Eval: frozen student + FastSAC residual actor (``WX_FSAC``); ``WX_FS_EVAL=mean`` (default) or ``sample``."""

    def __init__(self, env):
        super().__init__(env)
        ck = torch.load(os.path.expanduser(os.environ["WX_FSAC"]), map_location=env.device, weights_only=False)
        cfg = ck["cfg"]
        self.actor = Actor(cfg["odim"], cfg["adim"], tuple(cfg["hidden"]), cfg["bound"], cfg["log_std_min"],
                           cfg["log_std_max"], cfg["std0"]).to(env.device).eval()
        self.actor.load_state_dict(ck["actor"])
        self.na = Normalizer(cfg["odim"]).to(env.device)
        self.na.load_state_dict(ck["na"])
        self.mask = torch.ones(12, device=env.device)
        self.mask[RRES_FIXED] = 0.0
        self.mode = os.environ.get("WX_FS_EVAL", "mean")
        print(f"WX_FSAC eval {os.environ['WX_FSAC']} t={ck.get('t')} mode={self.mode} scale={RRES_SCALE}", flush=True)

    @torch.no_grad()
    def act(self):
        base = self.propose()
        x = self.na(rres_features(self, base))
        res = self.actor.mean_action(x) if self.mode == "mean" else self.actor.sample(x)[0]
        return self.commanded(base + RRES_SCALE * self.mask * res.float())
