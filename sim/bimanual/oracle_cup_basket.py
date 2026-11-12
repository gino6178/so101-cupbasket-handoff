"""Scripted oracle for BiTidy-CupBasket-SO101 (``BITIDY_SCRIPT=oracle_cup_basket:CupBasketScript``).

BOTH ARMS AT ONCE: arm s executes command slot s (``cup_basket_env_cfg``: the env assigns each (object, basket) pair of
the instruction to the nearest arm); an arm with an inactive slot, or done, waits at its idle point
(``BITIDY_CB_IDLE_FWD`` / ``_IDLE_Z``). Row mutex (``BITIDY_CB_MUTEX``): one arm at a time carries / places over the
baskets, the other holds its lifted vessel until the row is free.

RIM PINCH (the SO-101 jaws open only ~4.6 cm): the FIXED jaw goes inside the wall, the moving jaw outside. The grasp
azimuth u on the rim follows the jaws' current closing axis (fixed -> moving, horizontal part) until the descent, so the
wrist roll stays at home (the vessel is round). The IK steers the FIXED fingertip to ``FIX_IN`` inside the inner rim,
``PINCH_DEPTH`` below it.

Phases (:class:`RimArm`): 0 rise to ``TRANSIT_Z`` over the grasp point, 1 hover ``HOVER`` above it, 2 down, 3 close,
4 lift to the carry height (vessel bottom ``CARRY_CLEAR`` above every basket wall and every other vessel's rim),
5 carry over the stack axis (zone centre if the zone is empty), 6 lower until the vessel bottom is ``REL_ABOVE``
(cup / bowl; ``REL_EMPTY`` onto the floor) above its nested height and centred (steering the measured vessel, not the
pads), 7 open, 8 straight up ``UP``, 9 park at the idle point. Lost in the jaws during 4-6 -> open; back on the table
after the drop -> retry from 0 (``RETRIES``).

``BITIDY_DEBUG=1`` traces one env (``BITIDY_DEBUG_ENV``) every ``BITIDY_DEBUG_EVERY`` steps; ``BITIDY_ROWS=1`` prints
per-phase reached / forced rates of first episodes (``BITIDY_CB_PHASES``).
"""

from __future__ import annotations

import os

import torch

from . import cup_basket_env_cfg as T
from . import env_cfg as base
from .scripted import ArmIK

RISE, HOVER_P, DOWN, CLOSE, LIFT, CARRY, LOWER, OPEN, UP, PARK = range(10)
LAST = PARK
ARMS = ("robot_left", "robot_right")


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _col(v: torch.Tensor) -> torch.Tensor:
    return v.unsqueeze(-1)


class RimArm:
    """Rim-pinch pick and place for one arm (all envs; acts only where ``active``)."""

    TRANSIT_Z = _f("BITIDY_CB_TRANSIT_Z", 0.14)
    HOVER = _f("BITIDY_CB_HOVER", 0.04)
    PINCH_DEPTH = _f("BITIDY_CB_PINCH_DEPTH", 0.018)
    FIX_IN = _f("BITIDY_CB_FIX_IN", 0.0015)  # fixed fingertip this far inside the inner rim
    CARRY_CLEAR = _f("BITIDY_CB_CARRY_CLEAR", 0.015)
    ZMAX = _f("BITIDY_CB_ZMAX", 0.23)
    LOWER_MODE = os.environ.get("BITIDY_CB_LOWER", "measured")
    REL_ABOVE = (_f("BITIDY_CB_REL_CUP", 0.015), _f("BITIDY_CB_REL_BOWL", 0.02))
    """Release height above the nest (the outer jaw reaches PINCH_DEPTH below the held rim: bowls nest only ~6 mm apart,
    so a bowl is let go ~1.6 cm up and drops into the taper)."""
    REL_EMPTY = _f("BITIDY_CB_REL_EMPTY", 0.01)
    UP_DZ = _f("BITIDY_CB_UP", 0.05)
    TILT = _f("BITIDY_CB_TILT", 0.5)
    TILT_CARRY = _f("BITIDY_CB_TILT_CARRY", 0.5)
    PW = _f("BITIDY_CB_PW", 0.01)
    PW_CARRY = _f("BITIDY_CB_PW_CARRY", 0.005)
    ROLL = _f("BITIDY_CB_ROLL", 0.0)
    GAIN = _f("BITIDY_IK_GAIN", 1.0)
    NEAR = _f("BITIDY_IK_NEAR", 0.08)
    BIAS_CLAMP = _f("BITIDY_CB_BIAS_CLAMP", 0.04)
    FREE_SPEED = _f("BITIDY_CB_FREE_SPEED", 0.004)
    CARRY_SPEED = _f("BITIDY_CB_CARRY_SPEED", 0.0015)
    HOVER_TOL = _f("BITIDY_CB_HOVER_TOL", 0.008)
    GRASP_TOL = _f("BITIDY_CB_GRASP_TOL", 0.004)
    CENTRE = _f("BITIDY_CB_CENTRE", 0.003)
    OVER_OK = _f("BITIDY_CB_OVER_OK", 0.007)
    CLOSE_STEPS = int(_f("BITIDY_CB_CLOSE_STEPS", 30))
    OPEN_STEPS = int(_f("BITIDY_CB_OPEN_STEPS", 15))
    DROP_ANYWAY = int(_f("BITIDY_CB_DROP_ANYWAY", 90))
    PHASE_LIMIT = int(_f("BITIDY_PHASE_LIMIT", 150))
    LOST_STEPS = int(_f("BITIDY_CB_LOST_STEPS", 10))
    LAG = _f("BITIDY_CB_LAG", 0.025)
    STALL = int(_f("BITIDY_CB_STALL", 30))  # steps without 1 mm progress -> "close enough" (if within STALL_TOL)
    STALL_TOL = _f("BITIDY_CB_STALL_TOL", 0.03)
    STALL_TOL_DOWN = _f("BITIDY_CB_STALL_TOL_DOWN", 0.012)

    def __init__(self, env, arm: str):
        self.env, self.arm = env, arm
        self.ik = ArmIK(env, arm)
        n, dev = env.num_envs, env.device
        self.p = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros_like(self.p)
        self.lost_t = torch.zeros_like(self.p)
        self.prev_p = torch.full_like(self.p, -1)
        self.u = torch.zeros((n, 2), device=dev)
        self.u[:, 0] = 1.0
        self.gpad = torch.zeros((n, 3), device=dev)
        self.off = torch.zeros((n, 3), device=dev)
        self.off0 = torch.zeros((n, 3), device=dev)
        self.release = torch.zeros((n, 3), device=dev)
        self.carrot = torch.zeros((n, 3), device=dev)
        self.bias = torch.zeros((n, 3), device=dev)
        self.reached = torch.zeros((n, LAST + 1), dtype=torch.bool, device=dev)
        self.forced = torch.zeros_like(self.reached)
        self.stalled = torch.zeros_like(self.reached)
        self.best = torch.full((n,), 9.0, device=dev)
        self.stall_t = torch.zeros_like(self.p)

    def reset(self, ids: torch.Tensor) -> None:
        for t in (self.p, self.t, self.lost_t):
            t[ids] = 0
        self.prev_p[ids] = -1
        self.bias[ids] = 0.0
        self.reached[ids] = False
        self.forced[ids] = False
        self.stalled[ids] = False
        self.best[ids] = 9.0
        self.stall_t[ids] = 0

    def idle_point(self) -> torch.Tensor:
        b = self.ik.base_xy
        fwd = torch.tensor((1.0, 0.0) if self.arm == "robot_left" else (-1.0, 0.0), device=b.device)
        xy = b + fwd * _f("BITIDY_CB_IDLE_FWD", 0.09)
        return torch.cat((xy, torch.full_like(xy[:, :1], _f("BITIDY_CB_IDLE_Z", 0.16))), -1)

    def idle(self) -> torch.Tensor:
        """Back off to the idle point, jaws open (for envs where the other arm works)."""
        t = self.idle_point()
        act, _ = self.ik(t, self.ik.tilted(t, self.TILT), None, gain=0.5, point_weight=self.PW)
        return torch.cat((act * 0.5, torch.ones_like(act[:, :1])), -1)

    def act(self, active: torch.Tensor, c: dict) -> tuple[torch.Tensor, torch.Tensor]:
        env, p = self.env, self.p
        n = env.num_envs
        fixed, moving, pad = T.jaw_points(env, self.arm)
        self.ik.base_xy = self.ik.robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        obj, rc, typ = c["pos"], c["rim"], c["typ"]
        up = obj.new_tensor((0.0, 0.0, 1.0))

        def at(xy, z):
            return torch.cat((xy, z.unsqueeze(-1) if torch.is_tensor(z) else torch.full_like(xy[:, :1], z)), -1)

        # Grasp azimuth: follows the jaws' closing axis (fixed -> moving) until the descent starts.
        ax = moving[:, :2] - fixed[:, :2]
        self.u = torch.where(_col(p <= HOVER_P), torch.nn.functional.normalize(ax, dim=-1), self.u)
        rf = c["r_in"] - self.FIX_IN
        g_fixed = at(rc[:, :2] + self.u * _col(rf), rc[:, 2] - self.PINCH_DEPTH)
        self.gpad = torch.where(_col(p <= HOVER_P), g_fixed + (pad - fixed), self.gpad)  # locked from the descent on
        near = T.near_rim(env, c["slot"])
        self.off = torch.where(_col(((p >= CLOSE) & (p <= LOWER) & near) | (p < CLOSE)),
                               (obj - pad).clamp(-0.08, 0.08), self.off)
        carry_z = (c["clear_z"] - self.off[:, 2]).clamp(max=self.ZMAX)
        rel = torch.where(typ == T.CUP, self.REL_ABOVE[0], self.REL_ABOVE[1])
        rel = torch.where(c["depth"] > 0, rel, torch.full_like(rel, self.REL_EMPTY))
        # Pads relative to the vessel's bottom centre at the grasp (vessel upright on the table).
        self.off0 = torch.where(_col(p <= CLOSE), (pad - obj).clamp(-0.08, 0.08), self.off0)
        if self.LOWER_MODE == "upright":  # lower to where the pads would be with the vessel UPRIGHT at its nest: the
            # hanging (tilted) vessel touches down on its low edge and is pressed upright before the jaws open
            # (the grasp offset rotates with the wrist: radius from the grasp, direction = the jaws' current axis)
            r0 = torch.linalg.vector_norm(self.off0[:, :2], dim=-1, keepdim=True)
            u_now = torch.nn.functional.normalize(ax, dim=-1)
            place_xy = torch.where(_col(p >= LOWER), c["axis"] + r0 * u_now, c["axis"] - self.off[:, :2])
            lower_z = c["nest_z"] + rel + self.off0[:, 2]
        else:
            place_xy = c["axis"] - self.off[:, :2]
            lower_z = c["nest_z"] + rel - self.off[:, 2]
        self.release = torch.where(_col(p <= OPEN), pad, self.release)

        t = at(self.gpad[:, :2], self.TRANSIT_Z)
        t = torch.where(_col(p >= HOVER_P), self.gpad + up * self.HOVER, t)
        t = torch.where(_col(p >= DOWN), self.gpad, t)
        t = torch.where(_col(p >= LIFT), at(self.gpad[:, :2], carry_z), t)
        t = torch.where(_col(p >= CARRY), at(place_xy, carry_z), t)
        t = torch.where(_col(p >= LOWER), at(place_xy, lower_z), t)
        t = torch.where(_col(p >= OPEN), self.release, t)
        t = torch.where(_col(p >= UP), self.release + up * self.UP_DZ, t)
        t = torch.where(_col(p >= PARK), self.idle_point(), t)

        # Carrot for the free / carry legs (slow while carrying).
        changed = p != self.prev_p
        self.carrot = torch.where(_col(changed), pad, self.carrot)
        self.prev_p = p.clone()
        step = t - self.carrot
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        speed = torch.where((p == LIFT) | (p == CARRY) | (p == LOWER), self.CARRY_SPEED, self.FREE_SPEED)
        # The carrot waits while the pads lag more than LAG behind it (the arm sags far out: no low sweeps).
        self.carrot = self.carrot + step * (_col(speed) / dist.clamp(min=1e-6)).clamp(max=1.0)
        # Leash: the carrot stays within LAG of the pads (the arm sags far out: no runaway target, no low sweeps).
        lead = self.carrot - pad
        ln = torch.linalg.vector_norm(lead, dim=-1, keepdim=True)
        self.carrot = torch.where(ln > self.LAG, pad + lead * (self.LAG / ln.clamp(min=1e-6)), self.carrot)
        carrot_at = torch.linalg.vector_norm(self.carrot - t, dim=-1) < 1e-4
        use_carrot = (p == RISE) | (p == LIFT) | (p == CARRY) | (p == LOWER) | (p == UP) | (p == PARK)
        goal = torch.where(_col(use_carrot), self.carrot, t)
        grip = torch.where((p >= CLOSE) & (p < OPEN), -1.0, 1.0)

        nb = _col(torch.linalg.vector_norm(goal - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(nb, self.bias + 0.05 * (goal - pad), 0.9 * self.bias).clamp(-self.BIAS_CLAMP, self.BIAS_CLAMP)
        self.bias[:, 2] = torch.where(p <= DOWN, self.bias[:, 2].clamp(min=0.0), self.bias[:, 2])
        carrying = (p >= LIFT) & (p <= UP)
        tilt = torch.where(_col(carrying), self.ik.tilted(goal, self.TILT_CARRY), self.ik.tilted(goal, self.TILT))
        pw = torch.where(carrying, torch.full_like(rf, self.PW_CARRY), torch.full_like(rf, self.PW))
        roll = torch.full_like(rf, self.ROLL)
        act, _ = self.ik(goal + self.bias, tilt, roll, gain=self.GAIN, point_weight=pw)
        err = torch.linalg.vector_norm(t - pad, dim=-1)

        dxy = torch.linalg.vector_norm(obj[:, :2] - c["axis"], dim=-1)
        centred = dxy < self.CENTRE
        over_ok = dxy < self.OVER_OK
        # Phase clock: carrot phases only count once the carrot is at the waypoint (long carries are slow on purpose).
        self.t += (carrot_at | ~use_carrot).long()
        adv = torch.zeros_like(active)
        adv |= (p == RISE) & carrot_at & (err < 0.02)
        adv |= (p == HOVER_P) & (err < self.HOVER_TOL)
        adv |= (p == DOWN) & (err < self.GRASP_TOL)
        adv |= (p == CLOSE) & (self.t > self.CLOSE_STEPS)
        adv |= (p == LIFT) & carrot_at & (err < 0.01)
        adv |= (p == CARRY) & carrot_at & (err < 0.008)
        adv |= (p == LOWER) & carrot_at & (err < 0.005) & centred
        adv |= (p == LOWER) & over_ok & (self.t > self.DROP_ANYWAY)
        adv |= (p == OPEN) & (self.t > self.OPEN_STEPS)
        adv |= (p == UP) & (err < 0.01)
        # "Close enough and not improving": the arm cannot get nearer (sag / reach limit): move on.
        better = err < self.best - 0.001
        self.best = torch.where(better, err, self.best)
        self.stall_t = torch.where(better, 0, self.stall_t + 1)
        tol = torch.where(p == DOWN, self.STALL_TOL_DOWN, self.STALL_TOL)
        free = (p == RISE) | (p == HOVER_P) | (p == DOWN) | (p == LIFT) | (p == CARRY) | (p == UP)
        stall = free & (self.stall_t > self.STALL) & (err < tol)
        stall |= (p == LOWER) & over_ok & (self.stall_t > self.STALL) & (err < 0.015)
        stall &= ~adv
        adv = adv | stall
        hold_on = (p == LOWER) & ~over_ok
        moved = adv | ((self.t > self.PHASE_LIMIT) & ~hold_on & (p < LAST))
        self.lost_t = torch.where((p >= LIFT) & (p <= LOWER) & ~near, self.lost_t + 1, 0)
        lost = self.lost_t > self.LOST_STEPS
        moved &= active
        lost &= active
        idx = torch.arange(n, device=env.device)
        self.reached[idx, p] |= active
        fid = (moved & ~adv).nonzero().squeeze(-1)
        self.forced[fid, p[fid]] = True
        sid = (moved & stall).nonzero().squeeze(-1)
        self.stalled[sid, p[sid]] = True
        self.p = torch.where(moved, (p + 1).clamp(max=LAST), p)
        self.p = torch.where(lost, OPEN, self.p)
        self.t = torch.where(moved | lost, 0, self.t)
        self.best = torch.where(moved | lost, 9.0, self.best)
        self.stall_t = torch.where(moved | lost, 0, self.stall_t)
        self.lost_t = torch.where(lost, 0, self.lost_t)

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if (os.environ.get("BITIDY_DEBUG") and bool(active[k])
                and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 10)) == 0):
            r3 = lambda v: [round(x, 3) for x in v[k].tolist()]  # noqa: E731
            print(f"[cb {self.arm[6:]}] t={int(env.episode_length_buf[k])} p={int(p[k])} err={float(err[k]):.4f} pad={r3(pad)} "
                  f"goal={r3(goal)} obj={r3(obj)} rim={r3(rc)} off={r3(self.off)} near={bool(near[k])} "
                  f"fix={r3(fixed)} mov={r3(moving)} axis={r3(c['axis'])} nest_z={float(c['nest_z'][k]):.3f} depth={int(c['depth'][k])} "
                  f"flags={T.progress(env).flags[k].int().tolist()}", flush=True)
        return torch.cat((act, grip.unsqueeze(-1)), -1), err


class CupBasketScript:
    """Both arms at once: arm s works on command slot s (the env's nearest-arm assignment); an inactive slot's arm
    stays home. Only one arm at a time is over the basket row (carry / lower / open / up): the first to finish its lift
    takes the row (``BITIDY_CB_MUTEX=1``), the other holds its vessel at carry height until the row is free."""

    START_DELAY = int(_f("BITIDY_CB_START_DELAY", 5))
    RETRIES = int(_f("BITIDY_CB_RETRIES", 1))
    MUTEX = os.environ.get("BITIDY_CB_MUTEX", "1") == "1"

    def __init__(self, env):
        self.env = env
        self.arms = [RimArm(env, a) for a in ARMS]
        n, dev = env.num_envs, env.device
        self.retries = torch.zeros((n, 2), dtype=torch.long, device=dev)
        self.owner = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.first = torch.ones(n, dtype=torch.bool, device=dev)
        self.tot_reached = torch.zeros((2, LAST + 1), device=dev)
        self.tot_forced = torch.zeros((2, LAST + 1), device=dev)
        self.tot_stalled = torch.zeros((2, LAST + 1), device=dev)
        self.n_act = torch.zeros(2, device=dev)
        self.episodes = 0

    def reset(self, ids: torch.Tensor) -> None:
        if os.environ.get("BITIDY_ROWS"):
            self._account(ids)
        for a in self.arms:
            a.reset(ids)
        self.retries[ids] = 0
        self.owner[ids] = -1

    def _account(self, ids: torch.Tensor) -> None:
        ids = ids[self.first[ids]]
        if not len(ids):
            return
        self.first[ids] = False
        cmd = T.command(self.env)[ids]
        for s, a in enumerate(self.arms):
            act = (cmd[:, s, 2] == 1).unsqueeze(-1)
            self.n_act[s] += act.float().sum()
            self.tot_reached[s] += (a.reached[ids] & act).float().sum(0)
            self.tot_forced[s] += (a.forced[ids] & act).float().sum(0)
            self.tot_stalled[s] += (a.stalled[ids] & act).float().sum(0)
        self.episodes += len(ids)

        def r3(t, s):
            return [round(v, 3) for v in (t[s] / self.n_act[s].clamp(min=1)).tolist()]

        print(f"BITIDY_CB_PHASES n={self.episodes} " + " | ".join(
            f"{'LR'[s]}: reached={r3(self.tot_reached, s)} forced={r3(self.tot_forced, s)} stalled={r3(self.tot_stalled, s)}"
            for s in range(2)), flush=True)

    def act(self) -> torch.Tensor:
        env = self.env
        pos_all = T.objs_pos(env)
        g = T._geo_t(env.device)
        rims = pos_all[..., 2] + g["rim"][T.obj_type(env.device)].unsqueeze(0)
        started = env.episode_length_buf >= self.START_DELAY
        if os.environ.get("BITIDY_CB_FREEZE"):
            started = torch.zeros_like(started)
        outs, ctx = [], []
        for s, a in enumerate(self.arms):
            typ = T.slot_type(env, s)
            axis, nest_z, depth = T.nest_target(env, s, pos_all)
            mine = torch.arange(T.N_OBJ, device=env.device).unsqueeze(0) == T.slot_obj(env, s).unsqueeze(-1)
            top = torch.where(~mine, rims, torch.full_like(rims, T.B_TOP_Z)).max(-1).values.clamp(min=T.B_TOP_Z)
            c = {"slot": s, "pos": T.slot_pos(env, s, pos_all), "rim": T.rim_center(env, s), "typ": typ,
                 "r_in": g["r_in"][typ], "axis": axis, "nest_z": nest_z, "depth": depth,
                 "clear_z": top + RimArm.CARRY_CLEAR}
            ctx.append(c)
            active = T.slot_active(env, s) & started
            pick, _ = a.act(active, c)
            outs.append(torch.where(_col(active), pick, a.idle()))
        if self.MUTEX:
            # Row mutex: an arm enters CARRY only if it owns the row; ownership ends once that arm parks.
            for s, a in enumerate(self.arms):
                done = (self.owner == s) & ((a.p >= PARK) | ~T.slot_active(env, s))
                self.owner = torch.where(done, -1, self.owner)
            for s, a in enumerate(self.arms):
                want = (a.p == CARRY) & (self.owner != s)
                take = want & (self.owner == -1)
                self.owner = torch.where(take, s, self.owner)
                blocked = want & ~take
                a.p = torch.where(blocked, LIFT, a.p)
                a.t = torch.where(blocked, 0, a.t)
        # Retry: the vessel is back on the table (not over its target) after the arm let go.
        for s, a in enumerate(self.arms):
            c = ctx[s]
            on_table = (c["pos"][:, 2] < base.TABLE_TOP + 0.01) & (torch.linalg.vector_norm(c["pos"][:, :2] - c["axis"], dim=-1) > 0.03)
            retry = (a.p >= UP) & on_table & (self.retries[:, s] < self.RETRIES) & T.slot_active(env, s)
            a.p = torch.where(retry, RISE, a.p)
            a.t = torch.where(retry, 0, a.t)
            self.retries[:, s] += retry.long()
        if os.environ.get("BITIDY_ROWS"):
            ending = (env.episode_length_buf == int(env.max_episode_length) - 2).nonzero().squeeze(-1)
            if len(ending):
                prog = T.progress(env)
                for e in ending.tolist()[:40]:
                    parts = []
                    for s, a in enumerate(self.arms):
                        c = ctx[s]
                        parts.append(f"{'LR'[s]}: act={int(T.slot_active(env, s)[e])} p={int(a.p[e])} depth={int(c['depth'][e])} "
                                     f"obj={[round(v, 3) for v in c['pos'][e].tolist()]} axis={[round(v, 3) for v in c['axis'][e].tolist()]} "
                                     f"tilt={float(T.tilt(T.slot_quat(env, s))[e]):.2f} ok={bool(prog.slot_ok[e, s])}")
                    print(f"BITIDY_CB_END env={e} cmd={T.command(env)[e].tolist()} owner={int(self.owner[e])} " + " | ".join(parts)
                          + f" flags={prog.flags[e].int().tolist()}", flush=True)
                self._account(ending)
        return torch.cat(outs, -1)
