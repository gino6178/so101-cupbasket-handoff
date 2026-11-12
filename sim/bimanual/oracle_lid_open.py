"""Scripted oracle for LidOpen (``BITIDY_SCRIPT=oracle_lid_open:LidOpenScript``): the left arm takes the box's lid
off by its knob, holds it aside and level, puts it back and lets go. Uses the simulator's true lid pose.

Left phases: 0 hover over the knob (settle there, jaws aligned to the knob faces), 1 straight down to the knob,
2 close, 3 lift straight up, 4 aside, 5 hold aside, 6 back over the box, 7 lower onto the rim, 8 open,
9 back off straight up. While the lid is carried (3-7) the waypoint is for the KNOB, so a grasp that is a few mm
off-centre does not carry over into where the lid lands.

``BITIDY_ROWS=1`` prints per-phase reached/forced(timeout) rates and mean durations over finished episodes;
``BITIDY_DEBUG=1`` traces one environment (``BITIDY_DEBUG_ENV``).
"""

from __future__ import annotations

import os

import torch

from . import env_cfg as base
from isaaclab.utils.math import quat_apply

from .scripted import ArmIK, jaw_tips

N_PHASES = 10


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class LidOpenScript:
    HOVER = _f("BITIDY_LO_HOVER", 0.03)
    KNOB_PINCH = _f("BITIDY_LO_PINCH", 0.0)
    LIFT = _f("BITIDY_LO_LIFT", 0.05)
    ASIDE = _f("BITIDY_LO_ASIDE", 0.075)
    ASIDE_UP = _f("BITIDY_LO_ASIDE_UP", 0.04)
    # Aside toward +y (the left arm's side), away from where the right arm brings the cube in the merged task.
    ASIDE_DIR = tuple(float(v) for v in os.environ.get("BITIDY_LO_ASIDE_DIR", "0,1").split(","))
    PLACE_ABOVE = _f("BITIDY_LO_PLACE", 0.001)
    SPEED = _f("BITIDY_LO_SPEED", 0.0015)
    SPEED_FREE = _f("BITIDY_LO_SPEED_FREE", 0.0025)
    SPEED_SLOW = _f("BITIDY_LO_SPEED_SLOW", 0.001)
    LEVEL_GAIN = _f("BITIDY_LO_LEVEL_GAIN", 0.5)
    PW = _f("BITIDY_LO_PW", 0.004)
    PW_CARRY = _f("BITIDY_LO_PW_CARRY", 0.004)
    PITCH_GAIN = _f("BITIDY_LO_PITCH_GAIN", 1.0)
    PRE = _f("BITIDY_LO_PRE", 0.07)
    LOST = _f("BITIDY_LO_LOST", 0.02)
    PINCH_TOL = _f("BITIDY_LO_PINCH_TOL", 0.0025)
    HOVER_TOL = _f("BITIDY_LO_HOVER_TOL", 0.003)
    SETTLE = int(_f("BITIDY_LO_SETTLE", 5))
    HOLD = int(_f("BITIDY_LO_HOLD", 45))
    NEAR = _f("BITIDY_IK_NEAR", 0.08)
    ROLL_GAIN = _f("BITIDY_ROLL_GAIN", 0.3)
    GAIN = _f("BITIDY_IK_GAIN", 1.0)
    TILT = _f("BITIDY_TILT", 1.2)
    PHASE_LIMIT = int(_f("BITIDY_PHASE_LIMIT", 150))

    def __init__(self, env):
        from . import lid_open_env_cfg as task

        self.env, self.T = env, task
        self.arm = "robot_left"
        self.ik = ArmIK(env, self.arm)
        n, dev = env.num_envs, env.device
        self.p = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros_like(self.p)
        self.settle = torch.zeros_like(self.p)
        self.high = torch.ones(n, dtype=torch.bool, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_p = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.wp = torch.zeros((n, 3), device=dev)
        self.roll = torch.zeros(n, device=dev)
        self.bias = torch.zeros((n, 3), device=dev)
        self.knob = torch.zeros((n, 3), device=dev)
        self.hold_pad = torch.zeros((n, 3), device=dev)
        # Phase statistics over finished episodes.
        self.reach = torch.zeros((n, N_PHASES), dtype=torch.bool, device=dev)
        self.forced = torch.zeros_like(self.reach)
        self.dur = torch.zeros((n, N_PHASES), dtype=torch.long, device=dev)
        self.tot = {"n": 0, "reach": torch.zeros(N_PHASES, device=dev), "forced": torch.zeros(N_PHASES, device=dev),
                    "dur": torch.zeros(N_PHASES, device=dev)}

    def _stats(self, ids: torch.Tensor, commit: bool) -> None:
        tot = self.tot
        cnt = tot["n"] + len(ids)
        reach = tot["reach"] + self.reach[ids].float().sum(0)
        forced = tot["forced"] + self.forced[ids].float().sum(0)
        dur = tot["dur"] + self.dur[ids].float().sum(0)
        if commit:
            tot.update(n=cnt, reach=reach, forced=forced, dur=dur)
        r = lambda v: [round(x, 2) for x in v.tolist()]
        print(f"BITIDY_PHASES n={cnt} reached={r(reach / max(cnt, 1))} forced={r(forced / max(cnt, 1))} "
              f"dur={r(dur / reach.clamp(min=1))}", flush=True)

    def reset(self, ids: torch.Tensor) -> None:
        if os.environ.get("BITIDY_ROWS") and len(ids):
            self._stats(ids, True)
        for t in (self.p, self.t, self.settle, self.dur):
            t[ids] = 0
        self.reach[ids] = False
        self.forced[ids] = False
        self.roll[ids] = 0.0
        self.bias[ids] = 0.0
        self.high[ids] = True
        self.prev_p[ids] = -1
        self.retries[ids] = 0

    @staticmethod
    def _yaw(quat: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _face_err(self, yaw: torch.Tensor) -> torch.Tensor:
        fixed, moving = jaw_tips(self.env, self.arm)
        axis = moving - fixed
        jaw = torch.atan2(axis[:, 1], axis[:, 0])
        return torch.remainder(yaw - jaw + torch.pi / 4, torch.pi / 2) - torch.pi / 4

    def act(self) -> torch.Tensor:
        env, T, p = self.env, self.T, self.p
        n, dev = env.num_envs, env.device
        up = torch.tensor((0.0, 0.0, 1.0), device=dev)

        def col(v):
            return v.unsqueeze(-1)

        pad = base.pad_center(env, self.arm)
        knob_now = T.knob_pos(env)
        # Lock the knob once the descent starts, so a nudged lid is not chased.
        self.knob = torch.where(col(p <= 0), knob_now, self.knob)
        box = T.box_center(env)
        box_knob = torch.cat((box, torch.full((n, 1), T.LID_REST_Z + T.KNOB_OFFSET, device=dev)), -1)
        aside_dir = torch.tensor(self.ASIDE_DIR, device=dev).expand(n, -1)
        aside = box_knob + torch.cat((aside_dir * self.ASIDE, torch.zeros_like(box[:, :1])), -1) + up * self.ASIDE_UP

        pinch = self.knob + up * self.KNOB_PINCH
        hover = pinch + up * self.HOVER
        # Pad waypoints (free hand) and knob waypoints (lid in hand).
        # Come in high, then settle at the hover point (a low sweep drags the open jaws over the lid).
        high = col(self.high & (p == 0))
        pad_goal = torch.where(high, pinch + up * self.PRE, hover)
        pad_goal = torch.where(col(p >= 1), pinch, pad_goal)
        pad_goal = torch.where(col(p >= 8), self.hold_pad, pad_goal)
        pad_goal = torch.where(col(p >= 9), self.hold_pad + up * 0.06, pad_goal)
        knob_goal = torch.cat((self.knob[:, :2], self.knob[:, 2:] + self.LIFT), -1)
        knob_goal = torch.where(col(p >= 4), aside, knob_goal)
        knob_goal = torch.where(col(p >= 6), box_knob + up * self.LIFT, knob_goal)
        knob_goal = torch.where(col(p >= 7), box_knob + up * self.PLACE_ABOVE, knob_goal)
        carrying = (p >= 3) & (p <= 7)
        # Carrying: the controlled point is the knob, so the grasp offset does not shift where the lid goes.
        ctrl = torch.where(col(carrying), knob_now, pad)
        goal = torch.where(col(carrying), knob_goal, pad_goal)
        # A carrot moves from the controlled point toward the goal at a limited speed: no overshoot, no swinging lid.
        changed = p != self.prev_p
        self.wp = torch.where(col(changed), ctrl, self.wp)
        self.prev_p = p.clone()
        speed = torch.full((n,), self.SPEED, device=dev)
        speed = torch.where((p == 0) & self.high, self.SPEED_FREE, speed)
        speed = torch.where((p == 1) | (p == 7), self.SPEED_SLOW, speed)
        step = goal - self.wp
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        self.wp = self.wp + step * (col(speed) / dist.clamp(min=1e-6)).clamp(max=1.0)
        target = pad + self.wp - ctrl
        err = torch.linalg.vector_norm(goal - ctrl, dim=-1)
        # Keep the pad where it is while the jaws open.
        self.hold_pad = torch.where(col(p <= 7), pad, self.hold_pad)

        grip = torch.where((p >= 2) & (p < 8), -1.0, 1.0)
        # Jaw roll: keep the closing axis horizontal (feedback on its measured slope) the whole time the jaws work the
        # knob. Pinched across a level axis the lid hangs level below the knob (its plate is the pendulum mass); the
        # 5-DOF wrist cannot also turn the jaws to the knob's yaw, so the knob is turned on the plate to match.
        fixed, moving = jaw_tips(env, self.arm)
        axis = torch.nn.functional.normalize(moving - fixed, dim=-1)
        slope = axis[:, 2]
        face = self._face_err(self._yaw(T.lid_quat(env)) + T.KNOB_YAW)
        self.roll = torch.where((p <= 7) & (p != 2), self.roll - self.LEVEL_GAIN * slope, self.roll).clamp(-2.0, 2.0)

        # Integral term against gravity sag, only near the waypoint.
        near = col(torch.linalg.vector_norm(target - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (target - pad), 0.9 * self.bias).clamp(-0.03, 0.03)
        # Lid in hand: pitch the jaws about their (level) closing axis by the lid's tilt about that axis, so the lid stays
        # level whatever pose the arm takes (the knob does not turn in the pinch).
        direction = self.ik.tilted(target, self.TILT)
        point = torch.nn.functional.normalize(pad - (self.ik.robot.data.body_pos_w.torch[:, self.ik.body] - env.scene.env_origins), dim=-1)
        lid_up = quat_apply(T.lid_quat(env), up.expand(n, -1))
        u = lid_up - (lid_up * axis).sum(-1, keepdim=True) * axis
        zp = up - col(axis[:, 2]) * axis
        angle = torch.atan2((axis * torch.cross(u, zp, dim=-1)).sum(-1), (u * zp).sum(-1))
        a = col(self.PITCH_GAIN * angle)
        keep = point * torch.cos(a) + torch.cross(axis, point, dim=-1) * torch.sin(a) + axis * col((axis * point).sum(-1)) * (1 - torch.cos(a))
        held = col(carrying & T.lid_held(env) | carrying & (p == 3))
        direction = torch.where(held, keep, direction)
        weight = torch.where(carrying, self.PW_CARRY, self.PW)
        act, _ = self.ik(target + self.bias, direction, self.roll, gain=self.GAIN, point_weight=weight)
        # While the jaws close (2) or open (8) the arm holds still: only the moving jaw travels, so the pad midpoint
        # shifts by ~1 cm, and chasing it would sweep the fixed jaw into the knob and push the lid away.
        act = torch.where(col((p == 2) | (p == 8)), torch.zeros_like(act), act)

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 6)) == 0:
            r3 = lambda v: [round(x, 4) for x in v[k].tolist()]
            prog = T.progress(env)
            axis_z = slope
            rob = self.ik.robot
            jq = rob.data.joint_pos.torch[k].tolist()
            jt = rob.data.joint_pos_target.torch[k].tolist()
            lim = rob.data.soft_joint_pos_limits.torch[k].tolist()
            print(f"[lidq] t={int(env.episode_length_buf[k])} names={rob.joint_names} q={[round(v, 3) for v in jq]} "
                  f"tgt={[round(v, 3) for v in jt]} lim={[[round(a, 2), round(b, 2)] for a, b in lim]}", flush=True)
            print(f"[lido] t={int(env.episode_length_buf[k])} p={int(p[k])} err={float(err[k]):.4f} pad={r3(pad)} "
                  f"knob={r3(knob_now)} lid={r3(T.lid_pos(env))} tilt={float(T.lid_tilt(env)[k]):.3f} face={float(face[k]):.3f} "
                  f"open={float(base.gripper_openness(env, self.arm)[k]):.2f} held={bool(T.lid_held(env)[k])} "
                  f"roll={float(self.roll[k]):.3f} axis_z={float(axis_z[k]):.3f} bias={r3(self.bias)} flags={prog.flags[k].int().tolist()} succ={bool(prog.success[k])}", flush=True)

        # Phase transitions.
        self.t += 1
        self.high &= ~((p == 0) & (err < 0.015))
        hover_ok = (err < self.HOVER_TOL) & (slope.abs() < 0.05) & ~self.high
        self.settle = torch.where((p == 0) & hover_ok, self.settle + 1, 0)
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (self.settle >= self.SETTLE)
        adv |= (p == 1) & (err < self.PINCH_TOL)
        adv |= (p == 2) & (self.t > 15)
        adv |= (p == 3) & (err < 0.01)
        adv |= (p == 4) & (err < 0.008)
        adv |= (p == 5) & (self.t > self.HOLD)
        adv |= (p == 6) & (err < 0.006)
        adv |= (p == 7) & (err < 0.004)
        adv |= (p == 8) & (self.t > 15)
        if os.environ.get("BITIDY_ROWS"):
            done0 = ((p == 0) & adv).nonzero().squeeze(-1)
            if len(done0):
                jaw_yaw = torch.atan2(axis[done0, 1], axis[done0, 0])
                print(f"BITIDY_JAWYAW {[round(v, 3) for v in jaw_yaw.tolist()]} face={[round(v, 3) for v in face[done0].tolist()]}", flush=True)
        if os.environ.get("BITIDY_ROWS"):
            closing = ((p == 1) & adv).nonzero().squeeze(-1)
            for e in closing.tolist():
                print(f"BITIDY_CLOSE env={e} t={int(env.episode_length_buf[e])} face={float(face[e]):.3f} slope={float(slope[e]):.3f} "
                      f"padknob={[round(v, 4) for v in (pad[e] - knob_now[e]).tolist()]} lidxy={[round(v, 4) for v in (T.lid_pos(env)[e, :2] - box[e]).tolist()]}", flush=True)
        limit = torch.where(p == 0, 2 * self.PHASE_LIMIT, self.PHASE_LIMIT)
        moved = adv | ((self.t > limit) & (p < 9))
        idx = torch.arange(n, device=dev)
        self.reach[idx, p] = True
        self.dur[idx, p] += 1
        fids = (moved & ~adv).nonzero().squeeze(-1)
        if os.environ.get("BITIDY_ROWS"):
            for e in fids.tolist():
                print(f"BITIDY_FORCED env={e} phase={int(p[e])} t={int(env.episode_length_buf[e])} err={float(err[e]):.4f} tilt={float(T.lid_tilt(env)[e]):.3f} "
                      f"pad={[round(v, 4) for v in pad[e].tolist()]} knob={[round(v, 4) for v in knob_now[e].tolist()]} held={bool(T.lid_held(env)[e])}", flush=True)
        self.forced[fids, p[fids]] = True
        if os.environ.get("BITIDY_ROWS"):
            # Episodes that run to the time limit end together and the episode counter exits on that step, before
            # ``reset``: count them here (marked FINAL).
            last = (env.episode_length_buf >= env.max_episode_length - 2).nonzero().squeeze(-1)
            if len(last):
                print("BITIDY_PHASES_FINAL", flush=True)
                self._stats(last, False)
        nxt = (p + 1).clamp(max=N_PHASES - 1)
        # Lost the knob while carrying (it slipped out of the jaws): open and start over from the hover point
        # (re-locking the knob wherever the lid now is); while lowering onto the box, just let go.
        lost = (torch.linalg.vector_norm(pad - knob_now, dim=-1) > self.LOST) & (p >= 3) & (p <= 7) & (self.t > 3)
        if os.environ.get("BITIDY_ROWS"):
            for e in lost.nonzero().squeeze(-1).tolist():
                print(f"BITIDY_LOST env={e} phase={int(p[e])} t={int(env.episode_length_buf[e])} tilt={float(T.lid_tilt(env)[e]):.3f} "
                      f"pad={[round(v, 4) for v in pad[e].tolist()]} knob={[round(v, 4) for v in knob_now[e].tolist()]} "
                      f"open={float(base.gripper_openness(env, self.arm)[e]):.3f}", flush=True)
        self.retries += (lost & (p < 7)).long()
        nxt = torch.where(lost, torch.where(p < 7, 0, 8), nxt)
        moved = moved | lost
        self.high |= lost & (p < 7)
        self.p = torch.where(moved, nxt, p)
        self.t = torch.where(moved, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)
