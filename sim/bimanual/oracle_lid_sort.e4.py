"""Scripted oracle for BiTidy-LidSort-SO101 (``BITIDY_SCRIPT=oracle_lid_sort:LidSortScript``).

Facing layout (the task default, arms across the table, boxes in a row along y between them):

* LEFT arm = the verified ``oracle_lid_open.LidOpenScript`` (``BITIDY_LS_LID_SCRIPT=open``, default; ``row`` =
  :class:`RowLidScript`): lid j up by its knob, aside (``BITIDY_LS_ASIDE_MODE=dirs``: per row position, -y end toward
  -x/-y, middle -x, +y end toward +y; ``BITIDY_LS_ASIDE_DIRS``), held, then back on box j.
* RIGHT arm = :class:`FrontPlaceScript` (``BITIDY_LS_PLACER=front``, default; ``row`` = the previous
  :class:`RowPlaceScript`): pick cube i at its slot beside the -y row end, lift it straight up, then
  (``BITIDY_LS_PATH=row``, default) rise in place until the cube bottom is ``CROSS_CLEAR`` above the knob tops, go to
  the row line at the cube's y and along the row line over the box, lower, centre, drop, back off toward the base.
  ``BITIDY_LS_PATH=front`` instead carries in front of the row at rim height (never over a lid) -- the right arm
  cannot hold that height 13-16 cm from its base, so it is not the default.
* Coordination (``BITIDY_LS_CUBE_FIRST=1``, default): the right arm picks and lifts first; the left takes the lid off
  only once the cube is up beside the row (the held lid swings and slips in the knob pinch: keep the hold short); the
  right carries once the lid is held aside; the left puts the lid back once the cube rests in the box and the right
  jaws are clear.

With ``BITIDY_LS_FACING_X=0`` (side-by-side layout) :class:`LidSortScript` runs the previous oracle
(:class:`LidSortSideScript`).

``BITIDY_DEBUG=1`` traces one environment (``BITIDY_DEBUG_ENV``) every ``BITIDY_DEBUG_EVERY`` steps; ``BITIDY_ROWS=1``
prints forced phases, lost cubes and each environment's end state (``BITIDY_END``).
"""

from __future__ import annotations

import math
import os
import warnings

import torch

from isaaclab.utils.math import quat_apply

from . import env_cfg as base
from . import lid_sort_env_cfg as T
from .oracle_box_place import BoxPlaceScript
from .oracle_lid_open import LidOpenScript
from .scripted import jaw_tips

LEFT_HOLD = 5
RIGHT_LIFT, RIGHT_CARRY, RIGHT_PARK = 4, 5, 9


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _fold(a: torch.Tensor, period: float) -> torch.Tensor:
    """Angle folded to (-period/2, period/2]."""
    return torch.remainder(a + period / 2, period) - period / 2


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------- left arm: lid off, hold back, lid on


class RowLidScript(LidOpenScript):
    """LidOpen's phases (0 hover, 1 down, 2 close, 3 lift, 4 back, 5 hold, 6 over the box, 7 lower, 8 open, 9 up)
    with a vertical gripper whose roll lines the jaws up with the knob (see the module docstring)."""

    FWD = _f("BITIDY_LS_L_FWD", 0.0)  # forward tilt of the jaw direction (0: straight down)
    FACE_GAIN = _f("BITIDY_LS_L_FACE_GAIN", 0.3)
    YAW_GAIN = _f("BITIDY_LS_L_YAW_GAIN", 0.2)
    JAW_ANGLE = _f("BITIDY_LS_L_JAW_ANGLE", 0.0)  # closing axis yaw in the lid frame (0: along the box x axis)
    FACE_TOL = _f("BITIDY_LS_L_FACE_TOL", 0.06)
    LEVEL_K = _f("BITIDY_LS_L_LEVEL_K", 1.0)
    ROLL_MAX = _f("BITIDY_LS_L_ROLL_MAX", 2.8)
    PW = _f("BITIDY_LS_L_PW", 0.004)  # jaw-direction weight (straight down is out of reach at the row: ~15 deg forward)
    PW_CARRY = _f("BITIDY_LS_L_PW_CARRY", 0.02)
    SEAT_SHIFT = _f("BITIDY_LS_L_SEAT_SHIFT", 0.006)  # approach with the pads' midpoint this far toward the fixed jaw

    def __init__(self, env):
        super().__init__(env)
        self.T = T
        n, dev = env.num_envs, env.device
        self.aside_dir = torch.tensor((-1.0, 0.0), device=dev).expand(n, -1).clone()
        self.prev_approach = torch.zeros(n, dtype=torch.bool, device=dev)
        self.LIFT = _f("BITIDY_LS_L_LIFT", 0.03)
        self.ASIDE = _f("BITIDY_LS_L_ASIDE", 0.085)
        self.ASIDE_UP = _f("BITIDY_LS_L_ASIDE_UP", 0.025)

    def act(self) -> torch.Tensor:
        env, p = self.env, self.p
        n, dev = env.num_envs, env.device
        up = torch.tensor((0.0, 0.0, 1.0), device=dev)

        def col(v):
            return v.unsqueeze(-1)

        pad = base.pad_center(env, self.arm)
        knob_now = T.knob_pos(env)
        self.knob = torch.where(col(p <= 0), knob_now, self.knob)
        box = T.box_center(env)
        box_knob = torch.cat((box, torch.full((n, 1), T.LID_REST_Z + T.KNOB_OFFSET, device=dev)), -1)
        aside = box_knob + torch.cat((self.aside_dir * self.ASIDE, torch.zeros_like(box[:, :1])), -1) + up * self.ASIDE_UP

        pinch = self.knob + up * self.KNOB_PINCH
        hover = pinch + up * self.HOVER
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
        # Approaching the knob, optionally steer a point shifted toward the fixed jaw (less push when the jaws close).
        fixed_w, moving_w = jaw_tips(env, self.arm)
        jaw_axis = torch.nn.functional.normalize(moving_w - fixed_w, dim=-1)
        seat = pad - jaw_axis * self.SEAT_SHIFT
        approach = (p <= 1) & ~self.high
        ctrl = torch.where(col(carrying), knob_now, torch.where(col(approach), seat, pad))
        goal = torch.where(col(carrying), knob_goal, pad_goal)
        changed = (p != self.prev_p) | (approach != self.prev_approach)
        self.prev_approach = approach
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
        self.hold_pad = torch.where(col(p <= 7), pad, self.hold_pad)
        grip = torch.where((p >= 2) & (p < 8), -1.0, 1.0)

        # Roll: free hand -> closing axis onto the knob faces (along the box's x axis); lid in hand -> the lid turned
        # to the box's yaw (so the plate goes back square, clear of its neighbours).
        fixed, moving = jaw_tips(env, self.arm)
        axis = torch.nn.functional.normalize(moving - fixed, dim=-1)
        slope = axis[:, 2]
        jaw_yaw = torch.atan2(axis[:, 1], axis[:, 0])
        lid_yaw = _yaw(T.lid_quat(env))
        knob_yaw = T.target_box_yaw(env) if T.KNOB_FACETS > 1 else lid_yaw + T.KNOB_PLATE_YAW
        face = _fold(knob_yaw + self.JAW_ANGLE - jaw_yaw, math.pi)
        lid_err = _fold(T.target_box_yaw(env) - lid_yaw, math.pi / 2)
        # (only once the jaws point roughly down: far out, the jaw yaw is meaningless and the roll would wind up)
        point_down = -(pad - (self.ik.robot.data.body_pos_w.torch[:, self.ik.body] - env.scene.env_origins))
        down = torch.nn.functional.normalize(point_down, dim=-1)[:, 2] > 0.85
        d_roll = torch.where((p <= 1) & down, self.FACE_GAIN * face, torch.zeros_like(face))
        d_roll = torch.where((p >= 4) & (p <= 7), self.YAW_GAIN * lid_err, d_roll)
        self.roll = (self.roll + d_roll).clamp(-self.ROLL_MAX, self.ROLL_MAX)

        near = col(torch.linalg.vector_norm(target - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (target - pad), 0.9 * self.bias).clamp(-0.03, 0.03)
        direction = self.ik.tilted(target, self.FWD)
        # Lid in hand: tilt the jaw direction against the lid's tilt (the lid hangs rigidly in the pinch).
        point = torch.nn.functional.normalize(pad - (self.ik.robot.data.body_pos_w.torch[:, self.ik.body] - env.scene.env_origins), dim=-1)
        lid_up = quat_apply(T.lid_quat(env), up.expand(n, -1))
        lean = lid_up - col(lid_up[:, 2]) * up
        keep = torch.nn.functional.normalize(point + self.LEVEL_K * lean, dim=-1)
        held = col(carrying & T.lid_held(env) | carrying & (p == 3))
        direction = torch.where(held, keep, direction)
        weight = torch.where(carrying, self.PW_CARRY, self.PW)
        act, _ = self.ik(target + self.bias, direction, self.roll, gain=self.GAIN, point_weight=weight)
        act = torch.where(col((p == 2) | (p == 8)), torch.zeros_like(act), act)

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            r3 = lambda v: [round(x, 3) for x in v[k].tolist()]
            print(f"[lid] t={int(env.episode_length_buf[k])} p={int(p[k])} err={float(err[k]):.4f} pad={r3(pad)} knob={r3(knob_now)} "
                  f"lid={r3(T.lid_pos(env))} tilt={float(T.lid_tilt(env)[k]):.3f} face={float(face[k]):.3f} lid_err={float(lid_err[k]):.3f} "
                  f"roll={float(self.roll[k]):.2f} slope={float(slope[k]):.3f} held={bool(T.lid_held(env)[k])} "
                  f"point={r3(point)}", flush=True)

        self.t += 1
        self.high &= ~((p == 0) & (err < 0.015))
        hover_ok = (err < self.HOVER_TOL) & (face.abs() < self.FACE_TOL) & ~self.high
        self.settle = torch.where((p == 0) & hover_ok, self.settle + 1, 0)
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (self.settle >= self.SETTLE)
        adv |= (p == 1) & (err < self.PINCH_TOL)
        adv |= (p == 2) & (self.t > 15)
        adv |= (p == 3) & (err < 0.01)
        adv |= (p == 4) & (err < 0.008)
        adv |= (p == 5) & (self.t > self.HOLD)
        adv |= (p == 6) & (err < 0.006) & (lid_err.abs() < 0.05)
        adv |= (p == 7) & (err < 0.004)
        adv |= (p == 8) & (self.t > 15)
        limit = torch.where(p == 0, 2 * self.PHASE_LIMIT, self.PHASE_LIMIT)
        moved = adv | ((self.t > limit) & (p < 9))
        idx = torch.arange(n, device=dev)
        self.reach[idx, p] = True
        self.dur[idx, p] += 1
        fids = (moved & ~adv).nonzero().squeeze(-1)
        if os.environ.get("BITIDY_ROWS"):
            for e in fids.tolist()[:20]:
                print(f"BITIDY_FORCED_L env={e} phase={int(p[e])} t={int(env.episode_length_buf[e])} err={float(err[e]):.4f} "
                      f"face={float(face[e]):.3f} tilt={float(T.lid_tilt(env)[e]):.3f} pad={[round(v, 4) for v in pad[e].tolist()]} "
                      f"knob={[round(v, 4) for v in knob_now[e].tolist()]} held={bool(T.lid_held(env)[e])}", flush=True)
        self.forced[fids, p[fids]] = True
        nxt = (p + 1).clamp(max=9)
        lost = (torch.linalg.vector_norm(pad - knob_now, dim=-1) > self.LOST) & (p >= 3) & (p <= 7) & (self.t > 3)
        self.retries += (lost & (p < 7)).long()
        nxt = torch.where(lost, torch.where(p < 7, 0, 8), nxt)
        moved = moved | lost
        self.high |= lost & (p < 7)
        self.p = torch.where(moved, nxt, p)
        self.t = torch.where(moved, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)


# ---------------------------------------------------------------- right arm: pick, carry over the row, drop, park


class RowPlaceScript(BoxPlaceScript):
    """BoxPlace's phases (0 hover, 1 above the grasp, 2 down, 3 close, 4 lift, 5 carry, 6 lower, 7 open, 8 back off,
    9 park) on the commanded cube / box, with a level carry above the knob tops (see the module docstring)."""

    CARRY_CLEAR = _f("BITIDY_LS_CARRY_CLEAR", 0.012)
    TILT = _f("BITIDY_LS_R_TILT", 0.9)  # grasp tilt (the reach maps: 0.7 cannot lift at the row ends)
    TILT_BOX = _f("BITIDY_LS_R_TILT_BOX", 1.2)  # every other move  # cube bottom above the knob tops while carrying
    CARRY_SPEED = _f("BITIDY_LS_CARRY_SPEED", 0.003)  # carrot speed [m/step] of the carry
    PARK_DX = _f("BITIDY_LS_PARK_DX", 0.10)  # park this far from the box toward the right base, ...
    PARK_Z = _f("BITIDY_LS_PARK_Z", 0.13)  # ... with the pads this high
    BACK_UP = _f("BITIDY_LS_BACK_UP", 0.05)  # back off: pads straight up to this far above the rim
    FREE_SPEED = _f("BITIDY_LS_FREE_SPEED", 0.004)  # carrot speed of the free moves (approach, back off, park)
    READY = tuple(float(v) for v in os.environ.get("BITIDY_LS_READY", "0.25,0.5,0.18").split(","))
    """First approach waypoint: x, share of the cube's y, z (out of the left arm's way, toward the right base)."""
    TRANSIT_Z = _f("BITIDY_LS_TRANSIT_Z", 0.12)  # then over the cube at this height, then down to the hover point
    PARK_OUT = _f("BITIDY_LS_PARK_OUT", 0.02)  # park over the cube's start, this much further out along y

    def __init__(self, env):
        super().__init__(env)
        n, dev = env.num_envs, env.device
        self.carrot = torch.zeros((n, 3), device=dev)
        self.prev_p = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.sub = torch.zeros(n, dtype=torch.long, device=dev)
        self.sub_t = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_xy = torch.zeros((n, 2), device=dev)

    def reset(self, ids: torch.Tensor) -> None:
        super().reset(ids)
        self.prev_p[ids] = -1
        self.sub[ids] = 0
        self.sub_t[ids] = 0

    def act(self) -> torch.Tensor:
        env, p = self.env, self.p
        n = env.num_envs
        cube = T.cube_pos(env)
        pad = base.pad_center(env, B_ARM)
        up = cube.new_tensor((0.0, 0.0, 1.0))
        quat = T.target_view(env).scene["cube"].data.root_quat_w.torch
        cube_yaw = _yaw(quat)

        def col(v):
            return v.unsqueeze(-1)

        def at(xy, z):
            return torch.cat((xy, torch.full_like(xy[:, :1], z) if isinstance(z, float) else z), -1)

        box = T.target_box(env)
        box_yaw = T.target_box_yaw(env)
        held = torch.linalg.vector_norm(pad - cube, dim=-1) < 0.03
        self.offset = torch.where(col(held | (p <= 4)), (pad - cube).clamp(-0.02, 0.02), self.offset)
        offset = self.offset
        box_pad = box + offset[:, :2]
        carry_z = T.KNOB_TOP_Z + self.CARRY_CLEAR + T.CUBE_SIZE / 2 + offset[:, 2:]
        drop_z = T.RIM_Z + T.CUBE_SIZE / 2 + self.DROP_CLEAR + offset[:, 2:]
        self.grasp = torch.where(col(p <= 0), cube, self.grasp)
        self.start_xy = torch.where(col(env.episode_length_buf <= 1), cube[:, :2], self.start_xy)
        g = self.grasp
        # Approach: out toward the right base first (the arms start with their jaws side by side over the row), then
        # over the cube high, then down to the hover point.
        ready = torch.stack((torch.full_like(cube[:, 0], self.READY[0]), self.READY[1] * cube[:, 1],
                             torch.full_like(cube[:, 0], self.READY[2])), -1)
        t = torch.where(col(self.sub == 0), ready, at(cube[:, :2], self.TRANSIT_Z))
        t = torch.where(col(self.sub >= 2), cube + up * self.HOVER, t)
        t = torch.where(col(p >= 1), g + up * 0.03, t)
        t = torch.where(col(p >= 2), g + up * self.PINCH, t)
        t = torch.where(col(p >= 4), at(g[:, :2] + offset[:, :2], carry_z), t)
        t = torch.where(col(p >= 5), at(box_pad, carry_z), t)
        t = torch.where(col(p >= 6), at(box_pad, drop_z), t)
        self.release_xy = torch.where(col(p <= 7), pad[:, :2], self.release_xy)
        t = torch.where(col(p >= 8), at(self.release_xy, T.RIM_Z + self.BACK_UP), t)
        out = torch.sign(self.start_xy[:, 1:2]) * self.PARK_OUT
        park = torch.cat((self.start_xy[:, :1], self.start_xy[:, 1:2] + out), -1)
        t = torch.where(col(p >= 9), at(park, self.PARK_Z), t)
        # Free moves, lift and carry along a speed-limited carrot (straight lines: no swinging down into a lid).
        changed = p != self.prev_p
        self.carrot = torch.where(col(changed), pad, self.carrot)
        self.prev_p = p.clone()
        step = t - self.carrot
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        speed = torch.where((p == 4) | (p == 5), self.CARRY_SPEED, self.FREE_SPEED)
        self.carrot = self.carrot + step * (col(speed) / dist.clamp(min=1e-6)).clamp(max=1.0)
        goal = torch.where(col((p == 0) | (p == 4) | (p == 5) | (p == 8) | (p == 9)), self.carrot, t)
        grip = torch.where((p >= 3) & (p < 7), -1.0, 1.0)

        fixed, moving = jaw_tips(env, B_ARM)
        axis = moving - fixed
        face_err = self._fold(cube_yaw - torch.atan2(axis[:, 1], axis[:, 0]))
        box_err = self._fold(box_yaw - cube_yaw)
        d_roll = torch.where(p <= 2, self.ROLL_GAIN * face_err, torch.zeros_like(face_err))
        d_roll = torch.where((p == 5) | (p == 6), self.BOX_ROLL_GAIN * box_err, d_roll)
        self.roll = (self.roll + d_roll).clamp(-1.6, 1.6)
        self.roll = torch.where(p >= 8, 0.95 * self.roll, self.roll)

        near = col(torch.linalg.vector_norm(goal - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (goal - pad), 0.9 * self.bias).clamp(-self.BIAS_CLAMP, self.BIAS_CLAMP)
        self.bias[:, 2] = torch.where(p <= 2, self.bias[:, 2].clamp(min=0.0), self.bias[:, 2])
        tilt = torch.where(col((p >= 1) & (p <= 3)), self.ik.tilted(goal, self.TILT), self.ik.tilted(goal, self.TILT_BOX))
        act, _ = self.ik(goal + self.bias, tilt, self.roll, gain=self.GAIN)
        err = torch.linalg.vector_norm(t - pad, dim=-1)

        half = 0.5 * T.CUBE_SIZE * (torch.cos(box_err).abs() + torch.sin(box_err).abs())
        cube_off = T.in_box_frame(env, cube[:, :2]).abs().max(-1).values
        centred = (cube_off < T.BOX_INNER / 2 - half - self.CENTRE_TOL) & (cube_off < self.CENTRE_ABS)
        over_ok = cube_off < T.BOX_INNER / 2 - half

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            print(f"[place] t={int(env.episode_length_buf[k])} p={int(p[k])} err={float(err[k]):.4f} "
                  f"pad={[round(v, 3) for v in pad[k].tolist()]} goal={[round(v, 3) for v in goal[k].tolist()]} "
                  f"cube={[round(v, 3) for v in cube[k].tolist()]} off={float(cube_off[k]):.4f} held={bool(base.held_by(T.target_view(env), B_ARM)[k])}", flush=True)

        self.t += 1
        self.sub_t += 1
        sub_next = (p == 0) & (self.sub < 2) & ((err < 0.02) | (self.sub_t > 150))
        self.sub = torch.where(sub_next, self.sub + 1, self.sub)
        self.sub_t = torch.where(sub_next, 0, self.sub_t)
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (self.sub >= 2) & (err < self.HOVER_TOL)
        adv |= (p == 1) & (err < 0.004)
        adv |= (p == 2) & (err < self.GRASP_TOL)
        adv |= (p == 3) & (self.t > 15)
        adv |= (p == 4) & (err < 0.01)
        adv |= (p == 5) & (err < 0.008)
        adv |= (p == 6) & (err < 0.005) & centred & (box_err.abs() < 0.1)
        adv |= (p == 7) & (self.t > 15)
        adv |= (p == 8) & (err < 0.01)
        hold_on = (p == 6) & ~over_ok
        lost = ((p == 5) | (p == 6)) & ~held & (self.t > 10)
        moved = adv | ((self.t > self.PHASE_LIMIT) & ~hold_on & (p < 9) & ((p > 0) | (self.sub >= 2)))
        # The carry is long: give it time (its carrot alone needs ~0.3 m / CARRY_SPEED steps).
        moved = moved & ~((p == 5) & ~adv & (self.t <= 2 * self.PHASE_LIMIT + int(0.3 / self.CARRY_SPEED)))
        idx = torch.arange(n, device=env.device)
        self.reached[idx, p] = True
        fid = (moved & ~adv).nonzero().squeeze(-1)
        self.forced[fid, p[fid]] = True
        if os.environ.get("BITIDY_ROWS"):
            for e in fid.tolist()[:20]:
                print(f"BITIDY_FORCED_R env={e} phase={int(p[e])} t={int(env.episode_length_buf[e])} err={float(err[e]):.4f} "
                      f"pad={[round(v, 4) for v in pad[e].tolist()]} t={[round(v, 4) for v in t[e].tolist()]} "
                      f"cube={[round(v, 4) for v in cube[e].tolist()]}", flush=True)
        self.p = torch.where(moved, (p + 1).clamp(max=9), p)
        self.p = torch.where(lost, 7, self.p)
        moved = moved | lost
        self.t = torch.where(moved | sub_next, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)


class FrontPlaceScript(RowPlaceScript):
    """The default placer (``BITIDY_LS_PLACER=front``): BoxPlace's phase numbers, but the cube never passes over a
    box. Pick at the slot (beside the -y row end), lift straight up until the cube bottom is ``CARRY_RIM`` above the
    rims, then (phase 5, three legs in the target box's frame: u = row normal toward the right base, v = along the
    row) out to the front line u = ``FRONT`` at the cube's v, along the front line to the box's v, in over the box;
    lower, centre, open, back out along u to ``BACK_U`` at carry height and park there."""

    FRONT = _f("BITIDY_LS_FRONT", 0.065)
    CARRY_RIM = _f("BITIDY_LS_CARRY_RIM", 0.013)  # cube bottom above the rims while carrying
    BACK_U = _f("BITIDY_LS_BACK_U", 0.09)
    TILT_PLACE = _f("BITIDY_LS_R_TILT_PLACE", 1.2)  # jaw tilt from the lift on
    LEG_TOL = _f("BITIDY_LS_LEG_TOL", 0.015)
    CENTRE_ABS = _f("BITIDY_LS_CENTRE_ABS", 0.006)
    READY = tuple(float(v) for v in os.environ.get("BITIDY_LS_READY", "0.25,0.5,0.15").split(","))
    TRANSIT_Z = _f("BITIDY_LS_TRANSIT_Z", 0.11)
    PW_GRASP = _f("BITIDY_LS_R_PW_GRASP", 0.015)  # IK jaw-direction weight up to the grasp ...
    PW_PLACE = _f("BITIDY_LS_R_PW_PLACE", 0.005)  # ... and from the lift on (position first: the arm sagged ~2 cm)
    PW_APPROACH = _f("BITIDY_LS_R_PW_APPROACH", 0.004)  # ... and on the way to the hover point
    BIAS_CLAMP = _f("BITIDY_LS_R_BIAS_CLAMP", 0.04)
    GRASP_SHIFT = _f("BITIDY_LS_R_GRASP_SHIFT", 0.0)
    PATH = os.environ.get("BITIDY_LS_PATH", "row")  # front: in front of the row; row: along the row over the knobs
    CROSS_CLEAR = _f("BITIDY_LS_CROSS_CLEAR", 0.012)
    ACCUM_FROM = int(_f("BITIDY_LS_R_ACCUM_FROM", 99))  # 99: off (integral IK stalled at joint limits)
    ACCUM_TO = int(_f("BITIDY_LS_R_ACCUM_TO", 6))
    ACCUM_GAIN = _f("BITIDY_LS_R_ACCUM_GAIN", 0.3)
    LEG_ZTOL = _f("BITIDY_LS_LEG_ZTOL", 0.006)
    LEG_LIMIT = int(_f("BITIDY_LS_LEG_LIMIT", 90))
    PLACE_DIR = os.environ.get("BITIDY_LS_R_PLACE_DIR", "box")  # box: fixed jaw azimuth -u; base: away from the base

    def __init__(self, env):
        super().__init__(env)
        self.leg = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, ids: torch.Tensor) -> None:
        super().reset(ids)
        self.leg[ids] = 0

    def act(self) -> torch.Tensor:
        env, p = self.env, self.p
        n = env.num_envs
        cube = T.cube_pos(env)
        pad = base.pad_center(env, B_ARM)
        up = cube.new_tensor((0.0, 0.0, 1.0))
        cube_yaw = _yaw(T.target_view(env).scene["cube"].data.root_quat_w.torch)

        def col(v):
            return v.unsqueeze(-1)

        def at(xy, z):
            return torch.cat((xy, torch.full_like(xy[:, :1], z) if isinstance(z, float) else z), -1)

        box = T.target_box(env)
        box_yaw = T.target_box_yaw(env)
        ex = torch.stack((torch.cos(box_yaw), torch.sin(box_yaw)), -1)
        ey = torch.stack((-torch.sin(box_yaw), torch.cos(box_yaw)), -1)

        def uv(u, v):
            u = u if torch.is_tensor(u) else torch.full_like(box[:, 0], u)
            v = v if torch.is_tensor(v) else torch.full_like(box[:, 0], v)
            return box + col(u) * ex + col(v) * ey

        held = torch.linalg.vector_norm(pad - cube, dim=-1) < 0.03
        self.offset = torch.where(col(held | (p <= 4)), (pad - cube).clamp(-0.02, 0.02), self.offset)
        off = self.offset
        carry_z = T.RIM_Z + self.CARRY_RIM + T.CUBE_SIZE / 2 + off[:, 2:]
        drop_z = T.RIM_Z + T.CUBE_SIZE / 2 + self.DROP_CLEAR + off[:, 2:]
        self.grasp = torch.where(col(p <= 0), cube, self.grasp)
        self.start_xy = torch.where(col(env.episode_length_buf <= 1), cube[:, :2], self.start_xy)
        g = self.grasp
        v_cube = ((g[:, :2] - box) * ey).sum(-1)
        ready = torch.stack((torch.full_like(cube[:, 0], self.READY[0]), self.READY[1] * cube[:, 1],
                             torch.full_like(cube[:, 0], self.READY[2])), -1)
        t = torch.where(col(self.sub == 0), ready, at(cube[:, :2], self.TRANSIT_Z))
        t = torch.where(col(self.sub >= 2), cube + up * self.HOVER, t)
        t = torch.where(col(p >= 1), g + up * 0.03, t)
        t = torch.where(col(p >= 2), g + up * self.PINCH, t)
        t = torch.where(col(p >= 4), at(g[:, :2] + off[:, :2], carry_z), t)
        if self.PATH == "row":  # along the row line, the cube bottom CROSS_CLEAR above the knob tops
            legs = torch.stack((g[:, :2], uv(0.0, v_cube), box), 1)  # rise in place, to the row line, along it
            carry_z = torch.where(col(p >= 5), T.KNOB_TOP_Z + self.CROSS_CLEAR + T.CUBE_SIZE / 2 + off[:, 2:], carry_z)
        else:
            legs = torch.stack((uv(self.FRONT, v_cube), uv(self.FRONT, 0.0), box), 1)  # (n, 3, 2)
        leg_xy = legs[torch.arange(n, device=env.device), self.leg.clamp(max=2)] + off[:, :2]
        t = torch.where(col(p >= 5), at(leg_xy, carry_z), t)
        t = torch.where(col(p >= 6), at(box + off[:, :2], drop_z), t)
        back = uv(self.BACK_U, 0.0)
        t = torch.where(col(p >= 8), at(back, carry_z), t)
        changed = p != self.prev_p
        self.carrot = torch.where(col(changed), pad, self.carrot)
        self.prev_p = p.clone()
        self.leg = torch.where(changed, 0, self.leg)
        step = t - self.carrot
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        speed = torch.where((p == 4) | (p == 5), self.CARRY_SPEED, self.FREE_SPEED)
        self.carrot = self.carrot + step * (col(speed) / dist.clamp(min=1e-6)).clamp(max=1.0)
        goal = torch.where(col((p == 0) | (p == 4) | (p == 5) | (p == 8) | (p == 9)), self.carrot, t)
        grip = torch.where((p >= 3) & (p < 7), -1.0, 1.0)

        fixed, moving = jaw_tips(env, B_ARM)
        axis = moving - fixed
        face_err = self._fold(cube_yaw - torch.atan2(axis[:, 1], axis[:, 0]))
        box_err = self._fold(box_yaw - cube_yaw)
        d_roll = torch.where(p <= 2, self.ROLL_GAIN * face_err, torch.zeros_like(face_err))
        d_roll = torch.where((p == 5) | (p == 6), self.BOX_ROLL_GAIN * box_err, d_roll)
        self.roll = (self.roll + d_roll).clamp(-1.6, 1.6)
        self.roll = torch.where(p >= 8, 0.95 * self.roll, self.roll)

        near = col(torch.linalg.vector_norm(goal - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (goal - pad), 0.9 * self.bias).clamp(-self.BIAS_CLAMP, self.BIAS_CLAMP)
        self.bias[:, 2] = torch.where(p <= 2, self.bias[:, 2].clamp(min=0.0), self.bias[:, 2])
        tilt_v = torch.full_like(box[:, 0], self.TILT_BOX)
        tilt_v = torch.where((p >= 1) & (p <= 3), self.TILT, tilt_v)
        tilt_v = torch.where(p >= 4, self.TILT_PLACE, tilt_v)
        out = torch.nn.functional.normalize(goal[:, :2] - self.ik.base_xy, dim=-1)
        if self.PLACE_DIR == "box":  # from the lift on: jaws toward -u (into the row from the front), one fixed direction
            out = torch.where(col(p >= 4), -ex, out)
        tilt = torch.nn.functional.normalize(torch.cat((col(tilt_v) * out, -torch.ones_like(out[:, :1])), -1), dim=-1)
        pw = torch.where(p >= 4, self.PW_PLACE, self.PW_GRASP)
        pw = torch.where(p == 0, self.PW_APPROACH, pw)
        # Grasp: steer a point shifted toward the fixed jaw onto the cube (the moving jaw then pushes it less).
        shift = torch.nn.functional.normalize(axis, dim=-1) * self.GRASP_SHIFT
        ik_goal = goal + torch.where(col((p >= 1) & (p <= 3)), shift, torch.zeros_like(shift))
        act, _ = self.ik(ik_goal + self.bias, tilt, self.roll, gain=self.GAIN, point_weight=pw)
        if self.ACCUM_FROM < 99:  # integral action (joint targets stepped): no PD gravity sag while carrying
            acc, _ = self.ik(ik_goal, tilt, self.roll, gain=self.ACCUM_GAIN, point_weight=pw, accumulate=True)
            act = torch.where(col(p >= self.ACCUM_FROM) & col(p <= self.ACCUM_TO), acc, act)
        seat = pad - torch.where(col((p >= 1) & (p <= 3)), shift, torch.zeros_like(shift))
        err = torch.linalg.vector_norm(t - seat, dim=-1)

        half = 0.5 * T.CUBE_SIZE * (torch.cos(box_err).abs() + torch.sin(box_err).abs())
        cube_off = T.in_box_frame(env, cube[:, :2]).abs().max(-1).values
        centred = (cube_off < T.BOX_INNER / 2 - half - self.CENTRE_TOL) & (cube_off < self.CENTRE_ABS)
        over_ok = cube_off < T.BOX_INNER / 2 - half

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            print(f"[front] t={int(env.episode_length_buf[k])} p={int(p[k])} leg={int(self.leg[k])} err={float(err[k]):.4f} "
                  f"pad={[round(v, 3) for v in pad[k].tolist()]} goal={[round(v, 3) for v in goal[k].tolist()]} "
                  f"cube={[round(v, 3) for v in cube[k].tolist()]} off={float(cube_off[k]):.4f} held={bool(held[k])} "
                  f"jaw={[round(v, 2) for v in torch.nn.functional.normalize(pad - (self.ik.robot.data.body_pos_w.torch[:, self.ik.body] - env.scene.env_origins), dim=-1)[k].tolist()]} "
                  f"q={[round(v, 2) for v in self.ik.robot.data.joint_pos.torch[k].tolist()]}", flush=True)

        self.t += 1
        self.sub_t += 1
        sub_next = (p == 0) & (self.sub < 2) & ((err < 0.02) | (self.sub_t > 100))
        self.sub = torch.where(sub_next, self.sub + 1, self.sub)
        self.sub_t = torch.where(sub_next, 0, self.sub_t)
        # Carry legs: next leg once the carrot is at this leg's end and the pads are close.
        carrot_at = torch.linalg.vector_norm(self.carrot - t, dim=-1) < 1e-4
        z_ok = (pad[:, 2] - t[:, 2]).abs() < self.LEG_ZTOL  # the cube up at crossing height before it moves on
        leg_next = (p == 5) & (self.leg < 2) & carrot_at & (((err < self.LEG_TOL) & z_ok) | (self.t > self.LEG_LIMIT))
        self.leg = torch.where(leg_next, self.leg + 1, self.leg)
        self.t = torch.where(leg_next, 0, self.t)
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (self.sub >= 2) & (err < self.HOVER_TOL)
        adv |= (p == 1) & (err < 0.004)
        adv |= (p == 2) & (err < self.GRASP_TOL)
        adv |= (p == 3) & (self.t > 15)
        adv |= (p == 4) & (err < 0.01)
        adv |= (p == 5) & (self.leg == 2) & (err < 0.008)
        adv |= (p == 6) & (err < 0.005) & centred & (box_err.abs() < 0.1)
        adv |= (p == 7) & (self.t > 15)
        adv |= (p == 8) & (err < 0.01)
        hold_on = ((p == 6) & ~over_ok) | ((p == 5) & (self.leg < 2))
        lost = ((p == 5) | (p == 6)) & ~held & (self.t > 10)
        moved = adv | ((self.t > self.PHASE_LIMIT) & ~hold_on & (p < 9) & ((p > 0) | (self.sub >= 2)))
        idx = torch.arange(n, device=env.device)
        self.reached[idx, p] = True
        fid = (moved & ~adv).nonzero().squeeze(-1)
        self.forced[fid, p[fid]] = True
        if os.environ.get("BITIDY_ROWS"):
            for e in fid.tolist()[:20]:
                print(f"BITIDY_FORCED_R env={e} phase={int(p[e])} t={int(env.episode_length_buf[e])} err={float(err[e]):.4f} "
                      f"pad={[round(v, 4) for v in pad[e].tolist()]} t={[round(v, 4) for v in t[e].tolist()]} "
                      f"cube={[round(v, 4) for v in cube[e].tolist()]}", flush=True)
        self.p = torch.where(moved, (p + 1).clamp(max=9), p)
        if os.environ.get("BITIDY_ROWS"):
            for e in lost.nonzero().squeeze(-1).tolist()[:20]:
                print(f"BITIDY_LOST env={e} phase={int(p[e])} leg={int(self.leg[e])} t={int(env.episode_length_buf[e])} "
                      f"pad={[round(v, 3) for v in pad[e].tolist()]} goal={[round(v, 3) for v in goal[e].tolist()]} "
                      f"cube={[round(v, 3) for v in cube[e].tolist()]} box={[round(v, 3) for v in box[e].tolist()]}", flush=True)
        self.p = torch.where(lost, 7, self.p)
        moved = moved | lost
        self.t = torch.where(moved | sub_next, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)


B_ARM = "robot_right"


# ---------------------------------------------------------------- coordination


class LidSortScript:
    """Both arms on the facing layout (see the module docstring); the side-by-side layout runs :class:`LidSortSideScript`."""

    LID_SETTLE = int(_f("BITIDY_LP_LID_SETTLE", 10))
    GIVE_UP = int(_f("BITIDY_LP_GIVE_UP", 90))
    LIFT_STEPS = int(_f("BITIDY_LS_LIFT_STEPS", 30))
    LIFT_FORCE = int(_f("BITIDY_LS_LIFT_FORCE", 150))  # carry even if the lift height was not reached by then
    RIGHT_START = int(_f("BITIDY_LP_RIGHT_START", 1))
    RETRIES = int(_f("BITIDY_LS_RETRIES", 2))
    MISS_STEPS = int(_f("BITIDY_LS_MISS_STEPS", 40))
    GRASP_AFTER = int(_f("BITIDY_LS_GRASP_AFTER", 0))
    LIFT_AFTER = int(_f("BITIDY_LS_LIFT_AFTER", 0))
    CUBE_FIRST = os.environ.get("BITIDY_LS_CUBE_FIRST", "1") == "1"  # right picks and lifts before the lid comes off
    CUBE_FIRST_LIMIT = int(_f("BITIDY_LS_CUBE_FIRST_LIMIT", 1500))  # ... but the left starts anyway after this many steps  # the right arm goes down onto the cube once the left is in this phase
    # (holding a lifted cube for long lets it slip out of the jaws)

    def __new__(cls, env):
        if not T.FACING:
            return LidSortSideScript(env)
        return super().__new__(cls)

    def __init__(self, env):
        self.env = env
        view = T.target_view(env)
        self.lid_mode = os.environ.get("BITIDY_LS_LID_SCRIPT", "open")
        if self.lid_mode == "open":  # the verified LidOpen script as is (level jaws across the arm's reach plane)
            self.left = LidOpenScript(view)
            self.left.T = T.lid_open_shim()
            self.left.ASIDE = _f("BITIDY_LS_L_ASIDE", 0.075)
            self.left.ASIDE_UP = _f("BITIDY_LS_L_ASIDE_UP", 0.025)
            # The held lid swings in the knob pinch: move it slower and only call it lost at 3 cm.
            self.left.LOST = _f("BITIDY_LO_LOST", 0.03)
            self.left.SPEED = _f("BITIDY_LO_SPEED", 0.001)
            self.left.LIFT = _f("BITIDY_LS_L_LIFT", 0.03)
        else:
            self.left = RowLidScript(view)
        # back: -x; out: end boxes outward along y; dirs: per row position (-y end; middle; +y end), BITIDY_LS_ASIDE_DIRS
        self.ASIDE_MODE = os.environ.get("BITIDY_LS_ASIDE_MODE", "dirs")
        dirs = os.environ.get("BITIDY_LS_ASIDE_DIRS", "-1,-0.3;-1,0;-0.5,1")
        self.aside_dirs = torch.nn.functional.normalize(torch.tensor([[float(c) for c in d.split(",")] for d in dirs.split(";")],
                                                                     device=env.device), dim=-1)
        self.right = (FrontPlaceScript if os.environ.get("BITIDY_LS_PLACER", "front") == "front" else RowPlaceScript)(env)
        n, dev = env.num_envs, env.device
        self.l_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.aside_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.lift_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.park_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)
        self.cube_up = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, ids: torch.Tensor) -> None:
        self.cube_up[ids] = False
        if os.environ.get("BITIDY_ROWS"):
            R, L = self.right, self.left
            for e in ids.tolist()[:600]:
                print(f"BITIDY_END env={e} t={int(self.env.episode_length_buf[e])} R.p={int(R.p[e])} leg={int(getattr(R, 'leg', R.p)[e])} "
                      f"R.sub={int(R.sub[e])} L.p={int(L.p[e])} retries={int(self.retries[e])} lift_steps={int(self.lift_steps[e])} "
                      f"flags={self.env._bi_final[e].int().tolist()}", flush=True)
        self.left.reset(ids)
        self.right.reset(ids)
        self.l_started[ids] = False
        for t in (self.aside_steps, self.lift_steps, self.park_steps, self.retries):
            t[ids] = 0

    def act(self) -> torch.Tensor:
        env, L, R = self.env, self.left, self.right
        lid_ready = (L.p == LEFT_HOLD) & T.lid_aside(env) & T.lid_off_opening(env)
        self.aside_steps = torch.where(lid_ready, self.aside_steps + 1, 0)
        lid_ready = self.aside_steps >= self.LID_SETTLE

        parked = (R.p == RIGHT_PARK) & T.right_jaws_clear(env)
        self.park_steps = torch.where(parked, self.park_steps + 1, 0)
        if os.environ.get("BITIDY_LS_EARLY_DONE", "1") == "1":
            # The cube is in and the right jaws are out of the way (backing off / parking): the lid may go back.
            clear = (R.p >= 8) & T.right_jaws_clear(env) & T.cube_resting_inside(env)
            self.park_steps = torch.where(clear | parked, self.park_steps + 1, 0)
            right_done = (clear & (self.park_steps > 5)) | (parked & (self.park_steps > self.GIVE_UP))
        else:
            right_done = parked & (T.cube_resting_inside(env) | (self.park_steps > self.GIVE_UP))
        # Aside direction per env.
        n = env.num_envs
        back = torch.tensor((-1.0, 0.0), device=env.device).expand(n, -1)
        if self.ASIDE_MODE == "out":
            rel = T.target_box(env)[:, 1] - T.box_centers(env)[:, :, 1].mean(-1)
            side = torch.where(rel.abs() > 0.03, torch.sign(rel), torch.zeros_like(rel))
            out = torch.stack((torch.full_like(side, -0.5), side), -1)
            d = torch.where(side.unsqueeze(-1) != 0, torch.nn.functional.normalize(out, dim=-1), back)
        elif self.ASIDE_MODE == "dirs":
            ys = T.box_centers(env)[:, :, 1]
            rank = (ys < T.target_box(env)[:, 1:2] - 0.01).sum(-1)  # target's row position, 0 = -y end
            d = self.aside_dirs[rank.clamp(max=2)]
        else:
            d = back
        if self.lid_mode == "open":
            L.ASIDE_DIR = d
        else:
            L.aside_dir = d
        if os.environ.get("BITIDY_LS_TRACE"):
            first = (self.park_steps == 1) | ((L.p >= 6) & (L.t == 60))
            for e in first.nonzero().squeeze(-1).tolist()[:6]:
                r3 = lambda v: [round(x, 3) for x in v[e].tolist()]
                print(f"BITIDY_TRACE env={e} t={int(env.episode_length_buf[e])} L.p={int(L.p[e])} L.t={int(L.t[e])} R.p={int(R.p[e])} "
                      f"rdone={bool(right_done[e])} padL={r3(base.pad_center(env, 'robot_left'))} padR={r3(base.pad_center(env, 'robot_right'))} "
                      f"knob={r3(T.knob_pos(env))} tilt={float(T.lid_tilt(env)[e]):.3f} box={r3(T.target_box(env))} "
                      f"cube={r3(T.cube_pos(env))} rest={bool(T.cube_resting_inside(env)[e])} flags={T.progress(env).flags[e].int().tolist()}", flush=True)

        hold_on = (L.p == LEFT_HOLD) & ~right_done
        L.t = torch.where(hold_on, L.t.clamp(max=L.HOLD - 1), L.t)
        # The arms start with their jaws side by side over the row: the left waits until the right has moved out.
        self.l_started |= (R.sub >= 1) | (R.p > 0) | (env.episode_length_buf > 90)
        wait = ~self.l_started
        if self.CUBE_FIRST:  # the lid comes off only once the cube is up beside the row (short lid hold)
            self.cube_up |= (R.p == RIGHT_LIFT) & T.cube_held(env) & (self.lift_steps >= self.LIFT_STEPS)
            wait = wait | (~self.cube_up & (env.episode_length_buf < self.CUBE_FIRST_LIMIT))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_left = L.act()
        L.p = torch.where(wait, 0, L.p)
        L.t = torch.where(wait, 0, L.t)
        L.settle = torch.where(wait, 0, L.settle)
        L.prev_p = torch.where(wait, -1, L.prev_p)
        L.roll = torch.where(wait, 0.0, L.roll)
        L.high = L.high | wait
        a_left = torch.where(wait.unsqueeze(-1), torch.zeros_like(a_left), a_left)
        a_left[:, -1] = torch.where(wait, 1.0, a_left[:, -1])

        before = R.p.clone()
        a_right = R.act()
        # Hover over the cube until the lid is coming off.
        early = (before == 0) & (R.p == 1) & (L.p < self.GRASP_AFTER)
        R.p = torch.where(early, 0, R.p)
        # Jaws closed on the cube: keep it resting on the table (no slipping) until the lid is lifted.
        hold_low = (before == 3) & (R.p == 4) & (L.p < self.LIFT_AFTER)
        R.p = torch.where(hold_low, 3, R.p)
        R.t = torch.where(hold_low, 14, R.t)

        # Lifted cube: hold it at the carry height beside the row until the lid is held aside.
        self.lift_steps = torch.where(before == RIGHT_LIFT, self.lift_steps + 1, 0)
        lifted = (before == RIGHT_LIFT) & (R.p == RIGHT_CARRY)
        R.p = torch.where(lifted, RIGHT_LIFT, R.p)
        abort = (R.p == RIGHT_CARRY) & (L.p != LEFT_HOLD)
        R.p = torch.where(abort, RIGHT_LIFT, R.p)
        go = ((R.p == RIGHT_LIFT) & lid_ready & (lifted | (self.lift_steps > self.LIFT_FORCE))
              & (self.lift_steps >= self.LIFT_STEPS) & T.cube_held(env))
        R.p = torch.where(go, RIGHT_CARRY, R.p)
        R.t = torch.where(lifted | abort | go, 0, R.t)

        # Retry the pick: jaws closed on nothing, or the cube back on the table outside the box after the drop.
        held = T.cube_held(env)
        cube = T.cube_pos(env)
        missed = (R.p == RIGHT_LIFT) & (self.lift_steps > self.MISS_STEPS) & ~held
        on_table = (cube[:, 2] < base.CUBE_REST_Z + 0.005) & ~T.cube_over_opening(env)
        dropped = (R.p >= 8) & on_table & ~held
        retry = (missed | dropped) & (self.retries < self.RETRIES) & (L.p <= LEFT_HOLD)
        R.p = torch.where(retry, 0, R.p)
        R.t = torch.where(retry, 0, R.t)
        R.sub = torch.where(retry, 1, R.sub)
        self.lift_steps = torch.where(retry, 0, self.lift_steps)
        self.retries += retry.long()

        if os.environ.get("BITIDY_ROWS"):
            last = (env.episode_length_buf == int(env.max_episode_length) - 3).nonzero().squeeze(-1)
            for e in last.tolist():
                print(f"BITIDY_END env={e} R.p={int(R.p[e])} leg={int(getattr(R, 'leg', R.p)[e])} R.sub={int(R.sub[e])} "
                      f"L.p={int(L.p[e])} retries={int(self.retries[e])} lift_steps={int(self.lift_steps[e])} "
                      f"pad={[round(v, 3) for v in base.pad_center(env, 'robot_right')[e].tolist()]} "
                      f"cube={[round(v, 3) for v in T.cube_pos(env)[e].tolist()]} box={[round(v, 3) for v in T.target_box(env)[e].tolist()]} "
                      f"flags={T.progress(env).flags[e].int().tolist()}", flush=True)
        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            prog = T.progress(env)
            print(f"[ls] t={int(env.episode_length_buf[k])} cmd={T.command(env)[k].tolist()} L.p={int(L.p[k])} R.p={int(R.p[k])} "
                  f"lid_ready={bool(lid_ready[k])} right_done={bool(right_done[k])} box={[round(v, 3) for v in T.target_box(env)[k].tolist()]} "
                  f"flags={prog.flags[k].int().tolist()} succ={bool(prog.success[k])}", flush=True)
        return torch.cat((a_left, a_right), -1)


# ---------------------------------------------------------------- previous side-by-side oracle


class LidSortSideScript:
    """The previous oracle for the side-by-side layout (``BITIDY_LS_FACING_X=0``)."""

    LID_SETTLE = int(_f("BITIDY_LP_LID_SETTLE", 10))
    GIVE_UP = int(_f("BITIDY_LP_GIVE_UP", 90))
    LIFT_STEPS = int(_f("BITIDY_LP_LIFT_STEPS", 90))
    RIGHT_START = int(_f("BITIDY_LP_RIGHT_START", 1))
    ASIDE = _f("BITIDY_LP_ASIDE", 0.10)
    # Park (BoxPlace default): well clear of the boxes; a lower, more reachable park swept over neighbour lids.
    PARK = tuple(map(float, os.environ.get("BITIDY_LS_PARK", "0.14,-0.12,0.11").split(",")))
    # Missed grasps (the cube not in the jaws while lifting) and cubes dropped on the table: start the pick over.
    RETRIES = int(_f("BITIDY_LS_RETRIES", 2))
    # Lid aside direction per env: +y unless the held lid would hang over another box; then the first of these
    # directions (deg) whose aside spot is ``ASIDE_FREE`` clear of the other boxes (else the clearest one).
    ASIDE_ANGLES = tuple(float(a) for a in os.environ.get("BITIDY_LS_ASIDE_ANGLES", "90,60,120,30,150").split(","))
    ASIDE_FREE = _f("BITIDY_LS_ASIDE_FREE", 0.08)
    MISS_STEPS = int(_f("BITIDY_LS_MISS_STEPS", 25))
    # Carry via point: first to this far in front (-y) of the commanded box, at carry height, then over the box (the
    # cube approaches the row of boxes from in front of its own box, never across a neighbour). 0 = straight carry.
    VIA = _f("BITIDY_LS_VIA", 0.05)

    def __init__(self, env):
        self.env = env
        view = T.target_view(env)
        # The verified single-arm scripts read their task module (``LidOpenScript.T``, ``oracle_box_place.B``): hand
        # them LidSort's geometry and per-env commanded box / lid / cube instead (this process runs only LidSort).
        from . import oracle_box_place

        oracle_box_place.B = T.box_place_shim()
        self.left = LidOpenScript(view)
        self.left.T = T.lid_open_shim()
        if "BITIDY_LO_ASIDE" not in os.environ:
            self.left.ASIDE = self.ASIDE
        self.right = BoxPlaceScript(T.placer_view(env))
        if "BITIDY_PARK" not in os.environ:
            self.right.PARK = self.PARK
        # Lift the cube higher before carrying it past other boxes (reach-limited: ~10 cm pad height in practice).
        for attr, var, value in (("LIFT_ABOVE", "BITIDY_LIFT_ABOVE", 0.083), ("CARRY_ABOVE", "BITIDY_CARRY_ABOVE", 0.055)):
            if var not in os.environ:
                setattr(self.right, attr, value)
        n, dev = env.num_envs, env.device
        self.r_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.aside_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.lift_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.park_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)
        st = T._state(env)
        st["via"] = torch.zeros((n, 2), device=dev)
        st["via_on"] = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, ids: torch.Tensor) -> None:
        self.left.reset(ids)
        self.right.reset(ids)
        self.r_started[ids] = False
        for t in (self.aside_steps, self.lift_steps, self.park_steps, self.retries):
            t[ids] = 0
        T._state(self.env)["via_on"][ids] = False

    def _aside_dir(self) -> torch.Tensor:
        env = self.env
        dirs = torch.tensor([(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in self.ASIDE_ANGLES], device=env.device)
        spots = T.target_box(env).unsqueeze(1) + self.left.ASIDE * dirs.unsqueeze(0)  # (n, A, 2)
        boxes = T.box_centers(env)  # (n, 3, 2)
        d = torch.linalg.vector_norm(spots.unsqueeze(2) - boxes.unsqueeze(1), dim=-1)  # (n, A, 3)
        others = torch.arange(T.N, device=env.device).unsqueeze(0) != T.command(env)[:, 1:2]
        clear = torch.where(others.unsqueeze(1), d, torch.full_like(d, 1.0)).min(-1).values  # (n, A)
        ok = clear >= self.ASIDE_FREE
        pick = torch.where(ok.any(-1), ok.float().argmax(-1), clear.argmax(-1))
        return dirs[pick]

    def act(self) -> torch.Tensor:
        env, L, R = self.env, self.left, self.right
        if not os.environ.get("BITIDY_LO_ASIDE_DIR"):
            L.ASIDE_DIR = self._aside_dir()

        lid_ready = (L.p == LEFT_HOLD) & T.lid_aside(env) & T.lid_off_opening(env)
        self.aside_steps = torch.where(lid_ready, self.aside_steps + 1, 0)
        lid_ready = self.aside_steps >= self.LID_SETTLE

        parked = (R.p == RIGHT_PARK) & T.right_jaws_clear(env)
        self.park_steps = torch.where(parked, self.park_steps + 1, 0)
        right_done = parked & (T.cube_resting_inside(env) | (self.park_steps > self.GIVE_UP))

        hold_on = (L.p == LEFT_HOLD) & ~right_done
        L.t = torch.where(hold_on, L.t.clamp(max=L.HOLD - 1), L.t)
        with warnings.catch_warnings():  # LidOpenScript builds a tensor from ASIDE_DIR (here already a tensor)
            warnings.simplefilter("ignore")
            a_left = L.act()

        self.r_started |= L.p >= self.RIGHT_START
        wait = ~self.r_started
        before = R.p.clone()
        st = T._state(env)
        if self.VIA > 0:
            box = T.target_box(env)
            st["via"] = box - box.new_tensor((0.0, self.VIA))
            # Via on from the lift until the cube is within 1 cm (xy) of the via point during the carry.
            near_via = torch.linalg.vector_norm(T.cube_pos(env)[:, :2] - st["via"], dim=-1) < 0.012
            st["via_on"] = (R.p <= RIGHT_CARRY) & ~((R.p == RIGHT_CARRY) & near_via) & ~((R.p == RIGHT_CARRY) & ~st["via_on"])
        a_right = R.act()
        # The script may have counted the via point as "over the box": keep carrying, now to the box itself.
        at_via = st["via_on"] & (R.p == RIGHT_CARRY + 1) & (before == RIGHT_CARRY)
        R.p = torch.where(at_via, RIGHT_CARRY, R.p)
        R.t = torch.where(at_via, 0, R.t)
        st["via_on"] = st["via_on"] & ~at_via
        R.p = torch.where(wait, 0, R.p)
        R.t = torch.where(wait, 0, R.t)
        R.roll = torch.where(wait, 0.0, R.roll)
        R.bias = torch.where(wait.unsqueeze(-1), 0.0, R.bias)
        a_right = torch.where(wait.unsqueeze(-1), torch.zeros_like(a_right), a_right)
        a_right[:, -1] = torch.where(wait, 1.0, a_right[:, -1])

        self.lift_steps = torch.where(before == RIGHT_LIFT, self.lift_steps + 1, 0)
        blocked = (before == RIGHT_LIFT) & (R.p == RIGHT_CARRY) & ~lid_ready
        abort = (R.p == RIGHT_CARRY) & (L.p != LEFT_HOLD)
        R.p = torch.where(blocked | abort, RIGHT_LIFT, R.p)
        go = (R.p == RIGHT_LIFT) & lid_ready & (self.lift_steps >= self.LIFT_STEPS)
        R.p = torch.where(go, RIGHT_CARRY, R.p)
        R.t = torch.where(blocked | abort | go, 0, R.t)

        # Retry the pick: jaws closed on nothing (still not holding after MISS_STEPS of lifting), or the cube ended up
        # back on the table outside the box after the drop (right arm on its way up / parked).
        held = T.cube_held(env)
        cube = T.cube_pos(env)
        missed = (R.p == RIGHT_LIFT) & (self.lift_steps > self.MISS_STEPS) & ~held
        on_table = (cube[:, 2] < base.CUBE_REST_Z + 0.005) & ~T.cube_over_opening(env)
        dropped = (R.p >= 8) & on_table & ~held
        retry = (missed | dropped) & (self.retries < self.RETRIES) & (L.p <= LEFT_HOLD)
        R.p = torch.where(retry, 0, R.p)
        R.t = torch.where(retry, 0, R.t)
        self.lift_steps = torch.where(retry, 0, self.lift_steps)
        self.retries += retry.long()

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            prog = T.progress(env)
            r3 = lambda v: [round(x, 3) for x in v[k].tolist()]
            print(f"[ls] t={int(env.episode_length_buf[k])} cmd={T.command(env)[k].tolist()} L.p={int(L.p[k])} R.p={int(R.p[k])} "
                  f"lid_ready={bool(lid_ready[k])} right_done={bool(right_done[k])} box={r3(T.target_box(env))} "
                  f"lid={r3(T.lid_pos(env))} cube={r3(T.cube_pos(env))} padL={r3(base.pad_center(env, 'robot_left'))} "
                  f"padR={r3(base.pad_center(env, 'robot_right'))} flags={prog.flags[k].int().tolist()} "
                  f"succ={bool(prog.success[k])}", flush=True)
        return torch.cat((a_left, a_right), -1)
