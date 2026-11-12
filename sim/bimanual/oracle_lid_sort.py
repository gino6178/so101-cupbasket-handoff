"""Scripted oracle for BiTidy-LidSort-SO101 (``BITIDY_SCRIPT=oracle_lid_sort:LidSortScript``).

Facing layout (the task default, arms across the table, boxes in a row along y between them).

DEFAULT (``BITIDY_LS_LID_MODE=setdown``, :class:`LidSortSetDownScript`): the lid is never held while the cube moves.

* LEFT arm (:class:`LidTripScript`): lid j off by its knob and set down on the table next to the row
  (``BITIDY_LS_SPOTS``, per row position of box j: -y end -> beyond the -y end; middle and +y end -> beyond the +y
  end, the middle lid carried over the +y lid), let go, up and wait; once the cube rests in box j and the right jaws
  are clear, pick the lid up from the table again and put it back on box j.
* RIGHT arm (:class:`FrontPlaceScript`, path ``row``): pick cube i from its slot beside the -y end, lift; once the lid
  is down, rise in place, then carry along a line ``CROSS_U`` (3.5 cm) IN FRONT of the knobs, so the cube passes
  beside the other lids' knobs, over their plates only (``CROSS_CLEAR`` above the plate tops), slowly
  (``CARRY_SPEED``) and only while the pads are within ``ZGATE`` of the carry height (the arm sags); then in over box
  j along -u, lower, centre, drop; up out of the box and out in front of it, low (``BITIDY_LS_BACK=out``). A -y end
  target (next to the slots) is carried straight to.

Previous coordination (``BITIDY_LS_LID_MODE=hold``, :class:`LidSortScript`):

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

TAKE-OUT commands (v2, ``lid_sort_env_cfg.TAKEOUT``; cube i starts in box j with the lid on):
:class:`LidSortSetDownScript` with ``BITIDY_LS_OUT_ARM=left`` (default): the left arm sets lid j down (take-out lid
spots ``BITIDY_LS_OUT_SPOTS``), then :class:`CubeTripScript` lifts the cube out of box j with near-vertical jaws and
sets it on a free table spot in the cube region within its reach, then puts the lid back; the right arm stays at its
ready point. ``BITIDY_LS_OUT_ARM=right``: :class:`TakeOutScript` (right arm in-box grasp) -- measured ~0 %: from 21-25
cm out the right arm only reaches the row tilted ~45 deg and its gripper lands on the box's front wall.

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
    ACCUM_FROM = int(_f("BITIDY_LS_R_ACCUM_FROM", 99))  # 99: off (integral IK stalls at the joint limits)
    ZGATE = _f("BITIDY_LS_ZGATE", 0.003)
    CROSS_U = _f("BITIDY_LS_CROSS_U", 0.035)
    CARRY_SPEED = _f("BITIDY_LS_CARRY_SPEED", 0.0015)  # slow carry: less dynamic sag while crossing
    DROP_ANYWAY = int(_f("BITIDY_LS_DROP_ANYWAY", 90))
    DROP_U = _f("BITIDY_LS_DROP_U", 0.0)  # drop point offset along the row normal (toward the right base)
    PINCH = _f("BITIDY_PINCH", 0.0035)  # pads this far above the cube centre (a lower pinch holds through the long carry)
    ACCUM_TO = int(_f("BITIDY_LS_R_ACCUM_TO", 6))
    ACCUM_GAIN = _f("BITIDY_LS_R_ACCUM_GAIN", 0.3)
    LEG_ZTOL = _f("BITIDY_LS_LEG_ZTOL", 0.006)
    LEG_LIMIT = int(_f("BITIDY_LS_LEG_LIMIT", 90))
    PLACE_DIR = os.environ.get("BITIDY_LS_R_PLACE_DIR", "box")  # box: fixed jaw azimuth -u; base: away from the base
    BACK = os.environ.get("BITIDY_LS_BACK", "out")  # out: up, then out in front of the box low; front: to BACK_U at carry height
    BACK_UP_OUT = _f("BITIDY_LS_BACK_UP_OUT", 0.035)
    PARK_U = _f("BITIDY_LS_PARK_U", 0.10)
    PARK_Z_OUT = _f("BITIDY_LS_PARK_Z_OUT", 0.07)
    DROP_RIM = _f("BITIDY_LS_DROP_RIM", 0.0)  # release height computed for a rim at least this high (z, m)

    def __init__(self, env):
        super().__init__(env)
        self.leg = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.direct = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

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
        rim = max(T.RIM_Z, self.DROP_RIM)  # lower box walls: release as high as over the 4 cm walls (jaws clear the rim)
        carry_z = T.RIM_Z + self.CARRY_RIM + T.CUBE_SIZE / 2 + off[:, 2:]
        drop_z = rim + T.CUBE_SIZE / 2 + self.DROP_CLEAR + off[:, 2:]
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
            # rise in place, to the row line, along it ('direct': straight from above the slot to the box)
            # CROSS_U > 0: the crossing line runs CROSS_U in front of the knobs (row normal toward the right base): the
            # cube passes beside the knobs, over the plates only, then goes in over the box along -u.
            a = torch.where(col(self.direct), g[:, :2], uv(self.CROSS_U, v_cube))
            b = torch.where(col(self.direct), box, uv(self.CROSS_U, 0.0))
            legs = torch.stack((g[:, :2], a, b, box), 1)
            top = T.LID_REST_Z + T.LID_SIZE[2] / 2 if self.CROSS_U > 0 else T.KNOB_TOP_Z
            carry_z = torch.where(col(p >= 5), top + self.CROSS_CLEAR + T.CUBE_SIZE / 2 + off[:, 2:], carry_z)
        else:
            legs = torch.stack((uv(self.FRONT, v_cube), uv(self.FRONT, 0.0), box, box), 1)  # (n, 4, 2)
        leg_xy = legs[torch.arange(n, device=env.device), self.leg.clamp(max=3)] + off[:, :2]
        t = torch.where(col(p >= 5), at(leg_xy, carry_z), t)
        t = torch.where(col(p >= 6), at(uv(self.DROP_U, 0.0) + off[:, :2], drop_z), t)
        if self.BACK == "out":  # straight up out of the box, then out in front of it, low (high near the base is out of reach)
            self.release_xy = torch.where(col(p <= 7), pad[:, :2], self.release_xy)
            t = torch.where(col(p >= 8), at(self.release_xy, T.RIM_Z + self.BACK_UP_OUT), t)
            t = torch.where(col(p >= 9), at(uv(self.PARK_U, 0.0), self.PARK_Z_OUT), t)
        else:
            back = uv(self.BACK_U, 0.0)
            t = torch.where(col(p >= 8), at(back, carry_z), t)
        changed = p != self.prev_p
        self.carrot = torch.where(col(changed), pad, self.carrot)
        self.prev_p = p.clone()
        self.leg = torch.where(changed, 0, self.leg)
        step = t - self.carrot
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        speed = torch.where((p == 4) | (p == 5), self.CARRY_SPEED, self.FREE_SPEED)
        # Horizontal carry legs: the carrot waits while the pads hang more than ZGATE below the carry height (no low
        # crossing over a lid when the arm sags).
        speed = torch.where((p == 5) & (self.leg >= 1) & (t[:, 2] - pad[:, 2] > self.ZGATE), 0.0, speed)
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
        leg_next = (p == 5) & (self.leg < 3) & carrot_at & (((err < self.LEG_TOL) & z_ok) | (self.t > self.LEG_LIMIT))
        # (the drop below may only follow once the LAST leg's target was reached: err is to this step's leg target)
        last_leg = self.leg == 3
        self.leg = torch.where(leg_next, self.leg + 1, self.leg)
        self.t = torch.where(leg_next, 0, self.t)
        if os.environ.get("BITIDY_LS_TRACE_R") and 0 < int(env.episode_length_buf[0]) - int(os.environ["BITIDY_LS_TRACE_R"]) < 40:
            print(f"[tr] t={int(env.episode_length_buf[0])} p={int(p[0])} leg={int(self.leg[0])} Rt={int(self.t[0])} carrot_at={bool(carrot_at[0])} "
                  f"err={float(err[0]):.4f} z_ok={bool(z_ok[0])} tgt={[round(v, 3) for v in t[0].tolist()]} carrot={[round(v, 3) for v in self.carrot[0].tolist()]}", flush=True)
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (self.sub >= 2) & (err < self.HOVER_TOL)
        adv |= (p == 1) & (err < 0.004)
        adv |= (p == 2) & (err < self.GRASP_TOL)
        adv |= (p == 3) & (self.t > 15)
        adv |= (p == 4) & (err < 0.01)
        adv |= (p == 5) & last_leg & (err < 0.008)
        adv |= (p == 6) & (err < 0.005) & centred & (box_err.abs() < 0.1)
        adv |= (p == 6) & over_ok & (self.t > self.DROP_ANYWAY)  # cannot centre it (sag): drop it while it is over the opening
        adv |= (p == 7) & (self.t > 15)
        adv |= (p == 8) & (err < 0.01)
        hold_on = ((p == 6) & ~over_ok) | ((p == 5) & (self.leg < 3))
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
        if os.environ.get("BITIDY_LS_LID_MODE", "setdown") == "setdown":
            return LidSortSetDownScript(env)
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


# ---------------------------------------------------------------- lid set-down mode (the default)


def _spots(spec: str, dev) -> torch.Tensor:
    return torch.tensor([[float(c) for c in s.split(",")] for s in spec.split(";")], device=dev)


class LidTripScript(LidOpenScript):
    """Left arm, set-down mode: the lid is never held while the right arm works. Two knob pick-and-place trips:
    trip 0 takes lid j from its box to a spot on the table next to the row (``spot``, (u, v) in the box frame: u =
    row normal toward the right base, v = along the row) and lets go; the arm goes up and waits (phase 9). The
    coordinator then starts trip 1: the lid from the table back onto box j.

    Phases per trip: 0 hover over the knob, 1 down, 2 close, 3 lift (knob to ``CARRY_UP`` above its on-the-box height),
    4 to the corner point (u of the spot, v = 0: straight back / out, never over a neighbour), 5 over the destination,
    6 lower, 7 open, 8 up, 9 idle. While the lid is carried (3-6) the controlled point is the knob."""

    CARRY_UP = _f("BITIDY_LS_T_CARRY_UP", 0.03)
    UP = _f("BITIDY_LS_T_UP", 0.06)
    PRE1 = _f("BITIDY_LS_T_PRE1", 0.02)
    CLOSE = int(_f("BITIDY_LS_T_CLOSE", 15))
    SEAT_SHIFT = _f("BITIDY_LS_T_SEAT_SHIFT", 0.007)
    KEEP = os.environ.get("BITIDY_LS_T_KEEP", "end")
    """Keep the knob pinched while the lid rests on the table (no re-grasp): ``1`` always, ``end`` only for the -y end
    box's lid (default: the coordinator sets ``keep`` per env; 2048 envs: 53.8 % vs 51.6-52.2 % with ``0``), ``0`` never."""
    KEEP_SETTLE = int(_f("BITIDY_LS_T_KEEP_SETTLE", 40))
    SLOPE_TOL = 0.05  # hover: the jaws' closing axis this level (|z| of the unit axis)
    HOVER_LIMIT = 0  # steps before the hover (phase 0) is forced on; 0: 2 x PHASE_LIMIT
    SPEED_SLOW = _f("BITIDY_LO_SPEED_SLOW", 0.0007)  # down onto the knob / lid down onto the table or rim
    KNOB_PINCH = _f("BITIDY_LO_PINCH", -0.0015)  # pinch a little below the knob centre (fewer slips)  # trip 1 comes in low (a lid on the table may be near the arm's base)

    def __init__(self, env):
        super().__init__(env)
        self.T = T
        n, dev = env.num_envs, env.device
        self.trip = torch.zeros(n, dtype=torch.long, device=dev)
        self.keep = torch.full((n,), self.KEEP == "1", dtype=torch.bool, device=dev)
        self.prev_approach = torch.zeros(n, dtype=torch.bool, device=dev)
        self.spot = torch.zeros((n, 2), device=dev)
        self.spot[:, 0] = -0.08

    def reset(self, ids: torch.Tensor) -> None:
        super().reset(ids)
        self.trip[ids] = 0

    @property
    def down(self) -> torch.Tensor:
        """Trip 0 done: the lid lies on the table (let go, or -- KEEP -- still held by its knob)."""
        return (self.trip == 0) & ((self.p >= 8) | (self.keep & (self.p == 6) & (self.t > self.KEEP_SETTLE)))

    def start_trip(self, ids_mask: torch.Tensor) -> None:
        # KEEP: the jaws still hold the knob of the lid resting on the table -> lift it straight away (phase 3).
        in_jaws = torch.linalg.vector_norm(base.pad_center(self.env, self.arm) - T.knob_pos(self.env), dim=-1) < T.KNOB_HOLD_RADIUS
        keep = ids_mask & (self.p == 6) & in_jaws
        self.knob = torch.where(keep.unsqueeze(-1), T.knob_pos(self.env), self.knob)
        self.trip = torch.where(ids_mask, 1, self.trip)
        for t in (self.t, self.settle):
            t[ids_mask] = 0
        self.p = torch.where(ids_mask, torch.where(keep, 3, 0), self.p)
        self.prev_p[ids_mask] = -1
        self.high |= ids_mask & ~keep

    def act(self) -> torch.Tensor:
        env, p = self.env, self.p
        n, dev = env.num_envs, env.device
        up = torch.tensor((0.0, 0.0, 1.0), device=dev)

        def col(v):
            return v.unsqueeze(-1)

        def at(xy, z):
            return torch.cat((xy, torch.full_like(xy[:, :1], z) if isinstance(z, float) else col(z)), -1)

        pad = base.pad_center(env, self.arm)
        knob_now = T.knob_pos(env)
        self.knob = torch.where(col(p <= 0), knob_now, self.knob)
        box = T.target_box(env)
        yaw = T.target_box_yaw(env)
        ex = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
        ey = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
        rest_z = T.LID_REST_Z + T.KNOB_OFFSET
        table_z = base.TABLE_TOP + T.LID_SIZE[2] / 2 + T.KNOB_OFFSET
        carry_z = rest_z + self.CARRY_UP
        trip0 = self.trip == 0
        src_u = ((self.knob[:, :2] - box) * ex).sum(-1)
        corner_u = torch.where(trip0, self.spot[:, 0], src_u)
        corner = box + col(corner_u) * ex
        spot_xy = box + col(self.spot[:, 0]) * ex + col(self.spot[:, 1]) * ey
        dst_xy = torch.where(col(trip0), spot_xy, box)
        dst_z = torch.where(trip0, torch.full_like(src_u, table_z), torch.full_like(src_u, rest_z))

        pinch = self.knob + up * self.KNOB_PINCH
        hover = pinch + up * self.HOVER
        high = col(self.high & (p == 0))
        pre = torch.where(trip0, self.PRE, self.PRE1)
        pad_goal = torch.where(high, pinch + up * col(pre), hover)
        pad_goal = torch.where(col(p >= 1), pinch, pad_goal)
        pad_goal = torch.where(col(p >= 7), self.hold_pad, pad_goal)
        pad_goal = torch.where(col(p >= 8), self.hold_pad + up * self.UP, pad_goal)
        knob_goal = at(self.knob[:, :2], carry_z)
        knob_goal = torch.where(col(p >= 4), at(corner, carry_z), knob_goal)
        knob_goal = torch.where(col(p >= 5), at(dst_xy, carry_z), knob_goal)
        knob_goal = torch.where(col(p >= 6), at(dst_xy, dst_z + self.PLACE_ABOVE), knob_goal)
        carrying = (p >= 3) & (p <= 6)
        # Approaching the knob: steer a point shifted toward the fixed jaw (the closing moving jaw pushes the lid less).
        fixed_w, moving_w = jaw_tips(env, self.arm)
        seat = pad - torch.nn.functional.normalize(moving_w - fixed_w, dim=-1) * self.SEAT_SHIFT
        approach = (p <= 1) & ~self.high
        ctrl = torch.where(col(carrying), knob_now, torch.where(col(approach), seat, pad))
        goal = torch.where(col(carrying), knob_goal, pad_goal)
        changed = (p != self.prev_p) | (approach != self.prev_approach)
        self.prev_approach = approach
        self.wp = torch.where(col(changed), ctrl, self.wp)
        self.prev_p = p.clone()
        speed = torch.full((n,), self.SPEED, device=dev)
        speed = torch.where((p == 0) & self.high, self.SPEED_FREE, speed)
        speed = torch.where((p == 1) | (p == 6), self.SPEED_SLOW, speed)
        speed = torch.where(p >= 8, self.SPEED_FREE, speed)
        step = goal - self.wp
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        self.wp = self.wp + step * (col(speed) / dist.clamp(min=1e-6)).clamp(max=1.0)
        target = pad + self.wp - ctrl
        err = torch.linalg.vector_norm(goal - ctrl, dim=-1)
        self.hold_pad = torch.where(col(p <= 6), pad, self.hold_pad)
        grip = torch.where((p >= 2) & (p < 7), -1.0, 1.0)

        fixed, moving = jaw_tips(env, self.arm)
        axis = torch.nn.functional.normalize(moving - fixed, dim=-1)
        slope = axis[:, 2]
        self.roll = torch.where((p <= 6) & (p != 2), self.roll - self.LEVEL_GAIN * slope, self.roll).clamp(-2.0, 2.0)
        near = col(torch.linalg.vector_norm(target - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (target - pad), 0.9 * self.bias).clamp(-0.03, 0.03)
        direction = self.ik.tilted(target, self.TILT)
        point = torch.nn.functional.normalize(pad - (self.ik.robot.data.body_pos_w.torch[:, self.ik.body] - env.scene.env_origins), dim=-1)
        lid_up = quat_apply(T.lid_quat(env), up.expand(n, -1))
        u = lid_up - (lid_up * axis).sum(-1, keepdim=True) * axis
        zp = up - col(axis[:, 2]) * axis
        angle = torch.atan2((axis * torch.cross(u, zp, dim=-1)).sum(-1), (u * zp).sum(-1))
        a = col(self.PITCH_GAIN * angle)
        keep = point * torch.cos(a) + torch.cross(axis, point, dim=-1) * torch.sin(a) + axis * col((axis * point).sum(-1)) * (1 - torch.cos(a))
        in_jaws = torch.linalg.vector_norm(pad - knob_now, dim=-1) < T.KNOB_HOLD_RADIUS
        held = col(carrying & in_jaws | carrying & (p == 3))
        direction = torch.where(held, keep, direction)
        weight = torch.where(carrying, self.PW_CARRY, self.PW)
        act, _ = self.ik(target + self.bias, direction, self.roll, gain=self.GAIN, point_weight=weight)
        act = torch.where(col((p == 2) | (p == 7)), torch.zeros_like(act), act)

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            r3 = lambda v: [round(x, 3) for x in v[k].tolist()]
            print(f"[trip] t={int(env.episode_length_buf[k])} trip={int(self.trip[k])} p={int(p[k])} err={float(err[k]):.4f} "
                  f"pad={r3(pad)} knob={r3(knob_now)} goal={r3(goal)} tilt={float(T.lid_tilt(env)[k]):.3f}", flush=True)

        self.t += 1
        self.high &= ~((p == 0) & ((err < 0.015) | (self.t > 90)))
        hover_ok = (err < self.HOVER_TOL) & (slope.abs() < self.SLOPE_TOL) & ~self.high
        self.settle = torch.where((p == 0) & hover_ok, self.settle + 1, 0)
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (self.settle >= self.SETTLE)
        adv |= (p == 1) & (err < self.PINCH_TOL)
        adv |= (p == 2) & (self.t > self.CLOSE)
        adv |= (p == 3) & (err < 0.01)
        adv |= (p == 4) & (err < 0.008)
        adv |= (p == 5) & (err < 0.006)
        adv |= (p == 6) & (err < 0.004)
        adv |= (p == 7) & (self.t > 15)
        adv |= (p == 8) & (err < 0.01)
        limit = torch.where(p == 0, self.HOVER_LIMIT if self.HOVER_LIMIT > 0 else 2 * self.PHASE_LIMIT, self.PHASE_LIMIT)
        moved = adv | ((self.t > limit) & (p < 9))
        # keep: trip 0 stays lowered with the knob pinched (the coordinator starts trip 1 from there)
        moved = moved & ~(self.keep & trip0 & (p == 6))
        idx = torch.arange(n, device=dev)
        self.reach[idx, p] = True
        self.dur[idx, p] += 1
        fids = (moved & ~adv).nonzero().squeeze(-1)
        self.forced[fids, p[fids]] = True
        if os.environ.get("BITIDY_ROWS"):
            for e in fids.tolist()[:20]:
                print(f"BITIDY_FORCED_L env={e} trip={int(self.trip[e])} phase={int(p[e])} t={int(env.episode_length_buf[e])} "
                      f"err={float(err[e]):.4f} pad={[round(v, 4) for v in pad[e].tolist()]} knob={[round(v, 4) for v in knob_now[e].tolist()]}", flush=True)
        nxt = (p + 1).clamp(max=9)
        lost = (torch.linalg.vector_norm(pad - knob_now, dim=-1) > self.LOST) & carrying & (self.t > 3)
        if os.environ.get("BITIDY_ROWS"):
            for e in lost.nonzero().squeeze(-1).tolist()[:20]:
                print(f"BITIDY_LOST_L env={e} trip={int(self.trip[e])} phase={int(p[e])} t={int(env.episode_length_buf[e])}", flush=True)
        self.retries += (lost & (p < 6)).long()
        nxt = torch.where(lost, torch.where(p < 6, 0, 7), nxt)
        moved = moved | lost
        self.high |= lost & (p < 6)
        self.p = torch.where(moved, nxt, p)
        self.t = torch.where(moved, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)


class TakeOutScript(FrontPlaceScript):
    """Right arm, TAKE-OUT command (cube i inside box j, lid j coming off): BoxPlace's phase numbers with take-out
    targets. 0 approach (ready point, then WAIT in front of box j high until the coordinator lets it on once lid j is
    down on the table, then hover over the cube), 1 just above the rim over the cube, 2 straight down into the box to the
    pinch, 3 close, 4 straight up until the cube bottom clears the rim (``LIFT_CLEAR``), 5 carry legs (mirror of the
    put-in carry: -y end box straight to the spot, others out to the line ``CROSS_U`` in front of the knobs, along it,
    then to the spot), 6 lower onto the table at the spot, 7 open, 8 up, 9 park (above the spot, toward the right base).

    The spot is drawn per env (``pick_spot``) in the random-cube region: ``SPOT_R`` from the right base, ``SPOT_CUBE``
    from the other cubes, ``SPOT_BOX`` from every box centre and ``SPOT_LID`` from where lid j is set down, preferring
    spots near box j (short carry)."""

    TILT_IN = _f("BITIDY_LS_O_TILT", 1.2)  # jaw tilt for the in-box grasp (steeper than on the table: walls)
    PW_IN = _f("BITIDY_LS_O_PW", 0.005)  # IK jaw-direction weight while going into the box (keep the jaws steep)
    HOVER_IN = _f("BITIDY_LS_O_HOVER", 0.045)  # pads this far above the cube centre before going in
    ABOVE_IN = _f("BITIDY_LS_O_ABOVE", 0.03)  # phase 1: pads this far above the cube centre (rim: +2.75 cm)
    PINCH_IN = _f("BITIDY_LS_O_PINCH", 0.007)
    DOWN_SPEED = _f("BITIDY_LS_O_DOWN_SPEED", 0.002)  # carrot speed into the box
    LIFT_CLEAR = _f("BITIDY_LS_O_LIFT_CLEAR", 0.015)  # cube bottom above the rim / plate tops when lifted and carried
    LIFT_SPEED = _f("BITIDY_LS_O_LIFT_SPEED", 0.0015)
    PLACE_CLEAR = _f("BITIDY_LS_O_PLACE_CLEAR", 0.003)  # cube bottom above the table at release
    WAIT_U = _f("BITIDY_LS_O_WAIT_U", 0.07)  # wait point in front of box j (toward the right base) ...
    WAIT_Z = _f("BITIDY_LS_O_WAIT_Z", 0.10)  # ... this high, while the lid comes off
    PARK_UP = _f("BITIDY_LS_O_PARK_UP", 0.06)
    PARK_IN = _f("BITIDY_LS_O_PARK_X", 0.03)  # park this much further toward the right base (x) than the spot
    SPOT_R = tuple(float(v) for v in os.environ.get("BITIDY_LS_O_SPOT_R", "0.23,0.28").split(","))
    SPOT_CUBE = _f("BITIDY_LS_O_SPOT_CUBE", 0.06)
    SPOT_BOX = _f("BITIDY_LS_O_SPOT_BOX", 0.085)
    SPOT_LID = _f("BITIDY_LS_O_SPOT_LID", 0.075)
    SPOT_PREF = _f("BITIDY_LS_O_SPOT_PREF", 0.04)  # random tie-break (m) added to the distance to box j
    SPOT_FALLBACK = (0.24, -0.21)
    PLACE_ANYWAY = int(_f("BITIDY_LS_O_PLACE_ANYWAY", 60))
    APPROACH_SPEED = _f("BITIDY_LS_O_APPROACH_SPEED", 0.0015)
    JAW_SIDE = os.environ.get("BITIDY_LS_O_JAW_SIDE", "1") == "1"
    WAIT_READY = os.environ.get("BITIDY_LS_O_WAIT_READY", "1") == "1"
    CLOSED_APPROACH = os.environ.get("BITIDY_LS_O_CLOSED", "0") == "1"
    ACC_FROM_P = int(_f("BITIDY_LS_O_ACC_FROM", 0))  # integral IK in phases ACC_FROM..ACC_TO (from the hover on)
    ACC_TO_P = int(_f("BITIDY_LS_O_ACC_TO", 6))
    ACC_GAIN = _f("BITIDY_LS_O_ACC_GAIN", 0.3)
    APPROACH_ZGATE = _f("BITIDY_LS_O_APPROACH_ZGATE", 0.006)

    def __init__(self, env):
        super().__init__(env)
        n, dev = env.num_envs, env.device
        self.spot = torch.tensor(self.SPOT_FALLBACK, device=dev).expand(n, -1).clone()
        self.need_spot = torch.ones(n, dtype=torch.bool, device=dev)
        self.avoid = torch.full((n, 2), 10.0, device=dev)  # lid j's set-down spot (set by the coordinator)
        self.gate = torch.zeros(n, dtype=torch.bool, device=dev)  # True: wait (lid not down yet)
        self.spot_fallback = 0

    def reset(self, ids: torch.Tensor) -> None:
        super().reset(ids)
        self.need_spot[ids] = True

    def pick_spot(self, ids: torch.Tensor, tries: int = 512) -> None:
        env = self.env
        dev = env.device
        m = len(ids)
        cand = T._uniform((m, tries), T.CUBE_RX, T.CUBE_RY, dev)
        rb = torch.tensor(T.RIGHT_BASE, device=dev)
        r = torch.linalg.vector_norm(cand - rb, dim=-1)
        ok = (r >= self.SPOT_R[0]) & (r <= self.SPOT_R[1])
        boxes = T.box_centers(env)[ids]
        ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - boxes.unsqueeze(1), dim=-1) >= self.SPOT_BOX).all(-1)
        cubes = T._objs_pos(env, "cube")[ids, :, :2]
        others = T._others(env, 0)[ids]  # (m, 3)
        d = torch.linalg.vector_norm(cand.unsqueeze(2) - cubes.unsqueeze(1), dim=-1)  # (m, T, 3)
        ok &= ((d >= self.SPOT_CUBE) | ~others.unsqueeze(1)).all(-1)
        ok &= torch.linalg.vector_norm(cand - self.avoid[ids].unsqueeze(1), dim=-1) >= self.SPOT_LID
        box_j = T.target_box(env)[ids]
        score = torch.linalg.vector_norm(cand - box_j.unsqueeze(1), dim=-1) + torch.rand((m, tries), device=dev) * self.SPOT_PREF
        score = torch.where(ok, score, torch.full_like(score, 1e9))
        best = score.argmin(-1)
        found = ok.any(-1)
        pick = cand[torch.arange(m, device=dev), best]
        fb = torch.tensor(self.SPOT_FALLBACK, device=dev).expand(m, -1)
        self.spot[ids] = torch.where(found.unsqueeze(-1), pick, fb)
        self.spot_fallback += int((~found).sum())
        self.need_spot[ids] = False

    def act(self) -> torch.Tensor:
        env, p = self.env, self.p
        n = env.num_envs
        if bool(self.need_spot.any()):
            self.pick_spot(self.need_spot.nonzero().squeeze(-1))
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
        top = T.LID_REST_Z + T.LID_SIZE[2] / 2  # plate tops of the other (closed) boxes
        carry_z = top + self.LIFT_CLEAR + T.CUBE_SIZE / 2 + off[:, 2:]
        place_z = base.CUBE_REST_Z + self.PLACE_CLEAR + off[:, 2:]
        self.grasp = torch.where(col(p <= 0), cube, self.grasp)
        g = self.grasp
        spot = self.spot
        v_spot = ((spot - box) * ey).sum(-1)
        ready = torch.stack((torch.full_like(cube[:, 0], self.READY[0]), self.READY[1] * cube[:, 1],
                             torch.full_like(cube[:, 0], self.READY[2])), -1)
        if self.WAIT_READY:  # wait at the ready point (as the put-in approach), then over the cube high, then hover
            t = torch.where(col(self.sub == 0), ready, at(cube[:, :2], self.TRANSIT_Z))
        else:
            t = torch.where(col(self.sub == 0), ready, at(uv(self.WAIT_U, 0.0), self.WAIT_Z))
        t = torch.where(col(self.sub >= 2), cube + up * self.HOVER_IN, t)
        t = torch.where(col(p >= 1), g + up * self.ABOVE_IN, t)
        t = torch.where(col(p >= 2), g + up * self.PINCH_IN, t)
        t = torch.where(col(p >= 4), at(g[:, :2] + off[:, :2], carry_z), t)
        a = torch.where(col(self.direct), box, uv(self.CROSS_U, 0.0))
        b = torch.where(col(self.direct), spot, uv(self.CROSS_U, v_spot))
        legs = torch.stack((box, a, b, spot), 1)
        leg_xy = legs[torch.arange(n, device=env.device), self.leg.clamp(max=3)] + off[:, :2]
        t = torch.where(col(p >= 5), at(leg_xy, carry_z), t)
        t = torch.where(col(p >= 6), at(spot + off[:, :2], place_z), t)
        self.release_xy = torch.where(col(p <= 7), pad[:, :2], self.release_xy)
        t = torch.where(col(p >= 8), at(self.release_xy, base.CUBE_REST_Z + self.PARK_UP), t)
        park = self.release_xy + torch.tensor((self.PARK_IN, 0.0), device=env.device)
        t = torch.where(col(p >= 9), at(park, base.CUBE_REST_Z + self.PARK_UP), t)
        changed = p != self.prev_p
        self.carrot = torch.where(col(changed), pad, self.carrot)
        self.prev_p = p.clone()
        self.leg = torch.where(changed, 0, self.leg)
        step = t - self.carrot
        dist = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        speed = torch.full_like(box[:, 0], self.FREE_SPEED)
        # over to the hover point (near the reach limit): slow, and wait while the pads sag below its height
        speed = torch.where((p == 0) & (self.sub >= 2), self.APPROACH_SPEED, speed)
        speed = torch.where((p <= 1) & (self.sub >= 2) & (t[:, 2] - pad[:, 2] > self.APPROACH_ZGATE), 0.0, speed)
        speed = torch.where(p == 2, self.DOWN_SPEED, speed)
        speed = torch.where(p == 4, self.LIFT_SPEED, speed)
        speed = torch.where(p == 5, self.CARRY_SPEED, speed)
        speed = torch.where((p == 5) & (self.leg >= 1) & (t[:, 2] - pad[:, 2] > self.ZGATE), 0.0, speed)
        speed = torch.where(p == 6, self.DOWN_SPEED, speed)
        self.carrot = self.carrot + step * (col(speed) / dist.clamp(min=1e-6)).clamp(max=1.0)
        goal = self.carrot
        grip = torch.where((p >= 3) & (p < 7), -1.0, 1.0)
        if self.CLOSED_APPROACH:  # jaws closed (narrow) until the pads are over the cube inside the walls
            grip = torch.where((p <= 1) | ((p == 2) & (pad[:, 2] > T.RIM_Z + 0.005)), -1.0, grip)

        fixed, moving = jaw_tips(env, B_ARM)
        axis = moving - fixed
        jaw_yaw = torch.atan2(axis[:, 1], axis[:, 0])
        face_yaw = cube_yaw
        if self.JAW_SIDE:  # closing axis on the cube face nearest the row direction (jaws open sideways, not into the front wall)
            az = torch.atan2(box[:, 1] - self.ik.base_xy[:, 1], box[:, 0] - self.ik.base_xy[:, 0]) + math.pi / 2
            face_yaw = az + _fold(cube_yaw - az, math.pi / 2)
        face_err = _fold(face_yaw - jaw_yaw, math.pi) if self.JAW_SIDE else self._fold(face_yaw - jaw_yaw)
        d_roll = torch.where(p <= 2, self.ROLL_GAIN * face_err, torch.zeros_like(face_err))
        self.roll = (self.roll + d_roll).clamp(-1.6, 1.6)
        self.roll = torch.where(p >= 8, 0.95 * self.roll, self.roll)

        near = col(torch.linalg.vector_norm(goal - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (goal - pad), 0.9 * self.bias).clamp(-self.BIAS_CLAMP, self.BIAS_CLAMP)
        self.bias[:, 2] = torch.where(p <= 2, self.bias[:, 2].clamp(min=0.0), self.bias[:, 2])
        tilt_v = torch.full_like(box[:, 0], self.TILT_BOX)
        tilt_v = torch.where((self.sub >= 2) & (p <= 4), self.TILT_IN, tilt_v)
        tilt_v = torch.where(p >= 5, self.TILT_PLACE, tilt_v)
        out = torch.nn.functional.normalize(goal[:, :2] - self.ik.base_xy, dim=-1)
        tilt = torch.nn.functional.normalize(torch.cat((col(tilt_v) * out, -torch.ones_like(out[:, :1])), -1), dim=-1)
        pw = torch.full_like(box[:, 0], self.PW_PLACE)
        pw = torch.where((self.sub >= 2) & (p <= 4), self.PW_IN, pw)
        pw = torch.where(p == 0, torch.where(self.sub >= 2, self.PW_IN, self.PW_APPROACH), pw)
        act, _ = self.ik(goal + self.bias, tilt, self.roll, gain=self.GAIN, point_weight=pw)
        # Integral action (joint targets stepped, no PD gravity sag) from the hover over the box on: out there the arm
        # sags 4-6 cm under plain relative targets and drops in front of the box wall instead of reaching over it.
        acc_on = (self.sub >= 2) & (p >= self.ACC_FROM_P) & (p <= self.ACC_TO_P)
        if bool(acc_on.any()):
            acc, _ = self.ik(goal, tilt, self.roll, gain=self.ACC_GAIN, point_weight=pw, accumulate=True)
            act = torch.where(col(acc_on), acc, act)
        err = torch.linalg.vector_norm(t - pad, dim=-1)

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            print(f"[out] t={int(env.episode_length_buf[k])} p={int(p[k])} sub={int(self.sub[k])} leg={int(self.leg[k])} err={float(err[k]):.4f} "
                  f"pad={[round(v, 3) for v in pad[k].tolist()]} goal={[round(v, 3) for v in goal[k].tolist()]} "
                  f"cube={[round(v, 3) for v in cube[k].tolist()]} spot={[round(v, 3) for v in spot[k].tolist()]} held={bool(held[k])} "
                  f"face={float(face_err[k]):.3f} gate={bool(self.gate[k])} fixed={[round(v, 3) for v in fixed[k].tolist()]} "
                  f"moving={[round(v, 3) for v in moving[k].tolist()]} box={[round(v, 3) for v in box[k].tolist()]}", flush=True)

        self.t += 1
        self.sub_t += 1
        sub_next = (p == 0) & (self.sub < 2) & ((err < 0.02) | (self.sub_t > 100))
        sub_next &= ~((self.sub == (0 if self.WAIT_READY else 1)) & self.gate)  # wait until the lid is down
        at_hover = (self.sub >= 2) & ~sub_next  # (err above is to THIS step's target: the hover only once sub is 2)
        self.sub = torch.where(sub_next, self.sub + 1, self.sub)
        self.sub_t = torch.where(sub_next, 0, self.sub_t)
        carrot_at = torch.linalg.vector_norm(self.carrot - t, dim=-1) < 1e-4
        z_ok = (pad[:, 2] - t[:, 2]).abs() < self.LEG_ZTOL
        leg_next = (p == 5) & (self.leg < 3) & carrot_at & (((err < self.LEG_TOL) & z_ok) | (self.t > self.LEG_LIMIT))
        last_leg = self.leg == 3
        self.leg = torch.where(leg_next, self.leg + 1, self.leg)
        self.t = torch.where(leg_next, 0, self.t)
        on_table = cube[:, 2] < base.CUBE_REST_Z + self.PLACE_CLEAR + 0.002
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & at_hover & (err < self.HOVER_TOL) & (face_err.abs() < 0.08)
        adv |= (p == 1) & (err < 0.005)
        adv |= (p == 2) & carrot_at & (err < self.GRASP_TOL)
        adv |= (p == 3) & (self.t > 15)
        adv |= (p == 4) & carrot_at & (err < 0.01)
        adv |= (p == 5) & last_leg & carrot_at & (err < 0.008)
        adv |= (p == 6) & ((carrot_at & (err < 0.005)) | on_table | (carrot_at & (self.t > self.PLACE_ANYWAY)))
        adv |= (p == 7) & (self.t > 15)
        adv |= (p == 8) & (err < 0.01)
        hold_on = (p == 5) & (self.leg < 3)
        hold_on |= (p == 0) & self.gate
        lost = ((p == 5) | (p == 6)) & ~held & (self.t > 10) & ~on_table
        limit = torch.where((p == 2) | (p == 4) | (p == 6), 2 * self.PHASE_LIMIT, self.PHASE_LIMIT)
        moved = adv | ((self.t > limit) & ~hold_on & (p < 9) & ((p > 0) | at_hover))
        idx = torch.arange(n, device=env.device)
        self.reached[idx, p] = True
        fid = (moved & ~adv).nonzero().squeeze(-1)
        self.forced[fid, p[fid]] = True
        if os.environ.get("BITIDY_ROWS"):
            for e in fid.tolist()[:20]:
                print(f"BITIDY_FORCED_O env={e} phase={int(p[e])} t={int(env.episode_length_buf[e])} err={float(err[e]):.4f} "
                      f"pad={[round(v, 4) for v in pad[e].tolist()]} t={[round(v, 4) for v in t[e].tolist()]} "
                      f"cube={[round(v, 4) for v in cube[e].tolist()]}", flush=True)
        self.p = torch.where(moved, (p + 1).clamp(max=9), p)
        self.p = torch.where(lost, 7, self.p)
        moved = moved | lost
        self.t = torch.where(moved | sub_next, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)


_TR = T  # the real task module (this module's T is swapped for the shim while the cube trip runs)


class _CubeAsKnob:
    """Stand-in for this module's ``T`` while :class:`CubeTripScript` runs LidTripScript's phases: the commanded cube
    plays the knob (pinch point, hold radius) and the cube's carry / table heights replace the lid's."""

    def __init__(self, carry_center_z: float, carry_up: float):
        self._carry_z = carry_center_z
        self.KNOB_OFFSET = 0.0
        self.LID_SIZE = (_TR.LID_SIZE[0], _TR.LID_SIZE[1], _TR.CUBE_SIZE)  # table_z = TABLE_TOP + CUBE_SIZE / 2: cube resting
        self.LID_REST_Z = carry_center_z - carry_up  # carry_z = LID_REST_Z + CARRY_UP
        self.KNOB_HOLD_RADIUS = 0.02

    def __getattr__(self, name):
        return getattr(_TR, name)

    @staticmethod
    def knob_pos(env):
        return _TR.cube_pos(env)

    @staticmethod
    def lid_quat(env):
        return _TR.target_view(env).scene["cube"].data.root_quat_w.torch

    @staticmethod
    def lid_tilt(env):
        return _TR.lo._tilt(_CubeAsKnob.lid_quat(env))


class CubeTripScript(LidTripScript):
    """LEFT arm, TAKE-OUT (v2 default ``BITIDY_LS_OUT_ARM=left``): after lid j is set down, one more pick-and-place
    trip with LidTripScript's phases (0 hover, 1 down, 2 close, 3 lift, 4 corner, 5 over the spot, 6 lower, 7 open,
    8 up, 9 idle), the cube as the "knob": up out of box j to ``CARRY_Z`` (cube bottom clear of the closed lids' knob
    tops), over to a free spot in the cube region within the left arm's reach (``spot``, (u, v) in box j's frame, set
    by the coordinator), down onto the table, let go, up. The left arm reaches the row with near-vertical jaws, so it
    can pinch a cube between the 4 cm walls; the right arm cannot (it reaches the row only ~21 cm out, tilted ~45 deg,
    and its gripper lands on the front wall)."""

    CARRY_Z = _f("BITIDY_LS_C_CARRY_Z", 0.118)  # cube centre while carried (knob tops 0.091 + cube half + margin)
    PINCH = _f("BITIDY_LS_C_PINCH", 0.002)  # pads this far above the cube centre
    C_TILT = _f("BITIDY_LS_C_TILT", 0.15)  # jaw tilt (near vertical: walls; 0.15 grasped 0.62-0.65 vs 0.55 at 0.3)
    C_HOVER = _f("BITIDY_LS_C_HOVER", 0.045)  # first hover above the pinch point (above the rim)
    OPEN_NARROW = _f("BITIDY_LS_C_OPEN", 0.25)  # open command in the box: < T.NARROW_ABOVE = the narrow level

    def __init__(self, env):
        super().__init__(env)
        self.keep[:] = False
        self.KEEP = "0"
        self.TILT = self.C_TILT
        self.KNOB_PINCH = self.PINCH
        self.HOVER = self.C_HOVER
        self.PRE = self.C_HOVER + 0.02
        self.LOST = 0.03
        self.UP = _f("BITIDY_LS_C_UP", 0.12)  # after letting go: well up off the cube (then the coordinator homes the arm)
        self.SEAT_SHIFT = _f("BITIDY_LS_C_SEAT_SHIFT", 0.0)  # pads centred on the cube (a knob-style shift lands a jaw on its edge)
        self.shim = _CubeAsKnob(self.CARRY_Z, self.CARRY_UP)
        self.HOVER_TOL = _f("BITIDY_LS_C_HOVER_TOL", self.HOVER_TOL)
        self.PINCH_TOL = _f("BITIDY_LS_C_PINCH_TOL", self.PINCH_TOL)
        self.PHASE_LIMIT = int(_f("BITIDY_LS_C_LIMIT", self.PHASE_LIMIT))
        self.SETTLE = int(_f("BITIDY_LS_C_SETTLE", self.SETTLE))
        self.SLOPE_TOL = _f("BITIDY_LS_C_SLOPE_TOL", self.SLOPE_TOL)
        self.HOVER_LIMIT = int(_f("BITIDY_LS_C_HOVER_LIMIT", 120))  # the hover rarely settles (level vs square-to-cube): go on sooner
        self.LEVEL_GAIN = _f("BITIDY_LS_C_LEVEL_GAIN", self.LEVEL_GAIN)
        self.PW = _f("BITIDY_LS_C_PW", self.PW)  # IK jaw-direction weight (approach / grasp)
        # Tilt the jaws in a vertical plane that contains a cube face normal (the one nearest the arm's reach
        # direction): the level closing axis is then square to the cube's faces (BITIDY_LS_C_SNAP=1). With the plain
        # reach-direction tilt, levelling the closing axis and squaring it to the faces fight over the one wrist roll.
        if os.environ.get("BITIDY_LS_C_SNAP", "0") == "1":
            ik, plain = self.ik, self.ik.tilted

            def tilted(target, forward=0.7):
                d = target[:, :2] - ik.base_xy
                az = torch.atan2(d[:, 1], d[:, 0])
                cyaw = _yaw(_TR.target_view(env).scene["cube"].data.root_quat_w.torch)
                q = math.pi / 2
                snap = cyaw + torch.round((az - cyaw) / q) * q
                out = torch.stack((torch.cos(snap), torch.sin(snap)), -1)
                return torch.nn.functional.normalize(torch.cat((forward * out, -torch.ones_like(out[:, :1])), -1), dim=-1)

            ik.tilted = tilted

    def act(self) -> torch.Tensor:
        g = globals()
        real = g["T"]
        g["T"] = self.shim
        try:
            a = super().act()
        finally:
            g["T"] = real
        # open = the gripper's NARROW level (lid_sort_env_cfg.LEFT_OPEN_MODE=two): the open jaws fit between the walls
        a[:, -1] = torch.where(a[:, -1] >= 0, self.OPEN_NARROW, a[:, -1])
        return a


class LidSortSetDownScript:
    """Facing layout, set-down mode (default, ``BITIDY_LS_LID_MODE=setdown``): the left arm takes lid j off and sets
    it down on the table next to the row (spot per row position of box j: ``BITIDY_LS_SPOTS``, (u, v) for the -y end;
    middle; +y end) and goes up out of the way; meanwhile the right arm picks cube i from its slot in front of the row
    and lifts it; once the lid is down it carries the cube in front of the row (``FrontPlaceScript``, path ``front``:
    never over a lid), drops it into box j and backs off; then the left arm puts the lid back."""

    LIFT_STEPS = int(_f("BITIDY_LS_LIFT_STEPS", 20))
    LIFT_FORCE = int(_f("BITIDY_LS_LIFT_FORCE", 150))
    RETRIES = int(_f("BITIDY_LS_RETRIES", 2))
    MISS_STEPS = int(_f("BITIDY_LS_MISS_STEPS", 40))
    GIVE_UP = int(_f("BITIDY_LP_GIVE_UP", 150))
    DIRECT = os.environ.get("BITIDY_LS_DIRECT", "1") == "1"  # -y end box (next to the slots): carry straight to it
    GRASP_AFTER = int(_f("BITIDY_LS_GRASP_AFTER", 0))
    HOLD_LOW = os.environ.get("BITIDY_LS_HOLD_LOW", "0") == "1"  # cube kept on the table in the closed jaws until the lid is down

    def __init__(self, env):
        self.env = env
        n, dev = env.num_envs, env.device
        self.left = LidTripScript(T.target_view(env))
        self.left.LOST = _f("BITIDY_LO_LOST", 0.03)
        self.left.SPEED = _f("BITIDY_LO_SPEED", 0.001)
        self.left.HOVER_TOL = _f("BITIDY_LS_T_HOVER_TOL", self.left.HOVER_TOL)
        self.spots = _spots(os.environ.get("BITIDY_LS_SPOTS", "-0.035,-0.105;-0.02,0.173;-0.02,0.095"), dev)
        self.right = FrontPlaceScript(env)
        self.right.PATH = os.environ.get("BITIDY_LS_PATH", "row")
        # Take-out commands (v2): a second right-arm script; per env the command mode picks whose action is used.
        self.right_out = TakeOutScript(env)
        self.o_retries = torch.zeros(n, dtype=torch.long, device=dev)
        # Take-out with the LEFT arm (default): lid off (take-out lid spots), cube trip, lid back.
        self.OUT_ARM = os.environ.get("BITIDY_LS_OUT_ARM", "left")
        self.cube_left = CubeTripScript(T.target_view(env))
        self.cube_left.SPEED = _f("BITIDY_LS_C_SPEED", 0.0015)
        self.out_spots = _spots(os.environ.get("BITIDY_LS_OUT_SPOTS", "-0.085,-0.03;-0.02,0.173;-0.02,0.095"), dev)
        self.c_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.c_t0 = torch.zeros(n, dtype=torch.long, device=dev)  # step the cube trip started (diagnostics)
        self.drop_fallback = 0
        self.DROP_R = _f("BITIDY_LS_C_DROP_R", 0.25)  # drop spot within this distance of the left base
        self.DROP_BOX = _f("BITIDY_LS_C_DROP_BOX", 0.012)  # drop spot this much beyond OUT_BOX_DIST from every box
        self.C_FACE_GAIN = _f("BITIDY_LS_C_FACE_GAIN", 0.5)
        self.L_HOME_TOL = _f("BITIDY_LS_C_HOME_TOL", 0.015)
        self.L_HOME_LIMIT = int(_f("BITIDY_LS_C_HOME_LIMIT", 300))
        self.T_OPEN_PUT = _f("BITIDY_LS_T_OPEN_PUT", 0.25)  # put-in lid trips: narrow (as v2 before the two-level gripper)
        self.T_OPEN_OUT = _f("BITIDY_LS_T_OPEN_OUT", 1.0)  # take-out lid trips: full opening
        self.homed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wait_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.HOME_WAIT = os.environ.get("BITIDY_LS_C_HOME_WAIT", "1") == "1"
        self.WAIT_LIMIT = int(_f("BITIDY_LS_C_WAIT_LIMIT", 200))
        self.HOME_END = os.environ.get("BITIDY_LS_HOME_END", "0") == "1"  # 1: measured worse (other lids knocked)
        self.C_HOME = os.environ.get("BITIDY_LS_C_HOME", "joints")  # after the cube trip: joints (start pose) / wait
        self.HOME_TOL = _f("BITIDY_LS_C_HOME_JTOL", 0.08)  # rad, every left arm joint
        self.HOME_RATE = _f("BITIDY_LS_C_HOME_RATE", 0.6)  # joint-target step per control step, x 0.033 rad
        self.c_done_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.C_FACE_TOL = _f("BITIDY_LS_C_FACE_TOL", 0.05)
        self.C_FACE = os.environ.get("BITIDY_LS_C_FACE", "any")  # any: nearest cube face; row: the face along the row normal
        self.l_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lift_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.done_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.park_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)

    def _drop_spot(self, ids: torch.Tensor, tries: int = 512) -> torch.Tensor:
        """(u, v) in box j's frame of a free table spot for the taken-out cube: in the cube region (as the env's
        success test), within ``DROP_R`` of the left base, clear of the boxes, the other cubes and lid j."""
        env, dev, m = self.env, self.env.device, len(ids)
        mg = T.OUT_REGION_MARGIN - 0.008
        cand = T._uniform((m, tries), (T.CUBE_RX[0] - mg, T.CUBE_RX[1]), (T.CUBE_RY[0], T.CUBE_RY[1] + mg), dev)
        ok = torch.linalg.vector_norm(cand - self.left.ik.base_xy[ids].unsqueeze(1), dim=-1) <= self.DROP_R
        boxes = T.box_centers(env)[ids]
        ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - boxes.unsqueeze(1), dim=-1) >= T.OUT_BOX_DIST + self.DROP_BOX).all(-1)
        cubes = T._objs_pos(env, "cube")[ids, :, :2]
        others = T._others(env, 0)[ids]
        d = torch.linalg.vector_norm(cand.unsqueeze(2) - cubes.unsqueeze(1), dim=-1)
        ok &= ((d >= 0.06) | ~others.unsqueeze(1)).all(-1)
        ok &= torch.linalg.vector_norm(cand - T.lid_pos(env)[ids, :2].unsqueeze(1), dim=-1) >= 0.075
        box = T.target_box(env)[ids]
        score = torch.linalg.vector_norm(cand - box.unsqueeze(1), dim=-1) + torch.rand((m, tries), device=dev) * 0.02
        best = torch.where(ok, score, torch.full_like(score, 1e9)).argmin(-1)
        xy = cand[torch.arange(m, device=dev), best]
        found = ok.any(-1)
        self.drop_fallback += int((~found).sum())
        xy = torch.where(found.unsqueeze(-1), xy, torch.tensor((0.15, -0.14), device=dev).expand(m, -1))
        yaw = T.target_box_yaw(env)[ids]
        dxy = xy - box
        return torch.stack((torch.cos(yaw) * dxy[:, 0] + torch.sin(yaw) * dxy[:, 1],
                            -torch.sin(yaw) * dxy[:, 0] + torch.cos(yaw) * dxy[:, 1]), -1)

    def reset(self, ids: torch.Tensor) -> None:
        if os.environ.get("BITIDY_ROWS"):
            R, L = self.right, self.left
            for e in ids.tolist()[:600]:
                print(f"BITIDY_END env={e} R.p={int(R.p[e])} leg={int(R.leg[e])} R.sub={int(R.sub[e])} L.trip={int(L.trip[e])} "
                      f"L.p={int(L.p[e])} retries={int(self.retries[e])} L.retries={int(L.retries[e])} "
                      f"flags={self.env._bi_final[e].int().tolist()}", flush=True)
        self.left.reset(ids)
        self.right.reset(ids)
        self.right_out.reset(ids)
        self.cube_left.reset(ids)
        self.c_started[ids] = False
        self.homed[ids] = False
        self.wait_steps[ids] = 0
        self.c_t0[ids] = 0
        self.c_done_steps[ids] = 0
        self.l_started[ids] = False
        for t in (self.lift_steps, self.done_steps, self.park_steps, self.retries, self.o_retries):
            t[ids] = 0

    def act(self) -> torch.Tensor:
        env, L, R = self.env, self.left, self.right
        for s_ in (L, R, self.right_out):  # displaced arm bases (v2): the IK's base point follows the real root
            s_.ik.base_xy = s_.ik.robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        ys = T.box_centers(env)[:, :, 1]
        rank = (ys < T.target_box(env)[:, 1:2] - 0.01).sum(-1).clamp(max=2)
        # Lid spots for the middle box lie beyond the +y end box: follow that box when the gaps vary (DISP=wide).
        yaw_j = T.target_box_yaw(env)
        ey_j = torch.stack((-torch.sin(yaw_j), torch.cos(yaw_j)), -1)
        v_all = ((T.box_centers(env) - T.target_box(env).unsqueeze(1)) * ey_j.unsqueeze(1)).sum(-1)  # (n, 3)
        shift = torch.where(rank == 1, v_all.max(-1).values - T.ROW_PITCH, torch.zeros_like(yaw_j))
        dv = torch.stack((torch.zeros_like(shift), shift), -1)
        L.spot = torch.where((L.trip == 0).unsqueeze(-1), self.spots[rank] + dv, L.spot)
        if L.KEEP == "end":
            L.keep = torch.where((L.trip == 0) & (L.p <= 2), rank == 0, L.keep)
        R.direct = (rank == 0) & self.DIRECT
        lid_down = L.down & T.lid_off_opening(env)
        RO, out = self.right_out, T.takeout(env)
        C = self.cube_left
        left_out = out & (self.OUT_ARM == "left")
        L.spot = torch.where((left_out & (L.trip == 0)).unsqueeze(-1), self.out_spots[rank] + dv, L.spot)
        L.keep = L.keep & ~left_out
        C.ik.base_xy = L.ik.base_xy
        RO.direct = (rank == 0) & self.DIRECT
        yaw = T.target_box_yaw(env)
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        RO.avoid = T.target_box(env) + torch.stack((c * L.spot[:, 0] - s_ * L.spot[:, 1], s_ * L.spot[:, 0] + c * L.spot[:, 1]), -1)
        RO.gate = ((L.trip == 0) & ~lid_down) | left_out  # left take-out: the right arm stays at its ready point
        # Left take-out: once lid j is down and the left arm is up, the cube trip (box j -> a free table spot).
        start_c = left_out & ~self.c_started & L.down & (L.p == 9)
        if bool(start_c.any()):
            ids = start_c.nonzero().squeeze(-1)
            C.reset(ids)
            C.spot[ids] = self._drop_spot(ids)
            self.c_started |= start_c
            self.c_t0 = torch.where(start_c, env.episode_length_buf, self.c_t0)

        # Right done: the cube rests in the box (take-out: on the table at its spot) and the right jaws are clear (or
        # the right gave up while parked).
        clear_put = (R.p >= 8) & T.cube_resting_inside(env)
        # (left take-out: the lid trip starts once the left arm is back at its wait point above lid j -- going for the
        # lid on the table straight from the cube spot it falls short and lands on box j's wall)
        self.c_done_steps = torch.where(self.c_started & (C.p >= 9), self.c_done_steps + 1, 0)
        if self.C_HOME == "joints":
            # left take-out: after the cube trip the arm goes back to its start pose in joint space before the lid trip
            # (from wherever the cube trip left it, the knob IK got stuck in folded / straight-up poses and missed)
            robot = L.ik.robot
            jids = [int(j) for j in L.ik.joints] + [int(L.ik.roll)]
            q0 = robot.data.default_joint_pos.torch[:, jids]
            homed = (robot.data.joint_pos.torch[:, jids] - q0).abs().max(-1).values < self.HOME_TOL
            self.homed |= self.c_started & (C.p >= 9) & (homed | (self.c_done_steps > self.L_HOME_LIMIT))
            # then (BITIDY_LS_C_HOME_WAIT=1) up over lid j's table spot, where the put-in lid trip 1 starts from
            l_wait = L.hold_pad + torch.tensor((0.0, 0.0, L.UP), device=env.device)
            at_wait = torch.linalg.vector_norm(base.pad_center(env, "robot_left") - l_wait, dim=-1) < self.L_HOME_TOL
            self.wait_steps = torch.where(self.homed, self.wait_steps + 1, 0)
            l_home = self.homed & (at_wait | (self.wait_steps > self.WAIT_LIMIT) | (not self.HOME_WAIT))
        else:
            l_wait = L.hold_pad + torch.tensor((0.0, 0.0, L.UP), device=env.device)
            l_home = torch.linalg.vector_norm(base.pad_center(env, "robot_left") - l_wait, dim=-1) < self.L_HOME_TOL
            l_home |= self.c_done_steps > self.L_HOME_LIMIT
        clear_out = torch.where(left_out, self.c_started & (C.p >= 9) & l_home, (RO.p >= 8) & T.cube_placed_out(env))
        clear = torch.where(out, clear_out, clear_put) & T.right_jaws_clear(env)
        self.done_steps = torch.where(clear, self.done_steps + 1, 0)
        parked = torch.where(out, RO.p == RIGHT_PARK, R.p == RIGHT_PARK) & T.right_jaws_clear(env)
        self.park_steps = torch.where(parked, self.park_steps + 1, 0)
        right_done = (self.done_steps > 5) | (self.park_steps > self.GIVE_UP)
        start1 = L.down & ((L.p == 9) | (L.p == 6)) & right_done
        L.roll = torch.where(start1 & left_out, 0.0, L.roll)
        L.bias = torch.where((start1 & left_out).unsqueeze(-1), 0.0, L.bias)
        L.start_trip(start1)

        # The arms start with their jaws side by side over the row: the left waits until the right has moved out.
        self.l_started |= torch.where(out, (RO.sub >= 1) | (RO.p > 0), (R.sub >= 1) | (R.p > 0)) | (env.episode_length_buf > 90)
        wait = ~self.l_started
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
        if bool(left_out.any()):
            # cube faces: turn the (near-vertical) jaws onto them while coming down (the knob script only levels)
            fx, mv = jaw_tips(env, "robot_left")
            ax = mv - fx
            cyaw = _yaw(T.target_view(env).scene["cube"].data.root_quat_w.torch)
            if self.C_FACE == "row":
                # jaws on the cube's front / back faces (closing axis = the cube face normal nearest the row normal):
                # never across the side walls, where the open jaws straddle the 2 mm gap to a neighbour's lid plate
                q = math.pi / 2
                tgt = cyaw + torch.round((T.target_box_yaw(env) - cyaw) / q) * q
                face = _fold(tgt - torch.atan2(ax[:, 1], ax[:, 0]), math.pi)
            else:
                face = _fold(cyaw - torch.atan2(ax[:, 1], ax[:, 0]), math.pi / 2)
            C.roll = torch.where(C.p <= 1, C.roll + self.C_FACE_GAIN * face, C.roll)
            C.settle = torch.where((C.p == 0) & (face.abs() > self.C_FACE_TOL), 0, C.settle)  # aligned before going down
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a_cube = C.act()
            c_active = left_out & self.c_started & (C.p < 9) & L.down
            a_left = torch.where(c_active.unsqueeze(-1), a_cube, a_left)
            if self.C_HOME == "joints":
                homing = left_out & self.c_started & (C.p >= 9) & L.down & (L.trip == 0) & ~self.homed
                # and once lid j is back and let go: home again (the knob script's "up" stalled ~2 cm from the knob)
                homing |= left_out & (L.trip == 1) & (L.p >= 8) & self.HOME_END
                robot = L.ik.robot
                jids = [int(j) for j in L.ik.joints] + [int(L.ik.roll)]
                q0 = robot.data.default_joint_pos.torch[:, jids]
                d_home = ((q0 - robot.data.joint_pos_target.torch[:, jids]) / 0.033).clamp(-self.HOME_RATE, self.HOME_RATE)
                a_home = torch.cat((d_home, torch.ones_like(q0[:, :1])), -1)
                a_left = torch.where(homing.unsqueeze(-1), a_home, a_left)
                L.bias = torch.where(homing.unsqueeze(-1), 0.0, L.bias)

        # left gripper open level on the lid trips (lid_sort_env_cfg.LEFT_OPEN_MODE=two: >= 0.5 full, else narrow)
        t_open = torch.where(left_out, self.T_OPEN_OUT, self.T_OPEN_PUT)
        lid_trip = ~(left_out & self.c_started & (C.p < 9) & L.down)
        a_left[:, -1] = torch.where(lid_trip & (a_left[:, -1] >= 0), t_open, a_left[:, -1])

        before = R.p.clone()
        a_right = R.act()
        # Hover over the cube until the lid is on its way to the table (a long lifted hold lets the cube slip).
        early = (before == 0) & (R.p == 1) & (L.trip == 0) & (L.p < self.GRASP_AFTER)
        R.p = torch.where(early, 0, R.p)
        # Jaws closed on the cube: keep it on the table (no long lifted hold: it slips) until the lid is down.
        hold_low = self.HOLD_LOW & (before == 3) & (R.p == 4) & ~lid_down
        R.p = torch.where(hold_low, 3, R.p)
        R.t = torch.where(hold_low, 14, R.t)
        # Lifted cube: hold it up in front of the row until the lid is down on the table.
        self.lift_steps = torch.where(before == RIGHT_LIFT, self.lift_steps + 1, 0)
        lifted = (before == RIGHT_LIFT) & (R.p == RIGHT_CARRY)
        held = T.cube_held(env)
        go = ((before == RIGHT_LIFT) & lid_down & held & (self.lift_steps >= self.LIFT_STEPS)
              & (lifted | (self.lift_steps > self.LIFT_FORCE)))
        stay = (before == RIGHT_LIFT) & ~go
        R.p = torch.where(stay, RIGHT_LIFT, torch.where(go, RIGHT_CARRY, R.p))
        R.t = torch.where(go | (stay & lifted), 0, R.t)
        R.leg = torch.where(go, 0, R.leg)

        # Retry the pick: jaws closed on nothing, or the cube back on the table outside the box after the drop.
        cube = T.cube_pos(env)
        missed = (R.p == RIGHT_LIFT) & (self.lift_steps > self.MISS_STEPS) & ~held
        on_table = (cube[:, 2] < base.CUBE_REST_Z + 0.005) & ~T.cube_over_opening(env)
        dropped = (R.p >= 7) & on_table & ~held
        retry = (missed | dropped) & (self.retries < self.RETRIES) & (L.trip == 0)
        R.p = torch.where(retry, 0, R.p)
        R.t = torch.where(retry, 0, R.t)
        R.sub = torch.where(retry, 1, R.sub)
        self.lift_steps = torch.where(retry, 0, self.lift_steps)
        self.retries += retry.long()

        # Take-out envs: the take-out script's action; retry the grasp if the jaws closed on nothing or the cube fell
        # back into the box (while the lid is still off).
        a_out = RO.act()
        held_o = T.cube_held(env)
        o_missed = (RO.p == RIGHT_LIFT) & (RO.t > self.MISS_STEPS) & ~held_o
        o_fell = (RO.p >= 7) & T.cube_inside(env) & ~held_o & (cube[:, 2] < base.CUBE_REST_Z + 0.005)
        o_retry = out & (o_missed | o_fell) & (self.o_retries < self.RETRIES) & (L.trip == 0)
        RO.p = torch.where(o_retry, 0, RO.p)
        RO.t = torch.where(o_retry, 0, RO.t)
        RO.sub = torch.where(o_retry, 2, RO.sub)
        self.o_retries += o_retry.long()
        a_right = torch.where(out.unsqueeze(-1), a_out, a_right)

        if os.environ.get("BITIDY_ROWS"):
            last = (env.episode_length_buf == int(env.max_episode_length) - 3).nonzero().squeeze(-1)
            r3 = lambda v, e: [round(x, 3) for x in v[e].tolist()]
            padR, padL, lid = base.pad_center(env, "robot_right"), base.pad_center(env, "robot_left"), T.lid_pos(env)
            for e in last.tolist():
                print(f"BITIDY_SDEND env={e} rank={int(rank[e])} R.p={int(R.p[e])} leg={int(R.leg[e])} R.sub={int(R.sub[e])} "
                      f"L.trip={int(L.trip[e])} L.p={int(L.p[e])} retries={int(self.retries[e])} L.retries={int(L.retries[e])} "
                      f"held={bool(held[e])} padR={r3(padR, e)} cube={r3(cube, e)} box={r3(T.target_box(env), e)} padL={r3(padL, e)} "
                      f"lid={r3(lid, e)} flags={T.progress(env).flags[e].int().tolist()} c_t0={int(self.c_t0[e])} "
                      f"C.retries={int(C.retries[e])} C.p={int(C.p[e])} C.dur={C.dur[e].tolist()} L.dur={L.dur[e].tolist()}", flush=True)
        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            fx_, mv_ = jaw_tips(env, "robot_left")
            ax_ = torch.nn.functional.normalize(mv_ - fx_, dim=-1)
            cy_ = _yaw(T.target_view(env).scene["cube"].data.root_quat_w.torch)
            fc_ = _fold(cy_ - torch.atan2(ax_[:, 1], ax_[:, 0]), math.pi / 2)
            print(f"[sd] t={int(env.episode_length_buf[k])} slope={float(ax_[k, 2]):.3f} face={float(fc_[k]):.3f} C.roll={float(C.roll[k]):.2f} C.settle={int(C.settle[k])} "
                  f"cmd={T.command(env)[k].tolist()} L.trip={int(L.trip[k])} L.p={int(L.p[k])} "
                  f"C.p={int(C.p[k])} c_started={bool(self.c_started[k])} C.spot={[round(v, 3) for v in C.spot[k].tolist()]} "
                  f"cube={[round(v, 3) for v in T.cube_pos(env)[k].tolist()]} padL={[round(v, 3) for v in base.pad_center(env, 'robot_left')[k].tolist()]} "
                  f"baseL={[round(v, 4) for v in L.ik.base_xy[k].tolist()]} baseR={[round(v, 4) for v in R.ik.base_xy[k].tolist()]} "
                  f"pads={[round(v, 3) for v in base.pad_center(env, 'robot_right')[k].tolist()]} "
                  f"R.p={int(R.p[k])} leg={int(R.leg[k])} lid_down={bool(lid_down[k])} rdone={bool(right_done[k])} "
                  f"flags={T.progress(env).flags[k].int().tolist()} succ={bool(T.progress(env).success[k])}", flush=True)
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
