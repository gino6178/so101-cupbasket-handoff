"""Scripted oracle for BiTidy-BoxPlace-SO101: the right arm picks the cube and drops it into the open box.

Select with ``BITIDY_SCRIPT=oracle_box_place:BoxPlaceScript``. Phases (right arm only, the left arm holds still):

0 hover high over the cube, 1 above the (locked) grasp point, 2 down, 3 close, 4 lift, 5 carry high over the box,
6 lower until the cube bottom is just above the rim, centred on the measured cube (not the jaws), 7 open (the cube
drops in), 8 straight up, 9 park beside the arm's base (jaws well clear of the box, so a lid could go back on).

Jaw roll follows the cube faces by feedback until the jaws close; over the box it turns the gripped cube square to the
walls (feedback on the cube's measured yaw: the SO-101 can only point its jaws in the vertical plane through its base,
so without this the cube would arrive ~38 deg off the walls). The jaws never open unless the cube is over the opening.
The jaws are tilted more over the box (``TILT_BOX``) than at the grasp: with the grasp tilt the arm cannot lift the
cube high enough over the walls at the box.

``BITIDY_ROWS=1`` prints per-phase reached/forced rates (first episode per environment) and failure rows,
``BITIDY_DEBUG=1`` traces one environment (``BITIDY_DEBUG_ENV``).
"""

from __future__ import annotations

import os

import torch

from isaaclab.utils.math import quat_apply

from . import box_place_env_cfg as B
from . import env_cfg as base
from .scripted import ArmIK, jaw_tips

LAST = 9


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class BoxPlaceScript:
    HOVER = _f("BITIDY_HOVER", 0.07)  # first approach comes down from well above the cube (a low path knocks it away)
    PINCH = _f("BITIDY_PINCH", 0.008)  # pads this far above the cube centre (lower pinches squirt the cube away more)
    LIFT_ABOVE = _f("BITIDY_LIFT_ABOVE", 0.04)  # straight lift at the grasp point (full height is out of reach there)
    CARRY_ABOVE = _f("BITIDY_CARRY_ABOVE", 0.05)  # pad height above the rim while carrying (the arm sags short of it)
    DROP_CLEAR = _f("BITIDY_DROP_CLEAR", 0.004)  # cube bottom above the rim at release (measured cube, not the pad)
    UP_ABOVE = _f("BITIDY_UP_ABOVE", 0.04)
    BACK = _f("BITIDY_BACK", 0.04)
    TILT = _f("BITIDY_TILT", 0.7)
    TILT_BOX = _f("BITIDY_TILT_BOX", 1.2)
    ROLL_GAIN = _f("BITIDY_ROLL_GAIN", 0.3)
    BOX_ROLL_GAIN = _f("BITIDY_BOX_ROLL_GAIN", 0.2)
    GAIN = _f("BITIDY_IK_GAIN", 1.0)
    NEAR = _f("BITIDY_IK_NEAR", 0.08)
    BIAS_CLAMP = _f("BITIDY_BIAS_CLAMP", 0.03)
    HOVER_TOL = _f("BITIDY_HOVER_TOL", 0.012)
    GRASP_TOL = _f("BITIDY_GRASP_TOL", 0.004)
    CENTRE_TOL = _f("BITIDY_CENTRE_TOL", 0.003)  # footprint margin inside the opening
    CENTRE_ABS = _f("BITIDY_CENTRE_ABS", 0.0015)  # and the cube centre this close to the box centre
    PARK = tuple(map(float, os.environ.get("BITIDY_PARK", "0.14,-0.12,0.11").split(",")))  # well clear of the box
    PHASE_LIMIT = int(_f("BITIDY_PHASE_LIMIT", 120))

    def __init__(self, env):
        self.env = env
        self.ik = ArmIK(env, B.ARM)
        n, dev = env.num_envs, env.device
        self.p = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros_like(self.p)
        self.roll = torch.zeros(n, device=dev)
        self.bias = torch.zeros((n, 3), device=dev)
        self.grasp = torch.zeros((n, 3), device=dev)
        self.release_xy = torch.zeros((n, 2), device=dev)
        self.offset = torch.zeros((n, 3), device=dev)
        # Diagnostics (first episode per environment).
        self.reached = torch.zeros((n, LAST + 1), dtype=torch.bool, device=dev)
        self.forced = torch.zeros_like(self.reached)
        self.first = torch.ones(n, dtype=torch.bool, device=dev)
        self.tot_reached = torch.zeros(LAST + 1, device=dev)
        self.tot_forced = torch.zeros(LAST + 1, device=dev)
        self.episodes = 0
        self.fails = 0
        self.flat = 0
        self.last_cube = torch.zeros((n, 3), device=dev)
        self.start_cube = torch.zeros((n, 4), device=dev)

    @staticmethod
    def _yaw(quat: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @staticmethod
    def _fold(a: torch.Tensor) -> torch.Tensor:
        """Angle folded to (-45, 45] deg (cube 90-degree symmetry)."""
        return torch.remainder(a + torch.pi / 4, torch.pi / 2) - torch.pi / 4

    # ---------------------------------------------------------------- diagnostics

    def _account(self, ids: torch.Tensor, flags: torch.Tensor, why: str) -> None:
        """Fold the first episodes of ``ids`` into the per-phase totals and print a row per failure."""
        first = self.first[ids]
        ids, flags = ids[first], flags[first]
        if not len(ids):
            return
        self.first[ids] = False
        self.tot_reached += self.reached[ids].float().sum(0)
        self.tot_forced += self.forced[ids].float().sum(0)
        self.episodes += len(ids)
        # Successful episodes whose cube lies flat on the floor (the rest lean against an inner wall, below the rim).
        self.flat += int((flags[:, -1] & (self.last_cube[ids, 2] < base.CUBE_REST_Z + 0.003)).sum())
        for i in (~flags[:, -1]).nonzero().squeeze(-1).tolist():
            if self.fails < 60:
                e = int(ids[i])
                c, s = self.last_cube[e].tolist(), self.start_cube[e].tolist()
                print(f"BITIDY_FAIL {why} env={e} phase={int(self.p[e])} flags={flags[i].int().tolist()} "
                      f"forced={self.forced[e].int().tolist()} start=({s[0]:.3f},{s[1]:.3f},yaw {s[3]:.2f}) "
                      f"cube_end=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})", flush=True)
            self.fails += 1
        r = (self.tot_reached / self.episodes).tolist()
        f = (self.tot_forced / self.episodes).tolist()
        print(f"BITIDY_PHASES n={self.episodes} reached={[round(v, 3) for v in r]} forced={[round(v, 3) for v in f]} "
              f"flat_successes={self.flat}", flush=True)

    def reset(self, ids: torch.Tensor) -> None:
        if os.environ.get("BITIDY_ROWS"):
            self._account(ids, self.env._bi_final[ids], "ended")
        self.p[ids] = 0
        self.t[ids] = 0
        self.roll[ids] = 0.0
        self.bias[ids] = 0.0
        self.reached[ids] = False
        self.forced[ids] = False

    # ---------------------------------------------------------------- policy

    def act(self) -> torch.Tensor:
        env, p = self.env, self.p
        n = env.num_envs
        cube = base.cube_pos(env)
        pad = base.pad_center(env, B.ARM)
        up = cube.new_tensor((0.0, 0.0, 1.0))
        quat = env.scene["cube"].data.root_quat_w.torch
        cube_yaw = self._yaw(quat)
        fresh = env.episode_length_buf <= 1
        self.start_cube = torch.where(fresh.unsqueeze(-1), torch.cat((cube, cube_yaw.unsqueeze(-1)), -1), self.start_cube)
        self.last_cube = cube.clone()
        if os.environ.get("BITIDY_ROWS"):
            # Episodes about to time out (the episode counter exits before their reset reaches this script).
            ending = (env.episode_length_buf == env.max_episode_length - 2).nonzero().squeeze(-1)
            if len(ending):
                prog = B.progress(env)
                self._account(ending, torch.cat((prog.flags[ending], prog.success[ending].unsqueeze(-1)), -1), "timeout")

        def col(v):
            return v.unsqueeze(-1)

        def at(xy, z):
            return torch.cat((xy, torch.full_like(xy[:, :1], z)), -1)

        box = B.box_center(env)
        # Over the box, steer the measured cube (not the jaws) onto the opening: it sits off-centre in the jaws.
        # (frozen once the cube is out of the jaws, so a cube that slipped into the box does not drag the target down).
        held = torch.linalg.vector_norm(pad - cube, dim=-1) < 0.03
        self.offset = torch.where(col(held | (p <= 4)), (pad - cube).clamp(-0.02, 0.02), self.offset)
        offset = self.offset
        box_pad = box + offset[:, :2]
        drop_z = B.RIM_Z + B.CUBE_SIZE / 2 + self.DROP_CLEAR + offset[:, 2:]
        # Lock the grasp point once the descent starts, so a nudged cube is not chased across the table.
        self.grasp = torch.where(col(p <= 0), cube, self.grasp)
        g = self.grasp
        t = cube + up * self.HOVER
        t = torch.where(col(p >= 1), g + up * 0.03, t)
        t = torch.where(col(p >= 2), g + up * self.PINCH, t)
        t = torch.where(col(p >= 4), g + up * self.LIFT_ABOVE, t)
        t = torch.where(col(p >= 5), at(box_pad, B.RIM_Z + self.CARRY_ABOVE), t)
        t = torch.where(col(p >= 6), torch.cat((box_pad, drop_z), -1), t)
        self.release_xy = torch.where(col(p <= 7), pad[:, :2], self.release_xy)
        # Back off up and toward the arm's base (straight up is out of reach there; the jaw tips sagged into the box).
        back = torch.nn.functional.normalize(self.ik.base_xy - box, dim=-1)
        t = torch.where(col(p >= 8), at(self.release_xy + self.BACK * back, B.RIM_Z + self.UP_ABOVE), t)
        t = torch.where(col(p >= 9), cube.new_tensor(self.PARK).expand(n, -1), t)
        grip = torch.where((p >= 3) & (p < 7), -1.0, 1.0)

        # Roll: jaws onto the cube faces until they close; carrying and lowering, turn the cube square to the walls.
        fixed, moving = jaw_tips(env, B.ARM)
        axis = moving - fixed
        face_err = self._fold(cube_yaw - torch.atan2(axis[:, 1], axis[:, 0]))
        box_err = self._fold(-cube_yaw)
        d_roll = torch.where(p <= 2, self.ROLL_GAIN * face_err, torch.zeros_like(face_err))
        d_roll = torch.where((p == 5) | (p == 6), self.BOX_ROLL_GAIN * box_err, d_roll)
        self.roll = (self.roll + d_roll).clamp(-1.6, 1.6)
        # Backing off, unwind the wrist back to its natural roll.
        self.roll = torch.where(p >= 8, 0.95 * self.roll, self.roll)

        # Integral term against gravity sag, active near the waypoint only (so long reaches cannot wind it up).
        near = col(torch.linalg.vector_norm(t - pad, dim=-1) < self.NEAR)
        self.bias = torch.where(near, self.bias + 0.05 * (t - pad), 0.9 * self.bias).clamp(-self.BIAS_CLAMP, self.BIAS_CLAMP)
        # Coming down onto the cube it only ever lifts: a wound-up downward bias dragged the jaw tips across the cube.
        self.bias[:, 2] = torch.where(p <= 2, self.bias[:, 2].clamp(min=0.0), self.bias[:, 2])
        tilt = torch.where(col((p >= 5) & (p <= 8)), self.ik.tilted(t, self.TILT_BOX), self.ik.tilted(t, self.TILT))
        act, _ = self.ik(t + self.bias, tilt, self.roll, gain=self.GAIN)
        err = torch.linalg.vector_norm(t - pad, dim=-1)

        # The cube's footprint (it grows with yaw) lies over the opening: with margin to release, at all to time out.
        half = 0.5 * B.CUBE_SIZE * (torch.cos(box_err).abs() + torch.sin(box_err).abs())
        cube_off = (cube[:, :2] - box).abs().max(-1).values
        centred = (cube_off < B.BOX_INNER / 2 - half - self.CENTRE_TOL) & (cube_off < self.CENTRE_ABS)
        over_ok = cube_off < B.BOX_INNER / 2 - half

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 8)) == 0:
            tiltdeg = torch.rad2deg(torch.acos(torch.stack([quat_apply(quat, quat.new_tensor(a).expand(n, -1))[:, 2].abs()
                                                            for a in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))], -1).max(-1).values.clamp(max=1.0)))
            print(f"[box] env={k} t={int(env.episode_length_buf[k])} p={int(p[k])} err={float(err[k]):.4f} "
                  f"pad={[round(v, 3) for v in pad[k].tolist()]} cube={[round(v, 3) for v in cube[k].tolist()]} "
                  f"yaw={float(cube_yaw[k]):.2f} tilt={float(tiltdeg[k]):.0f}deg face={float(face_err[k]):.2f} "
                  f"off={float(cube_off[k]):.4f} held={bool(base.held_by(env, B.ARM)[k])} "
                  f"open={base.gripper_openness(env, B.ARM)[k].item():.2f} flags={B.progress(env).flags[k].int().tolist()}", flush=True)

        self.t += 1
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (err < self.HOVER_TOL)
        adv |= (p == 1) & (err < 0.004)
        adv |= (p == 2) & (err < self.GRASP_TOL)
        adv |= (p == 3) & (self.t > 15)
        adv |= (p == 4) & (err < 0.01)
        adv |= (p == 5) & (err < 0.008)
        adv |= (p == 6) & (err < 0.005) & centred & (box_err.abs() < 0.08)
        adv |= (p == 7) & (self.t > 15)
        adv |= (p == 8) & (err < 0.01)
        # Never let go of the cube anywhere but over the opening: the lowering phase times out only if it is over it.
        hold_on = (p == 6) & ~over_ok
        lost = ((p == 5) | (p == 6)) & ~held & (self.t > 10)
        moved = adv | ((self.t > self.PHASE_LIMIT) & ~hold_on & (p < LAST))
        idx = torch.arange(n, device=env.device)
        self.reached[idx, p] = True
        fid = (moved & ~adv).nonzero().squeeze(-1)
        self.forced[fid, p[fid]] = True
        self.p = torch.where(moved, (p + 1).clamp(max=LAST), p)
        self.p = torch.where(lost, 7, self.p)
        moved = moved | lost
        self.t = torch.where(moved, 0, self.t)
        return torch.cat((act, grip.unsqueeze(-1)), -1)
