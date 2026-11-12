"""Scripted oracle for BiTidy-LidPlace-SO101 (``BITIDY_SCRIPT=oracle_lid_place:LidPlaceScript``).

Runs the two verified single-arm oracles side by side, unchanged, and only coordinates them:

* LEFT arm = ``LidOpenScript`` (lid off by its knob, aside toward +y, hold level, back on, let go, back off).
* RIGHT arm = ``BoxPlaceScript`` (pick the cube, carry it over the box, square it to the walls, drop it in, park).

Both scripts work on this env as is: the scene names (``lid``, ``cube``, ``robot_left``/``robot_right``) and the
box/lid/cube geometry are the same as in their own tasks, which is all they read. The only parameter changed is
the left arm's aside distance (``ASIDE``, 7.5 -> 10 cm toward +y, see below).

Coordination:

1. The right arm holds its start pose (its script not started) until the left arm starts down onto the knob (left
   phase ``RIGHT_START``), then runs its script from the start, exactly as in BoxPlace, so it has the cube lifted
   about when the lid is aside. It may not start the carry over the box (its phase 4 -> 5) until the left arm is in
   its hold phase (5) with the lid measured aside (held, level, footprint clear of the opening) for ``LID_SETTLE``
   steps; past that it waits lifted at most ``LIFT_STEPS``. If the left arm leaves its hold (lost the lid, starts
   over) before the right arm is lowering into the box, the right arm goes back to waiting lifted.
2. The left arm stays in its hold phase (its 45-step hold and its phase timeout are suspended) until the cube rests
   inside the box and the right arm is parked with its jaws clear of the box; then it puts the lid back.
   If the right arm has given up (parked for ``GIVE_UP`` steps without the cube inside), the lid goes back anyway, so
   the scene ends tidy (the episode fails on the cube).

``BITIDY_DEBUG=1`` prints a coordination trace of one environment (``BITIDY_DEBUG_ENV``) every ``BITIDY_DEBUG_EVERY``
steps (plus both scripts' own traces); ``BITIDY_ROWS=1`` prints both scripts' phase statistics and the mean waits.
"""

from __future__ import annotations

import os

import torch

from . import env_cfg as base
from . import lid_place_env_cfg as T
from .oracle_box_place import BoxPlaceScript
from .oracle_lid_open import LidOpenScript

LEFT_HOLD = 5
RIGHT_LIFT, RIGHT_CARRY, RIGHT_PARK = 4, 5, 9


class LidPlaceScript:
    LID_SETTLE = int(os.environ.get("BITIDY_LP_LID_SETTLE", 10))
    GIVE_UP = int(os.environ.get("BITIDY_LP_GIVE_UP", 90))
    LIFT_STEPS = int(os.environ.get("BITIDY_LP_LIFT_STEPS", 30))
    RIGHT_START = int(os.environ.get("BITIDY_LP_RIGHT_START", 1))
    # Lid held 10 cm (not LidOpen's 7.5 cm) to +y of the box: at 7.5 cm the hanging lid's corner reaches to ~3.6 cm
    # from the box centre, close to the right jaws lowering the cube (cube-in rate +3% at 10 cm in A/B runs).
    ASIDE = float(os.environ.get("BITIDY_LP_ASIDE", 0.10))

    def __init__(self, env):
        self.env = env
        self.left = LidOpenScript(env)
        if "BITIDY_LO_ASIDE" not in os.environ:
            self.left.ASIDE = self.ASIDE
        self.right = BoxPlaceScript(env)
        n, dev = env.num_envs, env.device
        self.r_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.aside_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.lift_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.park_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.wait_r = torch.zeros(n, dtype=torch.long, device=dev)
        self.wait_l = torch.zeros(n, dtype=torch.long, device=dev)
        self.tot = {"n": 0, "wait_r": 0.0, "wait_l": 0.0}

    def reset(self, ids: torch.Tensor) -> None:
        if os.environ.get("BITIDY_ROWS") and len(ids):
            self.tot["n"] += len(ids)
            self.tot["wait_r"] += float(self.wait_r[ids].sum())
            self.tot["wait_l"] += float(self.wait_l[ids].sum())
            n = max(self.tot["n"], 1)
            print(f"BITIDY_COORD n={self.tot['n']} right_wait_s={self.tot['wait_r'] / n / 30:.2f} "
                  f"left_wait_s={self.tot['wait_l'] / n / 30:.2f}", flush=True)
        self.left.reset(ids)
        self.right.reset(ids)
        self.r_started[ids] = False
        for t in (self.aside_steps, self.lift_steps, self.park_steps, self.wait_r, self.wait_l):
            t[ids] = 0

    def act(self) -> torch.Tensor:
        env, L, R = self.env, self.left, self.right

        # Gate 1 input: lid held aside, level, clear of the opening, in the left arm's hold phase.
        lid_ready = (L.p == LEFT_HOLD) & T.lid_aside(env) & T.lid_off_opening(env)
        self.aside_steps = torch.where(lid_ready, self.aside_steps + 1, 0)
        lid_ready = self.aside_steps >= self.LID_SETTLE

        # Gate 2 input: cube resting inside, right arm parked with its jaws clear of the box.
        parked = (R.p == RIGHT_PARK) & T.right_jaws_clear(env)
        self.park_steps = torch.where(parked, self.park_steps + 1, 0)
        right_done = parked & (T.cube_resting_inside(env) | (self.park_steps > self.GIVE_UP))

        # Left: suspend the hold's clock (and so its timeout) until the right arm is done.
        hold_on = (L.p == LEFT_HOLD) & ~right_done
        L.t = torch.where(hold_on, L.t.clamp(max=L.HOLD - 1), L.t)
        self.wait_l += hold_on.long()
        a_left = L.act()

        # Right arm start. Its script is not run until the left arm starts down onto the knob; then it runs from the
        # start pose exactly as in BoxPlace. Every place it could wait mid-pick cost success in tests (512 eps each):
        # lifted at the grasp point (the edge of its reach: the arm sags, the anti-sag integral winds up and unwinds
        # when the carry starts, the jaws dip and drag the cube over the table: 72%), jaws open around the cube or
        # 3 cm above it (the rolling open jaws nudge the cube: 81-86%), at the 7 cm hover (after a long hover the
        # descent overshoots; BoxPlace alone with that wait drops from 95% to 89.6%).
        self.r_started |= L.p >= self.RIGHT_START
        wait = ~self.r_started
        before = R.p.clone()
        a_right = R.act()
        # Not started: undo the script's step (it starts from phase 0 when released) and hold the start pose.
        R.p = torch.where(wait, 0, R.p)
        R.t = torch.where(wait, 0, R.t)
        R.roll = torch.where(wait, 0.0, R.roll)
        R.bias = torch.where(wait.unsqueeze(-1), 0.0, R.bias)
        a_right = torch.where(wait.unsqueeze(-1), torch.zeros_like(a_right), a_right)
        a_right[:, -1] = torch.where(wait, 1.0, a_right[:, -1])

        # Carry over the box only once the lid is aside; back to waiting lifted if the left arm drops out of its
        # hold (lost the lid) before the lowering starts.
        self.lift_steps = torch.where(before == RIGHT_LIFT, self.lift_steps + 1, 0)
        blocked = (before == RIGHT_LIFT) & (R.p == RIGHT_CARRY) & ~lid_ready
        abort = (R.p == RIGHT_CARRY) & (L.p != LEFT_HOLD)
        R.p = torch.where(blocked | abort, RIGHT_LIFT, R.p)
        # The lift waypoint is at the edge of reach and may never count as reached: once the lid is ready, go after
        # ``LIFT_STEPS`` (BoxPlace's own lift takes ~20 steps).
        go = (R.p == RIGHT_LIFT) & lid_ready & (self.lift_steps >= self.LIFT_STEPS)
        R.p = torch.where(go, RIGHT_CARRY, R.p)
        R.t = torch.where(blocked | abort | go, 0, R.t)
        self.wait_r += (wait | blocked | abort).long()

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 15)) == 0:
            prog = T.progress(env)
            r3 = lambda v: [round(x, 3) for x in v[k].tolist()]
            print(f"[lp] t={int(env.episode_length_buf[k])} L.p={int(L.p[k])} R.p={int(R.p[k])} lid_ready={bool(lid_ready[k])} "
                  f"right_done={bool(right_done[k])} lid={r3(T.lid_pos(env))} cube={r3(T.cube_pos(env))} "
                  f"padL={r3(base.pad_center(env, 'robot_left'))} padR={r3(base.pad_center(env, 'robot_right'))} "
                  f"flags={prog.flags[k].int().tolist()} succ={bool(prog.success[k])}", flush=True)
        return torch.cat((a_left, a_right), -1)
