"""Scripted oracle V2 for BiTidy-CupBasket-SO101 (``BITIDY_SCRIPT=oracle_cup_basket_v2:CupBasketScriptV2``).

V2 = v1 (``oracle_cup_basket.CupBasketScript``) plus, with these defaults (each can be switched off by env var):
  * active re-uprighting while carrying (``BITIDY_CB2_NULL_K=0.3``, ``BITIDY_CB2_NULL_ROLL=1``): LIFT..UP use a
    position-only IK on pan/lift/elbow/flex/roll and spend the 2-D null space on turning the gripper with angular
    velocity NULL_K * (vessel_up x z) (measured vessel tilt) -> the vessel hangs much closer to upright when lowered;
  * RISE escape for far vessels (``BITIDY_CB2_RISE_XY=0.02``): the transit height is out of reach far out, the carrot
    never arrived and the arm hung in RISE forever; now it goes on once horizontally within 2 cm and stalled;
  * LIFT done on height alone (``BITIDY_CB2_LIFT_Z=0.015``);
  * cups released lower (``BITIDY_CB_REL_CUP=0.008``, v1 0.015).
Off switches: BITIDY_CB2_NULL_K=0 BITIDY_CB2_RISE_XY=0 BITIDY_CB2_LIFT_Z=0 BITIDY_CB_REL_CUP=0.015 -> v1 behaviour.
Experimental (default off): BITIDY_CB2_UP_K, BITIDY_CB2_PITCH_K, BITIDY_CB2_PRESS, BITIDY_CB2_ROLL_CUP/_BOWL,
BITIDY_CB2_FAR_R; diagnostics BITIDY_CB_TDIAG=1 (hang tilt), BITIDY_CB2_RSTATS=1 (by start range).

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
from .scripted import ARM_SCALE, ArmIK, _skew

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
    REL_ABOVE = (_f("BITIDY_CB_REL_CUP", 0.008), _f("BITIDY_CB_REL_BOWL", 0.02))
    """Release height above the nest (the outer jaw reaches PINCH_DEPTH below the held rim: bowls nest only ~6 mm apart,
    so a bowl is let go ~1.6 cm up and drops into the taper)."""
    REL_EMPTY = _f("BITIDY_CB_REL_EMPTY", 0.01)
    UP_DZ = _f("BITIDY_CB_UP", 0.05)
    TILT = _f("BITIDY_CB_TILT", 0.5)
    TILT_CARRY = _f("BITIDY_CB_TILT_CARRY", 0.5)
    PW = _f("BITIDY_CB_PW", 0.01)
    PW_CARRY = _f("BITIDY_CB_PW_CARRY", 0.005)
    ROLL = _f("BITIDY_CB_ROLL", 0.0)
    ROLL_CUP = _f("BITIDY_CB2_ROLL_CUP", ROLL)  # V2: per-type wrist roll offset (0.8 lines the jaws up radially)
    ROLL_BOWL = _f("BITIDY_CB2_ROLL_BOWL", ROLL)
    FAR_R = _f("BITIDY_CB2_FAR_R", 0.0)  # V2: start range (m) beyond which FAR_ROLL is used
    FAR_ROLL = _f("BITIDY_CB2_FAR_ROLL", 1.57)
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
    # V2: active re-uprighting while the vessel hangs in the jaws (LIFT..LOWER): the IK drives the gripper's angular
    # velocity (horizontal part) toward UP_K * (vessel_up x z), so the vessel is turned back upright; OPEN / UP hold the
    # orientation. 0 = off (v1 behaviour).
    UP_K = _f("BITIDY_CB2_UP_K", 0.0)
    UP_W = _f("BITIDY_CB2_UP_W", 0.05)
    UP_MAX = _f("BITIDY_CB2_UP_MAX", 0.03)  # max joint step per control step [rad]
    UP_FROM = int(_f("BITIDY_CB2_UP_FROM", 4))  # first servo phase (4 = LIFT, 3 = CLOSE)
    # V2: RISE escape (far vessels: the transit height is out of reach, the carrot never arrives and the phase clock
    # never ran): once the pads are within RISE_XY (horizontally) and have stalled, go on to HOVER.
    RISE_XY = _f("BITIDY_CB2_RISE_XY", 0.02)
    # V2: pitch re-uprighting through the standard 4-joint IK: the jaw pointing direction is turned about the
    # horizontal axis z x out (out = base -> vessel) by an integrated angle phi += PITCH_K * (vessel tilt about that
    # axis) during PITCH_FROM..LOWER, held during OPEN / UP. 0 = off.
    PITCH_K = _f("BITIDY_CB2_PITCH_K", 0.0)
    PITCH_MAX = _f("BITIDY_CB2_PITCH_MAX", 1.0)
    PITCH_PW = _f("BITIDY_CB2_PITCH_PW", 0.03)
    PITCH_FROM = int(_f("BITIDY_CB2_PITCH_FROM", 4))
    LIFT_Z = _f("BITIDY_CB2_LIFT_Z", 0.015)  # >0: LIFT is done once the pads are within LIFT_Z of the carry height (xy ignored)
    CARRY_STALL = int(_f("BITIDY_CB2_CARRY_STALL", 0))  # >0: CARRY gives up on the waypoint after this many stalled steps
    # V2: null-space re-uprighting: position-only IK (4 joints, 3 constraints) and the free joint direction is used to
    # turn the gripper with angular velocity NULL_K * (vessel_up x z) (LIFT..LOWER, once the vessel hangs free).
    NULL_K = _f("BITIDY_CB2_NULL_K", 0.3)
    NULL_MAX = _f("BITIDY_CB2_NULL_MAX", 0.02)
    NULL_ROLL = os.environ.get("BITIDY_CB2_NULL_ROLL", "1") == "1"
    # V2: PRESS placement. The rim pinch is a hinge about the rim tangent, so a held vessel hangs with its centre of mass
    # under the pinch (~40-70 deg tilt) and no wrist motion can hold it upright. LOWER first brings the hanging vessel
    # (centred over the stack axis) down until its low bottom edge is PRESS_GAP above the nest, then moves the pads to
    # where they would be with the vessel UPRIGHT at its nest (axis + grasp radius along the jaw axis, grasp height):
    # the vessel pivots on its bottom edge and is pressed upright, and only then do the jaws open.
    PRESS = os.environ.get("BITIDY_CB2_PRESS", "0") == "1"
    PRESS_GAP = _f("BITIDY_CB2_PRESS_GAP", 0.004)
    PRESS_DZ = _f("BITIDY_CB2_PRESS_DZ", 0.0)
    PRESS_TOL = _f("BITIDY_CB2_PRESS_TOL", 0.006)
    PRESS_TILT = _f("BITIDY_CB2_PRESS_TILT", 0.12)
    PRESS_STEPS = int(_f("BITIDY_CB2_PRESS_STEPS", 150))
    PRESS_R = _f("BITIDY_CB2_PRESS_R", 1.0)  # scale on the grasp radius of the upright pad pose
    PITCH_AIR = _f("BITIDY_CB2_PITCH_AIR", 0.0)
    PITCH_EMAX = _f("BITIDY_CB2_PITCH_EMAX", 1.0)
    PITCH_FF = _f("BITIDY_CB2_PITCH_FF", 0.0)  # feed-forward phi at the start of the servo (rad)

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
        self.phi = torch.zeros(n, device=dev)
        self.press = torch.zeros(n, dtype=torch.bool, device=dev)
        self.press_t = torch.zeros_like(self.p)
        self.far = torch.zeros(n, dtype=torch.bool, device=dev)

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
        self.phi[ids] = 0.0
        self.press[ids] = False
        self.press_t[ids] = 0

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

    def _tdiag(self, p_old, active, c, obj, fixed, moving) -> None:
        """Hang-tilt geometry at LIFT->CARRY and LOWER->OPEN: tilt, up-vector horizontal part along u / reach / tangent."""
        from isaaclab.utils.math import quat_apply
        if not hasattr(self, "_td"):
            self._td = {}
        upv = quat_apply(c["quat"], obj.new_tensor((0.0, 0.0, 1.0)).expand_as(obj))
        tl = T.tilt(c["quat"])
        out = torch.nn.functional.normalize(obj[:, :2] - self.ik.base_xy, dim=-1)
        u = self.u
        tg = torch.stack((-u[:, 1], u[:, 0]), -1)
        ax = torch.nn.functional.normalize(moving[:, :2] - fixed[:, :2], dim=-1)
        rng = torch.linalg.vector_norm(obj[:, :2] - self.ik.base_xy, dim=-1)
        for name, frm, to in (("lift", LIFT, CARRY), ("rel", LOWER, OPEN), ("grasp", CLOSE, LIFT)):
            m = active & (p_old == frm) & (self.p == to)
            if not bool(m.any()):
                continue
            rows = torch.stack((tl, (upv[:, :2] * u).sum(-1), (upv[:, :2] * out).sum(-1), (upv[:, :2] * tg).sum(-1),
                                (u * out).sum(-1), (ax * out).sum(-1), rng, c["typ"].float()), -1)[m]
            self._td.setdefault(name, []).append(rows.cpu())
        self._tn = getattr(self, "_tn", 0) + 1
        if self._tn % 400 == 0:
            for name, lst in self._td.items():
                r = torch.cat(lst)
                for typ in (0, 1):
                    q = r[r[:, 7] == typ]
                    if not len(q):
                        continue
                    big = (q[:, 0] > 0.3).float().mean()
                    near = q[q[:, 6] < 0.22]; far = q[q[:, 6] >= 0.22]
                    print(f"BITIDY_TDIAG {self.arm[6:]} {name} typ={typ} n={len(q)} tilt_mean={float(q[:, 0].mean()):.3f} "
                          f"tilt>0.3={float(big):.2f} up.u={float(q[:, 1].mean()):.3f} up.out={float(q[:, 2].mean()):.3f} "
                          f"up.tg={float(q[:, 3].mean()):.3f} |up.tg|={float(q[:, 3].abs().mean()):.3f} u.out={float(q[:, 4].mean()):.3f} "
                          f"ax.out={float(q[:, 5].mean()):.3f} tilt_near={float(near[:, 0].mean()) if len(near) else -1:.3f} "
                          f"tilt_far={float(far[:, 0].mean()) if len(far) else -1:.3f} n_far={len(far)}", flush=True)

    def ik_upright(self, target: torch.Tensor, w_des: torch.Tensor, gain: float, weight: float) -> torch.Tensor:
        """5-joint IK (pan, lift, elbow, flex, roll): pad -> target, gripper angular velocity (x, y) -> w_des."""
        env, ik = self.env, self.ik
        robot = ik.robot
        pad = base.pad_center(env, self.arm)
        body = robot.data.body_pos_w.torch[:, ik.body] - env.scene.env_origins
        cols = ik.jac_cols + [ik.roll + robot.num_base_dofs]
        jac = robot.data.body_link_jacobian_w.torch[:, ik.jac_body][:, :, cols]
        j_v, j_w = jac[:, :3], jac[:, 3:]
        j_pad = j_v - _skew(pad - body) @ j_w
        a = torch.cat((j_pad, weight * j_w[:, :2]), 1)
        b = torch.cat((gain * (target - pad), weight * w_des[:, :2]), 1).unsqueeze(-1)
        damping = 2e-4 * torch.eye(a.shape[1], device=a.device)
        dq = (a.transpose(1, 2) @ torch.linalg.solve(a @ a.transpose(1, 2) + damping, b)).squeeze(-1)
        dq = dq.clamp(-self.UP_MAX, self.UP_MAX)
        jidx = ik.joints + [ik.roll]
        q = robot.data.joint_pos.torch[:, jidx]
        q_target = robot.data.joint_pos_target.torch[:, jidx]
        return ((q + dq - q_target) / ARM_SCALE).clamp(-1.0, 1.0)

    def ik_null(self, target: torch.Tensor, omega: torch.Tensor, gain: float) -> torch.Tensor:
        """Position-only IK on pan/lift/elbow/flex (+ roll with NULL_ROLL) plus a null-space motion toward gripper
        angular velocity omega. Returns the arm actions (4, or 5 with the roll)."""
        env, ik = self.env, self.ik
        robot = ik.robot
        pad = base.pad_center(env, self.arm)
        body = robot.data.body_pos_w.torch[:, ik.body] - env.scene.env_origins
        cols = ik.jac_cols + ([ik.roll + robot.num_base_dofs] if self.NULL_ROLL else [])
        jidx = ik.joints + ([ik.roll] if self.NULL_ROLL else [])
        nj = len(cols)
        jac = robot.data.body_link_jacobian_w.torch[:, ik.jac_body][:, :, cols]
        j_v, j_w = jac[:, :3], jac[:, 3:]
        j_pad = j_v - _skew(pad - body) @ j_w
        eye3 = torch.eye(3, device=pad.device)
        jp_pinv = j_pad.transpose(1, 2) @ torch.linalg.inv(j_pad @ j_pad.transpose(1, 2) + 1e-4 * eye3)
        dq_pos = (jp_pinv @ (gain * (target - pad)).unsqueeze(-1)).squeeze(-1)
        null = torch.eye(nj, device=pad.device) - jp_pinv @ j_pad
        # best null-space motion for the horizontal angular velocity: min |J_w,xy N z - omega_xy|
        a = j_w[:, :2] @ null
        z = a.transpose(1, 2) @ torch.linalg.inv(a @ a.transpose(1, 2) + 1e-3 * torch.eye(2, device=pad.device)) @ omega[:, :2].unsqueeze(-1)
        dq_n = (null @ z).squeeze(-1).clamp(-self.NULL_MAX, self.NULL_MAX)
        dq = dq_pos + dq_n
        q = robot.data.joint_pos.torch[:, jidx]
        q_target = robot.data.joint_pos_target.torch[:, jidx]
        return ((q + dq - q_target) / ARM_SCALE).clamp(-1.0, 1.0)

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
        if self.PRESS:
            g = T._geo_t(env.device)
            tl = T.tilt(c["quat"])
            low_z = c["nest_z"] + self.PRESS_GAP + g["r_bot"][typ] * torch.sin(tl) - self.off[:, 2]
            lower_z = torch.where(self.press, lower_z, low_z)
            r0 = torch.linalg.vector_norm(self.off0[:, :2], dim=-1, keepdim=True) * self.PRESS_R
            u_now = torch.nn.functional.normalize(ax, dim=-1)
            up_xy = c["axis"] + r0 * u_now
            up_z = c["nest_z"] + self.off0[:, 2] + self.PRESS_DZ
            place_xy = torch.where(_col(self.press & (p == LOWER)), up_xy, place_xy)
            lower_z = torch.where(self.press & (p == LOWER), up_z, lower_z)
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
        roll = torch.where(typ == T.CUP, torch.full_like(rf, self.ROLL_CUP), torch.full_like(rf, self.ROLL_BOWL))
        if self.FAR_R > 0:  # far vessels: turn the jaws so the pinch is on the side (less reach), latched before DOWN
            rng = torch.linalg.vector_norm(obj[:, :2] - self.ik.base_xy, dim=-1)
            self.far = torch.where(p <= HOVER_P, rng > self.FAR_R, self.far)
            roll = torch.where(self.far, torch.full_like(rf, self.FAR_ROLL), roll)
        if self.PITCH_K > 0:
            from isaaclab.utils.math import quat_apply
            zz = obj.new_tensor((0.0, 0.0, 1.0)).expand_as(obj)
            upv = quat_apply(c["quat"], zz)
            out = torch.nn.functional.normalize(obj[:, :2] - self.ik.base_xy, dim=-1)
            a3 = torch.cat((-out[:, 1:2], out[:, 0:1], torch.zeros_like(out[:, :1])), -1)  # z x out
            e = (torch.cross(upv, zz, dim=-1) * a3).sum(-1)
            servo = (p >= self.PITCH_FROM) & (p <= LOWER) & active
            if self.PITCH_AIR > 0:  # only once the vessel hangs free (its bottom off the table / nest)
                ref = T._pick(T._state(self.env)["ref"], T.slot_obj(self.env, c["slot"]))
                servo &= (obj[:, 2] > ref[:, 2] + self.PITCH_AIR) | (p >= CARRY)
                servo &= ~((p == LOWER) & (obj[:, 2] < c["nest_z"] + 0.03))
            e = e.clamp(-self.PITCH_EMAX, self.PITCH_EMAX)
            first = servo & (self.phi == 0)
            self.phi = torch.where(first, torch.full_like(self.phi, self.PITCH_FF), self.phi)
            self.phi = torch.where(servo, (self.phi + self.PITCH_K * e).clamp(-self.PITCH_MAX, self.PITCH_MAX), self.phi)
            self.phi = torch.where((p < self.PITCH_FROM) | (p > UP), torch.zeros_like(self.phi), self.phi)
            # Rodrigues: rotate the jaw direction about a3 by phi
            cph, sph = torch.cos(self.phi).unsqueeze(-1), torch.sin(self.phi).unsqueeze(-1)
            d = tilt
            d_rot = d * cph + torch.cross(a3, d, dim=-1) * sph + a3 * (a3 * d).sum(-1, keepdim=True) * (1 - cph)
            on = (p >= self.PITCH_FROM) & (p <= UP)
            tilt = torch.where(_col(on), d_rot, tilt)
            pw = torch.where(on, torch.full_like(pw, self.PITCH_PW), pw)
        act, _ = self.ik(goal + self.bias, tilt, roll, gain=self.GAIN, point_weight=pw)
        if self.NULL_K > 0:
            from isaaclab.utils.math import quat_apply
            zz = obj.new_tensor((0.0, 0.0, 1.0)).expand_as(obj)
            upv = quat_apply(c["quat"], zz)
            on = (p >= LIFT) & (p <= UP)
            turn = (p <= LOWER) & active
            if self.PITCH_AIR > 0:
                ref = T._pick(T._state(self.env)["ref"], T.slot_obj(self.env, c["slot"]))
                turn &= (obj[:, 2] > ref[:, 2] + self.PITCH_AIR) | (p >= CARRY)
                turn &= ~((p == LOWER) & (obj[:, 2] < c["nest_z"] + 0.03))
            omega = torch.where(_col(turn), self.NULL_K * torch.cross(upv, zz, dim=-1), torch.zeros_like(upv))
            arm_null = self.ik_null(goal + self.bias, omega, self.GAIN)
            if not self.NULL_ROLL:
                arm_null = torch.cat((arm_null, act[:, 4:5]), -1)
            act = torch.where(_col(on), arm_null, act)
        if self.UP_K > 0:
            from isaaclab.utils.math import quat_apply
            zz = obj.new_tensor((0.0, 0.0, 1.0)).expand_as(obj)
            upv = quat_apply(c["quat"], zz)
            servo = (p >= self.UP_FROM) & (p <= UP)
            hold = p >= OPEN
            w_des = torch.where(_col(hold), torch.zeros_like(upv), self.UP_K * torch.cross(upv, zz, dim=-1))
            act_up = self.ik_upright(goal + self.bias, w_des, self.GAIN, self.UP_W)
            act = torch.where(_col(servo), act_up, act)
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
        if self.LIFT_Z > 0:
            zerr = (carry_z - pad[:, 2]).abs()
            adv |= (p == LIFT) & (zerr < self.LIFT_Z) & ((self.stall_t > 10) | carrot_at)
        if self.CARRY_STALL > 0:
            adv |= (p == CARRY) & (self.stall_t > self.CARRY_STALL) & over_ok
        if self.RISE_XY > 0:
            xy_err = torch.linalg.vector_norm(t[:, :2] - pad[:, :2], dim=-1)
            stall |= (p == RISE) & (self.stall_t > self.STALL) & (xy_err < self.RISE_XY)
        stall &= ~adv
        adv = adv | stall
        hold_on = (p == LOWER) & ~over_ok
        if self.PRESS:
            in_low = (p == LOWER) & active
            go = in_low & adv & ~self.press
            pressing = in_low & self.press
            self.press_t = torch.where(pressing, self.press_t + 1, 0)
            tl = T.tilt(c["quat"])
            done = pressing & (((err < self.PRESS_TOL) & (tl < self.PRESS_TILT) & (self.press_t > 10))
                               | (self.press_t > self.PRESS_STEPS))
            adv = torch.where(in_low, done, adv)
            stall = stall & ~in_low
            hold_on = hold_on & ~self.press
            self.press = (self.press | go) & (p == LOWER)
            self.t = torch.where(go, 0, self.t)
            self.best = torch.where(go, 9.0, self.best)
            self.stall_t = torch.where(go, 0, self.stall_t)
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

        if os.environ.get("BITIDY_CB_TDIAG"):
            self._tdiag(p, active, c, obj, fixed, moving)
        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if (os.environ.get("BITIDY_DEBUG") and bool(active[k])
                and int(env.episode_length_buf[k]) % int(os.environ.get("BITIDY_DEBUG_EVERY", 10)) == 0):
            r3 = lambda v: [round(x, 3) for x in v[k].tolist()]  # noqa: E731
            print(f"[cb {self.arm[6:]}] t={int(env.episode_length_buf[k])} p={int(p[k])} err={float(err[k]):.4f} pad={r3(pad)} "
                  f"goal={r3(goal)} obj={r3(obj)} rim={r3(rc)} off={r3(self.off)} near={bool(near[k])} "
                  f"fix={r3(fixed)} mov={r3(moving)} axis={r3(c['axis'])} nest_z={float(c['nest_z'][k]):.3f} depth={int(c['depth'][k])} "
                  f"flags={T.progress(env).flags[k].int().tolist()} phi={float(self.phi[k]):.3f} tilt={float(T.tilt(c['quat'])[k]):.3f} "
                  f"owner_t={int(self.t[k])} stall={int(self.stall_t[k])}", flush=True)
        return torch.cat((act, grip.unsqueeze(-1)), -1), err


class CupBasketScriptV2:
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
        if hasattr(self, "_rs_maxp"):
            self._rs_maxp[ids] = 0

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

    def _rstats(self, ctx) -> None:
        """First episodes: per-slot success / furthest phase by the vessel's start range from its arm's base."""
        env = self.env
        if not hasattr(self, "_rs"):
            self._rs_first = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            self._rs_maxp = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)
            self._rs = []
        for s, a in enumerate(self.arms):
            self._rs_maxp[:, s] = torch.maximum(self._rs_maxp[:, s], a.p)
        ending = (env.episode_length_buf == int(env.max_episode_length) - 2) & self._rs_first
        if not bool(ending.any()):
            return
        self._rs_first &= ~ending
        st = T._state(env)
        prog = T.progress(env)
        for s, a in enumerate(self.arms):
            m = ending & T.slot_active(env, s)
            ref = T._pick(st["ref"], T.slot_obj(env, s))
            rng = torch.linalg.vector_norm(ref[:, :2] - a.ik.base_xy, dim=-1)
            # basket distance from base
            brng = torch.linalg.vector_norm(ctx[s]["axis"] - a.ik.base_xy, dim=-1)
            self._rs.append(torch.stack((torch.full_like(rng, s), rng, brng, prog.slot_ok[:, s].float(),
                                         self._rs_maxp[:, s].float(), ctx[s]["typ"].float()), -1)[m].cpu())
        r = torch.cat(self._rs)
        out = []
        for lo in (0.0, 0.18, 0.21, 0.24, 0.27):
            q = r[(r[:, 1] >= lo) & (r[:, 1] < lo + (0.03 if lo else 0.18))]
            if len(q):
                out.append(f"start[{lo:.2f}] n={len(q)} ok={float(q[:, 3].mean()):.2f} rise_stuck={float((q[:, 4] == 0).float().mean()):.2f}")
        for lo in (0.0, 0.15, 0.20, 0.25):
            q = r[(r[:, 2] >= lo) & (r[:, 2] < lo + (0.15 if not lo else 0.05 if lo < 0.25 else 1))]
            if len(q):
                out.append(f"basket[{lo:.2f}] n={len(q)} ok={float(q[:, 3].mean()):.2f}")
        print("BITIDY_CB2_RANGE n=%d " % len(r) + " | ".join(out), flush=True)

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
                 "clear_z": top + RimArm.CARRY_CLEAR, "quat": T.slot_quat(env, s)}
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
        if os.environ.get("BITIDY_CB2_RSTATS"):
            self._rstats(ctx)
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
