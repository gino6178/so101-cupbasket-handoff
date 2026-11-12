"""Deterministic release filter for CupBasket executors (``BITIDY_RETREAT=1``): open fully, then retreat vertically.

Applied to the final 12-D action (after base + residual) in training (``residual.install_residual``) and in
``residual.ResidualScript`` (eval, voice chains), so train and deploy see the same executed actions.

Per arm (slot s): when its gripper command has been open for ``BITIDY_RETREAT_PERSIST`` (2) steps in a row while its slot's vessel is near the goal (slot active,
``lift`` milestone reached, vessel within ``BITIDY_RETREAT_XY`` (2 cm) of the stack axis / zone centre and less than
``BITIDY_RETREAT_Z`` (3 cm) above the nest height), the filter
  1. HOLD: zeroes that arm's five joint deltas and keeps the gripper open until the jaw joint has reached its open
     target (|q - q_target| < ``BITIDY_RETREAT_TOL`` rad, at least 2 steps) or ``BITIDY_RETREAT_HOLD`` steps passed;
  2. LIFT: drives the pad centre straight up (position-only damped least squares on pan/lift/elbow/wrist_flex, wrist
     roll held) to ``BITIDY_RETREAT_DZ`` (4 cm) above where it opened, at most ``BITIDY_RETREAT_LIFT`` steps;
then hands the arm back. The gripper channel stays open during both phases. Uses the true sim state (as the env's
milestones do).
"""

from __future__ import annotations

import os

import torch

from . import cup_basket_env_cfg as T
from . import env_cfg as base
from .scripted import ARM_SCALE, _skew


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def enabled() -> bool:
    return os.environ.get("BITIDY_RETREAT", "0") not in ("", "0")


class RetreatFilter:
    TRIG_XY = _f("BITIDY_RETREAT_XY", 0.02)
    TRIG_Z = _f("BITIDY_RETREAT_Z", 0.03)
    PERSIST = int(_f("BITIDY_RETREAT_PERSIST", 2))
    HOLD_MAX = int(_f("BITIDY_RETREAT_HOLD", 20))
    OPEN_TOL = _f("BITIDY_RETREAT_TOL", 0.05)
    DZ = _f("BITIDY_RETREAT_DZ", 0.04)
    LIFT_MAX = int(_f("BITIDY_RETREAT_LIFT", 40))
    GAIN = _f("BITIDY_RETREAT_GAIN", 1.0)

    def __init__(self, env):
        self.env = env
        n, dev = env.num_envs, env.device
        self.phase = torch.zeros((n, T.N_SLOT), dtype=torch.long, device=dev)  # 0 off, 1 hold, 2 lift
        self.t = torch.zeros((n, T.N_SLOT), dtype=torch.long, device=dev)
        self.anchor = torch.zeros((n, T.N_SLOT, 3), device=dev)
        self.prev_close = torch.zeros((n, T.N_SLOT), dtype=torch.bool, device=dev)
        self.open_run = torch.zeros((n, T.N_SLOT), dtype=torch.long, device=dev)
        self.arms = []
        for arm in T.ARMS:
            robot = env.scene[arm]
            body = robot.find_bodies("gripper")[0][0]
            joints = robot.find_joints(base.ARM_JOINTS, preserve_order=True)[0]
            self.arms.append({"robot": robot, "body": body, "jac_body": body - 1 if robot.is_fixed_base else body,
                              "joints": joints[:4], "cols": [j + robot.num_base_dofs for j in joints[:4]],
                              "grip": robot.find_joints("gripper")[0][0]})
        self.count = {"triggers": 0, "hold_steps": 0, "lift_steps": 0}

    def reset(self, ids: torch.Tensor) -> None:
        self.phase[ids] = 0
        self.t[ids] = 0
        self.prev_close[ids] = False
        self.open_run[ids] = 0

    def _lift_action(self, s: int, target: torch.Tensor) -> torch.Tensor:
        env, a = self.env, self.arms[s]
        robot = a["robot"]
        pad = base.pad_center(env, T.ARMS[s])
        body = robot.data.body_pos_w.torch[:, a["body"]] - env.scene.env_origins
        jac = robot.data.body_link_jacobian_w.torch[:, a["jac_body"]][:, :, a["cols"]]
        j = jac[:, :3] - _skew(pad - body) @ jac[:, 3:]
        b = (self.GAIN * (target - pad)).unsqueeze(-1)
        dq = (j.transpose(1, 2) @ torch.linalg.solve(j @ j.transpose(1, 2) + 1e-4 * torch.eye(3, device=j.device), b))
        return (dq.squeeze(-1) / ARM_SCALE).clamp(-1.0, 1.0)

    @torch.no_grad()
    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        env = self.env
        out = action.clone()
        pos = T.objs_pos(env)
        prog = T.progress(env)
        for s in range(T.N_SLOT):
            g, arm = 6 * s + 5, slice(6 * s, 6 * s + 5)
            a = self.arms[s]
            robot = a["robot"]
            pad = base.pad_center(env, T.ARMS[s])
            p = T.slot_pos(env, s, pos)
            axis, z, _ = T.nest_target(env, s, pos)
            near = (T.slot_active(env, s) & prog.stage(s, "lift")
                    & (torch.linalg.vector_norm(p[:, :2] - axis, dim=-1) < self.TRIG_XY) & (p[:, 2] < z + self.TRIG_Z))
            # the base's open command must persist PERSIST steps (a one-step flicker is not a release)
            self.open_run[:, s] = torch.where(action[:, g] >= 0, self.open_run[:, s] + 1, 0)
            start = (self.open_run[:, s] == self.PERSIST) & near & (self.phase[:, s] == 0)
            if bool(start.any()):
                self.anchor[:, s] = torch.where(start.unsqueeze(-1), pad, self.anchor[:, s])
                self.phase[:, s] = torch.where(start, 1, self.phase[:, s])
                self.t[:, s] = torch.where(start, 0, self.t[:, s])
                self.count["triggers"] += int(start.sum())
            hold = self.phase[:, s] == 1
            if bool(hold.any()):
                q = robot.data.joint_pos.torch[:, a["grip"]]
                qt = robot.data.joint_pos_target.torch[:, a["grip"]]
                opened = (self.t[:, s] >= 2) & ((q - qt).abs() < self.OPEN_TOL)
                to_lift = hold & (opened | (self.t[:, s] >= self.HOLD_MAX))
                self.phase[:, s] = torch.where(to_lift, 2, self.phase[:, s])
                self.t[:, s] = torch.where(to_lift, 0, self.t[:, s])
                hold = self.phase[:, s] == 1
                out[:, arm] = torch.where(hold.unsqueeze(-1), 0.0, out[:, arm])
                self.count["hold_steps"] += int(hold.sum())
            lift = self.phase[:, s] == 2
            if bool(lift.any()):
                tgt = self.anchor[:, s].clone()
                tgt[:, 2] += self.DZ
                done = lift & ((pad[:, 2] >= tgt[:, 2] - 0.005) | (self.t[:, s] >= self.LIFT_MAX))
                self.phase[:, s] = torch.where(done, 0, self.phase[:, s])
                lift = self.phase[:, s] == 2
                act4 = self._lift_action(s, tgt)
                arm_act = torch.cat((act4, torch.zeros_like(act4[:, :1])), -1)
                out[:, arm] = torch.where(lift.unsqueeze(-1), arm_act, out[:, arm])
                self.count["lift_steps"] += int(lift.sum())
            busy = self.phase[:, s] > 0
            out[:, g] = torch.where(busy, torch.ones_like(out[:, g]), out[:, g])
            self.t[:, s] += busy.long()
            self.prev_close[:, s] = out[:, g] < 0
        return out
