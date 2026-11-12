"""STS3215 servo model for the SO-101 arms (``BITIDY_CB_SERVO=1``), from published single-author measurements:

* so101-mjlab (step-response sysid): kp ~18 (16-40) N m/rad, kd 1.3-3, frictionloss 0.2, armature 0.02-0.045,
  effort 1.47 N m arm / 1.2 N m gripper.
* so101-real2sim: stiffness 16-62 (stock model ~1000); lag up to 40 deg on fast moves; deadzone 1 encoder tick.
* VatoLabs/STS3215_profile: the servo moves an internal target with a constant-acceleration (triangle) profile,
  a = 38.96 rad/s^2 at LeRobot settings; best constant delay without the profile ~20 ms.

Pipeline per 120 Hz goal write (the deploy contract of ``utils/deploy.py``: goal = present + scale * action):
  goal -> LeRobot clamp |goal - present| <= 10 deg -> per-episode delay 0-2 ticks (0-17 ms) -> 1-tick deadband
  (|goal - present| <= 1 tick: no new goal) -> internal target, constant-acceleration profile (a in [25, 50] rad/s^2,
  |v| <= VMAX) -> physics PD (kp log-uniform [16, 62], kd [1, 3], armature [0.02, 0.2], frictionloss [0.02, 0.2],
  effort 1.47 / 1.2 N m; set by reset events in ``cup_basket_env_cfg``).
The policy's ``joint_target`` observation is the written goal (what the real robot reports), not the internal target.

Env (all ranges "lo,hi"): BITIDY_CB_SERVO_ACC, _VMAX, _DELAY (ticks), _CLAMP_DEG, _TICKS (deadband in ticks).
"""

from __future__ import annotations

import math
import os

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction

ON = os.environ.get("BITIDY_CB_SERVO", "0") == "1"
PARTS = os.environ.get("BITIDY_CB_SERVO_PARTS", "profile,pd,effort").split(",")
"""profile: goal clamp / delay / deadband / acceleration profile; pd: kp kd armature frictionloss; effort: limits."""


def _pair(name: str, default: str) -> tuple[float, float]:
    a, b = (float(v) for v in os.environ.get(name, default).split(","))
    return a, b


ACC = _pair("BITIDY_CB_SERVO_ACC", "25,50")
VMAX = float(os.environ.get("BITIDY_CB_SERVO_VMAX", 4.7))  # rad/s, STS3215 no-load ~0.222 s / 60 deg
DELAY = tuple(int(v) for v in _pair("BITIDY_CB_SERVO_DELAY", "0,2"))  # 120 Hz ticks (8.3 ms)
CLAMP = math.radians(float(os.environ.get("BITIDY_CB_SERVO_CLAMP_DEG", 10.0)))
TICK = float(os.environ.get("BITIDY_CB_SERVO_TICKS", 1.0)) * 2 * math.pi / 4096
NSUB = 8  # profile integration steps per 120 Hz tick


class _Servo:
    """Goal -> internal target state for one action term (envs x joints)."""

    def _servo_init(self, relative: bool):
        n, j, dev = self.num_envs, self._num_joints, self.device
        ids = self._joint_ids
        if isinstance(ids, slice):
            ids = list(range(self._asset.num_joints))[ids]
        elif not isinstance(ids, (list, tuple, torch.Tensor)):
            import warp as wp

            ids = wp.to_torch(ids)
        self._jid = torch.as_tensor(ids, device=dev).long().flatten()
        self._relative = relative
        self.goal = torch.zeros(n, j, device=dev)
        self.s = torch.zeros(n, j, device=dev)
        self.v = torch.zeros(n, j, device=dev)
        self.buf = torch.zeros(DELAY[1] + 1, n, j, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.acc = torch.full((n, 1), sum(ACC) / 2, device=dev)
        self._servo_reset(None)
        if os.environ.get("BITIDY_CB_SERVO_LOG") and not getattr(type(self), "_logged", False):
            type(self)._logged = True
            d = self._asset.data

            def g(name):
                try:
                    x = getattr(d, name)
                    x = getattr(x, "torch", x)
                    return [round(float(v), 4) for v in x[0, self._jid]]
                except Exception as e:  # noqa: BLE001
                    return f"n/a ({type(e).__name__})"

            print(f"BITIDY_CB_SERVO asset={self.cfg.asset_name} joints={self._joint_names} "
                  + " ".join(f"{k}={g(k)}" for k in ("joint_stiffness", "joint_damping", "joint_armature",
                                                      "joint_friction_coeff", "joint_effort_limits",
                                                      "joint_vel_limits")), flush=True)

    def _q(self):
        return self._asset.data.joint_pos.torch[:, self._jid]

    def _servo_reset(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None or isinstance(env_ids, slice) \
            else torch.as_tensor(env_ids, device=self.device).long()
        if len(ids) == 0:
            return
        q = self._q()[ids]
        self.s[ids], self.v[ids], self.goal[ids] = q, 0.0, q
        self.buf[:, ids] = q.unsqueeze(0)
        self.delay[ids] = torch.randint(DELAY[0], DELAY[1] + 1, (len(ids),), device=self.device)
        self.acc[ids] = torch.empty(len(ids), 1, device=self.device).uniform_(*ACC)

    def _servo_apply(self, g: torch.Tensor):
        q = self._q()
        if "profile" not in PARTS:
            self.goal = g
            self._asset.set_joint_position_target_index(target=g, joint_ids=self._jid)
            return
        g = q + (g - q).clamp(-CLAMP, CLAMP)  # LeRobot max relative target
        self.goal = g
        self.buf = torch.cat((g.unsqueeze(0), self.buf[:-1]), 0)
        g = self.buf[self.delay, torch.arange(self.num_envs, device=self.device)]
        # deadzone: an error within one tick does not move the servo
        g = torch.where((g - q).abs() <= TICK, self.s, g)
        h = self._env.physics_dt / NSUB
        a = self.acc
        s, v = self.s, self.v
        for _ in range(NSUB):
            d = g - s
            brake = (torch.sign(v) == torch.sign(d)) & (d.abs() <= v * v / (2 * a) + 1e-6)
            acc = torch.where(brake, -torch.sign(v) * a, torch.sign(d) * a)
            v_new = (v + acc * h).clamp(-VMAX, VMAX)
            s_new = s + 0.5 * (v + v_new) * h
            # arrive: crossing the goal while slow enough to stop within one step
            cross = torch.sign(g - s_new) != torch.sign(d)
            stop = cross & (v_new.abs() <= a * h * 2)
            s = torch.where(stop | (d.abs() < 1e-5), g, s_new)
            v = torch.where(stop | (d.abs() < 1e-5), torch.zeros_like(v_new), v_new)
        self.s, self.v = s, v
        self._asset.set_joint_position_target_index(target=s, joint_ids=self._jid)


class ServoRelativeJointPositionAction(_Servo, RelativeJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._servo_init(True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._servo_reset(env_ids)

    def apply_actions(self):
        self._servo_apply(self.processed_actions + self._q())


class ServoBinaryJointPositionAction(_Servo, BinaryJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._servo_init(False)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._servo_reset(env_ids)

    def apply_actions(self):
        self._servo_apply(self.processed_actions)


def servo_joint_target(env, asset_cfg):
    """``joint_target`` observation; with the servo model, the written goal of each servo action term instead of the
    PD target (the internal profile target is not observable on the robot)."""
    from isaaclab_tutorial.tasks.place_vial import mdp

    t = mdp.joint_target(env, asset_cfg)
    if not ON:
        return t
    t = t.clone()
    cols = asset_cfg.joint_ids
    cols = list(range(t.shape[1])) if isinstance(cols, slice) else [int(c) for c in cols]
    for term in env.action_manager._terms.values():
        if isinstance(term, _Servo) and term.cfg.asset_name == asset_cfg.name:
            for k, jid in enumerate(term._jid.tolist()):
                jid = int(jid)
                if jid in cols:
                    t[:, cols.index(jid)] = term.goal[:, k]
    return t
