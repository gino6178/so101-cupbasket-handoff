"""Scripted demonstrations for the bimanual tasks (T1 handover first).

A waypoint state machine per environment drives each fingertip midpoint with damped least-squares IK on the
simulator's Jacobian (3 position rows + 2 pointing rows for the 5 arm joints) and emits actions in the task's own
action space (relative joint targets, scaled), so the recorded (observation, action) pairs can pretrain the PPO
actor by behaviour cloning.

Run with ``isaaclab play ... --external_callback isaaclab_tutorial.tasks.bimanual.scripted.install_scripted``; the
policy's actions are replaced by the script. ``$BITIDY_DEMO`` (path) saves ``obs``/``act``/``success`` as .npz,
``$BITIDY_VIDEO`` also records the tiled view (Render tasks), and the episode counter prints ``BITIDY_EVAL_RESULT``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from . import env_cfg as base

ARM_SCALE, GRIP_SCALE = 0.033, 0.02
DOWN = (0.0, 0.0, -1.0)
TOWARD_LEFT = (0.0, 1.0, 0.0)


def _skew(r: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(r[:, 0])
    return torch.stack(
        (
            torch.stack((z, -r[:, 2], r[:, 1]), -1),
            torch.stack((r[:, 2], z, -r[:, 0]), -1),
            torch.stack((-r[:, 1], r[:, 0], z), -1),
        ),
        1,
    )


class ArmIK:
    """Per-step joint deltas that move one arm's fingertip midpoint toward a target and point the jaws along a direction."""

    def __init__(self, env, arm: str):
        self.env, self.arm = env, arm
        robot = env.scene[arm]
        self.robot = robot
        self.body = robot.find_bodies("gripper")[0][0]
        self.jac_body = self.body - 1 if robot.is_fixed_base else self.body
        joints = robot.find_joints(base.ARM_JOINTS, preserve_order=True)[0]
        # IK moves pan/lift/elbow/wrist_flex; wrist_roll is set directly to line the jaws up with the object.
        self.joints, self.roll = joints[:4], joints[4]
        self.jac_cols = [j + robot.num_base_dofs for j in self.joints]
        self.roll_home = robot.data.default_joint_pos.torch[:, self.roll].clone()
        self.base_xy = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]

    def __call__(self, target: torch.Tensor, direction: torch.Tensor, roll: torch.Tensor | None = None,
                 gain: float = 0.35, point_weight: float = 0.015, accumulate: bool = False):
        env, robot = self.env, self.robot
        pad = base.pad_center(env, self.arm)
        body = robot.data.body_pos_w.torch[:, self.body] - env.scene.env_origins
        jac = robot.data.body_link_jacobian_w.torch[:, self.jac_body][:, :, self.jac_cols]
        j_v, j_w = jac[:, :3], jac[:, 3:]
        j_pad = j_v - _skew(pad - body) @ j_w
        axis = torch.nn.functional.normalize(pad - body, dim=-1)
        turn = torch.cross(axis, direction, dim=-1)
        # Pointing only constrains the two rotations that tilt the jaw axis; drop the component about the axis itself.
        j_turn = j_w - axis.unsqueeze(-1) * (axis.unsqueeze(1) @ j_w)
        # ``point_weight`` may be a float or one weight per environment.
        w = point_weight if isinstance(point_weight, float) else point_weight.view(-1, 1)
        a = torch.cat((j_pad, (w.unsqueeze(-1) if not isinstance(w, float) else w) * j_turn), 1)
        b = torch.cat((gain * (target - pad), w * gain * turn), 1).unsqueeze(-1)
        damping = 1e-4 * torch.eye(a.shape[1], device=a.device)
        dq = (a.transpose(1, 2) @ torch.linalg.solve(a @ a.transpose(1, 2) + damping, b)).squeeze(-1)
        # Relative actions move the joint *targets*; aim them at position + step so PD lag cannot wind up.
        q = robot.data.joint_pos.torch[:, self.joints]
        q_target = robot.data.joint_pos_target.torch[:, self.joints]
        # ``accumulate`` steps the targets themselves instead (integral action: gravity sag is driven out).
        arm = ((dq if accumulate else q + dq - q_target) / ARM_SCALE).clamp(-1.0, 1.0)
        roll_goal = self.roll_home if roll is None else self.roll_home + roll
        q_roll = robot.data.joint_pos_target.torch[:, self.roll]
        roll_act = ((roll_goal - q_roll) / ARM_SCALE).clamp(-1.0, 1.0)
        return torch.cat((arm, roll_act.unsqueeze(-1)), -1), torch.linalg.vector_norm(target - pad, dim=-1)

    def joint_step(self, target: torch.Tensor, gain: float = 0.25, max_step: float = 0.03) -> torch.Tensor:
        """New position targets for the four IK joints that move the fingertip midpoint toward ``target``."""
        env, robot = self.env, self.robot
        pad = base.pad_center(env, self.arm)
        body = robot.data.body_pos_w.torch[:, self.body] - env.scene.env_origins
        jac = robot.data.body_link_jacobian_w.torch[:, self.jac_body][:, :, self.jac_cols]
        a = jac[:, :3] - _skew(pad - body) @ jac[:, 3:]
        b = (gain * (target - pad)).unsqueeze(-1)
        dq = (a.transpose(1, 2) @ torch.linalg.solve(a @ a.transpose(1, 2) + 1e-4 * torch.eye(3, device=a.device), b))
        # Accumulate on the current targets so the PD drives' gravity sag is integrated away instead of persisting.
        return robot.data.joint_pos_target.torch[:, self.joints] + dq.squeeze(-1).clamp(-max_step, max_step)

    def tilted(self, target: torch.Tensor, forward: float = 0.7) -> torch.Tensor:
        """Jaw direction for a top grasp: down, tilted away from this arm's base like a natural SO-101 reach."""
        out = torch.nn.functional.normalize(target[:, :2] - self.base_xy, dim=-1)
        return torch.nn.functional.normalize(torch.cat((forward * out, -torch.ones_like(out[:, :1])), -1), dim=-1)


class HandoverScript:
    """T1: left grasps and lifts, right takes it from the side, left lets go, right places on the target."""

    HOVER = 0.06
    HANDOVER = (0.20, 0.02, 0.13)
    RIGHT_READY = (0.20, -0.10, 0.13)
    # Right pads grip the lower, -y half of the cube while the left jaws hold it from above.
    RIGHT_GRIP_OFFSET = (0.0, -0.004, -0.006)

    def __init__(self, env):
        self.env = env
        self.left, self.right = ArmIK(env, "robot_left"), ArmIK(env, "robot_right")
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.timer = torch.zeros_like(self.phase)
        self.down = torch.tensor(DOWN, device=env.device).expand(env.num_envs, -1)
        self.side = torch.tensor(TOWARD_LEFT, device=env.device).expand(env.num_envs, -1)

    def reset(self, ids: torch.Tensor) -> None:
        self.phase[ids] = 0
        self.timer[ids] = 0

    OPEN = 0.35
    PINCH_LIFT = 0.008
    ROLL_SIGN = float(os.environ.get("BITIDY_ROLL_SIGN", "1"))

    def _cube_roll(self) -> torch.Tensor:
        """Cube yaw folded to (-45, 45] deg, so the jaws close on a pair of faces."""
        x, y, z, w = self.env.scene["cube"].data.root_quat_w.torch.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        half = torch.pi / 4
        return self.ROLL_SIGN * (torch.remainder(yaw + half, 2 * half) - half)

    def _toward_open(self, env, arm: str) -> torch.Tensor:
        """Gripper action that settles the jaws at a moderate opening (wide open jaws strike the table)."""
        return ((self.OPEN - base.gripper_openness(env, arm)) * 20.0).clamp(-1.0, 1.0)

    def _const(self, value) -> torch.Tensor:
        return torch.tensor(value, device=self.env.device).expand(self.env.num_envs, -1)

    def act(self) -> torch.Tensor:
        env, p = self.env, self.phase
        n = env.num_envs
        cube = base.cube_pos(env)
        target = base.cube_pos(env).new_tensor((*base.TARGET, base.CUBE_REST_Z)).expand(n, -1)
        hover = torch.tensor((0.0, 0.0, self.HOVER), device=env.device)
        handover = self._const(self.HANDOVER)
        ready = self._const(self.RIGHT_READY)
        grip = cube + torch.tensor(self.RIGHT_GRIP_OFFSET, device=env.device)
        approach = grip + torch.tensor((0.0, -0.05, 0.0), device=env.device)
        left_up = cube + torch.tensor((0.0, 0.05, 0.07), device=env.device)

        # Left arm: waypoint per phase.
        # Pinch slightly above the cube centre so the fingertips clear the table on the way in.
        pinch = cube + torch.tensor((0.0, 0.0, self.PINCH_LIFT), device=env.device)
        left_target = torch.where((p == 0).unsqueeze(-1), cube + hover, pinch)
        left_target = torch.where((p >= 3).unsqueeze(-1), handover, left_target)
        left_target = torch.where((p >= 8).unsqueeze(-1), left_up, left_target)
        left_grip = torch.where((p >= 2) & (p < 8), -1.0, self._toward_open(env, "robot_left"))
        # Right arm.
        right_target = torch.where((p >= 4).unsqueeze(-1), approach, ready)
        right_target = torch.where((p >= 5).unsqueeze(-1), grip, right_target)
        right_target = torch.where((p >= 9).unsqueeze(-1), target + hover, right_target)
        right_target = torch.where((p >= 10).unsqueeze(-1), target + torch.tensor((0.0, 0.0, 0.003), device=env.device), right_target)
        right_target = torch.where((p >= 12).unsqueeze(-1), target + hover, right_target)
        right_grip = torch.where((p >= 6) & (p < 11), -1.0, self._toward_open(env, "robot_right"))
        right_dir = torch.where((p >= 9).unsqueeze(-1), self.right.tilted(right_target), self.side)

        left_act, left_err = self.left(left_target, self.left.tilted(left_target), self._cube_roll())
        right_act, right_err = self.right(right_target, right_dir)
        if os.environ.get("BITIDY_DEBUG") and int(self.timer.sum()) % 7 == 3:
            for ik in (self.left, self.right):
                r = ik.robot
                jac_all = r.data.body_link_jacobian_w.torch
                qd = r.data.joint_vel.torch
                pred = (jac_all[:, ik.jac_body] @ torch.cat((torch.zeros_like(qd[:, :r.num_base_dofs]), qd), -1).unsqueeze(-1)).squeeze(-1) if r.num_base_dofs else (jac_all[:, ik.jac_body] @ qd.unsqueeze(-1)).squeeze(-1)
                meas = torch.cat((r.data.body_link_lin_vel_w.torch[:, ik.body], r.data.body_link_ang_vel_w.torch[:, ik.body]), -1)
                print(f"[jac] {ik.arm} shape={tuple(jac_all.shape)} body={ik.body} jb={ik.jac_body} cols={ik.jac_cols} fixed={r.is_fixed_base} "
                      f"bodies={r.body_names} joints={r.joint_names} pred={pred[0].tolist()} meas={meas[0].tolist()}", flush=True)
        if os.environ.get("BITIDY_DEBUG") and int(self.timer.sum()) % 7 == 0:
            pad = base.pad_center(env, "robot_left")
            print(f"[script] phase={p[:4].tolist()} lerr={left_err[:4].tolist()} rerr={right_err[:4].tolist()} "
                  f"lpad={pad[0].tolist()} cube={cube[0].tolist()} open={base.gripper_openness(env, 'robot_left')[0].item():.2f}", flush=True)

        # Phase transitions: reach waypoint (error) or hold a timed grip.
        self.timer += 1
        reached_l, reached_r = left_err < 0.006, right_err < 0.006
        advance = torch.zeros_like(p, dtype=torch.bool)
        advance |= (p == 0) & reached_l
        advance |= (p == 1) & reached_l
        advance |= (p == 2) & (self.timer > 25)
        advance |= (p == 3) & (left_err < 0.01)
        advance |= (p == 4) & (right_err < 0.01)
        advance |= (p == 5) & reached_r
        advance |= (p == 6) & (self.timer > 25)
        advance |= (p == 7) & (self.timer > 1)
        advance |= (p == 8) & (self.timer > 20)
        advance |= (p == 9) & (right_err < 0.01)
        advance |= (p == 10) & reached_r
        advance |= (p == 11) & (self.timer > 20)
        stuck = self.timer > 150
        moved = advance | stuck
        self.phase = torch.where(moved, (p + 1).clamp(max=12), p)
        self.timer = torch.where(moved, 0, self.timer)

        return torch.cat(
            (left_act, left_grip.unsqueeze(-1), right_act, right_grip.unsqueeze(-1)), dim=-1
        )


class SortScript:
    """Zone sorting oracle: one arm picks the cube in its source zone and sets it down in the target zone.

    Uses the simulator's true cube pose, so it checks that the task, success test and rewards admit a clean solution
    (binary gripper: positive opens, negative closes).
    """

    HOVER = 0.06
    CARRY = 0.07
    PINCH_LIFT = float(os.environ.get("BITIDY_PINCH", "0.001"))
    ROLL_SIGN = HandoverScript.ROLL_SIGN

    def __init__(self, env):
        from . import sort_env_cfg as sort

        self.env, self.sort = env, sort
        self.arm = env.cfg.actions.arm_joints.asset_name
        self.ik = ArmIK(env, self.arm)
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.timer = torch.zeros_like(self.phase)
        self.roll = torch.zeros(env.num_envs, device=env.device)
        # Integral term: the PD drives sag under gravity, so aim past the waypoint by the accumulated error.
        self.bias = torch.zeros((env.num_envs, 3), device=env.device)

    def reset(self, ids: torch.Tensor) -> None:
        self.phase[ids] = 0
        self.timer[ids] = 0
        self.bias[ids] = 0.0
        self.roll[ids] = 0.0

    ROLL_GAIN = float(os.environ.get("BITIDY_ROLL_GAIN", "0.3"))
    GAIN = float(os.environ.get("BITIDY_IK_GAIN", "1.0"))
    NEAR = float(os.environ.get("BITIDY_IK_NEAR", "0.04"))
    ACCUMULATE = os.environ.get("BITIDY_IK_ACC", "0") == "1"
    PHASE_LIMIT = int(os.environ.get("BITIDY_PHASE_LIMIT", "120"))

    def _roll_error(self) -> torch.Tensor:
        """Cube yaw minus the jaws' closing-axis yaw (top view), folded to (-45, 45] deg."""
        from isaaclab.utils.math import quat_apply

        robot = self.env.scene[self.arm]
        tips = []
        for body, offset in (("gripper", base.FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", base.MOVING_FINGERTIP_OFFSET)):
            i = robot.find_bodies(body)[0][0]
            pos, quat = robot.data.body_pos_w.torch[:, i], robot.data.body_quat_w.torch[:, i]
            tips.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
        axis = tips[1] - tips[0]
        jaw_yaw = torch.atan2(axis[:, 1], axis[:, 0])
        x, y, z, w = self.env.scene["cube"].data.root_quat_w.torch.unbind(-1)
        cube_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        half = torch.pi / 4
        return torch.remainder(cube_yaw - jaw_yaw + half, 2 * half) - half

    def act(self) -> torch.Tensor:
        env, p = self.env, self.phase
        cube = base.cube_pos(env)
        rest = base.CUBE_REST_Z
        goal = self.sort.target_xy(env)
        fresh = env.episode_length_buf <= 1
        self.start = torch.where(fresh.unsqueeze(-1), cube[:, :2], getattr(self, "start", cube[:, :2]))
        self.goal = torch.where(fresh.unsqueeze(-1), goal, getattr(self, "goal", goal))
        # One row per episode on its last step (all time out together, and the episode counter exits right then):
        # where the cube started, where it had to go, and the progress flags reached.
        if os.environ.get("BITIDY_ROWS"):
            last = (env.episode_length_buf == env.max_episode_length - 2).nonzero().squeeze(-1)
            prog = self.sort.progress(env)
            flags = torch.cat((prog.flags[last], prog.success[last].unsqueeze(-1)), -1).int()
            for s, g, f in zip(self.start[last].tolist(), self.goal[last].tolist(), flags.tolist()):
                print(f"BITIDY_ROW {s[0]:.4f} {s[1]:.4f} {g[0]:.3f} {g[1]:.3f} " + " ".join(map(str, f)), flush=True)

        def at(xy, z):
            return torch.cat((xy, torch.full_like(xy[:, :1], z)), -1)

        # Jaw roll follows the cube until it is gripped, then stays fixed.
        free = p <= 0
        # Turn the wrist until the jaws' closing axis is normal to a pair of cube faces (feedback on the measured
        # axis, so the tilted approach pose needs no roll calibration); hold the roll once the jaws start closing.
        aligning = p <= 2
        self.roll = torch.where(aligning, self.roll + self.ROLL_GAIN * self._roll_error(), self.roll).clamp(-1.6, 1.6)
        # Lock the grasp point once the descent starts, so a nudged cube is not chased across the table.
        self.grasp = torch.where(free.unsqueeze(-1), cube, getattr(self, "grasp", cube))
        g = self.grasp
        target = cube + cube.new_tensor((0.0, 0.0, self.HOVER))
        target = torch.where((p >= 1).unsqueeze(-1), g + g.new_tensor((0.0, 0.0, 0.03)), target)
        target = torch.where((p >= 2).unsqueeze(-1), g + g.new_tensor((0.0, 0.0, self.PINCH_LIFT)), target)
        target = torch.where((p >= 4).unsqueeze(-1), at(cube[:, :2], rest + self.CARRY), target)
        target = torch.where((p >= 5).unsqueeze(-1), at(goal, rest + self.CARRY), target)
        target = torch.where((p >= 6).unsqueeze(-1), at(goal, rest + self.PINCH_LIFT + 0.004), target)
        target = torch.where((p >= 8).unsqueeze(-1), at(goal, rest + self.HOVER + 0.03), target)
        grip = torch.where((p >= 3) & (p < 7), -1.0, 1.0)
        pad = base.pad_center(env, self.arm)
        # Integrate only near the waypoint, so the long first reach cannot wind the bias up.
        near = (torch.linalg.vector_norm(target - pad, dim=-1) < self.NEAR).unsqueeze(-1)
        self.bias = torch.where(near, self.bias + 0.05 * (target - pad), 0.9 * self.bias).clamp(-0.02, 0.02)
        act, _ = self.ik(target + self.bias, self.ik.tilted(target), self.roll, gain=self.GAIN, accumulate=self.ACCUMULATE)
        err = torch.linalg.vector_norm(target - pad, dim=-1)
        # Trace the environment furthest along (``BITIDY_DEBUG_ENV`` pins one).
        k = int(os.environ.get("BITIDY_DEBUG_ENV", -1))
        k = int(p.argmax()) if k < 0 else k
        if os.environ.get("BITIDY_DEBUG") and int(self.timer[k]) % 4 == 0:
            print(f"[sort] env={k} phase=[{int(p[k])}] err=[{float(err[k])}] pad={pad[k].tolist()} cube={cube[k].tolist()} "
                  f"goal={goal[k].tolist()} open={base.gripper_openness(env, self.arm)[k].item():.2f} "
                  f"held=[{bool(base.held_by(env, self.arm)[k])}] rollerr={float(self._roll_error()[k]):.3f}", flush=True)

        self.timer += 1
        reached = err < 0.006
        advance = torch.zeros_like(p, dtype=torch.bool)
        advance |= (p == 0) & reached
        advance |= (p == 1) & (err < 0.004)
        advance |= (p == 2) & (err < 0.004)
        advance |= (p == 3) & (self.timer > 15)
        advance |= (p == 4) & (err < 0.01)
        advance |= (p == 5) & (err < 0.01)
        advance |= (p == 6) & (err < 0.008)
        advance |= (p == 7) & (self.timer > 15)
        moved = advance | (self.timer > self.PHASE_LIMIT)
        if os.environ.get("BITIDY_ROWS"):
            # Step at which each phase was entered, and whether it was left by its condition or by the time limit.
            if not hasattr(self, "enter"):
                self.enter = torch.full((env.num_envs, 9), -1, dtype=torch.long, device=env.device)
                self.forced = torch.zeros((env.num_envs, 9), dtype=torch.long, device=env.device)
            fresh_ids = (env.episode_length_buf <= 1).nonzero().squeeze(-1)
            self.enter[fresh_ids] = -1
            self.enter[fresh_ids, 0] = 0
            self.forced[fresh_ids] = 0
            nxt = (p + 1).clamp(max=8)
            ids = (moved & (p < 8)).nonzero().squeeze(-1)
            self.enter[ids, nxt[ids]] = env.episode_length_buf[ids]
            fids = (moved & ~advance & (p < 8)).nonzero().squeeze(-1)
            self.forced[fids, p[fids]] = 1
            if int(env.episode_length_buf[0]) == int(env.max_episode_length) - 3:
                e = self.enter.float()
                dur = [((e[:, i + 1] - e[:, i])[(e[:, i + 1] >= 0) & (e[:, i] >= 0)]).mean().item() for i in range(8)]
                reach = [(e[:, i] >= 0).float().mean().item() for i in range(9)]
                frc = self.forced.float().mean(0).tolist()
                print("BITIDY_PHASES dur=" + str([round(d, 1) for d in dur]) + " reached=" + str([round(r, 2) for r in reach])
                      + " forced=" + str([round(f, 2) for f in frc]), flush=True)
        self.phase = torch.where(moved, (p + 1).clamp(max=8), p)
        self.timer = torch.where(moved, 0, self.timer)
        return torch.cat((act, grip.unsqueeze(-1)), -1)


def jaw_tips(env, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    """World positions of the fixed and moving fingertip pads."""
    from isaaclab.utils.math import quat_apply

    robot = env.scene[arm]
    tips = []
    for body, offset in (("gripper", base.FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", base.MOVING_FINGERTIP_OFFSET)):
        i = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, i], robot.data.body_quat_w.torch[:, i]
        tips.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return tips[0], tips[1]


def _wrap_half_pi(a: torch.Tensor) -> torch.Tensor:
    """Angle folded to (-90, 90] deg: a jaw closing axis has no sign."""
    return torch.remainder(a + torch.pi / 2, torch.pi) - torch.pi / 2


class HandoverSortScript:
    """S3 oracle: A to C (or C to A) with a mid-air handover.

    The giver (the arm on the source side) top-grasps the cube and holds it at ``HANDOVER``; the receiver comes down
    from above onto the other face pair; the giver lets go and backs off; the receiver carries the cube to the target
    zone and sets it down. Jaw roll is aligned by feedback on the measured closing axis.
    """

    HOVER = 0.06
    CARRY = 0.06
    HANDOVER = tuple(map(float, os.environ.get("BITIDY_HANDOVER_POINT", "0.10,0.0,0.10").split(",")))
    GIVER_PINCH = float(os.environ.get("BITIDY_GIVER_PINCH", "0.005"))
    RECV_GRIP = float(os.environ.get("BITIDY_RECV_GRIP", "-0.003"))
    APPROACH = 0.05
    TILT = float(os.environ.get("BITIDY_TILT", "0.7"))
    GIVER_FLIP = float(os.environ.get("BITIDY_GIVER_FLIP", "0"))
    RECV_FLIP = float(os.environ.get("BITIDY_RECV_FLIP", "0"))
    ROLL_GAIN = float(os.environ.get("BITIDY_ROLL_GAIN", "0.3"))
    RECV_ROLL_GAIN = float(os.environ.get("BITIDY_RECV_ROLL_GAIN", "0.3"))
    RECV_POINT = float(os.environ.get("BITIDY_RECV_POINT", "0.05"))
    GAIN = float(os.environ.get("BITIDY_IK_GAIN", "1.0"))
    PHASE_LIMIT = int(os.environ.get("BITIDY_PHASE_LIMIT", "120"))
    LAST = 14

    def __init__(self, env):
        from . import sort_env_cfg as sort

        self.env, self.sort = env, sort
        self.ik = {"robot_left": ArmIK(env, "robot_left"), "robot_right": ArmIK(env, "robot_right")}
        n, dev = env.num_envs, env.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.timer = torch.zeros_like(self.phase)
        self.roll = {a: torch.zeros(n, device=dev) for a in self.ik}
        self.bias = {a: torch.zeros((n, 3), device=dev) for a in self.ik}
        self.grasp = torch.zeros((n, 3), device=dev)
        self.place = torch.zeros((n, 3), device=dev)
        self.left_gives = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, ids: torch.Tensor) -> None:
        self.phase[ids] = 0
        self.timer[ids] = 0
        for a in self.ik:
            self.roll[a][ids] = 0.0
            self.bias[a][ids] = 0.0

    def _normals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Horizontal cube face normals: n1 nearest x (shared closing axis), n2 (receiver approach), unsigned."""
        x, y, z, w = self.env.scene["cube"].data.root_quat_w.torch.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        a = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), -1)
        b = torch.stack((-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)), -1)
        a_first = (a[:, 0].abs() >= b[:, 0].abs()).unsqueeze(-1)
        return torch.where(a_first, a, b), torch.where(a_first, b, a)

    def _axis(self, arm: str) -> torch.Tensor:
        fixed, moving = jaw_tips(self.env, arm)
        return torch.nn.functional.normalize(moving - fixed, dim=-1)

    def act(self) -> torch.Tensor:
        env, p = self.env, self.phase
        n = env.num_envs
        cube = base.cube_pos(env)
        rest = base.CUBE_REST_Z
        goal = self.sort.target_xy(env)
        fresh = env.episode_length_buf <= 1
        self.left_gives = torch.where(fresh, cube[:, 1] > 0, self.left_gives)
        lg = self.left_gives
        giver_arm = {True: "robot_left", False: "robot_right"}

        def col(v):
            return v.unsqueeze(-1)

        def at(xy, z):
            return torch.cat((xy, torch.full_like(xy[:, :1], z)), -1)

        # A 5-DOF SO-101 can only point its jaws within the vertical plane through its base, and only a tilted top
        # grasp reaches well. At the handover point the two arms' radial directions are perpendicular (it lies on the
        # circle whose diameter joins the bases), so with natural roll the giver's jaws close across its radial, on
        # one face pair, and the receiver's jaws close across its own radial, on the other, free face pair.
        handover = cube.new_tensor(self.HANDOVER).expand(n, -1)
        up = cube.new_tensor((0.0, 0.0, 1.0))
        g_base = torch.where(col(lg), self.ik["robot_left"].base_xy, self.ik["robot_right"].base_xy)
        r_base = torch.where(col(lg), self.ik["robot_right"].base_xy, self.ik["robot_left"].base_xy)

        def radial(base_xy, point):
            d = torch.nn.functional.normalize(point[:, :2] - base_xy, dim=-1)
            return torch.cat((d, torch.zeros_like(d[:, :1])), -1)

        dg = radial(g_base, handover)
        g_perp = torch.stack((-dg[:, 1], dg[:, 0], torch.zeros_like(dg[:, 0])), -1)

        self.grasp = torch.where(col(p <= 0), cube, self.grasp)
        g = self.grasp
        # Giver waypoints.
        gt = cube + up * self.HOVER
        gt = torch.where(col(p >= 1), g + up * 0.03, gt)
        gt = torch.where(col(p >= 2), g + up * self.GIVER_PINCH, gt)
        gt = torch.where(col(p >= 4), at(g[:, :2], rest + self.CARRY), gt)
        gt = torch.where(col(p >= 5), handover, gt)
        gt = torch.where(col(p >= 10), handover - dg * 0.05 + up * 0.06, gt)
        g_grip = torch.where((p >= 3) & (p < 9), -1.0, 1.0)
        # Receiver waypoints: a top grasp RECV_GRIP from the cube centre, coming down from above.
        # It waits high and back until the giver holds still at the handover point, then lines up over the cube.
        grip_pt = cube + up * self.RECV_GRIP
        rt = handover + up * 0.10 - radial(r_base, handover) * 0.08
        # Line up behind the cube along its own jaw axis, not straight above it, where the giver's jaws are.
        r_dir = self.ik["robot_right"].tilted(grip_pt, self.TILT) * col(lg).float() + self.ik["robot_left"].tilted(grip_pt, self.TILT) * col(~lg).float()
        rt = torch.where(col(p >= 6), grip_pt - r_dir * self.APPROACH, rt)
        rt = torch.where(col(p >= 7), grip_pt, rt)
        # While closing and while the giver lets go, the receiver holds where it gripped.
        self.place = torch.where(col(p <= 7), grip_pt, self.place)
        rt = torch.where(col((p >= 8) & (p <= 10)), self.place, rt)
        rt = torch.where(col(p >= 11), at(goal, rest + self.CARRY + self.RECV_GRIP), rt)
        rt = torch.where(col(p >= 12), at(goal, rest + 0.003 + self.RECV_GRIP), rt)
        rt = torch.where(col(p >= 14), at(goal, rest + self.HOVER + 0.03), rt)
        r_grip = torch.where((p >= 8) & (p < 13), -1.0, 1.0)

        # Roll feedback (top view). Giver: jaws onto the nearest face pair while grasping (p <= 2), then across its
        # radial at the handover point (p 4-6), turning the gripped cube. Receiver: onto the nearest face pair.
        x, y, z, w = env.scene["cube"].data.root_quat_w.torch.unbind(-1)
        cube_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        targets, dirs, grips, weights, errs = {}, {}, {}, {}, {}
        for arm in self.ik:
            is_giver = lg if arm == "robot_left" else ~lg
            axis = self._axis(arm)
            jaw_yaw = torch.atan2(axis[:, 1], axis[:, 0])
            face_err = torch.remainder(cube_yaw - jaw_yaw + torch.pi / 4, torch.pi / 2) - torch.pi / 4
            perp_err = _wrap_half_pi(torch.atan2(g_perp[:, 1], g_perp[:, 0]) - jaw_yaw)
            g_err = torch.where(p <= 2, face_err, perp_err)
            r_err = face_err
            turn = (p >= 4) & (p <= 6) & (os.environ.get("BITIDY_GIVER_TURN", "1") == "1")
            align = torch.where(is_giver, (p <= 2) | turn, (p >= 6) & (p <= 7))
            gain = torch.where(is_giver, self.ROLL_GAIN * g_err, self.ROLL_GAIN * r_err)
            self.roll[arm] = torch.where(align, self.roll[arm] + gain, self.roll[arm]).clamp(-1.6, 1.6)
            target = torch.where(col(is_giver), gt, rt)
            targets[arm] = target
            dirs[arm] = self.ik[arm].tilted(target, self.TILT)
            grips[arm] = torch.where(is_giver, g_grip, r_grip)
            weights[arm] = torch.full_like(p, 0.015, dtype=torch.float32)
            errs[arm] = (g_err, r_err)

        acts, err = [], {}
        for arm in self.ik:
            pad = base.pad_center(env, arm)
            target = targets[arm]
            near = col(torch.linalg.vector_norm(target - pad, dim=-1) < 0.04)
            self.bias[arm] = torch.where(near, self.bias[arm] + 0.05 * (target - pad), 0.9 * self.bias[arm]).clamp(-0.02, 0.02)
            # Optional half turn of the wrist (same closing axis) to swing the wrist camera mount to the other side.
            is_giver = lg if arm == "robot_left" else ~lg
            flip = torch.where(is_giver, self.GIVER_FLIP * (p >= 4).float(), self.RECV_FLIP * torch.ones_like(p, dtype=torch.float32))
            a, _ = self.ik[arm](target + self.bias[arm], dirs[arm], self.roll[arm] + torch.pi * flip, gain=self.GAIN, point_weight=weights[arm])
            acts += [a, grips[arm].unsqueeze(-1)]
            err[arm] = torch.linalg.vector_norm(target - pad, dim=-1)
        g_e = torch.where(lg, err["robot_left"], err["robot_right"])
        r_e = torch.where(lg, err["robot_right"], err["robot_left"])

        k = int(os.environ.get("BITIDY_DEBUG_ENV", -1))
        k = int(p.argmax()) if k < 0 else k
        if os.environ.get("BITIDY_DEBUG") and int(self.timer[k]) % 4 == 0:
            ga, ra = ("robot_left", "robot_right") if bool(lg[k]) else ("robot_right", "robot_left")
            print(f"[hand] env={k} phase={int(p[k])} gerr={float(g_e[k]):.4f} rerr={float(r_e[k]):.4f} "
                  f"cube={[round(v, 3) for v in cube[k].tolist()]} gpad={[round(v, 3) for v in base.pad_center(env, ga)[k].tolist()]} "
                  f"rpad={[round(v, 3) for v in base.pad_center(env, ra)[k].tolist()]} groll={float(errs[ga][0][k]):.3f} "
                  f"rroll={float(errs[ra][1][k]):.3f} gheld={bool(base.held_by(env, ga)[k])} rheld={bool(base.held_by(env, ra)[k])} "
                  f"gopen={base.gripper_openness(env, ga)[k].item():.2f} ropen={base.gripper_openness(env, ra)[k].item():.2f}", flush=True)

        if os.environ.get("BITIDY_ROWS"):
            if not hasattr(self, "best"):
                self.best = torch.full((n, 2), 9.0, device=env.device)
            self.best[fresh] = 9.0
            at_hand = (p >= 5) & (p <= 7)
            self.best[:, 0] = torch.where(at_hand, torch.minimum(self.best[:, 0], g_e), self.best[:, 0])
            self.best[:, 1] = torch.where(p == 7, torch.minimum(self.best[:, 1], r_e), self.best[:, 1])
            if int(env.episode_length_buf[0]) == int(env.max_episode_length) - 3:
                print(f"BITIDY_HANDOVER_ERR giver_min={self.best[:, 0].median():.4f} recv_min={self.best[:, 1].median():.4f}", flush=True)
        self.timer += 1
        adv = torch.zeros_like(p, dtype=torch.bool)
        adv |= (p == 0) & (g_e < 0.006)
        adv |= (p == 1) & (g_e < 0.004)
        adv |= (p == 2) & (g_e < 0.004)
        adv |= (p == 3) & (self.timer > 15)
        adv |= (p == 4) & (g_e < 0.01)
        adv |= (p == 5) & (g_e < 0.01) & (self.timer > 20)
        adv |= (p == 6) & (r_e < 0.006)
        adv |= (p == 7) & (r_e < 0.004)
        adv |= (p == 8) & (self.timer > 15)
        adv |= (p == 9) & (self.timer > 15)
        adv |= (p == 10) & (g_e < 0.02)
        adv |= (p == 11) & (r_e < 0.01)
        adv |= (p == 12) & (r_e < 0.008)
        adv |= (p == 13) & (self.timer > 15)
        moved = adv | (self.timer > self.PHASE_LIMIT)
        if os.environ.get("BITIDY_ROWS"):
            if not hasattr(self, "reached"):
                self.reached = torch.zeros((n, self.LAST + 1), dtype=torch.bool, device=env.device)
                self.forced = torch.zeros_like(self.reached)
            fresh_ids = fresh.nonzero().squeeze(-1)
            self.reached[fresh_ids] = False
            self.forced[fresh_ids] = False
            self.reached[torch.arange(n, device=env.device), p] = True
            fids = (moved & ~adv).nonzero().squeeze(-1)
            self.forced[fids, p[fids]] = True
            if int(env.episode_length_buf[0]) == int(env.max_episode_length) - 3:
                print("BITIDY_PHASES reached=" + str([round(v, 2) for v in self.reached.float().mean(0).tolist()])
                      + " forced=" + str([round(v, 2) for v in self.forced.float().mean(0).tolist()]), flush=True)
        self.phase = torch.where(moved, (p + 1).clamp(max=self.LAST), p)
        self.timer = torch.where(moved, 0, self.timer)
        return torch.cat(acts, -1)


class Lid3Script:
    """BiLid3 oracle: the left arm takes the named box's lid off and holds it aside, the right arm drops the cube in,
    the left arm puts the lid back. Each arm runs its own waypoint machine; they wait on each other at two points.

    Left: 0 over knob, 1 down to knob, 2 close, 3 lift, 4 aside, 5 hold until the cube is in and the right arm is
    clear, 6 back over box, 7 lower onto box, 8 open, 9 back off.
    Right: 0 over cube, 1 above, 2 down, 3 close, 4 lift, 5 wait for the lid to be aside, 6 over box, 7 open
    (the cube drops in), 8 back off.
    """

    HOVER = 0.03
    PINCH = 0.001
    KNOB_PINCH = 0.002
    NEAR = float(os.environ.get("BITIDY_IK_NEAR", "0.08"))
    LIFT = 0.05
    ASIDE = 0.075
    ASIDE_UP = 0.06
    DROP_ABOVE = 0.035
    ROLL_GAIN = float(os.environ.get("BITIDY_ROLL_GAIN", "0.3"))
    GAIN = float(os.environ.get("BITIDY_IK_GAIN", "1.0"))
    TILT = float(os.environ.get("BITIDY_TILT", "0.7"))
    PHASE_LIMIT = int(os.environ.get("BITIDY_PHASE_LIMIT", "120"))

    def __init__(self, env):
        from . import lid3_env_cfg as lid3

        self.env, self.L = env, lid3
        self.ik = {"robot_left": ArmIK(env, "robot_left"), "robot_right": ArmIK(env, "robot_right")}
        n, dev = env.num_envs, env.device
        self.pl = torch.zeros(n, dtype=torch.long, device=dev)
        self.pr = torch.zeros_like(self.pl)
        self.tl = torch.zeros_like(self.pl)
        self.tr = torch.zeros_like(self.pl)
        self.roll = {a: torch.zeros(n, device=dev) for a in self.ik}
        self.bias = {a: torch.zeros((n, 3), device=dev) for a in self.ik}
        self.grasp = torch.zeros((n, 3), device=dev)

    def reset(self, ids: torch.Tensor) -> None:
        for t in (self.pl, self.pr, self.tl, self.tr):
            t[ids] = 0
        for a in self.ik:
            self.roll[a][ids] = 0.0
            self.bias[a][ids] = 0.0

    @staticmethod
    def _yaw(quat: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _face_err(self, arm: str, yaw: torch.Tensor) -> torch.Tensor:
        fixed, moving = jaw_tips(self.env, arm)
        axis = moving - fixed
        jaw = torch.atan2(axis[:, 1], axis[:, 0])
        return torch.remainder(yaw - jaw + torch.pi / 4, torch.pi / 2) - torch.pi / 4

    def _drive(self, arm: str, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = base.pad_center(self.env, arm)
        # Near full reach the arm sags, so the integral term works from further out than in the single-arm oracle.
        near = (torch.linalg.vector_norm(target - pad, dim=-1) < self.NEAR).unsqueeze(-1)
        self.bias[arm] = torch.where(near, self.bias[arm] + 0.05 * (target - pad), 0.9 * self.bias[arm]).clamp(-0.03, 0.03)
        act, _ = self.ik[arm](target + self.bias[arm], self.ik[arm].tilted(target, self.TILT), self.roll[arm], gain=self.GAIN)
        return act, torch.linalg.vector_norm(target - pad, dim=-1)

    def act(self) -> torch.Tensor:
        env, L = self.env, self.L
        pl, pr = self.pl, self.pr
        n = env.num_envs
        up = base.cube_pos(env).new_tensor((0.0, 0.0, 1.0))

        def col(v):
            return v.unsqueeze(-1)

        box = L.box_center(env)
        # Lock the knob once the descent starts, so a nudged lid is not chased.
        self.knob = torch.where(col(pl <= 0), L.knob_pos(env), getattr(self, "knob", L.knob_pos(env)))
        knob = self.knob
        box3 = torch.cat((box, torch.full((n, 1), L.LID_REST_Z + L.KNOB_OFFSET, device=env.device)), -1)
        toward_left = torch.nn.functional.normalize(self.ik["robot_left"].base_xy - box, dim=-1)
        aside = box3 + torch.cat((toward_left * self.ASIDE, torch.zeros_like(box[:, :1])), -1) + up * self.ASIDE_UP

        # Left arm: the named lid.
        lt = knob + up * self.HOVER
        lt = torch.where(col(pl >= 1), knob + up * self.KNOB_PINCH, lt)
        lt = torch.where(col(pl >= 3), box3 + up * (self.LIFT + self.KNOB_PINCH), lt)
        lt = torch.where(col(pl >= 4), aside + up * self.KNOB_PINCH, lt)
        lt = torch.where(col(pl >= 6), box3 + up * (self.LIFT + self.KNOB_PINCH), lt)
        lt = torch.where(col(pl >= 7), box3 + up * (self.KNOB_PINCH + 0.002), lt)
        lt = torch.where(col(pl >= 9), box3 + up * (self.LIFT + 0.03), lt)
        l_grip = torch.where((pl >= 2) & (pl < 8), -1.0, 1.0)
        lid_yaw = self._yaw(L.lid_quat(env))
        self.roll["robot_left"] = torch.where(pl <= 1, self.roll["robot_left"] + self.ROLL_GAIN * self._face_err("robot_left", lid_yaw),
                                              self.roll["robot_left"]).clamp(-1.6, 1.6)

        # Right arm: the cube.
        cube = base.cube_pos(env)
        self.grasp = torch.where(col(pr <= 0), cube, self.grasp)
        g = self.grasp
        rest = base.CUBE_REST_Z
        carry_z = L.TABLE_TOP + L.WALL_HEIGHT + 0.045
        over_box = torch.cat((box, torch.full((n, 1), L.TABLE_TOP + L.WALL_HEIGHT + self.DROP_ABOVE, device=env.device)), -1)
        rt = cube + up * self.HOVER
        rt = torch.where(col(pr >= 1), g + up * 0.03, rt)
        rt = torch.where(col(pr >= 2), g + up * self.PINCH, rt)
        rt = torch.where(col(pr >= 4), torch.cat((g[:, :2], torch.full((n, 1), carry_z, device=env.device)), -1), rt)
        rt = torch.where(col(pr >= 6), over_box, rt)
        retreat = torch.cat((g[:, :2], torch.full((n, 1), carry_z, device=env.device)), -1)
        rt = torch.where(col(pr >= 8), retreat, rt)
        r_grip = torch.where((pr >= 3) & (pr < 7), -1.0, 1.0)
        self.roll["robot_right"] = torch.where(pr <= 2, self.roll["robot_right"] + self.ROLL_GAIN * self._face_err("robot_right", self._yaw(env.scene["cube"].data.root_quat_w.torch)),
                                               self.roll["robot_right"]).clamp(-1.6, 1.6)

        l_act, le = self._drive("robot_left", lt)
        r_act, re = self._drive("robot_right", rt)
        right_clear = torch.linalg.vector_norm(base.pad_center(env, "robot_right")[:, :2] - box, dim=-1) > 0.07

        k = int(os.environ.get("BITIDY_DEBUG_ENV", 0))
        if os.environ.get("BITIDY_DEBUG") and int(env.episode_length_buf[k]) % 8 == 0:
            print(f"[lid3] env={k} t={int(env.episode_length_buf[k])} target={L.NAMES[int(L.target_index(env)[k])]} pl={int(pl[k])} pr={int(pr[k])} "
                  f"le={float(le[k]):.4f} re={float(re[k]):.4f} knob={[round(v, 3) for v in knob[k].tolist()]} "
                  f"lpad={[round(v, 3) for v in base.pad_center(env, 'robot_left')[k].tolist()]} cube={[round(v, 3) for v in cube[k].tolist()]} "
                  f"rpad={[round(v, 3) for v in base.pad_center(env, 'robot_right')[k].tolist()]} lidheld={bool(L.lid_held(env)[k])} "
                  f"clear={bool(L.opening_clear(env)[k])} tilt={float(L.lid_tilt(env)[k]):.2f} rheld={bool(base.held_by(env, 'robot_right')[k])} "
                  f"flags={L.progress(env).flags[k].int().tolist()}", flush=True)

        self.tl += 1
        self.tr += 1
        al = torch.zeros_like(pl, dtype=torch.bool)
        al |= (pl == 0) & (le < 0.006)
        al |= (pl == 1) & (le < 0.004)
        al |= (pl == 2) & (self.tl > 15)
        al |= (pl == 3) & (le < 0.01)
        al |= (pl == 4) & (le < 0.01)
        al |= (pl == 5) & (pr >= 8) & right_clear
        al |= (pl == 6) & (le < 0.008)
        al |= (pl == 7) & (le < 0.005)
        al |= (pl == 8) & (self.tl > 15)
        ar = torch.zeros_like(pr, dtype=torch.bool)
        ar |= (pr == 0) & (re < 0.006)
        ar |= (pr == 1) & (re < 0.004)
        ar |= (pr == 2) & (re < 0.004)
        ar |= (pr == 3) & (self.tr > 15)
        ar |= (pr == 4) & (re < 0.01)
        ar |= (pr == 5) & (pl >= 5)
        ar |= (pr == 6) & (re < 0.008)
        # Never let go of the cube anywhere but over the opening.
        stuck_over = (pr == 6) & (self.tr > self.PHASE_LIMIT) & (re > 0.015)
        ar |= (pr == 7) & (self.tr > 15)
        # Waits never time out; motions do.
        waiting_l, waiting_r = pl == 5, pr == 5
        ml = al | ((self.tl > self.PHASE_LIMIT) & ~waiting_l)
        mr = ar | ((self.tr > self.PHASE_LIMIT) & ~waiting_r & ~stuck_over)
        if os.environ.get("BITIDY_ROWS"):
            if not hasattr(self, "reach_l"):
                self.reach_l = torch.zeros((n, 10), dtype=torch.bool, device=env.device)
                self.reach_r = torch.zeros((n, 9), dtype=torch.bool, device=env.device)
                self.force_l = torch.zeros_like(self.reach_l)
                self.force_r = torch.zeros_like(self.reach_r)
            fresh = (env.episode_length_buf <= 1).nonzero().squeeze(-1)
            for t in (self.reach_l, self.reach_r, self.force_l, self.force_r):
                t[fresh] = False
            idx = torch.arange(n, device=env.device)
            self.reach_l[idx, pl] = True
            self.reach_r[idx, pr] = True
            fl = (ml & ~al).nonzero().squeeze(-1)
            fr = (mr & ~ar).nonzero().squeeze(-1)
            self.force_l[fl, pl[fl]] = True
            self.force_r[fr, pr[fr]] = True
            if int(env.episode_length_buf[0]) == int(env.max_episode_length) - 3:
                r2 = lambda t: [round(v, 2) for v in t.float().mean(0).tolist()]
                print(f"BITIDY_PHASES left reached={r2(self.reach_l)} forced={r2(self.force_l)} | right reached={r2(self.reach_r)} forced={r2(self.force_r)}", flush=True)
        self.pl = torch.where(ml, (pl + 1).clamp(max=9), pl)
        self.pr = torch.where(mr, (pr + 1).clamp(max=8), pr)
        self.tl = torch.where(ml, 0, self.tl)
        self.tr = torch.where(mr, 0, self.tr)
        return torch.cat((l_act, l_grip.unsqueeze(-1), r_act, r_grip.unsqueeze(-1)), -1)


class ReachScript:
    """Reach map: every environment sends both arms (tilted top-grasp pose, same IK as the oracles) to one random point
    and prints the remaining error after ``REACH_STEPS`` steps, to choose layouts both arms can actually serve."""

    REACH_STEPS = int(os.environ.get("BITIDY_REACH_STEPS", "400"))
    TILT = float(os.environ.get("BITIDY_TILT", "0.7"))

    def __init__(self, env):
        self.env = env
        self.ik = {"robot_left": ArmIK(env, "robot_left"), "robot_right": ArmIK(env, "robot_right")}
        n, dev = env.num_envs, env.device
        lo = torch.tensor([float(v) for v in os.environ.get("BITIDY_REACH_LO", "0.02,-0.22,0.05").split(",")], device=dev)
        hi = torch.tensor([float(v) for v in os.environ.get("BITIDY_REACH_HI", "0.32,0.22,0.16").split(",")], device=dev)
        self.pw = float(os.environ.get("BITIDY_REACH_PW", "0.015"))
        self.goal = lo + (hi - lo) * torch.rand((n, 3), device=dev)
        self.bias = {a: torch.zeros((n, 3), device=dev) for a in self.ik}
        self.t = 0

    def reset(self, ids) -> None:
        pass

    def act(self) -> torch.Tensor:
        # One arm per environment (even: left, odd: right); the other holds its start pose, so they never meet.
        if not hasattr(self, "home"):
            self.home = {a: base.pad_center(self.env, a).clone() for a in self.ik}
            self.left_turn = torch.arange(self.env.num_envs, device=self.env.device) % 2 == 0
        acts, errs = [], []
        for arm in self.ik:
            pad = base.pad_center(self.env, arm)
            active = self.left_turn if arm == "robot_left" else ~self.left_turn
            goal = torch.where(active.unsqueeze(-1), self.goal, self.home[arm])
            near = (torch.linalg.vector_norm(goal - pad, dim=-1) < 0.08).unsqueeze(-1)
            self.bias[arm] = torch.where(near, self.bias[arm] + 0.05 * (goal - pad), 0.9 * self.bias[arm]).clamp(-0.03, 0.03)
            a, _ = self.ik[arm](goal + self.bias[arm], self.ik[arm].tilted(goal, self.TILT), None, gain=1.0, point_weight=self.pw)
            acts += [a, torch.ones_like(a[:, :1])]
            errs.append(torch.where(active, torch.linalg.vector_norm(goal - pad, dim=-1), float("nan")))
        self.t += 1
        if self.t == self.REACH_STEPS:
            for g, el, er in zip(self.goal.tolist(), errs[0].tolist(), errs[1].tolist()):
                print(f"BITIDY_REACH {g[0]:.3f} {g[1]:.3f} {g[2]:.3f} {el:.4f} {er:.4f}", flush=True)
        return torch.cat(acts, -1)


class RandomScript:
    """Uniform random actions: the baseline the oracle's return must clearly beat."""

    def __init__(self, env):
        self.env = env
        self.dim = env.action_manager.total_action_dim

    def reset(self, ids) -> None:
        pass

    def act(self) -> torch.Tensor:
        return torch.rand((self.env.num_envs, self.dim), device=self.env.device) * 2 - 1


SCRIPTS = {"handover": HandoverScript, "sort": SortScript, "sort_handover": HandoverSortScript, "lid3": Lid3Script, "reach": ReachScript, "random": RandomScript}



def install_scripted() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import evaluation

    if not os.environ.get("BITIDY_VIDEO"):
        evaluation.install_episode_counter()
    else:
        from . import render

        render.install_recorder()
    inner_step = RslRlVecEnvWrapper.step
    demo_path = os.environ.get("BITIDY_DEMO")
    state: dict = {"obs": [], "act": [], "done": [], "succ": []}

    def step(self, actions):
        env = self.unwrapped
        if "script" not in state:
            state["script"] = _script_class(os.environ.get("BITIDY_SCRIPT", "handover"))(env)
            state["last_obs"] = self.get_observations()["policy"].clone()
            state["ret"] = torch.zeros(env.num_envs, device=env.device)
            state["returns"] = []
        script = state["script"]
        scripted = script.act()
        if demo_path:
            state["obs"].append(state["last_obs"].cpu())
            state["act"].append(scripted.cpu())
        result = inner_step(self, scripted)
        obs, dones = result[0], result[2].bool()
        # Episode returns, to compare the oracle against random actions.
        state["ret"] += result[1]
        # Finished episodes and how many succeeded (the termination reset stores each episode's final flags).
        if bool(dones.any()) and hasattr(env, "_bi_final"):
            state["episodes"] = state.get("episodes", 0) + int(dones.sum())
            state["wins"] = state.get("wins", 0) + int(env._bi_final[dones, -1].sum())
            print(f"BITIDY_DONE episodes={state['episodes']} successes={state['wins']} "
                  f"envs={dones.nonzero().squeeze(-1).tolist()[:8]} success={env._bi_final[dones, -1].int().tolist()[:8]}", flush=True)
        if bool(dones.any()):
            state["returns"].append(state["ret"][dones].cpu())
            state["ret"][dones] = 0.0
            done_returns = torch.cat(state["returns"])
            if len(state["returns"]) % 20 == 0:
                print(f"BITIDY_RETURN mean={done_returns.mean():.2f} median={done_returns.median():.2f} n={len(done_returns)}", flush=True)
        state["last_obs"] = obs["policy"].clone()
        if demo_path:
            state["done"].append(dones.cpu())
            state["succ"].append(env._bi_final[:, -1].cpu())
            if len(state["act"]) % 50 == 0:
                _save(demo_path, state)
        done_ids = dones.nonzero().squeeze(-1)
        if len(done_ids):
            script.reset(done_ids)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def _save(path: str, state: dict) -> None:
    np.savez_compressed(
        path,
        obs=torch.stack(state["obs"]).numpy(),
        act=torch.stack(state["act"]).numpy(),
        done=torch.stack(state["done"]).numpy(),
        success=torch.stack(state["succ"]).numpy(),
    )


def _script_class(name: str):
    """A name from ``SCRIPTS``, or ``module:Class`` for an oracle in its own module of this package."""
    if ":" not in name:
        return SCRIPTS[name]
    import importlib

    module, cls = name.split(":")
    return getattr(importlib.import_module(f"{__package__}.{module}"), cls)
