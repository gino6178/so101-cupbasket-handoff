"""Lid sort (LidSort): three lidded boxes and three coloured cubes; a command (cube i, box j) -- "put the red cube into
box B" -- says which cube goes into which box.

DEFAULT LAYOUT (sorting station, ``BITIDY_LS_FACING_X`` = 0.39): the arms face each other across the table, left base at
(-0.05, 0) facing +x, right base at (0.39, 0) facing -x (0.39 rather than 0.36: the right arm grasps and lifts far
better with the cube slots ~23-28 cm away). Three bigger boxes (6 cm inside, 4 cm walls, 2 mm apart) stand
in a straight row along y between the arms (row centre ``ROW_X``); each episode the boxes A/B/C are shuffled over the
three row positions, and the whole row is shifted by up to ``ROW_JITTER`` in x / y and turned by up to ``ROW_YAW``.
The three cubes are shuffled over ``CUBE_SLOTS`` beside the -y row end (each jittered by ``CUBE_SLOT_JITTER``; all
22-29 cm from the right base, where the right arm can lift a cube).

Oracle (``oracle_lid_sort``, set-down mode): the LEFT arm takes box j's lid off by its knob and sets it down on the
table beyond a row end, the RIGHT arm picks cube i, carries it along the row just in front of the other lids' knobs
(over their plates only) and drops it into box j, then backs out; the left arm picks the lid up again and puts it back
on box j. The other two lids and the other two cubes must stay where they are.
Full sorting = chaining commands (``voice_sort.py``).

``BITIDY_LS_FACING_X=0``: the previous side-by-side layout (arm bases at (-0.05, +-``BASE_Y``), small boxes at random
spots / slots, knob turned to the left arm), with its own geometry defaults; ``oracle_lid_sort.LidSortScript`` then
runs the previous oracle.

Built on the verified LidPlace halves (``lid_open_env_cfg`` / ``box_place_env_cfg``): same milestone flags (plus two
"disturbed" flags). :func:`target_view` wraps the env so that ``scene["lid"]`` / ``scene["cube"]`` resolve to the
commanded lid / cube of each environment. Boxes are kinematic rigid bodies (4 walls), moved at reset.

Command: uniform over the 3 x 3 (cube, box) pairs; ``BITIDY_LS_CMD=i,j`` fixes it.

Success: as LidPlace, for the commanded lid / cube / box, plus no other lid moved more than ``OTHER_LID_TOL`` and no
other cube more than ``OTHER_CUBE_TOL`` during the episode.

v2 (``BITIDY_LS_V2=1``; v1 unchanged without it): TAKE-OUT commands (``TAKEOUT``: cube i starts in box j with the lid
on; goal: cube i on the table in the cube region outside every box, lid j back on), random cube starts, partly sorted
starts, displacement (``BITIDY_LS_DISP`` none / small / medium / full: box row pose, arm base poses) and pose-estimation
noise on the policy obs. Command obs 6 -> 7 (mode bit), plus the arm base offsets (6): policy obs +7 dims.
For take-out the milestone flags keep their slots: cube_lift = lifted above the rim, cube_over = carried clear of box
j, cube_in = placed on the table in the region (``cube_placed_out``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import actions_cfg as _acfg
from isaaclab.envs.mdp.actions import binary_joint_actions as _bja
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    WORKSHOP_CONTACT_MATERIAL,
    _bind_workshop_contact_material,
)

from . import box_place_env_cfg as bp
from . import env_cfg as base
from . import lid_open_env_cfg as lo
from .env_cfg import CUBE_REST_Z, CUBE_SIZE, TABLE_TOP, held_by, pad_center


def _pair(name: str, default: str) -> tuple[float, float]:
    a, b = (float(v) for v in os.environ.get(name, default).split(","))
    return a, b


# ---------------------------------------------------------------- layout

FACING_X = float(os.environ.get("BITIDY_LS_FACING_X", "0.39"))
"""Right arm base x when the arms face each other (the default sorting-station layout); 0 = side-by-side layout."""
FACING = FACING_X > 0
BASE_Y = float(os.environ.get("BITIDY_LS_BASE_Y", "0.10"))
"""Side-by-side layout only: arm bases at (-0.05, +-BASE_Y)."""
N = 3
BOX_X = _pair("BITIDY_LS_BOX_X", "0.10,0.22")
BOX_Y = _pair("BITIDY_LS_BOX_Y", "-0.05,0.07")
BOX_MIN_DIST = float(os.environ.get("BITIDY_LS_BOX_DIST", "0.085"))
CUBE_X = _pair("BITIDY_LS_CUBE_X", "0.14,0.21")
CUBE_Y = _pair("BITIDY_LS_CUBE_Y", "-0.19,-0.10")
CUBE_MIN_DIST = float(os.environ.get("BITIDY_LS_CUBE_DIST", "0.06"))
CUBE_BOX_DIST = float(os.environ.get("BITIDY_LS_CUBE_BOX_DIST", "0.07"))
PATH_CLEAR = float(os.environ.get("BITIDY_LS_PATH_CLEAR", "0.0"))
FIXED_CMD = os.environ.get("BITIDY_LS_CMD")
# Fallback layout (used only if rejection sampling finds nothing for an env).
DEFAULT_BOXES = ((0.16, -0.07), (0.16, 0.0), (0.16, 0.07)) if FACING else ((0.13, -0.02), (0.23, -0.01), (0.17, 0.09))
DEFAULT_CUBES = ((0.20, -0.20), (0.24, -0.25), (0.26, -0.19)) if FACING else ((0.12, -0.15), (0.175, -0.16), (0.23, -0.15))

KNOB_PLATE_YAW = float(os.environ.get("BITIDY_LS_KNOB_YAW", "0.0" if FACING else "-0.57"))
"""Yaw of the knob on its plate. Facing layout: 0 (square to the plate; the left jaws, pointing straight down, are
rolled to it). Side-by-side: each lid is turned so its knob faces the left arm (see ``KNOB_FACE_OFFSET``)."""
LEFT_BASE = (-0.05, 0.0) if FACING else (-0.05, BASE_Y)
RIGHT_BASE = (FACING_X, 0.0) if FACING else (-0.05, -BASE_Y)
KNOB_FACE_OFFSET = lo.KNOB_YAW - math.atan2(lo.BOX[1] - 0.15, lo.BOX[0] + 0.05)
"""Knob yaw minus the base->box azimuth in the verified LidOpen layout (box (0.14, 0), left base (-0.05, 0.15)):
the jaws' level closing axis over a box keeps this offset from the azimuth of the box seen from the left base."""

# Box / lid geometry. Facing layout: bigger boxes (6 cm inside, 4 cm walls) for a 2.5 cm cube; the 6.4 cm lid plate
# rests 2 mm onto each 4 mm wall. Side-by-side layout: the previous small, low boxes.
_GEO = {"inner": ("0.056", "0.034"), "wall_h": ("0.040", "0.030"), "lid": ("0.060", "0.044"), "knob_h": ("0.015", "0.012"),
        "wall": ("0.006", "0.004")}


def _geo(var: str, key: str) -> float:
    return float(os.environ.get(var, _GEO[key][0 if FACING else 1]))


BOX_INNER = _geo("BITIDY_LS_BOX_INNER", "inner")
WALL = _geo("BITIDY_LS_WALL", "wall")
WALL_HEIGHT = _geo("BITIDY_LS_WALL_HEIGHT", "wall_h")
WALL_GAP = 0.001
LID_SIZE = (_geo("BITIDY_LS_LID", "lid"),) * 2 + (float(os.environ.get("BITIDY_LS_LID_T", "0.004")),)
KNOB_FACETS = int(os.environ.get("BITIDY_LS_KNOB_FACETS", "4" if FACING else "1"))
"""Facing layout: the knob is a round-ish post, ``KNOB_FACETS`` squares turned by 90/KNOB_FACETS deg each (its width
across any horizontal direction varies by ~2 %), so the jaws pinch it equally well from any yaw. 1: a square knob."""
_KNOB_W = 0.016 if KNOB_FACETS == 1 else 0.016 / math.sqrt(2)
KNOB_SIZE = (_KNOB_W, _KNOB_W, _geo("BITIDY_LS_KNOB_H", "knob_h"))
"""Square knob 16 x 16 mm (as LidOpen's); round-ish knob: ~16 mm across."""
KNOB_OFFSET = LID_SIZE[2] / 2 + KNOB_SIZE[2] / 2
RIM_Z = TABLE_TOP + WALL_HEIGHT
LID_REST_Z = RIM_Z + LID_SIZE[2] / 2
KNOB_TOP_Z = LID_REST_Z + LID_SIZE[2] / 2 + KNOB_SIZE[2]
ASIDE_CLEAR = float(os.environ.get("BITIDY_LS_ASIDE_CLEAR", "0.06" if FACING else "0.056"))
"""Lid centre this far (xy) from the box centre: its footprint clears the opening."""
CLEAR_XY = float(os.environ.get("BITIDY_LS_CLEAR_XY", "0.06" if FACING else "0.05"))
"""Right jaw tips at least this far (xy) from the box centre count as clear of the box (so the lid can go back)."""
CUBE_LIFT = bp.LIFT
TOP_MARGIN = float(os.environ.get("BITIDY_LS_TOP_MARGIN", "0.002"))
"""The cube's highest corner must be this far below the rim (as BoxPlace)."""
KNOB_HOLD_RADIUS, LIFT_HEIGHT, LEVEL_TILT = lo.KNOB_HOLD_RADIUS, lo.LIFT_HEIGHT, lo.LEVEL_TILT
SQUARE_LIDS = os.environ.get("BITIDY_LS_SQUARE_LIDS", "1") == "1"
"""Lids rest square to their boxes (yaw jitter only); the knob is turned on the plate by ``KNOB_PLATE_YAW``."""
LID_YAW_JITTER = float(os.environ.get("BITIDY_LS_LID_YAW", ("0.02" if FACING else "0.03") if SQUARE_LIDS else str(lo.LID_YAW)))
LID_JITTER = float(os.environ.get("BITIDY_LS_LID_JITTER", "0.0015" if FACING else str(lo.LID_JITTER)))
"""Lid offset on its box, uniform in +- this (x and y). Facing layout: neighbouring plates stay >= 2 mm apart."""
OTHER_LID_TOL = 0.01
OTHER_CUBE_TOL = 0.02
V2 = os.environ.get("BITIDY_LS_V2", "0") == "1"
"""LidSort v2 (one switch for the new defaults): random cube starts (``CUBE_RANDOM``, inner radius 0.23), partly sorted
starts (``PRESORT`` 0.5), take-out commands (``TAKEOUT`` 0.3) and the mode bit in the command observation. Every
setting can still be overridden by its own variable. Off (default): the v1 task, unchanged (6-dim command)."""
PRESORT = float(os.environ.get("BITIDY_LS_PRESORT", "0.5" if V2 else "0"))
"""Share of resets that start "partly sorted" (as a voice chain's later commands see the scene): 1 or 2 of the
non-commanded cubes already sit inside non-commanded boxes (one per box), those boxes' lids back on with
``PRESORT_LID_JITTER`` / ``PRESORT_LID_YAW`` offsets (as a lid put back by the arm). 0 (default): normal starts only."""
PRESORT_TWO = float(os.environ.get("BITIDY_LS_PRESORT_TWO", "0.5"))
"""Of the presorted starts, the share with both other cubes already sorted (else one)."""
PRESORT_LID_JITTER = float(os.environ.get("BITIDY_LS_PRESORT_LID_JITTER", "0.004"))
PRESORT_LID_YAW = float(os.environ.get("BITIDY_LS_PRESORT_LID_YAW", "0.06"))
PRESORT_CUBE_JITTER = float(os.environ.get("BITIDY_LS_PRESORT_CUBE_JITTER", "0.006"))
INBOX_U = float(os.environ.get("BITIDY_LS_INBOX_U", "0.0"))
"""Cubes starting inside a box sit this far from its centre along the row normal (u, + toward the right base)."""
INBOX_YAW = float(os.environ.get("BITIDY_LS_INBOX_YAW", str(math.pi / 4)))
"""Cubes starting inside a box: yaw uniform in +- this about the row's (default pi/4: any yaw, as on the table)."""
TAKEOUT = float(os.environ.get("BITIDY_LS_TAKEOUT", "0.3" if V2 else "0"))
"""Share of resets with a TAKE-OUT command (cube i starts inside box j, lid closed; goal: cube i resting on the table
in the cube region outside every box, lid j back on, nothing else moved). Else put-in (cube i into box j).
``BITIDY_LS_CMD=i,j[,m]`` fixes the command (m = 1: take-out)."""
MODE_BIT = os.environ.get("BITIDY_LS_MODE_BIT", "1" if (V2 or TAKEOUT > 0) else "0") == "1"
"""Append the command mode (0 put-in, 1 take-out) to ``command_onehot``: 6 -> 7 dims (policy obs +1)."""
OUT_BOX_DIST = float(os.environ.get("BITIDY_LS_OUT_BOX_DIST", "0.055"))
"""Take-out: the cube centre this far (xy) from every box centre counts as outside the boxes (outer half-width 3.4 cm
+ cube half-diagonal 1.8 cm)."""
OUT_REGION_MARGIN = float(os.environ.get("BITIDY_LS_OUT_MARGIN", "0.03"))
"""Take-out: the cube must end inside the random-cube region (``CUBE_RX`` x ``CUBE_RY``) grown by this margin."""
STABLE_STEPS = 10
EPISODE_S = float(os.environ.get("BITIDY_LID_SORT_EPISODE_S", ("100" if V2 else "75") if FACING else "60"))
"""Facing layout: 75 s (lid off and set down, cube carried over the row, lid picked up again and put back)."""
FLAGS = ("lid_grasp", "lid_lifted", "lid_aside", "cube_grasp", "cube_lift", "cube_over", "cube_in", "right_clear",
         "lid_back", "other_lid_moved", "other_cube_moved")
CUBE_COLORS = ((0.9, 0.12, 0.12), (0.1, 0.75, 0.2), (0.12, 0.3, 0.95))
"""cube 0 red, cube 1 green, cube 2 blue."""
BOX_COLORS = ((0.95, 0.85, 0.2), (0.65, 0.35, 0.85), (0.2, 0.8, 0.85))
"""box 0 (A) yellow, box 1 (B) purple, box 2 (C) cyan; each lid has its box's colour (a little darker)."""


# ---------------------------------------------------------------- per-env target view


def _state(env: ManagerBasedRLEnv) -> dict:
    if not hasattr(env, "_ls"):
        n, dev = env.num_envs, env.device
        env._ls = {
            "cmd": torch.zeros((n, 3), dtype=torch.long, device=dev),
            "lid_start": torch.zeros((n, N, 3), device=dev),
            "cube_start": torch.zeros((n, N, 3), device=dev),
            "box_yaw": torch.zeros((n, N), device=dev),
            "prev_lid": torch.zeros((n, N, 3), device=dev),
            "prev_cube": torch.zeros((n, N, 3), device=dev),
            "final_cmd": torch.zeros((n, 3), dtype=torch.long, device=dev),
            "seen": torch.zeros(n, dtype=torch.bool, device=dev),
            "stats": torch.zeros((N, N, 2), dtype=torch.long, device=dev),
            "mstats": torch.zeros((2, N, N, 2), dtype=torch.long, device=dev),  # per mode (put-in, take-out)
            "funnel": torch.zeros((2, len(FLAGS) + 2), dtype=torch.long, device=dev),  # per mode: n, flag counts, success
            "fallback": 0,
            "sampled": 0,
        }
    return env._ls


def command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n, 3) long: commanded cube index, box index, mode (0 put cube i into box j, 1 take cube i out of box j)."""
    real = env.unwrapped if isinstance(env, TargetView) else env
    return _state(real)["cmd"]


def takeout(env) -> torch.Tensor:
    """(n,) bool: the command is a take-out."""
    return command(env)[:, 2] == 1


def box_centers(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n, 3, 2) box centres in the env frame."""
    o = env.scene.env_origins[:, :2]
    return torch.stack([env.scene[f"box{j}"].data.root_pos_w.torch[:, :2] - o for j in range(N)], 1)


def _objs_pos(env: ManagerBasedRLEnv, kind: str) -> torch.Tensor:
    o = env.scene.env_origins
    return torch.stack([env.scene[f"{kind}{j}"].data.root_pos_w.torch - o for j in range(N)], 1)


def _objs_quat(env: ManagerBasedRLEnv, kind: str) -> torch.Tensor:
    return torch.stack([env.scene[f"{kind}{j}"].data.root_quat_w.torch for j in range(N)], 1)


def _pick(stacked: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return stacked[torch.arange(stacked.shape[0], device=stacked.device), idx]


def target_box(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _pick(box_centers(env), command(env)[:, 1])


def target_box_yaw(env) -> torch.Tensor:
    """(n,) yaw of the commanded box (the whole row's yaw in the facing layout; 0 side by side)."""
    real = env.unwrapped if isinstance(env, TargetView) else env
    return _pick(_state(real)["box_yaw"], command(real)[:, 1])


def in_box_frame(env, xy: torch.Tensor) -> torch.Tensor:
    """xy (n, 2) relative to the commanded box's centre, in the box's own axes."""
    d = xy - box_center(env)
    yaw = target_box_yaw(env)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]), -1)


class _Arr:
    __slots__ = ("torch",)

    def __init__(self, t: torch.Tensor):
        self.torch = t


class _Data:
    def __init__(self, objs, idx_fn):
        self._objs, self._idx = objs, idx_fn

    def _get(self, name: str) -> _Arr:
        return _Arr(_pick(torch.stack([getattr(o.data, name).torch for o in self._objs], 1), self._idx()))

    root_pos_w = property(lambda self: self._get("root_pos_w"))
    root_quat_w = property(lambda self: self._get("root_quat_w"))
    root_lin_vel_w = property(lambda self: self._get("root_lin_vel_w"))
    root_ang_vel_w = property(lambda self: self._get("root_ang_vel_w"))


class _Selected:
    """Read-only stand-in for one rigid object: per env, the commanded one of three."""

    def __init__(self, objs, idx_fn):
        self.data = _Data(objs, idx_fn)


class _Scene:
    def __init__(self, env):
        self._env = env
        scene = env.scene
        self._lid = _Selected([scene[f"lid{j}"] for j in range(N)], lambda: command(env)[:, 1])
        self._cube = _Selected([scene[f"cube{j}"] for j in range(N)], lambda: command(env)[:, 0])

    def __getitem__(self, name: str):
        if name == "lid":
            return self._lid
        if name == "cube":
            return self._cube
        return self._env.scene[name]

    def __getattr__(self, name: str):
        return getattr(self._env.scene, name)


class TargetView:
    """The env as the single-box scripts and accessors see it: ``scene["lid"]`` / ``scene["cube"]`` are the commanded
    lid / cube, ``box_center`` (of ``lid_open_env_cfg`` and ``box_place_env_cfg``) is the commanded box."""

    def __init__(self, env):
        self.__dict__["_env"] = env
        self.__dict__["scene"] = _Scene(env)

    @property
    def unwrapped(self):
        return self._env

    @property
    def _ls_target_box(self) -> torch.Tensor:
        return target_box(self._env)

    def __getattr__(self, name: str):
        return getattr(self.__dict__["_env"], name)


def placer_view(env) -> TargetView:
    """A separate view for the cube placer: its ``box_center`` may be the carry via point (the lid opener's not)."""
    view = TargetView(env)
    view.__dict__["placer"] = True
    return view


def target_view(env) -> TargetView:
    if isinstance(env, TargetView):
        return env
    if not hasattr(env, "_ls_view"):
        env._ls_view = TargetView(env)
    return env._ls_view


# ---------------------------------------------------------------- target accessors (real env or view in; commanded objects)


def _v(env) -> TargetView:
    return target_view(env)


def box_center(env):
    """The commanded box's centre; for the placer script (handed a ``TargetView``) a carry via point may stand in
    (``_state(env)["via"]``, set by the oracle while it routes the cube in front of the box row first)."""
    if isinstance(env, TargetView):
        real = env.unwrapped
        st = _state(real)
        if "via" in st and env.__dict__.get("placer"):
            return torch.where(st["via_on"].unsqueeze(-1), st["via"], target_box(real))
        return target_box(real)
    return target_box(env)


def lid_pos(env):
    v = _v(env)
    return v.scene["lid"].data.root_pos_w.torch - v.scene.env_origins


def lid_quat(env):
    return _v(env).scene["lid"].data.root_quat_w.torch


def lid_tilt(env):
    return lo._tilt(lid_quat(env))


def knob_pos(env):
    quat = lid_quat(env)
    return lid_pos(env) + quat_apply(quat, quat.new_tensor((0.0, 0.0, KNOB_OFFSET)).expand(quat.shape[0], -1))


def lid_held(env):
    """Left pads at the knob with the lid off the rim."""
    near = torch.linalg.vector_norm(pad_center(_v(env), "robot_left") - knob_pos(env), dim=-1) < KNOB_HOLD_RADIUS
    return near & (lid_pos(env)[:, 2] > LID_REST_Z + 0.01)


def lid_level(env):
    return lid_tilt(env) < LEVEL_TILT


def lid_lifted(env):
    return lid_held(env) & lid_level(env) & (lid_pos(env)[:, 2] > LID_REST_Z + LIFT_HEIGHT)


def lid_off_opening(env):
    return torch.linalg.vector_norm(lid_pos(env)[:, :2] - box_center(env), dim=-1) > ASIDE_CLEAR


def lid_aside(env):
    return lid_held(env) & lid_level(env) & lid_off_opening(env)


def lid_closed(env):
    pos = lid_pos(env)
    on_box = torch.linalg.vector_norm(pos[:, :2] - box_center(env), dim=-1) < 0.012
    return on_box & ((pos[:, 2] - LID_REST_Z).abs() < 0.006) & (lid_tilt(env) < 0.15)


def left_jaws_off(env):
    return torch.linalg.vector_norm(pad_center(_v(env), "robot_left") - knob_pos(env), dim=-1) > 0.03


def cube_pos(env):
    return base.cube_pos(_v(env))


def cube_held(env):
    """Held by the right jaws (take-out: by either arm's jaws; the oracle lifts it out of the box with the left)."""
    v = _v(env)
    return held_by(v, "robot_right") | (takeout(env) & held_by(v, "robot_left"))


def cube_over_opening(env, margin: float = 0.0):
    return (in_box_frame(env, cube_pos(env)[:, :2]).abs() < BOX_INNER / 2 - margin).all(-1)


def cube_inside(env):
    return cube_over_opening(env) & (cube_pos(env)[:, 2] < RIM_Z)


def cube_top_z(env):
    quat = _v(env).scene["cube"].data.root_quat_w.torch
    n = quat.shape[0]
    axes_z = torch.stack([quat_apply(quat, quat.new_tensor(a).expand(n, -1))[:, 2] for a in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))], -1)
    return cube_pos(env)[:, 2] + 0.5 * CUBE_SIZE * axes_z.abs().sum(-1)


def cube_resting_inside(env):
    speed = torch.linalg.vector_norm(_v(env).scene["cube"].data.root_lin_vel_w.torch, dim=-1)
    return cube_inside(env) & (cube_top_z(env) < RIM_Z - TOP_MARGIN) & (speed < 0.05)


def cube_placed_out(env):
    """Take-out goal: the commanded cube rests flat on the table inside the cube region, outside every box."""
    v = _v(env)
    c = cube_pos(env)
    speed = torch.linalg.vector_norm(v.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
    m = OUT_REGION_MARGIN
    region = ((c[:, 0] >= CUBE_RX[0] - m) & (c[:, 0] <= CUBE_RX[1] + m) & (c[:, 1] >= CUBE_RY[0] - m)
              & (c[:, 1] <= CUBE_RY[1] + m))
    off_boxes = (torch.linalg.vector_norm(c[:, None, :2] - box_centers(v.unwrapped), dim=-1) >= OUT_BOX_DIST).all(-1)
    return (c[:, 2] < CUBE_REST_Z + 0.004) & region & off_boxes & (speed < 0.05)


def cube_out_of_box(env):
    """Take-out: the cube's footprint is clear of box j's walls (xy)."""
    return torch.linalg.vector_norm(cube_pos(env)[:, :2] - box_center(env), dim=-1) >= OUT_BOX_DIST


def right_jaws_clear(env):
    v = _v(env)
    box = box_center(env)
    pts = (*bp.jaw_tips(v, "robot_right"), pad_center(v, "robot_right"))
    return torch.stack([torch.linalg.vector_norm(p[:, :2] - box, dim=-1) > CLEAR_XY for p in pts]).all(0)


class _Shim:
    """Module stand-in handed to the verified single-arm scripts (``LidOpenScript.T``, ``oracle_box_place.B``):
    their module's names, with LidSort's geometry and per-env commanded objects."""

    def __init__(self, module, **override):
        self.__dict__.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
        self.__dict__.update(override)


def lid_open_shim() -> _Shim:
    return _Shim(lo, box_center=box_center, lid_pos=lid_pos, lid_quat=lid_quat, lid_tilt=lid_tilt, knob_pos=knob_pos,
                 lid_held=lid_held, LID_REST_Z=LID_REST_Z, KNOB_OFFSET=KNOB_OFFSET, KNOB_YAW=KNOB_PLATE_YAW,
                 progress=progress)


def box_place_shim() -> _Shim:
    return _Shim(bp, box_center=box_center, RIM_Z=RIM_Z, BOX_INNER=BOX_INNER, CUBE_SIZE=CUBE_SIZE, progress=progress)


def _others(env, idx_col: int) -> torch.Tensor:
    """(n, 3) bool: True for the objects that are not commanded."""
    return torch.arange(N, device=env.device).unsqueeze(0) != command(env)[:, idx_col : idx_col + 1]


def other_lid_moved(env) -> torch.Tensor:
    d = torch.linalg.vector_norm(_objs_pos(env, "lid") - _state(env)["lid_start"], dim=-1)
    return ((d > OTHER_LID_TOL) & _others(env, 1)).any(-1)


def other_cube_moved(env) -> torch.Tensor:
    d = torch.linalg.vector_norm(_objs_pos(env, "cube") - _state(env)["cube_start"], dim=-1)
    return ((d > OTHER_CUBE_TOL) & _others(env, 0)).any(-1)


# ---------------------------------------------------------------- progress


class LidSortProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.lid_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cube_count = torch.zeros_like(self.lid_count)
        self.stable_count = torch.zeros_like(self.lid_count)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cmd = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
        # Diagnostics: milestone flags at the step another lid / cube was first disturbed (-1: never).
        self.olid_at = torch.full((num_envs, len(FLAGS) - 2), -1, dtype=torch.long, device=device)
        self.ocube_at = torch.full_like(self.olid_at, -1)
        # ... and where things were then: moved lid xyz, left pad, right pad, commanded cube (12).
        self.olid_info = torch.zeros((num_envs, 12), device=device)

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.lid_count[ids] = 0
        self.cube_count[ids] = 0
        self.stable_count[ids] = 0
        self.success[ids] = False
        self.olid_at[ids] = -1
        self.ocube_at[ids] = -1

    def update(self, env: ManagerBasedRLEnv) -> None:
        self.cmd = command(env).clone()
        self.lid_count = torch.where(lid_held(env), self.lid_count + 1, 0)
        lid_grasp = self.lid_count >= 2
        lifted = self.lid_grasp & lid_lifted(env)
        aside = self.lid_lifted & lid_aside(env)
        cube = cube_pos(env)
        held = cube_held(env)
        self.cube_count = torch.where(held, self.cube_count + 1, 0)
        cube_grasp = self.cube_count >= 2
        out = takeout(env)
        lid_off = lid_off_opening(env)
        above_rim = cube[:, 2] - CUBE_SIZE / 2 > RIM_Z
        # take-out: cube_lift = lifted above the rim, cube_over = carried out over the table, cube_in = placed there
        cube_lift = self.cube_grasp & held & torch.where(out, above_rim, cube[:, 2] > CUBE_REST_Z + CUBE_LIFT)
        over = self.cube_lift & held & torch.where(out, cube_out_of_box(env), cube_over_opening(env) & above_rim & lid_off)
        released = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube, dim=-1) > 0.03
        released &= ~out | (torch.linalg.vector_norm(pad_center(env, "robot_left") - cube, dim=-1) > 0.03)
        resting = torch.where(out, cube_placed_out(env), cube_resting_inside(env))
        cube_in = self.cube_over & resting & released & (lid_off | out)
        right_clear = self.cube_in & resting & right_jaws_clear(env)
        closed = lid_closed(env)
        back = self.lid_aside & closed
        lid_moved, cube_moved = other_lid_moved(env), other_cube_moved(env)
        stage = self.flags[:, :-2].long()
        first_move = (lid_moved & ~self.other_lid_moved).unsqueeze(-1)
        self.olid_at = torch.where(first_move, stage, self.olid_at)
        if os.environ.get("BITIDY_LS_LOG"):
            d = torch.linalg.vector_norm(_objs_pos(env, "lid") - _state(env)["lid_start"], dim=-1) * _others(env, 1)
            moved_lid = _pick(_objs_pos(env, "lid"), d.argmax(-1))
            info = torch.cat((moved_lid, pad_center(env, "robot_left"), pad_center(env, "robot_right"), cube), -1)
            self.olid_info = torch.where(first_move, info, self.olid_info)
        self.ocube_at = torch.where((cube_moved & ~self.other_cube_moved).unsqueeze(-1), stage, self.ocube_at)
        self.flags |= torch.stack((lid_grasp, lifted, aside, cube_grasp, cube_lift, over, cube_in, right_clear, back,
                                   lid_moved, cube_moved), -1)
        done = (self.lid_back & self.right_clear & closed & resting & left_jaws_off(env) & right_jaws_clear(env)
                & ~self.other_lid_moved & ~self.other_cube_moved)
        self.stable_count = torch.where(done, self.stable_count + 1, 0)
        self.success |= self.stable_count >= STABLE_STEPS


def progress(env: ManagerBasedRLEnv) -> LidSortProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = LidSortProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- layout sampling


def _seg_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
    return torch.linalg.vector_norm(a + t.unsqueeze(-1) * ab - p, dim=-1)


def _uniform(shape, lo_hi_x, lo_hi_y, dev) -> torch.Tensor:
    r = torch.rand((*shape, 2), device=dev)
    lo_ = torch.tensor((lo_hi_x[0], lo_hi_y[0]), device=dev)
    hi_ = torch.tensor((lo_hi_x[1], lo_hi_y[1]), device=dev)
    return lo_ + r * (hi_ - lo_)


def _min_pair(p: torch.Tensor) -> torch.Tensor:
    """Smallest pairwise distance among the 3 points of the last-but-one dim."""
    d = torch.linalg.vector_norm(p.unsqueeze(-2) - p.unsqueeze(-3), dim=-1)
    d = d + torch.eye(N, device=p.device) * 10.0
    return d.flatten(-2).min(-1).values


def _sequential(m: int, tries: int, xr, yr, min_dist: float, dev, avoid: torch.Tensor | None = None,
                avoid_dist: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """3 points per env, each drawn (``tries`` candidates, first valid kept) at least ``min_dist`` from the ones before
    and ``avoid_dist`` from every ``avoid`` point (m, k, 2). Returns points (m, 3, 2) and 'all found' (m,)."""
    pts = torch.zeros((m, N, 2), device=dev)
    found = torch.ones(m, dtype=torch.bool, device=dev)
    rows = torch.arange(m, device=dev)
    for i in range(N):
        cand = _uniform((m, tries), xr, yr, dev)  # (m, T, 2)
        ok = torch.ones((m, tries), dtype=torch.bool, device=dev)
        if i:
            ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - pts[:, None, :i], dim=-1) >= min_dist).all(-1)
        if avoid is not None:
            ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - avoid.unsqueeze(1), dim=-1) >= avoid_dist).all(-1)
        pts[:, i] = cand[rows, ok.float().argmax(-1)]
        found &= ok.any(-1)
    return pts, found


def sample_layout(n: int, dev, tries: int = 128, rounds: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Box and cube centres (n, 3, 2) satisfying the layout rules; also a bool (n,): fallback layout used."""
    boxes = torch.tensor(DEFAULT_BOXES, device=dev).expand(n, -1, -1).clone()
    cubes = torch.tensor(DEFAULT_CUBES, device=dev).expand(n, -1, -1).clone()
    todo = torch.ones(n, dtype=torch.bool, device=dev)
    others = torch.tensor([[k != j for k in range(N)] for j in range(N)], device=dev)  # [j, k]
    for _ in range(rounds):
        ids = todo.nonzero().squeeze(-1)
        if not len(ids):
            break
        m = len(ids)
        b, b_ok = _sequential(m, tries, BOX_X, BOX_Y, BOX_MIN_DIST, dev)
        c, c_ok = _sequential(m, tries, CUBE_X, CUBE_Y, CUBE_MIN_DIST, dev, avoid=b, avoid_dist=CUBE_BOX_DIST)
        # Every carry line cube i -> box j must clear every other box k.
        ci = c[:, :, None, None, :]  # (m, i, 1, 1, 2)
        bj = b[:, None, :, None, :]  # (m, 1, j, 1, 2)
        bk = b[:, None, None, :, :]  # (m, 1, 1, k, 2)
        sd = _seg_dist(bk, ci, bj)  # (m, i, j, k)
        clear = ((sd >= PATH_CLEAR) | ~others.view(1, 1, N, N)).flatten(1).all(-1)
        good = b_ok & c_ok & clear
        gids = ids[good]
        boxes[gids] = b[good]
        cubes[gids] = c[good]
        todo[gids] = False
    return boxes, cubes, todo


def _parse_slots(name: str, default: str):
    v = os.environ.get(name, default)
    return None if v in ("", "random") else tuple(tuple(float(c) for c in pt.split(",")) for pt in v.split(";"))


_BOX_SLOTS = os.environ.get("BITIDY_LS_BOX_SLOTS", "row" if FACING else "0.11,0.0;0.165,0.0;0.22,0.0")
BOX_SLOTS = "row" if _BOX_SLOTS == "row" else _parse_slots("BITIDY_LS_BOX_SLOTS", _BOX_SLOTS)
"""Box spots "x,y;x,y;x,y": each episode the boxes A/B/C are shuffled over them, each jittered by
``SLOT_JITTER_X`` / ``SLOT_JITTER_Y`` ("random": rejection-sampled layout instead)."""
CUBE_SLOTS = _parse_slots("BITIDY_LS_CUBE_SLOTS", "0.20,-0.20;0.24,-0.25;0.26,-0.19" if FACING else "0.165,-0.13;0.215,-0.115;0.155,-0.19")
SLOT_JITTER_X = float(os.environ.get("BITIDY_LS_SLOT_JITTER_X", "0.005"))
SLOT_JITTER_Y = float(os.environ.get("BITIDY_LS_SLOT_JITTER_Y", "0.01"))
CUBE_SLOT_JITTER = float(os.environ.get("BITIDY_LS_CUBE_SLOT_JITTER", "0.008" if FACING else "0.005"))


CUBE_RANDOM = os.environ.get("BITIDY_LS_CUBE_RANDOM", "1" if V2 else "0") == "1"
"""Facing layout: cube starts drawn at random in the right arm's grasp region instead of the 3 fixed slots."""
CUBE_RX = _pair("BITIDY_LS_CUBE_RX", "0.15,0.30")
CUBE_RY = _pair("BITIDY_LS_CUBE_RY", "-0.29,-0.14")
CUBE_RR = _pair("BITIDY_LS_CUBE_RR", "0.23,0.30" if V2 else "0.21,0.30")
"""Distance band from the right arm base (the right arm lifts well 22-29 cm out, poorly close to its base)."""
CUBE_R_MIN = float(os.environ.get("BITIDY_LS_CUBE_R_MIN", "0.05"))
"""Minimum centre distance between cubes (2.5 cm cubes, random yaw: >= 3.6 cm avoids overlap, 5 cm leaves the jaws room)."""
CUBE_R_BOX = float(os.environ.get("BITIDY_LS_CUBE_R_BOX", "0.075"))
"""Minimum distance from every box centre (box outer half-width 3.4 cm + cube half-diagonal + jaw room)."""


CUBE_R_SPOT = float(os.environ.get("BITIDY_LS_CUBE_R_SPOT", "0.07" if V2 else "0"))
"""v2: cubes at least this far from the spot where the oracle sets the -y end box's lid down (``SPOT0``)."""
SPOT0 = _pair("BITIDY_LS_SPOT0", "-0.035,-0.105")
"""(u, v) of that lid spot in the -y end box's frame (u = row normal toward the right base, v = along the row)."""


def lid_spot0(boxes: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """(n, 2) where the oracle sets the -y end box's lid down."""
    end = _pick(boxes, boxes[..., 1].argmin(-1))
    c, s = torch.cos(yaw), torch.sin(yaw)
    u, v = SPOT0
    return end + torch.stack((c * u - s * v, s * u + c * v), -1)


def _random_cubes(boxes: torch.Tensor, yaw: torch.Tensor | None = None, tries: int = 256) -> torch.Tensor:
    """Cube centres (n, 3, 2) drawn uniformly in CUBE_RX x CUBE_RY, kept to the CUBE_RR band around the right base,
    ``CUBE_R_MIN`` apart and ``CUBE_R_BOX`` from every box; falls back to the slots where nothing valid is found."""
    n, dev = boxes.shape[0], boxes.device
    pts = torch.zeros((n, N, 2), device=dev)
    found = torch.ones(n, dtype=torch.bool, device=dev)
    rows = torch.arange(n, device=dev)
    rb = torch.tensor(RIGHT_BASE, device=dev)
    for i in range(N):
        cand = _uniform((n, tries), CUBE_RX, CUBE_RY, dev)
        r = torch.linalg.vector_norm(cand - rb, dim=-1)
        ok = (r >= CUBE_RR[0]) & (r <= CUBE_RR[1])
        ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - boxes.unsqueeze(1), dim=-1) >= CUBE_R_BOX).all(-1)
        if CUBE_R_SPOT > 0:
            spot = lid_spot0(boxes, torch.zeros(n, device=dev) if yaw is None else yaw)
            ok &= torch.linalg.vector_norm(cand - spot.unsqueeze(1), dim=-1) >= CUBE_R_SPOT
        if i:
            ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - pts[:, None, :i], dim=-1) >= CUBE_R_MIN).all(-1)
        pts[:, i] = cand[rows, ok.float().argmax(-1)]
        found &= ok.any(-1)
    if not bool(found.all()):
        pts[~found] = _slots(CUBE_SLOTS, (CUBE_SLOT_JITTER, CUBE_SLOT_JITTER), int((~found).sum()), dev)
    return pts


ROW_X = float(os.environ.get("BITIDY_LS_ROW_X", "0.16"))
ROW_Y = float(os.environ.get("BITIDY_LS_ROW_Y", "0.0"))
ROW_GAP = float(os.environ.get("BITIDY_LS_ROW_GAP", "0.01"))
ROW_PITCH = BOX_INNER + 2 * WALL + ROW_GAP
"""Facing layout: box centres ``ROW_PITCH`` apart along the row (2 mm between neighbouring walls)."""
# Displacement (v2: no fixtures on the real robot, so the arms, the row and the cubes are never placed exactly).
# ``BITIDY_LS_DISP`` picks a preset (row xy, row yaw, base xy, base yaw); each value has its own variable to tune it.
_DISP = {"none": ("0.01", "0.05", "0", "0"), "small": ("0.01", "0.05", "0.005", "0.017"),
         "medium": ("0.02", "0.10", "0.01", "0.033"), "full": ("0.03", "0.15", "0.015", "0.05"),
         "wide": ("0.03", "0.15", "0.015", "0.05")}
DISP = os.environ.get("BITIDY_LS_DISP", "medium" if V2 else "none")
"""v2 default ``medium`` (``full`` cost put-in ~15 points of oracle success)."""
_d = _DISP[DISP]
ROW_JITTER = _pair("BITIDY_LS_ROW_JITTER", f"{_d[0]},{_d[0]}")
"""Whole-row shift, uniform in +- these (x, y), per episode (v1: 1 cm; v2 full: 3 cm)."""
ROW_YAW = float(os.environ.get("BITIDY_LS_ROW_YAW", _d[1]))
"""Whole-row yaw, uniform in +- this (rad), per episode (each box and lid turned with it; v1: 0.05; v2 full: 0.15)."""
BASE_JITTER = float(os.environ.get("BITIDY_LS_BASE_JITTER", _d[2]))
"""Each arm base shifted by uniform +- this in x and y at reset (articulation root pose; v2 full: 1.5 cm)."""
BASE_YAW = float(os.environ.get("BITIDY_LS_BASE_YAW", _d[3]))
_wide = DISP == "wide"
GAP_EXTRA = _pair("BITIDY_LS_GAP_EXTRA", "0,0.03" if _wide else "0,0")
"""Each gap between neighbouring boxes widened by uniform [lo, hi] (m), independently: the boxes are not evenly spaced."""
BOX_OFF = float(os.environ.get("BITIDY_LS_BOX_OFF", "0.015" if _wide else "0"))
"""Each box shifted across the row (toward / away from the arms) by uniform +- this (m), independently."""
BOX_YAW = float(os.environ.get("BITIDY_LS_BOX_YAW", "0.10" if _wide else "0"))
"""Each box (and its lid) turned by uniform +- this (rad) on top of the row yaw, independently."""
"""Each arm base turned by uniform +- this (rad) at reset (v2 full: 0.05)."""
BASE_OBS = os.environ.get("BITIDY_LS_BASE_OBS", "1" if V2 else "0") == "1"
"""Policy / critic obs get both arms' base offsets (dx, dy, dyaw each: +6 dims). Object poses stay in the env (table)
frame, so an unobserved base offset would not be learnable; on the real robot this is the measured base calibration."""
OBS_NOISE = float(os.environ.get("BITIDY_LS_OBS_NOISE", "0.004" if V2 else "0"))
"""Policy obs only (not the critic, not the oracle): Gaussian noise (std, m) on every observed object position
(box centres, lids, cubes, knob), as a stand-in for camera pose estimation."""
OBS_YAW_NOISE = float(os.environ.get("BITIDY_LS_OBS_YAW_NOISE", "0.05" if V2 else "0"))
"""... and on the observed lid / cube yaw (std, rad)."""
_ARM_YAW0 = {"robot_left": math.pi / 2, "robot_right": -math.pi / 2}


def _row(n: int, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Facing layout: box centres (n, 3, 2) of a row along y (boxes shuffled over the row positions, row shifted and
    turned as a whole; with ``GAP_EXTRA`` / ``BOX_OFF`` / ``BOX_YAW`` each box also moved and turned on its own), the
    row yaw (n,) and each box's yaw (n, 3)."""
    perm = torch.argsort(torch.rand((n, N), device=dev), dim=-1)  # box k -> row position perm[k]
    gaps = ROW_PITCH + GAP_EXTRA[0] + torch.rand((n, N - 1), device=dev) * (GAP_EXTRA[1] - GAP_EXTRA[0])
    slot = torch.cat((torch.zeros((n, 1), device=dev), gaps.cumsum(-1)), -1)  # (n, 3) row positions
    slot = slot - slot.mean(-1, keepdim=True)
    along = torch.gather(slot, 1, perm)  # (n, 3)
    across = (torch.rand((n, N), device=dev) * 2 - 1) * BOX_OFF
    yaw = (torch.rand(n, device=dev) * 2 - 1) * ROW_YAW
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    centre = torch.tensor((ROW_X, ROW_Y), device=dev) + (torch.rand((n, 2), device=dev) * 2 - 1) * torch.tensor(ROW_JITTER, device=dev)
    boxes = centre.unsqueeze(1) + torch.stack((-s * along + c * across, c * along + s * across), -1)
    box_yaw = yaw.unsqueeze(-1) + (torch.rand((n, N), device=dev) * 2 - 1) * BOX_YAW
    return boxes, yaw, box_yaw


def _slots(slots, jitter, n: int, dev) -> torch.Tensor:
    base_ = torch.tensor(slots, device=dev)  # (3, 2)
    perm = torch.argsort(torch.rand((n, N), device=dev), dim=-1)  # object k -> slot perm[k]
    return base_[perm] + (torch.rand((n, N, 2), device=dev) * 2 - 1) * torch.tensor(jitter, device=dev)


# ---------------------------------------------------------------- events


def _place_bases(env, ids: torch.Tensor) -> None:
    """Arm bases displaced (writes the fixed root joint's parent pose under Newton); offsets kept for the obs."""
    st = _state(env)
    n, dev = len(ids), env.device
    if "base_off" not in st:
        st["base_off"] = torch.zeros((env.num_envs, 6), device=dev)
    for a, (name, nominal) in enumerate((("robot_left", LEFT_BASE), ("robot_right", RIGHT_BASE))):
        dxy = (torch.rand((n, 2), device=dev) * 2 - 1) * BASE_JITTER
        dyaw = (torch.rand(n, device=dev) * 2 - 1) * BASE_YAW
        st["base_off"][ids, 3 * a: 3 * a + 2] = dxy
        st["base_off"][ids, 3 * a + 2] = dyaw
        if BASE_JITTER <= 0 and BASE_YAW <= 0:
            continue
        yaw = _ARM_YAW0[name] + dyaw
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = torch.tensor(nominal, device=dev) + dxy
        pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        pose[:, :3] += env.scene.env_origins[ids]
        env.scene[name].write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)


def base_offsets(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n, 6): left dx, dy, dyaw, right dx, dy, dyaw of the arm bases from their nominal poses."""
    st = _state(env)
    if "base_off" not in st:
        st["base_off"] = torch.zeros((env.num_envs, 6), device=env.device)
    return st["base_off"]


INBOX = PRESORT > 0 or TAKEOUT > 0
"""Some cubes may start inside boxes (presorted others, or the take-out cube)."""


def reset_objects(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: new layout, new command, boxes (kinematic) moved, lids settled on them knob toward the left arm,
    cubes flat with random yaw."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    n, dev = len(ids), env.device
    st = _state(env)
    # The termination term's reset (it logs the finished episode) runs after this event: keep the old layout for it.
    st["prev_lid"][ids] = st["lid_start"][ids]
    st["prev_cube"][ids] = st["cube_start"][ids]
    if BOX_SLOTS is None or CUBE_SLOTS is None:
        boxes, cubes, fell_back = sample_layout(n, dev)
    else:
        boxes = cubes = torch.zeros((n, N, 2), device=dev)
        fell_back = torch.zeros(n, dtype=torch.bool, device=dev)
    row_yaw = torch.zeros(n, device=dev)
    box_yaw = None
    if BOX_SLOTS == "row":  # facing layout: a straight row of boxes, shuffled, shifted and turned as a whole
        boxes, row_yaw, box_yaw = _row(n, dev)
    elif BOX_SLOTS is not None:  # slot layout: boxes / cubes shuffled over fixed slots, each jittered
        boxes = _slots(BOX_SLOTS, (SLOT_JITTER_X, SLOT_JITTER_Y), n, dev)
    if CUBE_SLOTS is not None:
        cubes = _slots(CUBE_SLOTS, (CUBE_SLOT_JITTER, CUBE_SLOT_JITTER), n, dev)
    if CUBE_RANDOM:
        cubes = _random_cubes(boxes, row_yaw)
    if os.environ.get("BITIDY_LS_PROBE"):  # capability map: one box / cube in play, the others parked far away
        px, py = _pair("BITIDY_LS_PROBE_X", "0.08,0.26"), _pair("BITIDY_LS_PROBE_Y", "-0.08,0.12")
        boxes[:, 0] = _uniform((n,), px, py, dev)
        boxes[:, 1] = torch.tensor((0.40, 0.30), device=dev)
        boxes[:, 2] = torch.tensor((0.40, -0.30), device=dev)
        cubes[:, 0] = _uniform((n,), CUBE_X, CUBE_Y, dev)
        cubes[:, 1] = torch.tensor((0.40, 0.04), device=dev)
        cubes[:, 2] = torch.tensor((0.40, -0.04), device=dev)
    if os.environ.get("BITIDY_LS_LAYOUT"):  # debugging: one fixed layout "bx0,by0,bx1,by1,bx2,by2,cx0,cy0,...,cy2"
        v = torch.tensor([float(x) for x in os.environ["BITIDY_LS_LAYOUT"].split(",")], device=dev).view(2, N, 2)
        boxes[:], cubes[:] = v[0], v[1]
        row_yaw[:] = float(os.environ.get("BITIDY_LS_LAYOUT_YAW", "0"))
        box_yaw = None
    if box_yaw is None:
        box_yaw = row_yaw.unsqueeze(-1).expand(-1, N).clone()
    st["box_yaw"][ids] = box_yaw
    if BASE_JITTER > 0 or BASE_YAW > 0 or BASE_OBS:
        _place_bases(env, ids)
    st["fallback"] += int(fell_back.sum())
    st["sampled"] += n
    if FIXED_CMD:
        fixed = [int(v) for v in FIXED_CMD.split(",")]
        st["cmd"][ids] = torch.tensor((fixed + [0])[:3], device=dev)
    else:
        st["cmd"][ids, :2] = torch.randint(0, N, (n, 2), device=dev)
        st["cmd"][ids, 2] = (torch.rand(n, device=dev) < TAKEOUT).long()
    origins = env.scene.env_origins[ids]
    zeros = torch.zeros((n, 6), device=dev)
    # Partly sorted starts (BITIDY_LS_PRESORT): in_box[r, i] = box that cube i already sits in (-1: in its slot).
    in_box = torch.full((n, N), -1, dtype=torch.long, device=dev)
    cmd = st["cmd"][ids]
    rows = torch.arange(n, device=dev)
    out = cmd[:, 2] == 1
    in_box[rows[out], cmd[out, 0]] = cmd[out, 1]  # take-out: the commanded cube starts inside box j
    if PRESORT > 0:
        pre = torch.rand(n, device=dev) < PRESORT
        two = torch.rand(n, device=dev) < PRESORT_TWO
        ar = torch.arange(N, device=dev).expand(n, -1)
        # the two other cubes / boxes, in random order
        oc = ar[ar != cmd[:, :1]].view(n, N - 1)
        ob = ar[ar != cmd[:, 1:2]].view(n, N - 1)
        oc = torch.where((torch.rand(n, device=dev) < 0.5).unsqueeze(-1), oc.flip(-1), oc)
        ob = torch.where((torch.rand(n, device=dev) < 0.5).unsqueeze(-1), ob.flip(-1), ob)
        in_box[rows[pre], oc[pre, 0]] = ob[pre, 0]
        both = pre & two
        in_box[rows[both], oc[both, 1]] = ob[both, 1]
        st["presorted"] = st.get("presorted", 0) + int(pre.sum())
    lid_over = (in_box.unsqueeze(-1) == torch.arange(N, device=dev).view(1, 1, N)).any(1)  # (n, box) holds a cube
    for j in range(N):
        byaw = box_yaw[:, j]
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = boxes[:, j]
        pose[:, 2] = TABLE_TOP
        pose[:, 5], pose[:, 6] = torch.sin(byaw / 2), torch.cos(byaw / 2)
        pose[:, :3] += origins
        env.scene[f"box{j}"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[f"box{j}"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)
        # Lid: knob square to the left arm's approach (yaw folded into +-45 deg: the plate and knob are square).
        az = torch.atan2(boxes[:, j, 1] - LEFT_BASE[1], boxes[:, j, 0] - LEFT_BASE[0])
        yaw = (byaw if SQUARE_LIDS else az + KNOB_FACE_OFFSET - KNOB_PLATE_YAW) + (torch.rand(n, device=dev) * 2 - 1) * LID_YAW_JITTER
        yaw = torch.remainder(yaw + math.pi / 4, math.pi / 2) - math.pi / 4
        pose = torch.zeros((n, 7), device=dev)
        if INBOX:  # a lid put back by the arm: larger offsets
            yaw = torch.where(lid_over[:, j], yaw + (torch.rand(n, device=dev) * 2 - 1) * PRESORT_LID_YAW, yaw)
        jit = torch.where(lid_over[:, j:j + 1], PRESORT_LID_JITTER, LID_JITTER)
        pose[:, :2] = boxes[:, j] + (torch.rand((n, 2), device=dev) * 2 - 1) * jit
        pose[:, 2] = LID_REST_Z
        pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        st["lid_start"][ids, j] = pose[:, :3]
        pose[:, :3] += origins
        env.scene[f"lid{j}"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[f"lid{j}"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)
        yaw = (torch.rand(n, device=dev) * 2 - 1) * (math.pi / 4)
        if INBOX and INBOX_YAW < math.pi / 4:  # in-box cubes: yaw within +-INBOX_YAW of the row
            in_yaw = _pick(box_yaw, in_box[:, j].clamp(min=0))
            yaw = torch.where(in_box[:, j] >= 0, in_yaw + (torch.rand(n, device=dev) * 2 - 1) * INBOX_YAW, yaw)
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = cubes[:, j]
        if INBOX:  # presorted / take-out cube: inside its box (no floor: it rests on the table between the walls)
            b = in_box[:, j]
            inside = _pick(boxes, b.clamp(min=0)) + (torch.rand((n, 2), device=dev) * 2 - 1) * PRESORT_CUBE_JITTER
            if INBOX_U != 0.0:  # in-box cubes shifted along the row normal (u, toward the right base)
                in_yaw = _pick(box_yaw, b.clamp(min=0))
                inside = inside + INBOX_U * torch.stack((torch.cos(in_yaw), torch.sin(in_yaw)), -1)
            pose[:, :2] = torch.where((b >= 0).unsqueeze(-1), inside, cubes[:, j])
        pose[:, 2] = CUBE_REST_Z
        pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        st["cube_start"][ids, j] = pose[:, :3]
        pose[:, :3] += origins
        env.scene[f"cube{j}"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[f"cube{j}"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)


# ---------------------------------------------------------------- observations


def _noise(t: torch.Tensor, on: bool) -> torch.Tensor:
    return t + torch.randn_like(t) * OBS_NOISE if on and OBS_NOISE > 0 else t


def _yaw_noise(q: torch.Tensor, on: bool) -> torch.Tensor:
    """Quaternions (..., 4, xyzw) turned about z by Gaussian noise (std ``OBS_YAW_NOISE``)."""
    if not on or OBS_YAW_NOISE <= 0:
        return q
    a = torch.randn(q.shape[:-1], device=q.device) * OBS_YAW_NOISE
    dq = torch.stack((torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2), torch.cos(a / 2)), -1)
    return quat_mul(dq.reshape(-1, 4), q.reshape(-1, 4)).reshape(q.shape)


def scene_state(env: ManagerBasedRLEnv, noisy: bool = False) -> torch.Tensor:
    """All box centres (6), all lid poses (3 x pos+quat = 21), all cube poses (21). ``noisy`` (policy obs): pose
    estimation noise (``OBS_NOISE`` / ``OBS_YAW_NOISE``)."""
    n = env.num_envs
    return torch.cat((_noise(box_centers(env), noisy).reshape(n, -1),
                      torch.cat((_noise(_objs_pos(env, "lid"), noisy), _yaw_noise(_objs_quat(env, "lid"), noisy)), -1).reshape(n, -1),
                      torch.cat((_noise(_objs_pos(env, "cube"), noisy), _yaw_noise(_objs_quat(env, "cube"), noisy)), -1).reshape(n, -1)), -1)


def command_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot cube command (3) then one-hot box command (3); with ``MODE_BIT`` (v2) also the mode (1: take cube i out
    of box j; put-in commands keep the v1 encoding plus a 0)."""
    cmd = command(env)
    parts = [torch.nn.functional.one_hot(cmd[:, 0], N), torch.nn.functional.one_hot(cmd[:, 1], N)]
    if MODE_BIT:
        parts.append(cmd[:, 2:3])
    return torch.cat(parts, -1).float()


def target_relative(env: ManagerBasedRLEnv, noisy: bool = False) -> torch.Tensor:
    """Commanded objects relative to the jaws / their goals, and their velocities (as LidPlace's relative + state).
    ``noisy`` (policy obs): object positions with pose estimation noise."""
    n = env.num_envs
    box = _noise(target_box(env), noisy)
    box_lid = torch.cat((box, torch.full((n, 1), LID_REST_Z, device=env.device)), -1)
    box_floor = torch.cat((box, torch.full((n, 1), TABLE_TOP, device=env.device)), -1)
    cube = _noise(cube_pos(env), noisy)
    v = target_view(env)
    return torch.cat((_noise(knob_pos(env), noisy) - pad_center(env, "robot_left"), _noise(lid_pos(env), noisy) - box_lid,
                      cube - pad_center(env, "robot_right"), cube - box_floor,
                      v.scene["lid"].data.root_lin_vel_w.torch, v.scene["cube"].data.root_lin_vel_w.torch), -1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards (for completeness; the task is for BC)


def reach_knob(env: ManagerBasedRLEnv) -> torch.Tensor:
    d = torch.linalg.vector_norm(pad_center(env, "robot_left") - knob_pos(env), dim=-1)
    return torch.where(progress(env).lid_lifted, 1.0, 1.0 - torch.tanh(d / 0.08))


def reach_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    d = torch.linalg.vector_norm(pad_center(env, "robot_right") - cube_pos(env), dim=-1)
    return torch.where(progress(env).cube_grasp, 1.0, 1.0 - torch.tanh(d / 0.08))


def carry_cube(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    above = torch.cat((target_box(env), torch.full((env.num_envs, 1), RIM_Z + 0.03, device=env.device)), -1)
    d = torch.linalg.vector_norm(cube_pos(env) - above, dim=-1)
    active = cube_held(env) & ~progress(env).cube_in & lid_off_opening(env) & ~takeout(env)
    return torch.where(active, torch.exp(-d / std), 0.0)


def close_lid(env: ManagerBasedRLEnv, std: float = 0.03) -> torch.Tensor:
    rest = torch.cat((target_box(env), torch.full((env.num_envs, 1), LID_REST_Z, device=env.device)), -1)
    d = torch.linalg.vector_norm(lid_pos(env) - rest, dim=-1)
    return torch.where(progress(env).right_clear, torch.exp(-d / std) * torch.exp(-lid_tilt(env) / 0.2), 0.0)


def disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (other_lid_moved(env) | other_cube_moved(env)).float()


class milestones(ManagerTermBase):
    WEIGHTS = (1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 3.0, 6.0, 0.0, 0.0)

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, len(FLAGS)), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        flags = progress(env).flags
        reward = ((flags & ~self.paid).float() * flags.new_tensor(self.WEIGHTS, dtype=torch.float32)).sum(-1)
        self.paid = flags.clone()
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).success.float()


# ---------------------------------------------------------------- terminations


class lid_sort_success(ManagerTermBase):
    """Latches the episode's flags into ``env._bi_final`` at reset (last column = success) and keeps per-command
    success counts of every environment's first episode (printed as ``BITIDY_LS_CMD`` lines)."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        progress(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env = self._env
        ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice)
               else torch.as_tensor(env_ids, device=env.device))
        p = progress(env)
        env._bi_final[ids] = torch.cat((p.flags[ids], p.success[ids].unsqueeze(-1)), dim=-1)
        st = _state(env)
        st["final_cmd"][ids] = p.cmd[ids]
        # First episodes only, and only ones that ran (the initial reset has episode_length 0).
        first = ids[~st["seen"][ids] & (env.episode_length_buf[ids] > 0)]
        if len(first):
            st["seen"][first] = True
            c = p.cmd[first]
            st["stats"][:, :, 0].index_put_((c[:, 0], c[:, 1]), torch.ones(len(first), dtype=torch.long, device=env.device), accumulate=True)
            st["stats"][:, :, 1].index_put_((c[:, 0], c[:, 1]), p.success[first].long(), accumulate=True)
            ones = torch.ones(len(first), dtype=torch.long, device=env.device)
            st["mstats"][..., 0].index_put_((c[:, 2], c[:, 0], c[:, 1]), ones, accumulate=True)
            st["mstats"][..., 1].index_put_((c[:, 2], c[:, 0], c[:, 1]), p.success[first].long(), accumulate=True)
            fin = torch.cat((ones.unsqueeze(-1), env._bi_final[first].long()), -1)
            st["funnel"].index_add_(0, c[:, 2], fin)
            if os.environ.get("BITIDY_LS_LOG"):
                fin = env._bi_final[first].int().cpu()
                lids, cubes_ = st["prev_lid"][first].cpu(), st["prev_cube"][first].cpu()
                for r, e in enumerate(first.tolist()):
                    print(f"BITIDY_LS_EP env={e} cmd={c[r].tolist()} boxes={[round(v, 3) for v in lids[r, :, :2].flatten().tolist()]} "
                          f"cubes={[round(v, 3) for v in cubes_[r, :, :2].flatten().tolist()]} flags={fin[r].tolist()} "
                          f"olid_at={p.olid_at[e].tolist()} ocube_at={p.ocube_at[e].tolist()} olid_info={[round(v, 3) for v in p.olid_info[e].tolist()]}", flush=True)
            s = st["stats"].cpu()
            per_box = [round(float(s[:, j, 1].sum()) / max(int(s[:, j, 0].sum()), 1), 3) for j in range(N)]
            per_cube = [round(float(s[i, :, 1].sum()) / max(int(s[i, :, 0].sum()), 1), 3) for i in range(N)]
            print(f"BITIDY_LS_CMD n={int(s[..., 0].sum())} per_box={per_box} per_cube={per_cube} "
                  f"n_box={[int(s[:, j, 0].sum()) for j in range(N)]} n_cube={[int(s[i, :, 0].sum()) for i in range(N)]} "
                  f"pairs={s[..., 1].tolist()}/{s[..., 0].tolist()} fallback={st['fallback']}/{st['sampled']}", flush=True)
            if TAKEOUT > 0 or FIXED_CMD:
                ms, fu = st["mstats"].cpu(), st["funnel"].cpu()
                for m, name in ((0, "put"), (1, "out")):
                    nm = max(int(ms[m, ..., 0].sum()), 1)
                    pb = [round(float(ms[m, :, j, 1].sum()) / max(int(ms[m, :, j, 0].sum()), 1), 3) for j in range(N)]
                    pc = [round(float(ms[m, i, :, 1].sum()) / max(int(ms[m, i, :, 0].sum()), 1), 3) for i in range(N)]
                    funnel = {k: round(float(fu[m, 1 + q]) / max(int(fu[m, 0]), 1), 3) for q, k in enumerate(list(FLAGS) + ["success"])}
                    print(f"BITIDY_LS_MODE mode={name} n={int(ms[m, ..., 0].sum())} success={round(float(ms[m, ..., 1].sum()) / nm, 3)} "
                          f"per_box={pb} per_cube={pc} n_box={[int(ms[m, :, j, 0].sum()) for j in range(N)]} funnel={funnel}", flush=True)
        p.reset(ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        p = progress(env)
        p.update(env)
        if os.environ.get("BITIDY_NO_SUCCESS_END"):
            return torch.zeros_like(p.success)
        return p.success.clone()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A lid or a cube left the table."""
    z = torch.cat((_objs_pos(env, "lid")[..., 2], _objs_pos(env, "cube")[..., 2]), -1)
    return z.min(-1).values < TABLE_TOP - 0.05


# ---------------------------------------------------------------- scene


@clone
def _spawn_box(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Open-top box (4 walls, no floor: it stands on the table) as one kinematic rigid body; origin at the bottom centre."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.2)
    h = BOX_INNER / 2 + WALL / 2
    walls = (("front", (h, 0.0), (WALL, BOX_INNER + 2 * WALL)), ("back", (-h, 0.0), (WALL, BOX_INNER + 2 * WALL)),
             ("left", (0.0, h), (BOX_INNER, WALL)), ("right", (0.0, -h), (BOX_INNER, WALL)))
    for name, (x, y), (sx, sy) in walls:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        # Walls float 1 mm over the table (rim height unchanged): no kinematic box / kinematic desk contacts.
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, (WALL_HEIGHT + WALL_GAP) / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, WALL_HEIGHT - WALL_GAP))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@clone
def _spawn_lid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """LidOpen's lid (plate + 16 mm cube knob, one rigid body) with the knob turned by ``cfg.knob_yaw`` on the plate."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass)
    parts = [("plate", LID_SIZE, (0.0, 0.0, 0.0), 0.0)]
    parts += [(f"knob{k}" if k else "knob", KNOB_SIZE, (0.0, 0.0, KNOB_OFFSET), cfg.knob_yaw + k * math.pi / 2 / KNOB_FACETS)
              for k in range(KNOB_FACETS)]
    for name, size, offset, yaw in parts:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
        xform.AddRotateZOp().Set(math.degrees(yaw))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@configclass
class LidSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_lid
    mass: float = 0.015
    color: tuple[float, float, float] = (0.85, 0.25, 0.25)
    knob_yaw: float = lo.KNOB_YAW


@configclass
class BoxSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_box
    color: tuple[float, float, float] = (0.95, 0.95, 0.95)


def _box(j: int) -> RigidObjectCfg:
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Box{j}", spawn=BoxSpawnCfg(color=BOX_COLORS[j]),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(*DEFAULT_BOXES[j], TABLE_TOP)))


def _lid(j: int) -> RigidObjectCfg:
    color = tuple(0.8 * c for c in BOX_COLORS[j])
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Lid{j}", spawn=LidSpawnCfg(color=color, knob_yaw=KNOB_PLATE_YAW),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(*DEFAULT_BOXES[j], LID_REST_Z)))


def _cube(j: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Cube{j}",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.02),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_COLORS[j]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*DEFAULT_CUBES[j], CUBE_REST_Z)),
    )


@configclass
class LidSortSceneCfg(base.BiSceneCfg):
    # Default: the arms face each other across the table (left at (-0.05, 0) facing +x, right at (FACING_X, 0) facing
    # -x), set here explicitly so the task does not depend on env_cfg.BITIDY_FACING_X. FACING_X 0: side by side.
    if FACING:
        robot_left = base._arm("robot_left", 0.0, x=LEFT_BASE[0])
        robot_right = base._arm("robot_right", 0.0, x=FACING_X, facing_back=True)
    else:
        robot_left = base._arm("robot_left", BASE_Y)
        robot_right = base._arm("robot_right", -BASE_Y)
    cube = None
    target_pad = None
    right_fixed_jaw_cube = None
    right_moving_jaw_cube = None
    box0 = _box(0)
    box1 = _box(1)
    box2 = _box(2)
    lid0 = _lid(0)
    lid1 = _lid(1)
    lid2 = _lid(2)
    cube0 = _cube(0)
    cube1 = _cube(1)
    cube2 = _cube(2)


# ---------------------------------------------------------------- config


LEFT_OPEN_DEG = float(os.environ.get("BITIDY_LS_LEFT_OPEN_DEG", "10" if V2 else "0"))
"""v2: the LEFT gripper's NARROW opening (deg; 0 = off). The default full opening, 14.6 deg (~4.6 cm between the
fingertips), has an outer span that does not fit between the 5.6 cm box walls; the left arm lifts take-out cubes out
of the boxes with the narrow opening."""
LEFT_OPEN_MODE = os.environ.get("BITIDY_LS_LEFT_OPEN_MODE", "two")
"""``two`` (default): two open levels on the left gripper action -- action >= ``NARROW_ABOVE`` (e.g. +1): full open
(14.6 deg, knob grasps); 0 <= action < ``NARROW_ABOVE`` (e.g. +0.25): ``LEFT_OPEN_DEG`` (inside a box); < 0: closed.
``fixed``: every open command is ``LEFT_OPEN_DEG`` (the previous v2 behaviour; the knob re-grasp slipped)."""
NARROW_ABOVE = 0.5


class TwoLevelGripperAction(_bja.BinaryJointPositionAction):
    """Binary gripper with a second, narrower open level for small non-negative actions (see ``LEFT_OPEN_MODE``)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._narrow_command = torch.full_like(self._open_command, cfg.narrow_open)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = actions.float()
        self._processed_actions = torch.where(
            a < 0, self._close_command, torch.where(a < self.cfg.narrow_above, self._narrow_command, self._open_command))


@configclass
class TwoLevelGripperActionCfg(_acfg.BinaryJointPositionActionCfg):
    class_type: type = TwoLevelGripperAction
    narrow_open: float = 0.0
    narrow_above: float = NARROW_ABOVE


@configclass
class LidSortActionsCfg:
    left_arm, left_gripper = base._arm_actions("robot_left")
    if LEFT_OPEN_DEG > 0 and LEFT_OPEN_MODE == "fixed":
        left_gripper = left_gripper.replace(open_command_expr={"gripper": math.radians(LEFT_OPEN_DEG)})
    elif LEFT_OPEN_DEG > 0:
        left_gripper = TwoLevelGripperActionCfg(
            asset_name="robot_left", joint_names=["gripper"], open_command_expr=dict(left_gripper.open_command_expr),
            close_command_expr=dict(left_gripper.close_command_expr), narrow_open=math.radians(LEFT_OPEN_DEG))
    right_arm, right_gripper = base._arm_actions("robot_right")


@configclass
class LidSortPolicyCfg(base.PolicyCfg):
    cube = ObsTerm(func=scene_state)
    relative = ObsTerm(func=target_relative)
    command = ObsTerm(func=command_onehot)
    progress = ObsTerm(func=progress_flags)


@configclass
class LidSortCriticCfg(LidSortPolicyCfg):
    """Privileged (noise-free) state; v2 also the arm base offsets."""

    base = ObsTerm(func=base_offsets) if BASE_OBS else None


@configclass
class LidSortNoisyPolicyCfg(LidSortCriticCfg):
    """v2 policy obs: object poses with pose-estimation noise (``OBS_NOISE``); v1 (no noise): identical to the critic."""

    cube = ObsTerm(func=scene_state, params={"noisy": True})
    relative = ObsTerm(func=target_relative, params={"noisy": True})


@configclass
class LidSortObservationsCfg:
    policy: LidSortPolicyCfg = LidSortNoisyPolicyCfg()
    critic: LidSortPolicyCfg = LidSortCriticCfg()


@configclass
class LidSortEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    objects = EventTerm(func=reset_objects, mode="reset")


@configclass
class LidSortRewardsCfg:
    reach_knob = RewTerm(func=reach_knob, weight=1.0)
    reach_cube = RewTerm(func=reach_cube, weight=1.0)
    carry = RewTerm(func=carry_cube, weight=0.5)
    close = RewTerm(func=close_lid, weight=0.5)
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    disturbed = RewTerm(func=disturbed, weight=-1.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    left_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot_right")})
    left_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_left")})
    right_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_right")})


@configclass
class LidSortTerminationsCfg:
    success = DoneTerm(func=lid_sort_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LidSortEnvCfg(base.BiHandoverEnvCfg):
    scene: LidSortSceneCfg = LidSortSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    actions: LidSortActionsCfg = LidSortActionsCfg()
    observations: LidSortObservationsCfg = LidSortObservationsCfg()
    events: LidSortEventsCfg = LidSortEventsCfg()
    rewards: LidSortRewardsCfg = LidSortRewardsCfg()
    terminations: LidSortTerminationsCfg = LidSortTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S
        # 3 lids on walls + 3 cubes: more contact rows than the single-box tasks' MJWarp buffers allow.
        for name in ("newton_mjwarp", "default"):
            physics = getattr(self.sim.physics, name, None)
            solver = getattr(physics, "solver_cfg", None)
            if solver is not None and hasattr(solver, "njmax"):
                solver.njmax = int(os.environ.get("BITIDY_LS_NJMAX", 1200 if FACING else 800))
                solver.nconmax = int(os.environ.get("BITIDY_LS_NCONMAX", 600 if FACING else 400))

    def play_mode(self):
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
