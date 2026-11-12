"""Cup / bowl sorting into open baskets with nesting, BOTH ARMS AT ONCE (BiTidy-CupBasket-SO101).

Scene (facing layout, as LidSort): left arm base (-0.05, 0) facing +x, right arm base (0.39, 0) facing -x. Three open
kinematic baskets A/B/C (``IN_L`` x ``IN_W`` inside, default 16 x 11 cm, 3 cm walls, thin floor) stand in a row along
y at x ~ ``ROW_X``, long axis along x: the CUP zone is the -x half, the BOWL zone the +x half. Six vessels start on the
table beside the row: objects 0-2 cups (red, green, blue), 3-5 bowls (red, green, blue).

Vessels are compound rigid bodies (``_spawn_vessel``): a ring of ``nseg`` tilted box wall segments + two crossed square
floor plates; same-type vessels nest physically. Nominal nesting step = wall / sin(taper): cups ~20 mm, bowls ~5.6 mm.

COMMAND = one instruction with up to two (object, basket) pairs, e.g. "紅碗放B，綠杯放A". The env assigns each pair to
an arm (nearest arm; if both objects are nearest the same arm, the 2 x 2 assignment with the smaller total distance), so
the command is stored per ARM SLOT: ``cmd[:, s] = (object, basket, active)``, slot 0 = left arm, slot 1 = right arm.
A single instruction leaves one slot inactive (``P_SINGLE``). ``BITIDY_CB_CMD=k0,j0,k1,j1`` fixes the pairs (k1 = -1:
single). Both arms work at the same time.

Success (joint): every active slot's vessel rests upright in its basket's zone of its type (nested on that zone's
stack if it holds same-type vessels: xy within ``NEST_XY`` of the stack axis, z in the nesting window, tilt <
``TILT_MAX``), no other vessel moved more than ``OTHER_TOL`` (the joined stacks excepted), all jaws clear of the
commanded vessels, for ``STABLE_STEPS``. Per-arm success is also latched (stats).

Chaining: ``_state(env)`` keeps per-object ``loc`` (basket or -1 = table) and ``level`` at reset; "partly done" starts
(``PRESORT``) put non-commanded vessels in baskets (stacked) so each commanded zone holds 0, 1 or 2 same-type vessels.

Randomization (``BITIDY_CB_DISP`` none / medium / wide, default wide; LidSort's wide scheme): basket order shuffled,
row shift / yaw, gaps widened, each basket offset / turned, arm bases displaced (offsets in the obs), random vessel
starts in the arms' reach bands (each commanded vessel in the band of the arm it is meant for), pose noise on the policy
obs. ``BITIDY_CB_OBJ=fixed``: the ``FIXED_SLOTS`` (no row / base randomization with ``BITIDY_CB_DISP=none``).
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import _bind_workshop_contact_material

from . import env_cfg as base
from . import servo
from .env_cfg import TABLE_TOP, pad_center

_JOINTS, _ARM_JOINTS = list(base.JOINTS), list(base.ARM_JOINTS)


def _f(name: str, default) -> float:
    return float(os.environ.get(name, default))


def _pair(name: str, default: str) -> tuple[float, float]:
    a, b = (float(v) for v in os.environ.get(name, default).split(","))
    return a, b


# ---------------------------------------------------------------- objects

N_OBJ, N_BASKET, N_SLOT = 6, 3, 2
CUP, BOWL = 0, 1
OBJ_TYPE = (CUP, CUP, CUP, BOWL, BOWL, BOWL)
OBJ_NAMES = ("red cup", "green cup", "blue cup", "red bowl", "green bowl", "blue bowl")
OBJ_COLORS = ((0.9, 0.12, 0.12), (0.1, 0.7, 0.2), (0.12, 0.3, 0.95)) * 2
"""Colour + type name an object: "紅杯" = 0, "綠杯" = 1, "藍杯" = 2, "紅碗" = 3, "綠碗" = 4, "藍碗" = 5."""
BASKET_NAMES = ("A", "B", "C")
BASKET_COLORS = ((0.86, 0.86, 0.82), (0.66, 0.5, 0.32), (0.32, 0.34, 0.38))
"""A light grey, B wood, C dark grey."""
ARMS = ("robot_left", "robot_right")


def _geo(prefix: str, r_top, r_bot, h, t, floor, mass, nseg) -> dict:
    g = {"r_top": _f(f"{prefix}_RTOP", r_top), "r_bot": _f(f"{prefix}_RBOT", r_bot), "h": _f(f"{prefix}_H", h),
         "t": _f(f"{prefix}_T", t), "floor": _f(f"{prefix}_FLOOR", floor), "mass": _f(f"{prefix}_MASS", mass),
         "nseg": int(_f(f"{prefix}_NSEG", nseg))}
    g["alpha"] = math.atan2(g["r_top"] - g["r_bot"], g["h"])  # wall taper from vertical
    g["z0"] = 0.5 * g["t"] * math.sin(g["alpha"]) + 0.0005  # wall bottoms just above the plate bottom
    g["rim"] = g["z0"] + g["h"]  # rim height above the object origin (bottom centre)
    g["pitch"] = g["t"] / math.sin(g["alpha"])  # nesting step, faces aligned
    g["pitch_max"] = g["pitch"] + g["r_top"] * (1.0 / math.cos(math.pi / g["nseg"]) - 1.0) / math.tan(g["alpha"])
    g["r_in"] = g["r_top"] - g["t"] / math.cos(g["alpha"])  # inner rim radius (apothem)
    return g


CUP_GEO = _geo("BITIDY_CB_CUP", 0.025, 0.018, 0.055, 0.0025, 0.003, 0.008, 16)
"""Cup: 50 mm top OD, 36 mm bottom OD, 55 mm tall, 2.5 mm walls (taper ~7.3 deg, nesting step ~20 mm), 8 g."""
BOWL_GEO = _geo("BITIDY_CB_BOWL", 0.040, 0.0225, 0.035, 0.0025, 0.003, 0.010, 16)
"""Bowl: 80 mm top OD, 45 mm bottom OD, 35 mm tall, 2.5 mm walls (taper ~26.6 deg, nesting step ~5.6 mm), 10 g."""
GEO = (CUP_GEO, BOWL_GEO)

# ---------------------------------------------------------------- baskets / layout

BASE_SEP = _f("BITIDY_CB_BASE_SEP", 0.44)
"""Distance between the two arm bases along x (default 0.44: left (-0.05, 0), right (0.39, 0))."""
BASE_MID = _f("BITIDY_CB_BASE_MID", 0.17)
FACING_X = _f("BITIDY_CB_FACING_X", round(BASE_MID + BASE_SEP / 2, 6))
"""Right arm base x (faces -x); left arm base x is ``BITIDY_CB_LEFT_X``. Both override BASE_SEP / BASE_MID."""
LEFT_BASE = (_f("BITIDY_CB_LEFT_X", round(BASE_MID - BASE_SEP / 2, 6)), 0.0)
BASE_SEP = FACING_X - LEFT_BASE[0]
RIGHT_BASE = (FACING_X, 0.0)
BASES = (LEFT_BASE, RIGHT_BASE)
_ARM_YAW0 = {"robot_left": math.pi / 2, "robot_right": -math.pi / 2}

IN_L = _f("BITIDY_CB_IN_L", 0.16)  # inside length (basket u axis, along x at yaw 0)
IN_W = _f("BITIDY_CB_IN_W", 0.11)  # inside width (v axis, along the row)
B_WALL = _f("BITIDY_CB_WALL", 0.003)
B_WALL_H = _f("BITIDY_CB_WALL_H", 0.03)
B_FLOOR = _f("BITIDY_CB_FLOOR", 0.003)
WALL_GAP = 0.001  # kinematic basket floats 1 mm over the kinematic desk
FLOOR_Z = TABLE_TOP + WALL_GAP + B_FLOOR
"""Top of the basket floor: a vessel in a basket rests with its origin here."""
B_TOP_Z = TABLE_TOP + B_WALL_H
ZONE_MARGIN = _f("BITIDY_CB_ZONE_MARGIN", 0.008)
ZONE_U = (_f("BITIDY_CB_ZONE_CUP", -IN_L / 2 + CUP_GEO["r_top"] + ZONE_MARGIN),
          _f("BITIDY_CB_ZONE_BOWL", IN_L / 2 - BOWL_GEO["r_top"] - ZONE_MARGIN))
"""Zone centres along the basket's u axis (-x half: cups, toward the left arm; +x half: bowls)."""

ROW_X = _f("BITIDY_CB_ROW_X", 0.16)
ROW_Y = _f("BITIDY_CB_ROW_Y", 0.0)
ROW_GAP = _f("BITIDY_CB_ROW_GAP", 0.008)
ROW_PITCH = IN_W + 2 * B_WALL + ROW_GAP
_DISP = {  # row xy, row yaw, base xy, base yaw, gap extra hi, box off, box yaw
    "none": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "medium": (0.02, 0.08, 0.01, 0.033, 0.01, 0.01, 0.05),
    "wide": (0.03, 0.15, 0.015, 0.05, 0.015, 0.015, 0.10),
}
DISP = os.environ.get("BITIDY_CB_DISP", "wide")
_d = _DISP[DISP]
ROW_JITTER = _pair("BITIDY_CB_ROW_JITTER", f"{_d[0]},{_d[0]}")
ROW_YAW = _f("BITIDY_CB_ROW_YAW", _d[1])
BASE_JITTER = _f("BITIDY_CB_BASE_JITTER", _d[2])
BASE_YAW = _f("BITIDY_CB_BASE_YAW", _d[3])
GAP_EXTRA = _pair("BITIDY_CB_GAP_EXTRA", f"0,{_d[4]}")
BOX_OFF = _f("BITIDY_CB_BOX_OFF", _d[5])
BOX_YAW = _f("BITIDY_CB_BOX_YAW", _d[6])
OBS_NOISE = _f("BITIDY_CB_OBS_NOISE", 0.004 if DISP != "none" else 0.0)
OBS_YAW_NOISE = _f("BITIDY_CB_OBS_YAW_NOISE", 0.05 if DISP != "none" else 0.0)

OBJ_MODE = os.environ.get("BITIDY_CB_OBJ", "fixed" if DISP == "none" else "random")
OBJ_X = _pair("BITIDY_CB_OBJ_X", f"{round(LEFT_BASE[0] - 0.03, 6)},{round(FACING_X + 0.03, 6)}")  # default -0.08,0.42
OBJ_Y = _pair("BITIDY_CB_OBJ_Y", "0.12,0.34")  # |y| range (both sides of the row)
OBJ_BAND = _pair("BITIDY_CB_OBJ_BAND", "0.16,0.27")
"""A start must lie this far (m) from the nominal base of the arm meant to pick it (any arm for the others)."""
OBJ_SEP = _f("BITIDY_CB_OBJ_SEP", 0.015)  # extra gap between vessel rims
OBJ_BASKET_CLR = _f("BITIDY_CB_OBJ_BASKET_CLR", 0.015)  # gap between a vessel rim and a basket's outer wall
FIXED_SLOTS = ((0.06, 0.235), (0.29, 0.235), (0.06, -0.235), (-0.04, 0.25), (0.38, 0.25), (0.33, -0.25))
"""``BITIDY_CB_OBJ=fixed`` (and the sampling fallback): object k starts at slot k (cups 0/2 + bowl 3 by the left arm,
cup 1 + bowls 4/5 by the right arm)."""
FIXED_ARM = (0, 1, 0, 0, 1, 1)

P_SINGLE = _f("BITIDY_CB_P_SINGLE", 0.15)
"""Share of single-pair instructions (one slot inactive: that arm stays home)."""
PRESORT = _f("BITIDY_CB_PRESORT", 0.6)
DEPTH_P = tuple(float(v) for v in os.environ.get("BITIDY_CB_DEPTH_P", "0.34,0.33,0.33").split(","))
"""Partly done starts: P(each commanded zone already holds 0 / 1 / 2 same-type vessels), as far as vessels allow."""
PRESORT_OTHER = _f("BITIDY_CB_PRESORT_OTHER", 0.4)
STACK_JITTER = _f("BITIDY_CB_STACK_JITTER", 0.002)
FIXED_CMD = os.environ.get("BITIDY_CB_CMD")

# ---------------------------------------------------------------- success

NEST_XY = _f("BITIDY_CB_NEST_XY", 0.01)
ZONE_XY = _f("BITIDY_CB_ZONE_XY", 0.025)
Z_TOL = _f("BITIDY_CB_Z_TOL", 0.006)
TILT_MAX = _f("BITIDY_CB_TILT_MAX", 0.2)
OTHER_TOL = _f("BITIDY_CB_OTHER_TOL", 0.015)
JAW_CLEAR = _f("BITIDY_CB_JAW_CLEAR", 0.012)
LIFT = _f("BITIDY_CB_LIFT", 0.03)
OVER_XY = _f("BITIDY_CB_OVER_XY", 0.015)
SETTLE_STEPS = int(_f("BITIDY_CB_SETTLE", 6))
STABLE_STEPS = 10
EPISODE_S = _f("BITIDY_CB_EPISODE_S", 60)
_STAGES = ("grasp", "lift", "over", "placed", "released")
FLAGS = tuple(f"{a}_{s}" for a in ("L", "R") for s in _STAGES) + ("other_moved",)
"""Per arm slot: grasp, lift, over, placed, released; then other_moved (11)."""
MILESTONE_W = tuple(float(v) for v in os.environ.get("BITIDY_CB_MILESTONE_W", "1,2,3,5,3,-5").split(","))
"""Weights of the 5 per-arm stages (paid once per arm) and of other_moved."""
TILT_PEN = _f("BITIDY_CB_TILT_PEN", 0.0)
"""> 0: one-time penalty (reward -TILT_PEN, once per arm slot and episode) when the slot's vessel rests tipped in a
basket: tilt > ``TILT_PEN_TH`` inside any basket's footprint, below its rim + 7 cm, slower than 5 cm/s, for
``TILT_PEN_STEPS`` steps in a row. 0 (default): no reward term."""
TILT_PEN_TH = _f("BITIDY_CB_TILT_PEN_TH", 0.2)
TILT_PEN_STEPS = int(_f("BITIDY_CB_TILT_PEN_STEPS", 5))
DIAG = bool(os.environ.get("BITIDY_CB_DIAG"))
"""Failure classification of first-episode slots (``BITIDY_CB_DIAG`` JSON line), see :func:`_diag_features`."""

# ---------------------------------------------------------------- state


def _state(env) -> dict:
    env = getattr(env, "unwrapped", env)
    if not hasattr(env, "_cb"):
        n, dev = env.num_envs, env.device
        env._cb = {
            "cmd": torch.zeros((n, N_SLOT, 3), dtype=torch.long, device=dev),  # per arm slot: object, basket, active
            "basket": torch.zeros((n, N_BASKET, 3), device=dev),  # x, y, yaw (env frame)
            "loc": torch.full((n, N_OBJ), -1, dtype=torch.long, device=dev),  # basket at reset (-1: table)
            "level": torch.zeros((n, N_OBJ), dtype=torch.long, device=dev),
            "depth": torch.zeros((n, N_SLOT), dtype=torch.long, device=dev),  # same-type vessels in each slot's zone
            "ref": torch.zeros((n, N_OBJ, 3), device=dev),  # settled start positions
            "base_off": torch.zeros((n, 6), device=dev),
            "seen": torch.zeros(n, dtype=torch.bool, device=dev),
            # first-episode stats [.., (n, success)]
            "s_joint": torch.zeros((3, 2), dtype=torch.long, device=dev),  # by number of active slots (index 1, 2); 0 all
            "s_arm": torch.zeros((N_SLOT, 2), dtype=torch.long, device=dev),
            "s_basket": torch.zeros((N_BASKET, 2), dtype=torch.long, device=dev),
            "s_type": torch.zeros((2, 2), dtype=torch.long, device=dev),
            "s_depth": torch.zeros((3, 2), dtype=torch.long, device=dev),
            "s_type_depth": torch.zeros((2, 3, 2), dtype=torch.long, device=dev),
            "funnel": torch.zeros(len(FLAGS) + 2, dtype=torch.long, device=dev),
            "fallback": 0,
            "sampled": 0,
            "swapped": 0,
        }
    return env._cb


def _geo_t(dev):
    return {k: torch.tensor([CUP_GEO[k], BOWL_GEO[k]], device=dev, dtype=torch.float32)
            for k in ("r_top", "r_bot", "h", "rim", "pitch", "pitch_max", "r_in", "t")}


_TYPE = {}


def obj_type(dev) -> torch.Tensor:
    key = str(dev)
    if key not in _TYPE:
        _TYPE[key] = torch.tensor(OBJ_TYPE, device=dev)
    return _TYPE[key]


def _pick(stacked: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return stacked[torch.arange(stacked.shape[0], device=stacked.device), idx]


def command(env) -> torch.Tensor:
    """(n, 2, 3): per arm slot (0 left, 1 right): object, basket, active."""
    return _state(env)["cmd"]


def slot_active(env, s: int) -> torch.Tensor:
    return command(env)[:, s, 2] == 1


def objs_pos(env) -> torch.Tensor:
    o = env.scene.env_origins
    return torch.stack([env.scene[f"obj{i}"].data.root_pos_w.torch - o for i in range(N_OBJ)], 1)


def objs_quat(env) -> torch.Tensor:
    return torch.stack([env.scene[f"obj{i}"].data.root_quat_w.torch for i in range(N_OBJ)], 1)


def objs_vel(env) -> torch.Tensor:
    return torch.stack([env.scene[f"obj{i}"].data.root_lin_vel_w.torch for i in range(N_OBJ)], 1)


def baskets(env) -> torch.Tensor:
    """(n, 3, 3) basket x, y (sim) and yaw (as placed)."""
    o = env.scene.env_origins[:, :2]
    xy = torch.stack([env.scene[f"basket{j}"].data.root_pos_w.torch[:, :2] - o for j in range(N_BASKET)], 1)
    return torch.cat((xy, _state(env)["basket"][..., 2:]), -1)


def to_basket(b: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    d = xy - b[..., :2]
    c, s = torch.cos(b[..., 2]), torch.sin(b[..., 2])
    return torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), -1)


def from_basket(b: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(b[..., 2]), torch.sin(b[..., 2])
    return b[..., :2] + torch.stack((c * uv[..., 0] - s * uv[..., 1], s * uv[..., 0] + c * uv[..., 1]), -1)


def zone_center(b: torch.Tensor, typ: torch.Tensor) -> torch.Tensor:
    u = torch.where(typ == CUP, ZONE_U[CUP], ZONE_U[BOWL]).to(b.dtype)
    return from_basket(b, torch.stack((u, torch.zeros_like(u)), -1))


def tilt(quat: torch.Tensor) -> torch.Tensor:
    x, y = quat[..., 0], quat[..., 1]
    return torch.acos((1 - 2 * (x * x + y * y)).clamp(-1.0, 1.0))


def slot_obj(env, s: int) -> torch.Tensor:
    return command(env)[:, s, 0]


def slot_type(env, s: int) -> torch.Tensor:
    return obj_type(env.device)[slot_obj(env, s)]


def slot_pos(env, s: int, pos: torch.Tensor | None = None) -> torch.Tensor:
    return _pick(objs_pos(env) if pos is None else pos, slot_obj(env, s))


def slot_quat(env, s: int) -> torch.Tensor:
    return _pick(objs_quat(env), slot_obj(env, s))


def slot_basket(env, s: int) -> torch.Tensor:
    return _pick(baskets(env), command(env)[:, s, 1])


def commanded(env) -> torch.Tensor:
    """(n, 6) bool: objects named by an active slot."""
    cmd = command(env)
    ar = torch.arange(N_OBJ, device=env.device).view(1, 1, -1)
    return ((cmd[..., 0:1] == ar) & (cmd[..., 2:3] == 1)).any(1)


def stack_members(env, s: int) -> torch.Tensor:
    """(n, 6) bool: vessels that started in slot s's zone (same basket, same type), commanded ones excluded."""
    st = _state(env)
    typ = obj_type(env.device)
    same = typ.unsqueeze(0) == slot_type(env, s).unsqueeze(-1)
    return (st["loc"] == command(env)[:, s, 1:2]) & same & ~commanded(env) & slot_active(env, s).unsqueeze(-1)


def nest_target(env, s: int, pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slot s's goal: stack axis xy (n, 2), nominal origin z (n,), depth (n,). Depth 0: zone centre on the floor; else on
    the top vessel of the zone's stack (its current pose)."""
    st = _state(env)
    pos = objs_pos(env) if pos is None else pos
    typ = slot_type(env, s)
    g = _geo_t(env.device)
    members = stack_members(env, s)
    depth = members.sum(-1)
    top = torch.where(members, st["level"], -1).argmax(-1)
    top_pos = _pick(pos, top)
    zc = zone_center(slot_basket(env, s), typ)
    has = depth > 0
    axis = torch.where(has.unsqueeze(-1), top_pos[:, :2], zc)
    z = torch.where(has, top_pos[:, 2] + g["pitch"][typ], torch.full_like(zc[:, 0], FLOOR_Z))
    return axis, z, depth


# ---------------------------------------------------------------- predicates


def rim_center(env, s: int) -> torch.Tensor:
    q = slot_quat(env, s)
    up = torch.zeros((q.shape[0], 3), device=env.device)
    up[:, 2] = _geo_t(env.device)["rim"][slot_type(env, s)]
    return slot_pos(env, s) + quat_apply(q, up)


def jaw_points(env, arm: str) -> list[torch.Tensor]:
    """Fixed tip, moving tip, pad centre of ``arm`` (env frame)."""
    robot = env.scene[arm]
    pts = []
    for body, offset in (("gripper", base.FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", base.MOVING_FINGERTIP_OFFSET)):
        i = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, i], robot.data.body_quat_w.torch[:, i]
        pts.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)) - env.scene.env_origins)
    pts.append(0.5 * (pts[0] + pts[1]))
    return pts


def near_rim(env, s: int) -> torch.Tensor:
    """Slot s's arm has its pads at slot s's vessel rim."""
    rc = rim_center(env, s)
    pad = pad_center(env, ARMS[s])
    r = _geo_t(env.device)["r_top"][slot_type(env, s)]
    radial = torch.linalg.vector_norm(pad[:, :2] - rc[:, :2], dim=-1)
    return ((radial - r).abs() < 0.015) & ((pad[:, 2] - rc[:, 2]).abs() < 0.025)


def held(env, s: int) -> torch.Tensor:
    lifted = slot_pos(env, s)[:, 2] > _pick(_state(env)["ref"], slot_obj(env, s))[:, 2] + 0.008
    return near_rim(env, s) & lifted & slot_active(env, s)


def jaws_clear(env, s: int) -> torch.Tensor:
    p = slot_pos(env, s)
    typ = slot_type(env, s)
    g = _geo_t(env.device)
    r, top = g["r_top"][typ] + JAW_CLEAR, p[:, 2] + g["rim"][typ] + JAW_CLEAR
    inside = torch.zeros_like(typ, dtype=torch.bool)
    for arm in ARMS:
        for q in jaw_points(env, arm):
            inside |= (torch.linalg.vector_norm(q[:, :2] - p[:, :2], dim=-1) < r) & (q[:, 2] < top) & (q[:, 2] > p[:, 2] - 0.01)
    return ~inside


def placed_ok(env, s: int, pos: torch.Tensor | None = None) -> torch.Tensor:
    pos = objs_pos(env) if pos is None else pos
    p = slot_pos(env, s, pos)
    typ = slot_type(env, s)
    g = _geo_t(env.device)
    axis, z, depth = nest_target(env, s, pos)
    dxy = torch.linalg.vector_norm(p[:, :2] - axis, dim=-1)
    xy_ok = torch.where(depth > 0, dxy < NEST_XY, dxy < ZONE_XY)
    z_ok = (p[:, 2] > z - Z_TOL) & (p[:, 2] < z + (g["pitch_max"] - g["pitch"])[typ] + Z_TOL)
    speed = torch.linalg.vector_norm(_pick(objs_vel(env), slot_obj(env, s)), dim=-1)
    return xy_ok & z_ok & (tilt(slot_quat(env, s)) < TILT_MAX) & (speed < 0.05)


def in_baskets(env, p: torch.Tensor, margin: float = 0.01, z_above: float = 0.07) -> torch.Tensor:
    """(n, 3) bool: point p (n, 3) inside basket j's footprint (+ margin) and below its rim + z_above."""
    uv = to_basket(baskets(env), p[:, None, :2])
    half = p.new_tensor((IN_L / 2 + margin, IN_W / 2 + margin))
    return (uv.abs() < half).all(-1) & (p[:, 2:3] < B_TOP_Z + z_above)


def _clearance(b: torch.Tensor, xy: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rim-to-inner-wall clearance (m) of a vessel of top radius r centred at xy in basket b (negative: over the wall)."""
    uv = to_basket(b, xy).abs()
    return torch.minimum(IN_L / 2 - uv[..., 0], IN_W / 2 - uv[..., 1]) - r


DIAG_KEYS = ("tilt", "dxy", "dz", "in_cmd", "in_any", "held", "placed_ok", "clear", "tclear", "speed")


def _diag_features(env, s: int, pos: torch.Tensor) -> torch.Tensor:
    """(n, len(DIAG_KEYS)) end-of-step state of slot s's vessel, for the failure classification."""
    p = slot_pos(env, s, pos)
    typ = slot_type(env, s)
    g = _geo_t(env.device)
    axis, z, _ = nest_target(env, s, pos)
    b = slot_basket(env, s)
    inb = in_baskets(env, p)
    in_cmd = _pick(inb, command(env)[:, s, 1])
    speed = torch.linalg.vector_norm(_pick(objs_vel(env), slot_obj(env, s)), dim=-1)
    return torch.stack((tilt(slot_quat(env, s)), torch.linalg.vector_norm(p[:, :2] - axis, dim=-1), p[:, 2] - z,
                        in_cmd.float(), inb.any(-1).float(), held(env, s).float(), placed_ok(env, s, pos).float(),
                        _clearance(b, p[:, :2], g["r_top"][typ]), _clearance(b, axis, g["r_top"][typ]), speed), -1)


def other_moved(env) -> torch.Tensor:
    st = _state(env)
    d = torch.linalg.vector_norm(objs_pos(env) - st["ref"], dim=-1)
    free = ~commanded(env) & ~stack_members(env, 0) & ~stack_members(env, 1)
    started = env.episode_length_buf >= SETTLE_STEPS
    return ((d > OTHER_TOL) & free).any(-1) & started


# ---------------------------------------------------------------- progress


class CupBasketProgress:
    def __init__(self, num_envs: int, device):
        self.flags = torch.zeros((num_envs, len(FLAGS)), dtype=torch.bool, device=device)
        self.held_count = torch.zeros((num_envs, N_SLOT), dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.slot_stable = torch.zeros((num_envs, N_SLOT), dtype=torch.long, device=device)
        self.slot_ok = torch.zeros((num_envs, N_SLOT), dtype=torch.bool, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cmd = torch.zeros((num_envs, N_SLOT, 3), dtype=torch.long, device=device)
        if DIAG:  # end-of-step slot state (BITIDY_CB_DIAG); the termination reset reads the last one
            self.diag = torch.zeros((num_envs, N_SLOT, len(DIAG_KEYS)), device=device)

    def __getattr__(self, name):
        if name in FLAGS:
            return self.flags[:, FLAGS.index(name)]
        raise AttributeError(name)

    def stage(self, s: int, name: str) -> torch.Tensor:
        return self.flags[:, 5 * s + _STAGES.index(name)]

    def reset(self, ids) -> None:
        self.flags[ids] = False
        self.held_count[ids] = 0
        self.stable_count[ids] = 0
        self.slot_stable[ids] = 0
        self.slot_ok[ids] = False
        self.success[ids] = False

    def update(self, env: ManagerBasedRLEnv) -> None:
        st = _state(env)
        settling = env.episode_length_buf < SETTLE_STEPS
        pos = objs_pos(env)
        st["ref"] = torch.where(settling.view(-1, 1, 1), pos, st["ref"])
        self.cmd = st["cmd"].clone()
        moved = other_moved(env)
        new = torch.zeros_like(self.flags)
        all_done = ~moved & (self.cmd[:, :, 2] == 1).any(-1)
        for s in range(N_SLOT):
            act = slot_active(env, s)
            h = held(env, s) & ~settling
            self.held_count[:, s] = torch.where(h, self.held_count[:, s] + 1, 0)
            p = slot_pos(env, s, pos)
            ref = _pick(st["ref"], slot_obj(env, s))
            axis, _, _ = nest_target(env, s, pos)
            grasp = self.held_count[:, s] >= 3
            lift = self.stage(s, "grasp") & h & (p[:, 2] > ref[:, 2] + LIFT)
            over = self.stage(s, "lift") & h & (torch.linalg.vector_norm(p[:, :2] - axis, dim=-1) < OVER_XY)
            ok = placed_ok(env, s, pos)
            placed = self.stage(s, "over") & ok
            clear = jaws_clear(env, s)
            released = self.stage(s, "placed") & ok & clear
            new[:, 5 * s: 5 * s + 5] = torch.stack((grasp, lift, over, placed, released), -1) & act.unsqueeze(-1)
            done = self.stage(s, "over") & ok & clear
            self.slot_stable[:, s] = torch.where(done, self.slot_stable[:, s] + 1, 0)
            self.slot_ok[:, s] |= self.slot_stable[:, s] >= STABLE_STEPS
            all_done &= done | ~act
        new[:, -1] = moved
        if os.environ.get("BITIDY_CB_LOG"):
            first = (moved & ~self.other_moved).nonzero().squeeze(-1)
            if len(first):
                d = torch.linalg.vector_norm(pos - st["ref"], dim=-1)
                for e in first.tolist()[:20]:
                    print(f"BITIDY_CB_MOVED env={e} t={int(env.episode_length_buf[e])} obj={int(d[e].argmax())} "
                          f"loc={st['loc'][e].tolist()} d={[round(v, 3) for v in d[e].tolist()]} cmd={st['cmd'][e].tolist()} "
                          f"flags={self.flags[e].int().tolist()}", flush=True)
        if LOG2:
            _log2(self, env, pos, st, settling)
        self.flags |= new
        if DIAG:
            self.diag = torch.stack([_diag_features(env, s, pos) for s in range(N_SLOT)], 1)
        self.stable_count = torch.where(all_done, self.stable_count + 1, 0)
        self.success |= self.stable_count >= STABLE_STEPS


LOG2 = bool(os.environ.get("BITIDY_CB_LOG2"))
TIP_LOG = _f("BITIDY_CB_TIP_LOG", 0.6)


def _log2(p: "CupBasketProgress", env, pos: torch.Tensor, st: dict, settling: torch.Tensor) -> None:
    """Eval-only cause logging (BITIDY_CB_LOG2). BITIDY_CB_BUMP: first time a free vessel moves > 5 mm, distances
    (horizontal gap to its rim, height above its rim) from each arm's jaws, each arm's lowest nearby link and each held
    vessel. BITIDY_CB_TIP: first time a commanded vessel tilts > TILT_MAX, its state."""
    n, dev = env.num_envs, env.device
    if not hasattr(p, "l2_bump"):
        p.l2_bump = torch.zeros((n, N_OBJ), dtype=torch.bool, device=dev)
        p.l2_tip = torch.zeros((n, N_SLOT), dtype=torch.bool, device=dev)
    fresh = env.episode_length_buf <= 1
    p.l2_bump[fresh] = False
    p.l2_tip[fresh] = False
    g = _geo_t(dev)
    typ = obj_type(dev)
    r_o, rim_o = g["r_top"][typ], g["rim"][typ]
    stage = [p.flags[:, 5 * s: 5 * s + 5].sum(-1) for s in range(N_SLOT)]
    names = ("reach",) + _STAGES

    def gap(pts, e, o):  # pts (k, 3): min horizontal gap to vessel o's rim, and height of that point above the rim
        dxy = torch.linalg.vector_norm(pts[:, :2] - pos[e, o, :2], dim=-1) - r_o[o]
        i = int(dxy.argmin())
        return round(float(dxy[i]), 3), round(float(pts[i, 2] - pos[e, o, 2] - rim_o[o]), 3)

    d = torch.linalg.vector_norm(pos - st["ref"], dim=-1)
    free = ~commanded(env) & ~stack_members(env, 0) & ~stack_members(env, 1)
    hit = (d > 0.005) & free & ~settling.unsqueeze(-1) & ~p.l2_bump
    if hit.any():
        jaws = {a: torch.stack(jaw_points(env, a), 1) for a in ARMS}
        links = {a: env.scene[a].data.body_pos_w.torch - env.scene.env_origins.unsqueeze(1) for a in ARMS}
        for e, o in hit.nonzero().tolist()[:40]:
            rec = {"t": int(env.episode_length_buf[e]), "obj": o, "typ": int(typ[o]), "loc": int(st["loc"][e, o]),
                   "stage": [names[int(stage[s][e])] for s in range(N_SLOT)]}
            for s, a in enumerate(ARMS):
                rec[f"jaw{s}"] = gap(jaws[a][e], e, o)
                rec[f"link{s}"] = gap(links[a][e], e, o)
                if bool(slot_active(env, s)[e]):
                    so = int(command(env)[e, s, 0])
                    dxy = float(torch.linalg.vector_norm(pos[e, so, :2] - pos[e, o, :2])) - float(r_o[so] + r_o[o])
                    rec[f"ves{s}"] = (round(dxy, 3), round(float(pos[e, so, 2] - pos[e, o, 2] - rim_o[o]), 3),
                                      bool(held(env, s)[e]))
            print("BITIDY_CB_BUMP " + str(rec), flush=True)
        p.l2_bump |= hit
    # per-slot carry / release / grasp-attempt state (aggregated per failure class in _diag_account)
    if not hasattr(p, "l2_ng"):
        p.l2_ng = torch.zeros((n, N_SLOT, 3), device=dev)  # min pad-to-rim distance, max push before grasp, closed-on-air steps
        p.l2_ht = torch.zeros((n, N_SLOT), device=dev)  # max tilt while held
        p.l2_rel = torch.full((n, N_SLOT, 5), -1.0, device=dev)  # first release: tilt, dxy, dz, clearance, in target
        p.l2_was = torch.zeros((n, N_SLOT), dtype=torch.bool, device=dev)
    p.l2_ng[fresh] = p.l2_ng.new_tensor((1.0, 0.0, 0.0))
    p.l2_ht[fresh] = 0.0
    p.l2_rel[fresh] = -1.0
    p.l2_was[fresh] = False
    act = env.action_manager.action
    for s in range(N_SLOT):
        tl = tilt(slot_quat(env, s))
        sp = slot_pos(env, s, pos)
        ax, zt, _ = nest_target(env, s, pos)
        styp = slot_type(env, s)
        ref = _pick(st["ref"], slot_obj(env, s))
        pad = torch.stack(jaw_points(env, ARMS[s]), 1).mean(1)
        dist = torch.sqrt((torch.linalg.vector_norm(pad[:, :2] - sp[:, :2], dim=-1) - g["r_top"][styp]) ** 2
                          + (pad[:, 2] - sp[:, 2] - g["rim"][styp]) ** 2)
        act_s = slot_active(env, s) & ~settling
        pre = act_s & ~p.stage(s, "grasp")
        p.l2_ng[:, s, 0] = torch.where(act_s, torch.minimum(p.l2_ng[:, s, 0], dist), p.l2_ng[:, s, 0])
        p.l2_ng[:, s, 1] = torch.where(pre, torch.maximum(p.l2_ng[:, s, 1], torch.linalg.vector_norm(sp[:, :2] - ref[:, :2], dim=-1)),
                                       p.l2_ng[:, s, 1])
        hs = held(env, s) & act_s
        p.l2_ng[:, s, 2] += (pre & (act[:, 6 * s + 5] < 0) & ~hs).float()
        p.l2_ht[:, s] = torch.where(hs, torch.maximum(p.l2_ht[:, s], tl), p.l2_ht[:, s])
        rel = p.l2_was[:, s] & ~hs & act_s & (p.l2_rel[:, s, 0] < 0)
        if rel.any():
            inb = _pick(in_baskets(env, sp, margin=0.0, z_above=0.05), command(env)[:, s, 1])
            f = torch.stack((tl, torch.linalg.vector_norm(sp[:, :2] - ax, dim=-1), sp[:, 2] - zt,
                             _clearance(slot_basket(env, s), sp[:, :2], g["r_top"][styp]), inb.float()), -1)
            p.l2_rel[:, s] = torch.where(rel.unsqueeze(-1), f, p.l2_rel[:, s])
        p.l2_was[:, s] = hs
        tip = (tl > TIP_LOG) & slot_active(env, s) & ~settling & ~p.l2_tip[:, s]
        if tip.any():
            b = slot_basket(env, s)
            inb = _pick(in_baskets(env, sp, margin=0.0, z_above=0.05), command(env)[:, s, 1])
            clr = _clearance(b, sp[:, :2], g["r_top"][slot_type(env, s)])
            ref = _pick(st["ref"], slot_obj(env, s))
            vel = _pick(objs_vel(env), slot_obj(env, s))
            for e in tip.nonzero().squeeze(-1).tolist()[:40]:
                so = int(command(env)[e, s, 0])
                others = [o for o in range(N_OBJ) if o != so]
                near = min(float(torch.linalg.vector_norm(pos[e, so, :2] - pos[e, o, :2])) - float(r_o[so] + r_o[o])
                           for o in others)
                other_jaw = gap(torch.stack(jaw_points(env, ARMS[1 - s]), 1)[e], e, so)
                rec = {"t": int(env.episode_length_buf[e]), "slot": s, "typ": int(slot_type(env, s)[e]),
                       "stage": names[int(stage[s][e])], "held": bool(held(env, s)[e]),
                       "dxy": round(float(torch.linalg.vector_norm(sp[e, :2] - ax[e])), 3), "dz": round(float(sp[e, 2] - zt[e]), 3),
                       "htilt": round(float(p.l2_ht[e, s]), 2), "rel": [round(float(v), 3) for v in p.l2_rel[e, s].tolist()],
                       "lift": round(float(sp[e, 2] - ref[e, 2]), 3), "in_tgt": bool(inb[e]),
                       "clr": round(float(clr[e]), 3), "speed": round(float(torch.linalg.vector_norm(vel[e])), 3),
                       "depth": int(st["depth"][e, s]), "near_ves": round(near, 3), "other_jaw": other_jaw}
                print("BITIDY_CB_TIP " + str(rec), flush=True)
            p.l2_tip[:, s] |= tip


def progress(env: ManagerBasedRLEnv) -> CupBasketProgress:
    if not hasattr(env, "_bi_progress"):
        env._bi_progress = CupBasketProgress(env.num_envs, env.device)
        env._bi_final = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
        env._bi_flag_names = FLAGS
    return env._bi_progress


# ---------------------------------------------------------------- layout sampling


def _row(n: int, dev) -> torch.Tensor:
    """Basket poses (n, 3, 3): a row along y (LidSort's wide scheme)."""
    perm = torch.argsort(torch.rand((n, N_BASKET), device=dev), dim=-1)
    gaps = ROW_PITCH + GAP_EXTRA[0] + torch.rand((n, N_BASKET - 1), device=dev) * (GAP_EXTRA[1] - GAP_EXTRA[0])
    slot = torch.cat((torch.zeros((n, 1), device=dev), gaps.cumsum(-1)), -1)
    slot = slot - slot.mean(-1, keepdim=True)
    along = torch.gather(slot, 1, perm)
    across = (torch.rand((n, N_BASKET), device=dev) * 2 - 1) * BOX_OFF
    yaw = (torch.rand(n, device=dev) * 2 - 1) * ROW_YAW
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    centre = torch.tensor((ROW_X, ROW_Y), device=dev) + (torch.rand((n, 2), device=dev) * 2 - 1) * torch.tensor(ROW_JITTER, device=dev)
    xy = centre.unsqueeze(1) + torch.stack((-s * along + c * across, c * along + s * across), -1)
    byaw = yaw.unsqueeze(-1) + (torch.rand((n, N_BASKET), device=dev) * 2 - 1) * BOX_YAW
    return torch.cat((xy, byaw.unsqueeze(-1)), -1)


def _sample_objects(bk: torch.Tensor, skip: torch.Tensor, arm: torch.Tensor, tries: int = 768):
    """Table starts (n, 6, 2) for the vessels not in ``skip``: in the OBJ_X x +-OBJ_Y region, clear of every basket and
    of each other, within ``OBJ_BAND`` of base ``arm[r, i]`` (0 left, 1 right, -1 either). Falls back to the fixed
    slots per env (counted)."""
    n, dev = bk.shape[0], bk.device
    pts = torch.full((n, N_OBJ, 2), 10.0, device=dev)
    found = torch.ones(n, dtype=torch.bool, device=dev)
    rows = torch.arange(n, device=dev)
    bases = torch.tensor(BASES, device=dev)
    radius = [GEO[OBJ_TYPE[i]]["r_top"] for i in range(N_OBJ)]
    half = torch.tensor((IN_L / 2 + B_WALL, IN_W / 2 + B_WALL), device=dev)
    for i in (3, 4, 5, 0, 1, 2):
        r = torch.rand((n, tries, 2), device=dev)
        x = OBJ_X[0] + r[..., 0] * (OBJ_X[1] - OBJ_X[0])
        ay = OBJ_Y[0] + r[..., 1] * (OBJ_Y[1] - OBJ_Y[0])
        sign = torch.where(torch.rand((n, tries), device=dev) < 0.5, -1.0, 1.0)
        cand = torch.stack((x, sign * ay), -1)
        dist = torch.linalg.vector_norm(cand.unsqueeze(2) - bases.view(1, 1, 2, 2), dim=-1)  # (n, T, 2)
        inband = (dist >= OBJ_BAND[0]) & (dist <= OBJ_BAND[1])
        a = arm[:, i].view(-1, 1)
        ok = torch.where(a < 0, inband.any(-1), torch.where(a == 0, inband[..., 0], inband[..., 1]))
        uv = to_basket(bk.unsqueeze(1), cand.unsqueeze(2))
        ok &= ~((uv.abs() < half + radius[i] + OBJ_BASKET_CLR).all(-1)).any(-1)
        need = torch.tensor([radius[i] + radius[o] + OBJ_SEP for o in range(N_OBJ)], device=dev)
        ok &= (torch.linalg.vector_norm(cand.unsqueeze(2) - pts.unsqueeze(1), dim=-1) >= need).all(-1)
        pts[:, i] = cand[rows, ok.float().argmax(-1)]
        found &= ok.any(-1) | skip[:, i]
        pts[:, i] = torch.where(skip[:, i:i + 1], torch.full_like(pts[:, i], 10.0), pts[:, i])
    fixed = torch.tensor(FIXED_SLOTS, device=dev).expand(n, -1, -1)
    pts = torch.where(found.view(-1, 1, 1), pts, fixed)
    return pts, ~found


def _plan(n: int, fixed_slots: bool) -> tuple[list, list, list, list, list]:
    """Per env (python, CPU): command pairs [(k, j)] (1 or 2), loc, level, depth per pair, intended arm per object."""
    cmds, locs, levels, depths, arms = [], [], [], [], []
    for _ in range(n):
        if FIXED_CMD:
            v = [int(x) for x in FIXED_CMD.split(",")]
            pairs = [(v[0], v[1])] + ([(v[2], v[3])] if len(v) > 3 and v[2] >= 0 else [])
        else:
            single = random.random() < P_SINGLE
            ks = random.sample(range(N_OBJ), 1 if single else 2)
            pairs = []
            for k in ks:
                j = random.randrange(N_BASKET)
                if pairs and OBJ_TYPE[k] == OBJ_TYPE[pairs[0][0]] and j == pairs[0][1]:  # never the same zone
                    j = random.choice([b for b in range(N_BASKET) if b != j])
                pairs.append((k, j))
        cmd_objs = {k for k, _ in pairs}
        loc, level = [-1] * N_OBJ, [0] * N_OBJ
        pre = random.random() < PRESORT
        stacks: dict = {}
        dlist = []
        for k, j in pairs:
            tk = OBJ_TYPE[k]
            pool = [o for o in range(N_OBJ) if OBJ_TYPE[o] == tk and o not in cmd_objs and loc[o] < 0]
            random.shuffle(pool)
            d = random.choices(range(len(DEPTH_P)), weights=DEPTH_P)[0] if pre else 0
            d = min(d, len(pool))
            stacks[(j, tk)] = pool[:d]
            for lv, o in enumerate(pool[:d]):
                loc[o], level[o] = j, lv
            dlist.append(d)
        if pre:
            zones = {(j, OBJ_TYPE[k]) for k, j in pairs}
            rest = [o for o in range(N_OBJ) if o not in cmd_objs and loc[o] < 0]
            random.shuffle(rest)
            for o in rest:
                if random.random() < PRESORT_OTHER:
                    choices = [b for b in range(N_BASKET) if (b, OBJ_TYPE[o]) not in zones]
                    b = random.choice(choices)
                    s = stacks.setdefault((b, OBJ_TYPE[o]), [])
                    loc[o], level[o] = b, len(s)
                    s.append(o)
        # Intended arm: pair 0 -> left, pair 1 -> right (single: random arm); the env re-assigns by distance anyway.
        arm = [-1] * N_OBJ
        if fixed_slots:
            arm = list(FIXED_ARM)
        elif len(pairs) == 2:
            arm[pairs[0][0]], arm[pairs[1][0]] = 0, 1
        else:
            arm[pairs[0][0]] = random.randrange(2)
        cmds.append(pairs)
        locs.append(loc)
        levels.append(level)
        depths.append(dlist)
        arms.append(arm)
    return cmds, locs, levels, depths, arms


def _place_bases(env, ids: torch.Tensor) -> None:
    st = _state(env)
    n, dev = len(ids), env.device
    for a, name in enumerate(ARMS):
        dxy = (torch.rand((n, 2), device=dev) * 2 - 1) * BASE_JITTER
        dyaw = (torch.rand(n, device=dev) * 2 - 1) * BASE_YAW
        st["base_off"][ids, 3 * a: 3 * a + 2] = dxy
        st["base_off"][ids, 3 * a + 2] = dyaw
        if BASE_JITTER <= 0 and BASE_YAW <= 0:
            continue
        yaw = _ARM_YAW0[name] + dyaw
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = torch.tensor(BASES[a], device=dev) + dxy
        pose[:, 5], pose[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        pose[:, :3] += env.scene.env_origins[ids]
        env.scene[name].write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)


def base_offsets(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _state(env)["base_off"]


def _pick_rows(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t[torch.arange(t.shape[0], device=t.device).unsqueeze(-1), idx]


def reset_objects(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: basket row, instruction (1-2 pairs), partly done stacks, vessel starts, arm assignment."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    n, dev = len(ids), env.device
    st = _state(env)
    bk = _row(n, dev)
    st["basket"][ids] = bk
    _place_bases(env, ids)
    pairs, locs, levels, depths, arms = _plan(n, OBJ_MODE == "fixed")
    loc = torch.tensor(locs, device=dev)
    level = torch.tensor(levels, device=dev)
    st["loc"][ids], st["level"][ids] = loc, level
    if OBJ_MODE == "fixed":
        xy = torch.tensor(FIXED_SLOTS, device=dev).expand(n, -1, -1).clone()
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
    else:
        xy, fell = _sample_objects(bk, loc >= 0, torch.tensor(arms, device=dev))
    st["fallback"] += int(fell.sum())
    st["sampled"] += n
    # Arm assignment (nearest arm; both nearest the same arm -> the 2 x 2 assignment with the smaller total distance).
    bases = torch.tensor(BASES, device=dev).unsqueeze(0) + st["base_off"][ids].view(n, 2, 3)[..., :2]  # (n, 2, 2)
    dist_all = torch.linalg.vector_norm(xy.unsqueeze(2) - bases.unsqueeze(1), dim=-1).tolist()  # (n, 6, 2)
    cmd = [[[0, 0, 0], [0, 0, 0]] for _ in range(n)]
    dep = [[0, 0] for _ in range(n)]
    swapped = 0
    for r in range(n):
        pr = pairs[r]
        dist = [dist_all[r][k] for k, _ in pr]
        if len(pr) == 1:
            order = [0 if dist[0][0] <= dist[0][1] else 1]
            slots = [(order[0], 0)]
        else:
            keep = dist[0][0] + dist[1][1] <= dist[0][1] + dist[1][0]
            swapped += 0 if keep else 1
            slots = [(0, 0), (1, 1)] if keep else [(0, 1), (1, 0)]
        for a, p in slots:
            cmd[r][a] = [pr[p][0], pr[p][1], 1]
            dep[r][a] = depths[r][p]
    st["swapped"] += swapped
    st["cmd"][ids] = torch.tensor(cmd, dtype=torch.long, device=dev)
    st["depth"][ids] = torch.tensor(dep, dtype=torch.long, device=dev)
    yaw = (torch.rand((n, N_OBJ), device=dev) * 2 - 1) * math.pi
    z = torch.full((n, N_OBJ), TABLE_TOP, device=dev)
    typ = obj_type(dev)
    pitch = torch.tensor([CUP_GEO["pitch"], BOWL_GEO["pitch"]], device=dev)[typ]
    inb = loc >= 0
    b = _pick_rows(bk, loc.clamp(min=0))
    zc = zone_center(b, typ.expand(n, -1))
    stack_yaw = (torch.rand((n, N_BASKET, 2), device=dev) * 2 - 1) * math.pi
    syaw = stack_yaw[torch.arange(n, device=dev).unsqueeze(-1), loc.clamp(min=0), typ.expand(n, -1)]
    jit = (torch.rand((n, N_OBJ, 2), device=dev) * 2 - 1) * STACK_JITTER
    xy = torch.where(inb.unsqueeze(-1), zc + jit, xy)
    z = torch.where(inb, FLOOR_Z + level * pitch + 0.0005, z)
    yaw = torch.where(inb, syaw + (torch.rand((n, N_OBJ), device=dev) * 2 - 1) * 0.02, yaw)
    origins = env.scene.env_origins[ids]
    zeros = torch.zeros((n, 6), device=dev)
    for j in range(N_BASKET):
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = bk[:, j, :2]
        pose[:, 2] = TABLE_TOP
        pose[:, 5], pose[:, 6] = torch.sin(bk[:, j, 2] / 2), torch.cos(bk[:, j, 2] / 2)
        pose[:, :3] += origins
        env.scene[f"basket{j}"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[f"basket{j}"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)
    for i in range(N_OBJ):
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :2] = xy[:, i]
        pose[:, 2] = z[:, i]
        pose[:, 5], pose[:, 6] = torch.sin(yaw[:, i] / 2), torch.cos(yaw[:, i] / 2)
        st["ref"][ids, i] = pose[:, :3]
        pose[:, :3] += origins
        env.scene[f"obj{i}"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[f"obj{i}"].write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)


# ---------------------------------------------------------------- observations


def _noise(t: torch.Tensor, on: bool, std: float | None = None) -> torch.Tensor:
    std = OBS_NOISE if std is None else std
    return t + torch.randn_like(t) * std if on and std > 0 else t


def _yaw_noise(q: torch.Tensor, on: bool) -> torch.Tensor:
    if not on or OBS_YAW_NOISE <= 0:
        return q
    a = torch.randn(q.shape[:-1], device=q.device) * OBS_YAW_NOISE
    dq = torch.stack((torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2), torch.cos(a / 2)), -1)
    return quat_mul(dq.reshape(-1, 4), q.reshape(-1, 4)).reshape(q.shape)


def scene_state(env: ManagerBasedRLEnv, noisy: bool = False) -> torch.Tensor:
    """All vessel poses (6 x pos+quat = 42), all baskets (3 x x, y, cos yaw, sin yaw = 12): 54."""
    n = env.num_envs
    bk = baskets(env)
    yaw = _noise(bk[..., 2], noisy, OBS_YAW_NOISE)
    b = torch.cat((_noise(bk[..., :2], noisy), torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)), -1)
    o = torch.cat((_noise(objs_pos(env), noisy), _yaw_noise(objs_quat(env), noisy)), -1)
    return torch.cat((o.reshape(n, -1), b.reshape(n, -1)), -1)


def target_relative(env: ManagerBasedRLEnv, noisy: bool = False) -> torch.Tensor:
    """Per arm slot: its vessel's rim centre relative to that arm's pads (3), its goal (stack axis, nest height) relative
    to the vessel (3), the vessel's velocity (3); zeros for an inactive slot: 18."""
    parts = []
    for s in range(N_SLOT):
        rc = _noise(rim_center(env, s), noisy)
        axis, z, _ = nest_target(env, s)
        goal = _noise(torch.cat((axis, z.unsqueeze(-1)), -1), noisy)
        p = slot_pos(env, s)
        vel = _pick(objs_vel(env), slot_obj(env, s))
        v = torch.cat((rc - pad_center(env, ARMS[s]), goal - p, vel), -1)
        parts.append(v * slot_active(env, s).unsqueeze(-1).float())
    return torch.cat(parts, -1)


def command_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per arm slot (left, right): object one-hot (6), basket one-hot (3), active bit (1): 20 (inactive: all zero)."""
    cmd = command(env)
    parts = []
    for s in range(N_SLOT):
        act = (cmd[:, s, 2:3] == 1).float()
        parts += [torch.nn.functional.one_hot(cmd[:, s, 0], N_OBJ).float() * act,
                  torch.nn.functional.one_hot(cmd[:, s, 1], N_BASKET).float() * act, act]
    return torch.cat(parts, -1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).flags.float()


# ---------------------------------------------------------------- rewards (sparse: one-time milestones + success)


class milestones(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, len(FLAGS)), dtype=torch.bool, device=self.device)
        w = list(MILESTONE_W[:5]) * N_SLOT + [MILESTONE_W[5]]
        self.w = torch.tensor(w, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        flags = progress(env).flags
        reward = ((flags & ~self.paid).float() * self.w).sum(-1)
        self.paid = flags.clone()
        return reward


class tilt_penalty(ManagerTermBase):
    """1 on the step an active slot's vessel has rested tipped (> TILT_PEN_TH) in a basket for TILT_PEN_STEPS steps; at
    most once per slot and episode (weight -TILT_PEN). Not dense: nothing to farm, and it can only lower the return of
    episodes that end with a tipped vessel (which never succeed: success needs tilt < TILT_MAX)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, N_SLOT), dtype=torch.bool, device=self.device)
        self.count = torch.zeros((self.num_envs, N_SLOT), dtype=torch.long, device=self.device)
        self.total = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.count[ids] = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        pos = objs_pos(env)
        vel = objs_vel(env)
        started = env.episode_length_buf >= SETTLE_STEPS
        out = torch.zeros(self.num_envs, device=self.device)
        for s in range(N_SLOT):
            p = slot_pos(env, s, pos)
            speed = torch.linalg.vector_norm(_pick(vel, slot_obj(env, s)), dim=-1)
            bad = (slot_active(env, s) & started & (tilt(slot_quat(env, s)) > TILT_PEN_TH) & in_baskets(env, p).any(-1)
                   & (speed < 0.05))
            self.count[:, s] = torch.where(bad, self.count[:, s] + 1, 0)
            fire = (self.count[:, s] >= TILT_PEN_STEPS) & ~self.paid[:, s]
            self.paid[:, s] |= fire
            out += fire.float()
        return out


DISTURB_PEN = _f("BITIDY_CB_DISTURB_PEN", 0.0)
"""> 0: potential-based penalty on the displacement of vessels the policy must not move (not commanded, not a member of
a target stack): Phi = sum over them of clamp(d - DISTURB_TOL, 0, DISTURB_CAP), reward = -(Phi_t - Phi_{t-1}) with
weight DISTURB_PEN per metre. Telescopes to -DISTURB_PEN * Phi_final per episode (nothing to farm) but is paid on the
step of the bump, from 3 mm on, long before other_moved (OTHER_TOL) fires. 0 (default): no reward term."""
DISTURB_TOL = _f("BITIDY_CB_DISTURB_TOL", 0.003)
DISTURB_CAP = _f("BITIDY_CB_DISTURB_CAP", 0.03)
WALL_PEN = _f("BITIDY_CB_WALL_PEN", 0.0)
"""> 0: per-step penalty while a held vessel is lowered into its target basket (inside the footprint, below rim +
5 cm): clamp(WALL_CLR - rim-to-inner-wall clearance, 0) metres, weight WALL_PEN. Tipping is concentrated at clearance
< 5 mm (rim catches the wall). 0 (default): no reward term."""
WALL_CLR = _f("BITIDY_CB_WALL_CLR", 0.008)


HTILT_PEN = _f("BITIDY_CB_HTILT_PEN", 0.0)
"""> 0: per-step penalty while a slot's vessel is held: clamp(tilt - HTILT_TOL, 0) rad, weight HTILT_PEN. A rim pinch
lets the vessel hang tilted (hinge): a swinging, low-hanging vessel bumps neighbours and catches the basket wall."""
HTILT_TOL = _f("BITIDY_CB_HTILT_TOL", 0.2)
LOW_PEN = _f("BITIDY_CB_LOW_PEN", 0.0)
"""> 0: per-step penalty on the horizontal speed (beyond LOW_VTOL m/s) of a held vessel that is still within LOW_H of its
start height and not over its target basket: lift first, then traverse. Weight LOW_PEN per m/s."""
LOW_H = _f("BITIDY_CB_LOW_H", 0.06)
LOW_VTOL = _f("BITIDY_CB_LOW_VTOL", 0.03)


def held_tilt_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    out = torch.zeros(env.num_envs, device=env.device)
    for s in range(N_SLOT):
        ex = (tilt(slot_quat(env, s)) - HTILT_TOL).clamp(0.0, 1.5)
        out += torch.where(held(env, s), torch.nan_to_num(ex), torch.zeros_like(out))
    return out  # weight -HTILT_PEN


def low_traverse_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    pos, vel, st = objs_pos(env), objs_vel(env), _state(env)
    out = torch.zeros(env.num_envs, device=env.device)
    for s in range(N_SLOT):
        p = slot_pos(env, s, pos)
        low = p[:, 2] < _pick(st["ref"], slot_obj(env, s))[:, 2] + LOW_H
        over_tgt = _pick(in_baskets(env, p, margin=0.02, z_above=0.3), command(env)[:, s, 1])
        v = torch.linalg.vector_norm(_pick(vel, slot_obj(env, s))[:, :2], dim=-1)
        bad = held(env, s) & low & ~over_tgt
        out += torch.where(bad, torch.nan_to_num(v - LOW_VTOL).clamp(0.0, 1.0), torch.zeros_like(out))
    return out  # weight -LOW_PEN


class disturb_penalty(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.prev[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        st = _state(env)
        started = env.episode_length_buf >= SETTLE_STEPS
        d = torch.linalg.vector_norm(objs_pos(env) - st["ref"], dim=-1)
        free = ~commanded(env) & ~stack_members(env, 0) & ~stack_members(env, 1)
        phi = ((d - DISTURB_TOL).clamp(0.0, DISTURB_CAP) * free.float()).sum(-1)
        phi = torch.where(started, torch.nan_to_num(phi), torch.zeros_like(phi))
        out = phi - self.prev  # weight is -DISTURB_PEN
        self.prev = phi
        return out


def wall_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    pos = objs_pos(env)
    g = _geo_t(env.device)
    out = torch.zeros(env.num_envs, device=env.device)
    for s in range(N_SLOT):
        p = slot_pos(env, s, pos)
        in_cmd = _pick(in_baskets(env, p, margin=0.0, z_above=0.05), command(env)[:, s, 1])
        clr = _clearance(slot_basket(env, s), p[:, :2], g["r_top"][slot_type(env, s)])
        bad = held(env, s) & in_cmd
        out += torch.where(bad, (WALL_CLR - torch.nan_to_num(clr)).clamp(0.0, 0.05), torch.zeros_like(out))
    return out  # weight is -WALL_PEN


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return progress(env).success.float()


# ---------------------------------------------------------------- terminations / stats


_TCLEAR_BINS = (0.005, 0.01, 0.015, 0.02)


def _diag_class(fl: list, ok: bool, d: dict, depth: int) -> str:
    """Failure class of one active slot at the end of its first episode."""
    grasp = fl[0]
    if ok:
        return "ok"
    if not grasp:
        return "never_grasped"
    if d["tilt"] > TILT_MAX and d["in_any"] > 0.5:
        return "tipped"
    if d["held"] > 0.5:
        return "held_timeout"
    if d["placed_ok"] > 0.5:
        return "not_released"
    if d["in_cmd"] > 0.5:
        return "not_nested" if depth > 0 else "off_zone"
    if d["in_any"] > 0.5:
        return "wrong_basket"
    return "dropped"


def _diag_account(env, first: torch.Tensor, p: "CupBasketProgress") -> None:
    """BITIDY_CB_DIAG: classify every active slot of the first episodes; print one cumulative JSON line."""
    import json

    st = _state(env)
    D = st.setdefault("diag", {"episodes": 0, "joint_fail": 0, "ep_other_moved": 0, "ep_other_moved_fail": 0,
                               "slots": 0, "cls": {}, "cls_type": [{}, {}], "cls_depth": [{}, {}, {}],
                               "tipped_type_depth": [[0, 0, 0], [0, 0, 0]], "slots_type_depth": [[0, 0, 0], [0, 0, 0]],
                               "tipped_after_placed": 0, "ok_slot_in_failed_ep": 0, "single_pair_fail": [0, 0],
                               "tclear_bins": list(_TCLEAR_BINS), "tclear_all": [[0] * 5, [0] * 5],
                               "tclear_tipped": [[0] * 5, [0] * 5], "tipped_final_clear_sum": 0.0})
    c = p.cmd[first].cpu().tolist()
    f = p.diag[first].cpu().tolist()
    fl = p.flags[first].cpu().tolist()
    oks = p.slot_ok[first].cpu().tolist()
    succ = p.success[first].cpu().tolist()
    dep = st["depth"][first].clamp(max=2).cpu().tolist()
    l2 = hasattr(p, "l2_ng")
    if l2:
        ng, ht, rl = p.l2_ng[first].cpu().tolist(), p.l2_ht[first].cpu().tolist(), p.l2_rel[first].cpu().tolist()
        L = D.setdefault("l2", {})
    for e in range(len(c)):
        D["episodes"] += 1
        D["joint_fail"] += int(not succ[e])
        D["ep_other_moved"] += int(fl[e][-1])
        D["ep_other_moved_fail"] += int(fl[e][-1] and not succ[e])
        nact = sum(1 for s in range(N_SLOT) if c[e][s][2] == 1)
        if not succ[e]:
            D["single_pair_fail"][0 if nact == 1 else 1] += 1
        for s in range(N_SLOT):
            if c[e][s][2] != 1:
                continue
            d = dict(zip(DIAG_KEYS, f[e][s]))
            typ, depth = OBJ_TYPE[c[e][s][0]], dep[e][s]
            k = _diag_class(fl[e][5 * s: 5 * s + 5], bool(oks[e][s]), d, depth)
            D["slots"] += 1
            for tab in (D["cls"], D["cls_type"][typ], D["cls_depth"][depth]):
                tab[k] = tab.get(k, 0) + 1
            D["slots_type_depth"][typ][depth] += 1
            b = sum(1 for t in _TCLEAR_BINS if d["tclear"] >= t)
            D["tclear_all"][typ][b] += 1
            if k == "ok" and not succ[e]:
                D["ok_slot_in_failed_ep"] += 1
            if k == "tipped":
                D["tipped_type_depth"][typ][depth] += 1
                D["tclear_tipped"][typ][b] += 1
                D["tipped_after_placed"] += int(fl[e][5 * s + 3])
                D["tipped_final_clear_sum"] += d["clear"]
            if l2:  # per class: grasp attempts, carry tilt, first-release state (BITIDY_CB_LOG2)
                q = L.setdefault(k, {"n": 0, "mind": [0, 0, 0, 0], "push2cm": 0, "air_any": 0, "air_sum": 0,
                                     "htilt": [0, 0, 0, 0], "rel_n": 0, "rel_tilt": [0, 0, 0, 0], "rel_dz": [0, 0, 0, 0],
                                     "rel_dxy": [0, 0, 0, 0], "rel_clr5mm": 0, "rel_in": 0})
                m, pu, air = ng[e][s]
                q["n"] += 1
                q["mind"][sum(1 for t in (0.01, 0.02, 0.04) if m >= t)] += 1
                q["push2cm"] += int(pu > 0.02)
                q["air_any"] += int(air > 0)
                q["air_sum"] += int(air)
                q["htilt"][sum(1 for t in (0.2, 0.4, 0.7) if ht[e][s] >= t)] += 1
                r = rl[e][s]
                if r[0] >= 0:
                    q["rel_n"] += 1
                    q["rel_tilt"][sum(1 for t in (0.1, 0.2, 0.4) if r[0] >= t)] += 1
                    q["rel_dz"][sum(1 for t in (0.01, 0.025, 0.05) if r[2] >= t)] += 1
                    q["rel_dxy"][sum(1 for t in (0.005, 0.01, 0.02) if r[1] >= t)] += 1
                    q["rel_clr5mm"] += int(r[3] < 0.005)
                    q["rel_in"] += int(r[4] > 0.5)
    print("BITIDY_CB_DIAG=" + json.dumps(D), flush=True)


def _rate(t: torch.Tensor) -> list:
    t = t.cpu()
    return [round(float(r[1]) / max(int(r[0]), 1), 3) for r in t]


class cup_basket_success(ManagerTermBase):
    """Latches each episode's flags into ``env._bi_final`` (last column = joint success) and prints first-episode success:
    joint (all / single / pair instructions), per arm, per basket, per type, per stack depth, funnel
    (``BITIDY_CB_STATS``). Per-pair numbers count each active slot (latched per-arm success)."""

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
        first = ids[~st["seen"][ids] & (env.episode_length_buf[ids] > 0)]
        if len(first):
            st["seen"][first] = True
            c = p.cmd[first]
            ok = p.success[first].long()
            one = torch.ones_like(ok)
            nact = (c[:, :, 2] == 1).sum(-1).clamp(1, 2)
            for idx in (torch.zeros_like(nact), nact):
                st["s_joint"][..., 0].index_put_((idx,), one, accumulate=True)
                st["s_joint"][..., 1].index_put_((idx,), ok, accumulate=True)
            for s in range(N_SLOT):
                act = c[:, s, 2] == 1
                if not bool(act.any()):
                    continue
                cs, so = c[act, s], p.slot_ok[first][act, s].long()
                o1 = torch.ones_like(so)
                typ = obj_type(env.device)[cs[:, 0]]
                d = st["depth"][first][act, s].clamp(max=2)
                arm = torch.full_like(so, s)
                for key, idx in (("s_arm", (arm,)), ("s_basket", (cs[:, 1],)), ("s_type", (typ,)), ("s_depth", (d,)),
                                 ("s_type_depth", (typ, d))):
                    st[key][..., 0].index_put_(idx, o1, accumulate=True)
                    st[key][..., 1].index_put_(idx, so, accumulate=True)
            if DIAG:
                _diag_account(env, first, p)
            st["funnel"] += torch.cat((one.sum().view(1), env._bi_final[first].long().sum(0)))
            fu = st["funnel"].cpu()
            funnel = {k: round(float(fu[1 + q]) / max(int(fu[0]), 1), 3) for q, k in enumerate(list(FLAGS) + ["success"])}
            sj = st["s_joint"].cpu()
            print(f"BITIDY_CB_STATS n={int(sj[0, 0])} joint={_rate(sj)[0]} joint_single={_rate(sj)[1]} "
                  f"joint_pair={_rate(sj)[2]} n_single_pair={sj[1:, 0].tolist()} per_arm(L,R)={_rate(st['s_arm'])} "
                  f"n_arm={st['s_arm'][:, 0].tolist()} per_basket={_rate(st['s_basket'])} per_type(cup,bowl)={_rate(st['s_type'])} "
                  f"per_depth(0,1,2)={_rate(st['s_depth'])} n_depth={st['s_depth'][:, 0].tolist()} "
                  f"type_x_depth={[_rate(t) for t in st['s_type_depth']]} funnel={funnel} "
                  f"fallback={st['fallback']}/{st['sampled']} swapped={st['swapped']}", flush=True)
        p.reset(ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        p = progress(env)
        p.update(env)
        if os.environ.get("BITIDY_NO_SUCCESS_END"):
            return torch.zeros_like(p.success)
        return p.success.clone()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (objs_pos(env)[..., 2] < TABLE_TOP - 0.05).any(-1)


# ---------------------------------------------------------------- scene


@clone
def _spawn_vessel(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Tapered cup / bowl as one rigid body: ``nseg`` tilted box wall segments (polygon ring) and two crossed square
    floor plates; origin at the bottom centre."""
    from pxr import Gf, UsdGeom, UsdPhysics

    g = GEO[cfg.kind]
    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(g["mass"])
    a, t, h, nseg = g["alpha"], g["t"], g["h"], g["nseg"]
    length = h / math.cos(a)
    rc = 0.5 * (g["r_top"] + g["r_bot"]) - 0.5 * t / math.cos(a)
    width = 2 * g["r_top"] * math.tan(math.pi / nseg) + 0.0005
    parts = []
    for i in range(nseg):
        th = 2 * math.pi * i / nseg
        parts.append((f"w{i}", (rc * math.cos(th), rc * math.sin(th), g["z0"] + h / 2), math.degrees(th), math.degrees(a),
                      (t, width, length)))
    side = math.sqrt(2) * (g["r_bot"] - 0.001)
    for i, rz in enumerate((0.0, 45.0)):
        parts.append((f"floor{i}", (0.0, 0.0, g["floor"] / 2), rz, 0.0, (side, side, g["floor"])))
    for name, pos, rz, ry, size in parts:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddRotateZOp().Set(rz)
        xform.AddRotateYOp().Set(ry)
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@clone
def _spawn_basket(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Open basket (4 low walls + floor) as one kinematic rigid body floating 1 mm over the desk; origin at the bottom
    centre, length along local x."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.3)
    hu, hv, wz = IN_L / 2 + B_WALL / 2, IN_W / 2 + B_WALL / 2, (B_WALL_H + WALL_GAP) / 2
    parts = (("front", (hu, 0.0, wz), (B_WALL, IN_W + 2 * B_WALL, B_WALL_H - WALL_GAP)),
             ("back", (-hu, 0.0, wz), (B_WALL, IN_W + 2 * B_WALL, B_WALL_H - WALL_GAP)),
             ("left", (0.0, hv, wz), (IN_L, B_WALL, B_WALL_H - WALL_GAP)),
             ("right", (0.0, -hv, wz), (IN_L, B_WALL, B_WALL_H - WALL_GAP)),
             ("floor", (0.0, 0.0, WALL_GAP + B_FLOOR / 2), (IN_L, IN_W, B_FLOOR)))
    for name, pos, size in parts:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        shade = 0.8 if name == "floor" else 1.0
        cube.CreateDisplayColorAttr([Gf.Vec3f(*(shade * c for c in cfg.color))])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_workshop_contact_material(prim, prim_path)
    return prim


@configclass
class VesselSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_vessel
    kind: int = 0
    color: tuple[float, float, float] = (0.9, 0.1, 0.1)


@configclass
class BasketSpawnCfg(sim_utils.SpawnerCfg):
    func = _spawn_basket
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)


_DEFAULT_BASKETS = ((ROW_X, -ROW_PITCH), (ROW_X, 0.0), (ROW_X, ROW_PITCH))


def _basket(j: int) -> RigidObjectCfg:
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Basket{j}", spawn=BasketSpawnCfg(color=BASKET_COLORS[j]),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(*_DEFAULT_BASKETS[j], TABLE_TOP)))


def _obj(i: int) -> RigidObjectCfg:
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Obj{i}", spawn=VesselSpawnCfg(kind=OBJ_TYPE[i], color=OBJ_COLORS[i]),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(*FIXED_SLOTS[i], TABLE_TOP)))


@configclass
class CupBasketSceneCfg(base.BiSceneCfg):
    robot_left = base._arm("robot_left", 0.0, x=LEFT_BASE[0])
    robot_right = base._arm("robot_right", 0.0, x=FACING_X, facing_back=True)
    cube = None
    target_pad = None
    right_fixed_jaw_cube = None
    right_moving_jaw_cube = None
    basket0 = _basket(0)
    basket1 = _basket(1)
    basket2 = _basket(2)
    obj0 = _obj(0)
    obj1 = _obj(1)
    obj2 = _obj(2)
    obj3 = _obj(3)
    obj4 = _obj(4)
    obj5 = _obj(5)


# ---------------------------------------------------------------- config

OPEN_DEG = _f("BITIDY_CB_OPEN_DEG", 0)
"""> 0: both grippers open only this far (deg) instead of the full 14.6 deg."""


@configclass
class CupBasketActionsCfg:
    left_arm, left_gripper = base._arm_actions("robot_left")
    right_arm, right_gripper = base._arm_actions("robot_right")
    if OPEN_DEG > 0:
        left_gripper = left_gripper.replace(open_command_expr={"gripper": math.radians(OPEN_DEG)})
        right_gripper = right_gripper.replace(open_command_expr={"gripper": math.radians(OPEN_DEG)})
    if servo.ON:
        # BITIDY_CB_SERVO=1: STS3215 servo model on the goal side (servo.py); action dims and semantics unchanged
        left_arm = left_arm.replace(class_type=servo.ServoRelativeJointPositionAction)
        right_arm = right_arm.replace(class_type=servo.ServoRelativeJointPositionAction)
        left_gripper = left_gripper.replace(class_type=servo.ServoBinaryJointPositionAction)
        right_gripper = right_gripper.replace(class_type=servo.ServoBinaryJointPositionAction)


@configclass
class CupBasketCriticCfg(base.PolicyCfg):
    """Privileged (noise-free): joints etc. (base PolicyCfg), scene 54, per-arm relative 18, per-arm command 20, flags 11,
    base offsets 6."""

    cube = ObsTerm(func=scene_state)
    relative = ObsTerm(func=target_relative)
    command = ObsTerm(func=command_onehot)
    progress = ObsTerm(func=progress_flags)
    base = ObsTerm(func=base_offsets)
    if servo.ON:
        # joint_target = the written servo goal, as the real robot reports it
        left_joint_target = ObsTerm(func=servo.servo_joint_target,
                                    params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=_JOINTS)})
        right_joint_target = ObsTerm(func=servo.servo_joint_target,
                                     params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=_JOINTS)})


@configclass
class CupBasketPolicyCfg(CupBasketCriticCfg):
    """Same terms with pose-estimation noise on the vessel / basket poses."""

    cube = ObsTerm(func=scene_state, params={"noisy": True})
    relative = ObsTerm(func=target_relative, params={"noisy": True})


@configclass
class CupBasketObservationsCfg:
    policy: CupBasketCriticCfg = CupBasketPolicyCfg()
    critic: CupBasketCriticCfg = CupBasketCriticCfg()


PHYS_PARTS = [p for p in os.environ.get("BITIDY_CB_PHYS", "").split(",") if p]
"""BITIDY_CB_PHYS=gains,joints,mass,mat: physics randomization on reset (default off = no randomization). Same terms as
wrist_student/wx_env.py. BITIDY_CB_PHYS_GAIN=<half width> (default 0.3 -> stiffness/damping x[0.7,1.3]) for a curriculum;
BITIDY_CB_PHYS_JF=lo,hi joint friction scale (default 0.6,2.0); BITIDY_CB_PHYS_ARM=lo,hi armature scale (0.8,1.3)."""


def _phys_events() -> dict:
    from isaaclab.envs import mdp as imdp
    from isaaclab.managers import SceneEntityCfg

    g = _f("BITIDY_CB_PHYS_GAIN", 0.3)
    gr = (1.0 - g, 1.0 + g)
    jf, ja = _pair("BITIDY_CB_PHYS_JF", "0.6,2.0"), _pair("BITIDY_CB_PHYS_ARM", "0.8,1.3")
    arm_j = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    ev = {}
    for arm in ARMS:
        if "gains" in PHYS_PARTS:
            for key, joints in (("gains", arm_j), ("grip_gains", ["gripper"])):
                ev[f"{arm}_{key}"] = EventTerm(func=imdp.randomize_actuator_gains, mode="reset", params={
                    "asset_cfg": SceneEntityCfg(arm, joint_names=joints), "stiffness_distribution_params": gr,
                    "damping_distribution_params": gr, "operation": "scale", "distribution": "uniform"})
        if "joints" in PHYS_PARTS:
            ev[f"{arm}_joints"] = EventTerm(func=imdp.randomize_joint_parameters, mode="reset", params={
                "asset_cfg": SceneEntityCfg(arm, joint_names=arm_j + ["gripper"]), "friction_distribution_params": jf,
                "armature_distribution_params": ja, "operation": "scale", "distribution": "uniform"})
    for i in range(N_OBJ):
        if "mass" in PHYS_PARTS:
            ev[f"obj{i}_mass"] = EventTerm(func=imdp.randomize_rigid_body_mass, mode="reset", params={
                "asset_cfg": SceneEntityCfg(f"obj{i}"), "mass_distribution_params": (0.7, 1.3), "operation": "scale"})
        if "mat" in PHYS_PARTS:
            ev[f"obj{i}_mat"] = EventTerm(func=imdp.randomize_rigid_body_material, mode="reset", params={
                "asset_cfg": SceneEntityCfg(f"obj{i}"), "static_friction_range": (0.3, 0.9),
                "dynamic_friction_range": (0.3, 0.9), "restitution_range": (0.0, 0.02), "num_buckets": 64})
    return ev


def _set_effort_limit(env, env_ids, asset_cfg, limit: float):
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    n = len(ids) if not isinstance(ids, slice) else asset.num_joints
    asset.write_joint_effort_limit_to_sim_index(limits=torch.full((env.num_envs, n), limit, device=env.device),
                                                joint_ids=ids)


def _servo_events() -> dict:
    """BITIDY_CB_SERVO=1: per-episode STS3215 PD / joint parameters in absolute units (measured ranges, servo.py).
    BITIDY_CB_SERVO_KP (log-uniform), _KD, _ARMATURE, _FRICTION ranges; _EFFORT arm, _EFFORT_G gripper (N m)."""
    from isaaclab.envs import mdp as imdp
    from isaaclab.managers import SceneEntityCfg

    kp, kd = _pair("BITIDY_CB_SERVO_KP", "16,62"), _pair("BITIDY_CB_SERVO_KD", "1,3")
    arm_r, fr = _pair("BITIDY_CB_SERVO_ARMATURE", "0.02,0.2"), _pair("BITIDY_CB_SERVO_FRICTION", "0.02,0.2")
    eff, eff_g = _f("BITIDY_CB_SERVO_EFFORT", 1.47), _f("BITIDY_CB_SERVO_EFFORT_G", 1.2)
    ev = {}
    parts = servo.PARTS
    # BITIDY_CB_SERVO_PD=scale: randomize around this follower's per-joint SysID values instead of the shared absolute
    # ranges (published fits used absolute step targets; our deploy contract re-anchors the goal at 120 Hz, so joint
    # friction / gripper kp of the SysID asset carry the holding behaviour). Ranges: _KP_S (log-uniform), _KD_S, _ARM_S,
    # _FRIC_S.
    if os.environ.get("BITIDY_CB_SERVO_PD", "abs") == "scale":
        kp, kd = _pair("BITIDY_CB_SERVO_KP_S", "0.6,1.6"), _pair("BITIDY_CB_SERVO_KD_S", "0.6,1.6")
        arm_r, fr = _pair("BITIDY_CB_SERVO_ARM_S", "0.8,2.0"), _pair("BITIDY_CB_SERVO_FRIC_S", "0.6,1.5")
    op = "scale" if os.environ.get("BITIDY_CB_SERVO_PD", "abs") == "scale" else "abs"
    for arm in ARMS:
        allj = SceneEntityCfg(arm, joint_names=_JOINTS)
        ev[f"{arm}_servo_kp"] = EventTerm(func=imdp.randomize_actuator_gains, mode="reset", params={
            "asset_cfg": allj, "stiffness_distribution_params": kp, "operation": op, "distribution": "log_uniform"})
        ev[f"{arm}_servo_kd"] = EventTerm(func=imdp.randomize_actuator_gains, mode="reset", params={
            "asset_cfg": allj, "damping_distribution_params": kd, "operation": op, "distribution": "uniform"})
        ev[f"{arm}_servo_joint"] = EventTerm(func=imdp.randomize_joint_parameters, mode="reset", params={
            "asset_cfg": allj, "friction_distribution_params": fr, "armature_distribution_params": arm_r,
            "operation": op, "distribution": "uniform"})
        ev[f"{arm}_servo_eff"] = EventTerm(func=_set_effort_limit, mode="startup", params={
            "asset_cfg": SceneEntityCfg(arm, joint_names=_ARM_JOINTS), "limit": eff})
        ev[f"{arm}_servo_eff_g"] = EventTerm(func=_set_effort_limit, mode="startup", params={
            "asset_cfg": SceneEntityCfg(arm, joint_names=["gripper"]), "limit": eff_g})
    # sub-parts for ablation: kp / kd / arm (armature) / fric (frictionloss)
    for arm in ARMS:
        allj = SceneEntityCfg(arm, joint_names=_JOINTS)
        ev[f"{arm}_servo_arm"] = EventTerm(func=imdp.randomize_joint_parameters, mode="reset", params={
            "asset_cfg": allj, "armature_distribution_params": arm_r, "operation": op, "distribution": "uniform"})
        ev[f"{arm}_servo_fric"] = EventTerm(func=imdp.randomize_joint_parameters, mode="reset", params={
            "asset_cfg": allj, "friction_distribution_params": fr, "operation": op, "distribution": "uniform"})
    keep = (("pd", ("_servo_kp", "_servo_kd", "_servo_joint")), ("effort", ("_servo_eff", "_servo_eff_g")),
            ("kp", ("_servo_kp",)), ("kd", ("_servo_kd",)), ("arm", ("_servo_arm",)), ("fric", ("_servo_fric",)))
    return {k: v for k, v in ev.items() if any(p in parts and k.endswith(suf) for p, sufs in keep for suf in sufs)}


@configclass
class CupBasketEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    objects = EventTerm(func=reset_objects, mode="reset")
    if PHYS_PARTS:
        locals().update(_phys_events())
    if servo.ON:
        locals().update(_servo_events())
    if PHYS_PARTS and os.environ.get("BITIDY_CB_PHYS_LOG"):
        print(f"BITIDY_CB_PHYS parts={PHYS_PARTS} gain=+-{_f('BITIDY_CB_PHYS_GAIN', 0.3)}", flush=True)


@configclass
class CupBasketRewardsCfg:
    milestones = RewTerm(func=milestones, weight=10.0)
    success = RewTerm(func=success, weight=200.0)
    dropped = RewTerm(func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": ["dropped"]})
    if TILT_PEN > 0:
        tilt = RewTerm(func=tilt_penalty, weight=-TILT_PEN)
    if DISTURB_PEN > 0:
        disturb = RewTerm(func=disturb_penalty, weight=-DISTURB_PEN)
    if WALL_PEN > 0:
        wall = RewTerm(func=wall_penalty, weight=-WALL_PEN)
    if HTILT_PEN > 0:
        htilt = RewTerm(func=held_tilt_penalty, weight=-HTILT_PEN)
    if LOW_PEN > 0:
        low = RewTerm(func=low_traverse_penalty, weight=-LOW_PEN)


@configclass
class CupBasketTerminationsCfg:
    success = DoneTerm(func=cup_basket_success)
    dropped = DoneTerm(func=dropped)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CupBasketEnvCfg(base.BiHandoverEnvCfg):
    scene: CupBasketSceneCfg = CupBasketSceneCfg(num_envs=4096, env_spacing=1.2, replicate_physics=True)
    actions: CupBasketActionsCfg = CupBasketActionsCfg()
    observations: CupBasketObservationsCfg = CupBasketObservationsCfg()
    events: CupBasketEventsCfg = CupBasketEventsCfg()
    rewards: CupBasketRewardsCfg = CupBasketRewardsCfg()
    terminations: CupBasketTerminationsCfg = CupBasketTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S
        for name in ("newton_mjwarp", "default"):
            physics = getattr(self.sim.physics, name, None)
            solver = getattr(physics, "solver_cfg", None)
            if solver is not None and hasattr(solver, "njmax"):
                solver.njmax = int(os.environ.get("BITIDY_CB_NJMAX", 3000))
                solver.nconmax = int(os.environ.get("BITIDY_CB_NCONMAX", 1500))

    def play_mode(self):
        requested = self.scene.num_envs
        base.ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = requested
