"""Wrist-camera variant of BiTidy-CupBasket-SO101 for a deployable student (BiTidy-CupBasket-Wrist-SO101).

Same physics, layout sampling, rewards and success rule as ``cup_basket_env_cfg.CupBasketEnvCfg`` (imported, not
edited), plus
  * two wrist RGB cameras (``wrist_left`` / ``wrist_right``, ``WX_RES`` x ``WX_RES``) on each arm's gripper link at
    the asset's camera mount (``gripper/wowrobo_2MP_camera``), with a per-episode mount pose jitter;
  * an observation group ``coarse`` with exactly the layout of the teacher's ``policy`` group (163), but the vessel
    and basket poses replaced by what an overhead RGB camera gives (``Coarse``): per-episode calibration bias +
    temporally correlated 1-2 cm noise, 10 Hz refresh, freezes at the last value while an arm is over the vessel
    (occlusion) and at random drop-outs, orientation reduced to upright / lying (quantised tilt), no velocity, nest
    height from the known stack depth instead of the measured top vessel.
The teacher keeps reading the ``policy`` / ``critic`` groups (privileged), so DAgger labels are unchanged.

``WX_CAMS=0`` builds the same task without cameras (teacher / coarse-teacher baselines, no-image student).
"""

from __future__ import annotations

import json
import math
import os

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import _camera_renderer_cfg

from . import cup_basket_env_cfg as cb
from . import wx_real as R
from .env_cfg import pad_center


def _f(name: str, default) -> float:
    return float(os.environ.get(name, default))


CAMS = os.environ.get("WX_CAMS", "1") == "1"
# WX_SYNTH=1: no cameras at all; the localisation is the synthetic error model of ``Coarse.update`` fitted to the fused
# top + wrist pipeline (fast RL / DAgger at thousands of envs); the other WX_REAL factors (PROP, LAT, PHYS) stay on.
SYNTH = os.environ.get("WX_SYNTH", "0") == "1"
TOP = (os.environ.get("WX_TOP", "0") == "1" or R.REAL) and not SYNTH  # coarse poses from the overhead RGB camera + perception (wx_top) instead of synthetic noise
RES = int(_f("WX_RES", 88))
FOCAL = _f("WX_FOCAL", 11.0)  # mm on a 20.955 mm aperture: ~87 deg horizontal FOV
HIDE_HOUSING = os.environ.get("WX_HIDE_HOUSING", "1") == "1"
MOUNT_POS = _f("WX_MOUNT_POS", 0.004)  # m, per-episode mount translation jitter (uniform +-)
MOUNT_ROT = math.radians(_f("WX_MOUNT_ROT_DEG", 4.0))

# coarse overhead-camera model
SIG_XY = (_f("WX_SIG_LO", 0.01), _f("WX_SIG_HI", 0.02))  # per-episode noise std range (m)
SIG_Z = _f("WX_SIG_Z_SCALE", 0.6)  # z noise = this x xy noise
RHO = _f("WX_RHO", 0.7)  # AR(1) coefficient of the noise (per refresh)
BIAS = _f("WX_BIAS", 0.005)  # per-episode calibration bias std (m, xy)
BK_SIG, BK_BIAS = _f("WX_BK_SIG", 0.003), _f("WX_BK_BIAS", 0.006)
BK_YAW_SIG, BK_YAW_BIAS = _f("WX_BK_YAW_SIG", 0.03), _f("WX_BK_YAW_BIAS", 0.05)  # rad
HOLD = int(_f("WX_HOLD", 3))  # control steps per overhead refresh (30 Hz / 3 = 10 Hz)
ATTACH_XY = _f("WX_ATTACH_XY", 0.05)  # top-camera mode: grasp attachment radius (m)
HOLD_OFFSET = os.environ.get("WX_HOLD_OFFSET", "1") == "1"  # held vessel keeps its grasp-time offset from the pads
# deployable progress flags (WX_DEPLOY_FLAGS=1): thresholds looser than the privileged ones (estimates are ~1-2 cm off)
DF_OVER, DF_LIFT, DF_PLACE_Z, DF_CLEAR = _f("WX_DF_OVER", 0.03), _f("WX_DF_LIFT", 0.03), _f("WX_DF_PLACE_Z", 0.02), _f("WX_DF_CLEAR", 0.025)
OCC_XY, OCC_P = _f("WX_OCC_XY", 0.05), _f("WX_OCC_P", 0.6)  # arm over vessel: refresh dropped with prob OCC_P
FRZ_P, FRZ_LEN = _f("WX_FRZ_P", 0.004), (int(_f("WX_FRZ_MIN", 15)), int(_f("WX_FRZ_MAX", 60)))
TIP = _f("WX_TIP", 0.6)  # rad: tilt above this is reported as lying (90 deg), below as upright


# ---------------------------------------------------------------- coarse overhead-camera estimate


class Coarse:
    """Per-env coarse estimate of the 6 vessel poses and 3 basket poses, updated once per control step."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.n, self.dev, self.env = n, dev, env
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.bias, self.ar, self.pos = z(n, cb.N_OBJ, 3), z(n, cb.N_OBJ, 3), z(n, cb.N_OBJ, 3)
        self.quat = z(n, cb.N_OBJ, 4)
        self.quat[..., 3] = 1.0
        self.sig = z(n, 1, 1)
        self.frz = torch.zeros((n, cb.N_OBJ), dtype=torch.long, device=dev)
        self.bk_bias, self.bk_ar, self.bk = z(n, cb.N_BASKET, 3), z(n, cb.N_BASKET, 3), z(n, cb.N_BASKET, 3)
        self.phase = torch.randint(0, max(HOLD, 1), (n,), device=dev)
        self.init = torch.ones(n, dtype=torch.bool, device=dev)
        self.key = None
        self.top = None
        if TOP:
            from .wx_top import TopPerception

            self.top = TopPerception(env)
            self.init_top = torch.ones(n, dtype=torch.bool, device=dev)
        self.fuser = None
        from . import wx_fuse

        if self.top is not None and wx_fuse.FUSED:  # model-based top / wrist localisation + Kalman fusion
            self.fuser = wx_fuse.Fuser(self, env)
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        k = len(ids)
        if k == 0:
            return
        self.sig[ids] = torch.empty((k, 1, 1), device=self.dev).uniform_(*SIG_XY)
        b = torch.randn((k, cb.N_OBJ, 3), device=self.dev) * BIAS
        b[..., 2] = 0.0
        self.bias[ids] = b
        self.ar[ids] = 0.0
        self.frz[ids] = 0
        bb = torch.randn((k, cb.N_BASKET, 3), device=self.dev)
        self.bk_bias[ids] = bb * bb.new_tensor((BK_BIAS, BK_BIAS, BK_YAW_BIAS))
        self.bk_ar[ids] = 0.0
        self.init[ids] = True
        self.key = None
        if self.top is not None:
            self.top.reset(ids)
            self.init_top[ids] = True
        if getattr(self, "fuser", None) is not None:
            self.fuser.reset(ids)
            if R.WFUSE:  # the shared wrist lens / appearance / latency model starts its new episode with the env
                R.shared_wrist(self.env).reset(ids)

    @torch.no_grad()
    def update(self, env) -> None:
        key = int(env.common_step_counter)
        if key == self.key:
            return
        self.key = key
        n, dev = self.n, self.dev
        if self.fuser is not None:
            self.fuser.update(env, key)
            return
        if self.top is not None:
            self._update_top(env, key)
            return
        p = cb.objs_pos(env)
        q = cb.objs_quat(env)
        refresh = ((key + self.phase) % max(HOLD, 1) == 0).unsqueeze(-1).expand(n, cb.N_OBJ)
        # temporally correlated measurement noise (advances on refresh only)
        innov = torch.randn_like(self.ar) * self.sig * math.sqrt(1 - RHO * RHO)
        innov[..., 2] *= SIG_Z
        ar = torch.where(self.init.view(-1, 1, 1), torch.randn_like(self.ar) * self.sig * self.ar.new_tensor((1, 1, SIG_Z)),
                         RHO * self.ar + innov)
        self.ar = torch.where(refresh.unsqueeze(-1) | self.init.view(-1, 1, 1), ar, self.ar)
        meas = p + self.bias + self.ar
        # occlusion: an arm's pads over the vessel (within OCC_XY, above its origin) drop a refresh with prob OCC_P
        over = torch.zeros((n, cb.N_OBJ), dtype=torch.bool, device=dev)
        for arm in cb.ARMS:
            pad = pad_center(env, arm)
            dxy = torch.linalg.vector_norm(pad[:, None, :2] - p[..., :2], dim=-1)
            over |= (dxy < OCC_XY) & (pad[:, None, 2] > p[..., 2] + 0.02)
        drop = over & (torch.rand((n, cb.N_OBJ), device=dev) < OCC_P)
        start = (torch.rand((n, cb.N_OBJ), device=dev) < FRZ_P) & (self.frz == 0)
        self.frz = torch.where(start, torch.randint(FRZ_LEN[0], FRZ_LEN[1] + 1, (n, cb.N_OBJ), device=dev), (self.frz - 1).clamp(min=0))
        upd = (refresh & ~drop & (self.frz == 0)) | self.init.unsqueeze(-1)
        if SYNTH:  # as in the camera pipelines: a held vessel follows the pads (FK + grasp-time offset), see _attach_held
            valid = env.episode_length_buf >= 1
            if not hasattr(self, "held"):
                self.held = torch.zeros((n, cb.N_SLOT), dtype=torch.bool, device=dev)
            self.held &= valid.unsqueeze(-1)
            ar_ = torch.arange(n, device=dev)
            for s_ in range(cb.N_SLOT):
                upd[ar_, cb.slot_obj(env, s_)] &= ~self.held[:, s_] | self.init
        self.pos = torch.where(upd.unsqueeze(-1), meas, self.pos)
        # orientation: upright (identity) or lying along the measured direction (90 deg about the horizontal axis)
        ez = torch.zeros_like(p)
        ez[..., 2] = 1.0
        up = quat_apply(q.reshape(-1, 4), ez.reshape(-1, 3)).reshape(n, cb.N_OBJ, 3)
        h = torch.nn.functional.normalize(up[..., :2] + 1e-6, dim=-1)
        s2 = math.sqrt(0.5)
        lying = torch.stack((-h[..., 1] * s2, h[..., 0] * s2, torch.zeros_like(h[..., 0]), torch.full_like(h[..., 0], s2)), -1)
        upright = torch.zeros_like(lying)
        upright[..., 3] = 1.0
        qe = upright if SYNTH else torch.where((up[..., 2] < math.cos(TIP)).unsqueeze(-1), lying, upright)  # the real pipeline reports upright only
        self.quat = torch.where(upd.unsqueeze(-1), qe, self.quat)
        # baskets (static): bias + small correlated noise, refreshed at 10 Hz
        bk = cb.baskets(env)
        brefresh = refresh[:, :1].unsqueeze(-1) | self.init.view(-1, 1, 1)
        binn = torch.randn_like(self.bk_ar) * self.bk_ar.new_tensor((BK_SIG, BK_SIG, BK_YAW_SIG))
        self.bk_ar = torch.where(brefresh, RHO * self.bk_ar + math.sqrt(1 - RHO * RHO) * binn, self.bk_ar)
        self.bk = torch.where(brefresh, bk + self.bk_bias + self.bk_ar, self.bk)
        self.init[:] = False
        if SYNTH:
            self._attach_held(env)

    def _update_top(self, env, key: int) -> None:
        """Overhead camera perception (wx_top): one synchronous refresh every HOLD steps for all envs, plus a refresh
        as soon as a reset env has its first post-reset frame (episode step >= 1)."""
        from .wx_top import TIP_RATIO, VIS_MIN

        dev = self.dev
        t = env.episode_length_buf
        valid = t >= 1
        need = self.init_top & valid
        if not hasattr(self, "held"):
            self.held = torch.zeros((self.n, cb.N_SLOT), dtype=torch.bool, device=dev)
        self.held &= valid.unsqueeze(-1)
        if not (key % max(HOLD, 1) == 0 or bool(need.any())):
            self._attach_held(env)
            return
        m = self.top.measure(valid)
        vis = m["vis"]
        upd = valid.unsqueeze(-1) & ((vis > VIS_MIN) | (self.init_top.unsqueeze(-1) & (vis > 0.05)))
        for s_ in range(cb.N_SLOT):  # a vessel held by its arm follows the pads (FK), not the occluded camera view
            k = cb.slot_obj(env, s_)
            upd[torch.arange(self.n, device=dev), k] &= ~self.held[:, s_]
        bk, bk_ok = self.top.baskets(m, valid)
        self.bk = torch.where(bk_ok.unsqueeze(-1), bk, self.bk)
        xy = m["xy"]
        # diagnostics (gap ladder): replace parts of the camera estimate by the sim ground truth
        if os.environ.get("WX_DBG_TRUEBK") == "1":
            self.bk = cb.baskets(env).clone()
        if os.environ.get("WX_DBG_TRUEVES") == "1":
            xy = cb.objs_pos(env)[..., :2].clone()
            upd = valid.unsqueeze(-1).expand_as(upd).clone()
            for s_ in range(cb.N_SLOT):
                k = cb.slot_obj(env, s_)
                upd[torch.arange(self.n, device=dev), k] &= ~self.held[:, s_]
        # resting surface: basket floor inside an estimated basket footprint, else the table
        uv = cb.to_basket(self.bk.unsqueeze(1), xy.unsqueeze(2))  # (n, 6, 3, 2)
        inside = (uv.abs() < uv.new_tensor((cb.IN_L / 2, cb.IN_W / 2))).all(-1).any(-1)
        z = torch.where(inside, torch.full_like(xy[..., 0], cb.FLOOR_Z), torch.full_like(xy[..., 0], cb.TABLE_TOP))
        typ = cb.obj_type(dev)
        tipped = m["ratio"] > torch.where(typ == cb.CUP, TIP_RATIO[0], TIP_RATIO[1])
        a = m["axis"]
        s2 = math.sqrt(0.5)
        lying = torch.stack((-a[..., 1] * s2, a[..., 0] * s2, torch.zeros_like(a[..., 0]), torch.full_like(a[..., 0], s2)), -1)
        upright = torch.zeros_like(lying)
        upright[..., 3] = 1.0
        qe = torch.where(tipped.unsqueeze(-1), lying, upright)
        self.pos = torch.where(upd.unsqueeze(-1), torch.cat((xy, z.unsqueeze(-1)), -1), self.pos)
        # never seen this episode (hidden in a stack since the start): the zone centre of the basket the system put it
        # in (known from its own command history, cb state "loc"); table objects keep the last value
        self.seen = torch.where(valid.unsqueeze(-1), getattr(self, "seen", torch.zeros_like(upd)) | upd, torch.zeros_like(upd))
        loc = cb._state(env)["loc"]
        typ = cb.obj_type(dev).expand_as(loc)
        zc = cb.zone_center(self.bk.gather(
            1, loc.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)), typ)
        hid = ~self.seen & (loc >= 0) & valid.unsqueeze(-1)
        self.pos = torch.where(hid.unsqueeze(-1), torch.cat((zc, torch.full_like(zc[..., :1], cb.FLOOR_Z)), -1), self.pos)
        self.quat = torch.where(upd.unsqueeze(-1), qe, self.quat)
        self.tipped = torch.where(upd, tipped, getattr(self, "tipped", torch.zeros_like(tipped)))
        self._attach_held(env)
        self.top.account(m, upd, self.pos[..., :2], self.tipped, self.bk, valid)
        self.init_top &= ~valid
        self.init[:] = False


def _attach_held_impl(self, env) -> None:
    """Deployable grasp bookkeeping (gripper command + FK only): when an arm closes its gripper while its commanded
    vessel's estimate is within ATTACH_XY of its pads, the vessel is held until the gripper opens; while held its
    estimate is the pad centre minus the rim height, upright, updated every control step."""
    act = env.action_manager.action
    g = cb._geo_t(self.dev)
    ar = torch.arange(self.n, device=self.dev)
    if not hasattr(self, "hoff"):
        self.hoff = torch.zeros(self.n, cb.N_SLOT, 2, device=self.dev)
    for s_ in range(cb.N_SLOT):
        closed = act[:, 6 * s_ + 5] < 0
        k = cb.slot_obj(env, s_)
        pad = pad_center(env, cb.ARMS[s_])
        est = self.pos[ar, k]
        near = torch.linalg.vector_norm(est[:, :2] - pad[:, :2], dim=-1) < ATTACH_XY
        was = self.held[:, s_].clone()
        self.held[:, s_] = closed & cb.slot_active(env, s_) & (self.held[:, s_] | near)
        typ = cb.slot_type(env, s_)
        rim = g["rim"][typ]
        h = self.held[:, s_]
        xy = pad[:, :2]
        if HOLD_OFFSET:  # rim pinch: the vessel centre sits beside the pads; keep the grasp-time offset in the jaw frame
            jp = cb.jaw_points(env, cb.ARMS[s_])
            ex = torch.nn.functional.normalize((jp[1] - jp[0])[:, :2], dim=-1)
            ey = torch.stack((-ex[:, 1], ex[:, 0]), -1)
            d = est[:, :2] - pad[:, :2]
            loc = torch.stack(((d * ex).sum(-1), (d * ey).sum(-1)), -1)
            r = (g["r_top"][typ] + 0.01).unsqueeze(-1)
            loc = loc * (r / torch.linalg.vector_norm(loc, dim=-1, keepdim=True).clamp(min=1e-6)).clamp(max=1.0)
            self.hoff[:, s_] = torch.where((h & ~was).unsqueeze(-1), loc, self.hoff[:, s_])
            xy = pad[:, :2] + self.hoff[:, s_, :1] * ex + self.hoff[:, s_, 1:] * ey
        newp = torch.cat((xy, (pad[:, 2] - rim).unsqueeze(-1)), -1)
        self.pos[ar, k] = torch.where(h.unsqueeze(-1), newp, est)
        if self.top is not None:  # keep the camera tracker's seed with the carried vessel (re-identification on release)
            self.top.c0[ar, k] = torch.where(h.unsqueeze(-1), pad[:, :2], self.top.c0[ar, k])
        up = torch.zeros_like(self.quat[ar, k])
        up[:, 3] = 1.0
        self.quat[ar, k] = torch.where(h.unsqueeze(-1), up, self.quat[ar, k])


Coarse._attach_held = _attach_held_impl


def coarse(env) -> Coarse:
    env = getattr(env, "unwrapped", env)
    if not hasattr(env, "_wx_coarse"):
        env._wx_coarse = Coarse(env)
    c = env._wx_coarse
    c.update(env)
    return c


def deploy_flags(env) -> torch.Tensor:
    """Progress flags (layout of ``cb.FLAGS``, 11) from deployable signals only: the coarse / fused estimate (what the
    policy sees), the grasp bookkeeping (gripper command + FK) and the commanded task. Sticky per episode like the
    privileged ones; other_moved is unknown (0). Computed once per control step."""
    env = getattr(env, "unwrapped", env)
    c = coarse(env)
    if getattr(c, "df_key", None) == c.key and hasattr(c, "dflags"):
        return c.dflags
    c.df_key = c.key
    v = _view(env)
    n, dev = env.num_envs, env.device
    t = env.episode_length_buf
    if not hasattr(c, "dflags"):
        c.dflags = torch.zeros(n, len(cb.FLAGS), device=dev)
        c.dref = torch.zeros(n, cb.N_OBJ, device=dev)
        c.dheld_n = torch.zeros(n, cb.N_SLOT, dtype=torch.long, device=dev)
    fresh = t <= 1
    c.dflags[fresh] = 0.0
    settling = t < cb.SETTLE_STEPS
    c.dref = torch.where(settling.unsqueeze(-1), v.pos[..., 2], c.dref)
    held = getattr(c, "held", torch.zeros(n, cb.N_SLOT, dtype=torch.bool, device=dev)) & ~settling.unsqueeze(-1)
    c.dheld_n = torch.where(held, c.dheld_n + 1, 0)
    g = cb._geo_t(dev)
    act = env.action_manager.action
    ar = torch.arange(n, device=dev)
    f = c.dflags.bool()
    new = torch.zeros_like(f)
    for s in range(cb.N_SLOT):
        k = cb.slot_obj(env, s)
        typ = cb.slot_type(env, s)
        p = v.pos[ar, k]
        stage = lambda name: f[:, 5 * s + cb._STAGES.index(name)]  # noqa: E731
        h = held[:, s]
        members = cb.stack_members(env, s)
        depth = members.sum(-1)
        top = torch.where(members, cb._state(env)["level"], -1).argmax(-1)
        top_p = cb._pick(v.pos, top)
        zc = cb.zone_center(cb._pick(v.bk, cb.command(env)[:, s, 1]), typ)
        axis = torch.where((depth > 0).unsqueeze(-1), top_p[:, :2], zc)
        gz = torch.where(depth > 0, top_p[:, 2] + g["pitch"][typ], torch.full_like(zc[:, 0], cb.FLOOR_Z))
        dxy = torch.linalg.vector_norm(p[:, :2] - axis, dim=-1)
        grasp = c.dheld_n[:, s] >= 3
        lift = stage("grasp") & h & (p[:, 2] > c.dref[ar, k] + DF_LIFT)
        over = stage("lift") & h & (dxy < DF_OVER)
        placed = stage("over") & (dxy < DF_OVER) & (p[:, 2] < gz + DF_PLACE_Z)
        opened = act[:, 6 * s + 5] >= 0
        pad = pad_center(env, cb.ARMS[s])
        rimz = p[:, 2] + g["rim"][typ]
        away = (torch.linalg.vector_norm(pad[:, :2] - p[:, :2], dim=-1) > g["r_top"][typ] + DF_CLEAR) | (pad[:, 2] > rimz + DF_CLEAR)
        released = stage("placed") & opened & ~h & away
        new[:, 5 * s: 5 * s + 5] = torch.stack((grasp, lift, over, placed, released), -1) & cb.slot_active(env, s).unsqueeze(-1)
    c.dflags = (f | new).float()
    if os.environ.get("WX_DF_STATS") == "1":  # agreement with the privileged flags (sim truth)
        tr = cb.progress(env).flags
        est = c.dflags.bool()
        if not hasattr(c, "dfs"):
            c.dfs = torch.zeros(4, len(cb.FLAGS), device=dev)  # tp, fp, fn, n
        c.dfs += torch.stack(((tr & est).sum(0), (~tr & est).sum(0), (tr & ~est).sum(0), torch.full_like(tr[0], n, dtype=torch.long))).float()
        c.dfs_k = getattr(c, "dfs_k", 0) + 1
        if c.dfs_k % 300 == 0:
            tp, fp, fn_ = c.dfs[0], c.dfs[1], c.dfs[2]
            out = {nm: (round(float(tp[i] / (tp[i] + fn_[i]).clamp(min=1)), 3), round(float(fp[i] / (tp[i] + fp[i]).clamp(min=1)), 3))
                   for i, nm in enumerate(cb.FLAGS[:10])}
            print("WX_DF_STATS recall,false_frac " + json.dumps(out), flush=True)
    return c.dflags


def coarse_reset(env, env_ids) -> None:
    env = getattr(env, "unwrapped", env)
    if not hasattr(env, "_wx_coarse"):
        env._wx_coarse = Coarse(env)
    env._wx_coarse.reset(env_ids)


class _Delayed:
    """View of a Coarse estimate delayed by a per-env number of control steps (top-camera latency)."""

    def __init__(self, c, env):
        n, dev = c.n, c.dev
        if not hasattr(c, "hist_pos"):
            c.hist_pos = c.pos.unsqueeze(0).repeat(5, 1, 1, 1)
            c.hist_quat = c.quat.unsqueeze(0).repeat(5, 1, 1, 1)
            c.hist_bk = c.bk.unsqueeze(0).repeat(5, 1, 1, 1)
            c.hist_key = None
            c.lat = torch.randint(2, 5, (n,), device=dev)
        if c.hist_key != c.key:
            c.hist_pos = torch.cat((c.pos.unsqueeze(0), c.hist_pos[:-1]), 0)
            c.hist_quat = torch.cat((c.quat.unsqueeze(0), c.hist_quat[:-1]), 0)
            c.hist_bk = torch.cat((c.bk.unsqueeze(0), c.hist_bk[:-1]), 0)
            new = env.episode_length_buf <= 1  # a new episode has no older estimates: use the current ones
            if bool(new.any()):
                c.hist_pos[:, new] = c.pos[new].unsqueeze(0)
                c.hist_quat[:, new] = c.quat[new].unsqueeze(0)
                c.hist_bk[:, new] = c.bk[new].unsqueeze(0)
                c.lat[new] = torch.randint(2, 5, (int(new.sum()),), device=dev)
            c.hist_key = c.key
        ar = torch.arange(n, device=dev)
        self.pos, self.quat, self.bk = c.hist_pos[c.lat, ar], c.hist_quat[c.lat, ar], c.hist_bk[c.lat, ar]


class _TruePose:
    """Diagnostics (``WX_DBG_TRUEPOSE``): the policy view with parts replaced by the sim truth.
    all = every vessel pose; held = the carried vessels' poses; quat = every vessel orientation."""

    def __init__(self, v, env, mode: str):
        parts = set(mode.split(","))  # combinable, e.g. "all,bk"; bk = true basket poses
        self.bk = cb.baskets(env).clone() if "bk" in parts else v.bk
        pos, quat = v.pos.clone(), v.quat.clone()
        tp, tq = cb.objs_pos(env), cb.objs_quat(env)
        if "all" in parts:
            pos, quat = tp.clone(), tq.clone()
        if "pos" in parts:  # true positions, estimated (upright / lying) orientations as in training
            pos = tp.clone()
        if "quat" in parts:
            quat = tq.clone()
        if "held" in parts:
            ar = torch.arange(env.num_envs, device=env.device)
            c = coarse(env)
            for s_ in range(cb.N_SLOT):
                k = cb.slot_obj(env, s_)
                h = c.held[:, s_].unsqueeze(-1) if hasattr(c, "held") else torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
                pos[ar, k] = torch.where(h, tp[ar, k], pos[ar, k])
                quat[ar, k] = torch.where(h, tq[ar, k], quat[ar, k])
        if "heldpos" in parts:  # carried vessels' true positions only
            ar = torch.arange(env.num_envs, device=env.device)
            c = coarse(env)
            for s_ in range(cb.N_SLOT):
                k = cb.slot_obj(env, s_)
                h = c.held[:, s_].unsqueeze(-1) if hasattr(c, "held") else torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
                pos[ar, k] = torch.where(h, tp[ar, k], pos[ar, k])
        self.pos, self.quat = pos, quat


def _view(env):
    c = coarse(env)
    uenv = getattr(env, "unwrapped", env)
    # the fused estimate already applies every measurement with its own latency (wx_fuse)
    v = _Delayed(c, uenv) if R.on("LAT") and c.fuser is None else c
    from . import wx_fuse

    if wx_fuse.STATS and getattr(c, "stats_key", None) != c.key:
        c.stats_key = c.key
        if not hasattr(c, "fstats"):
            c.fstats = wx_fuse.FuseStats(uenv)
        c.fstats(uenv, v, c)
    mode = os.environ.get("WX_DBG_TRUEPOSE")
    if mode:
        return _TruePose(v, uenv, mode)
    return v


def coarse_scene(env) -> torch.Tensor:
    """Layout of ``cb.scene_state`` (54) from the coarse estimate."""
    c = _view(env)
    n = env.num_envs
    b = torch.cat((c.bk[..., :2], torch.cos(c.bk[..., 2:3]), torch.sin(c.bk[..., 2:3])), -1)
    o = torch.cat((c.pos, c.quat), -1)
    return torch.cat((o.reshape(n, -1), b.reshape(n, -1)), -1)


def coarse_relative(env) -> torch.Tensor:
    """Layout of ``cb.target_relative`` (18) from the coarse estimate; velocity unknown (zeros)."""
    return coarse_relative_from(env, _view(env))


def coarse_relative_from(env, c) -> torch.Tensor:
    g = cb._geo_t(env.device)
    parts = []
    for s in range(cb.N_SLOT):
        k = cb.slot_obj(env, s)
        typ = cb.slot_type(env, s)
        p, q = cb._pick(c.pos, k), cb._pick(c.quat, k)
        up = torch.zeros_like(p)
        up[:, 2] = g["rim"][typ]
        rim = p + quat_apply(q, up)
        members = cb.stack_members(env, s)
        depth = members.sum(-1)
        top = torch.where(members, cb._state(env)["level"], -1).argmax(-1)
        top_p = cb._pick(c.pos, top)
        zc = cb.zone_center(cb._pick(c.bk, cb.command(env)[:, s, 1]), typ)
        axis = torch.where((depth > 0).unsqueeze(-1), top_p[:, :2], zc)
        z = cb.FLOOR_Z + depth.float() * g["pitch"][typ]
        goal = torch.cat((axis, z.unsqueeze(-1)), -1)
        v = torch.cat((rim - pad_center(env, cb.ARMS[s]), goal - p, torch.zeros_like(p)), -1)
        parts.append(v * cb.slot_active(env, s).unsqueeze(-1).float())
    return torch.cat(parts, -1)


# ---------------------------------------------------------------- config


WSEG_COLLECT = os.environ.get("WX_WSEG_COLLECT") == "1"  # wrist-segmenter data: raw semantic ids on the wrist cameras


def _wrist(arm: str) -> CameraCfg:
    real = R.REAL
    focal = 20.955 / 2 / math.tan(math.radians(R.WRIST_RENDER_HFOV / 2)) if real else FOCAL
    kw = {}
    types = ["rgb"]
    rcfg = _camera_renderer_cfg()
    if WSEG_COLLECT:
        types.append("semantic_segmentation")
        kw["colorize_semantic_segmentation"] = False
        for name in ("newton_renderer", "default", "ovrtx"):
            r = getattr(rcfg, name, None)
            for flag in ("colorize_instance_segmentation", "colorize_semantic_segmentation"):
                if r is not None and hasattr(r, flag):
                    setattr(r, flag, False)
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{arm}/gripper/wowrobo_2MP_camera/pinhole",
        spawn=PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.005, 5.0)),
        offset=CameraCfg.OffsetCfg(convention="opengl"),
        data_types=types,
        width=R.WRIST_RENDER_W if real else RES,
        height=R.WRIST_RENDER_H if real else RES,
        update_period=0.0,
        update_latest_camera_pose=True,
        renderer_cfg=rcfg,
        background_color=(0.3, 0.3, 0.3),
        **kw,
    )


def _desk_visual():
    from isaaclab.assets import VisualMaterialCfg

    return VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Desk/geometry/material", spawn=None)


def _distractor(i: int, xy, size, color):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/Distractor{i}",
                        spawn=sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(xy[0], xy[1], cb.TABLE_TOP + size[2] / 2)))


DESK_COLORS = ((0.86, 0.86, 0.83), (0.62, 0.50, 0.36), (0.36, 0.25, 0.16), (0.45, 0.45, 0.45), (0.07, 0.07, 0.07),
               (0.16, 0.19, 0.26), (0.55, 0.45, 0.35))  # white, light wood, dark wood, grey, black, dark mat, sim default


def _top_cam():
    from .wx_top import top_camera_cfg

    return top_camera_cfg()


@configclass
class WristSceneCfg(cb.CupBasketSceneCfg):
    if CAMS:
        wrist_left = _wrist("robot_left")
        wrist_right = _wrist("robot_right")
    if TOP:
        top_cam = _top_cam()
    if R.on("VIS"):
        desk_visual = _desk_visual()
        distractor0 = _distractor(0, (0.40, 0.385), (0.09, 0.05, 0.04), (0.92, 0.78, 0.10))  # yellow box
        distractor1 = _distractor(1, (-0.08, -0.39), (0.12, 0.07, 0.02), (0.93, 0.93, 0.93))  # white box / paper
        distractor2 = _distractor(2, (0.05, 0.385), (0.06, 0.06, 0.08), (0.45, 0.25, 0.55))  # purple tin

    def __post_init__(self):
        sp = getattr(super(), "__post_init__", None)
        if sp is not None:
            sp()
        # semantic labels for the segmenter's ground truth (Newton segmentation only reports labelled prims)
        for i in range(cb.N_OBJ):
            getattr(self, f"obj{i}").spawn.semantic_tags = [("class", f"obj{i}")]
        for j in range(cb.N_BASKET):
            getattr(self, f"basket{j}").spawn.semantic_tags = [("class", f"basket{j}")]
        self.robot_left.spawn.semantic_tags = [("class", "robot")]
        self.robot_right.spawn.semantic_tags = [("class", "robot")]
        if CAMS and HIDE_HOUSING:
            # Newton ray tracing ignores the near clipping plane: hide the housing the camera sits in
            self.robot_left.spawn.hide_camera_housing = True
            self.robot_right.spawn.hide_camera_housing = True


@configclass
class CoarsePolicyCfg(cb.CupBasketCriticCfg):
    """Teacher ``policy`` layout (163) with the coarse overhead-camera poses."""

    cube = ObsTerm(func=coarse_scene)
    relative = ObsTerm(func=coarse_relative)


@configclass
class WristObservationsCfg(cb.CupBasketObservationsCfg):
    coarse: CoarsePolicyCfg = CoarsePolicyCfg()


def _mount(arm: str, cam: str) -> EventTerm:
    return EventTerm(func=mdp.randomize_wrist_camera_mount, mode="reset",
                     params={"sensor_cfg": SceneEntityCfg(cam), "asset_cfg": SceneEntityCfg(arm, body_names=["gripper"]),
                             "position_range": MOUNT_POS, "rotation_range": MOUNT_ROT})


def _phys_events() -> dict:
    from isaaclab.envs import mdp as imdp

    ev = {}
    parts = os.environ.get("WX_PHYS_PARTS", "gains,joints,mass,mat").split(",")
    arm_j = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for arm in cb.ARMS:
        ev[f"{arm}_gains"] = EventTerm(func=imdp.randomize_actuator_gains, mode="reset", params={
            "asset_cfg": SceneEntityCfg(arm, joint_names=arm_j), "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3), "operation": "scale", "distribution": "uniform"})
        ev[f"{arm}_grip_gains"] = EventTerm(func=imdp.randomize_actuator_gains, mode="reset", params={
            "asset_cfg": SceneEntityCfg(arm, joint_names=["gripper"]), "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3), "operation": "scale", "distribution": "uniform"})
        ev[f"{arm}_joints"] = EventTerm(func=imdp.randomize_joint_parameters, mode="reset", params={
            "asset_cfg": SceneEntityCfg(arm, joint_names=arm_j + ["gripper"]), "friction_distribution_params": tuple(float(v) for v in os.environ.get("WX_PHYS_JF", "0.6,2.0").split(",")),
            "armature_distribution_params": (0.8, 1.3), "operation": "scale", "distribution": "uniform"})
    for i in range(cb.N_OBJ):
        ev[f"obj{i}_mass"] = EventTerm(func=imdp.randomize_rigid_body_mass, mode="reset", params={
            "asset_cfg": SceneEntityCfg(f"obj{i}"), "mass_distribution_params": (0.7, 1.3), "operation": "scale"})
        ev[f"obj{i}_mat"] = EventTerm(func=imdp.randomize_rigid_body_material, mode="reset", params={
            "asset_cfg": SceneEntityCfg(f"obj{i}"), "static_friction_range": (0.3, 0.9), "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.02), "num_buckets": 64})
    keep = {"gains": ("_gains", "_grip_gains"), "joints": ("_joints",), "mass": ("_mass",), "mat": ("_mat",)}
    return {k: v for k, v in ev.items() if any(k.endswith(suf) for p in parts for suf in keep.get(p, ()))
            and not (k.endswith("_grip_gains") and "gains" not in parts)}


def _vis_events() -> dict:
    from isaaclab.envs import mdp as imdp

    return {"desk_color": EventTerm(func=imdp.randomize_visual_material, mode="reset", params={
        "materials": SceneEntityCfg("desk_visual"), "channels": {"color": {"choices": [list(c) for c in DESK_COLORS]}}})}


@configclass
class WristEventsCfg(cb.CupBasketEventsCfg):
    coarse = EventTerm(func=coarse_reset, mode="reset")
    if R.on("PHYS"):
        locals().update(_phys_events())
    if R.on("VIS"):
        locals().update(_vis_events())
    if CAMS and (MOUNT_POS > 0 or MOUNT_ROT > 0):
        mount_left = _mount("robot_left", "wrist_left")
        mount_right = _mount("robot_right", "wrist_right")


@configclass
class WristEnvCfg(cb.CupBasketEnvCfg):
    scene: WristSceneCfg = WristSceneCfg(num_envs=512, env_spacing=1.2, replicate_physics=True)
    observations: WristObservationsCfg = WristObservationsCfg()
    events: WristEventsCfg = WristEventsCfg()


def _render_cfg():
    from .render import with_camera

    return with_camera(WristEnvCfg)


def __getattr__(name: str):
    if name == "WristRenderEnvCfg":
        cfg = _render_cfg()
        globals()[name] = cfg
        return cfg
    raise AttributeError(name)
