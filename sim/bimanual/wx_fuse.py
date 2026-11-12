"""Model-based vessel localisation from the top and wrist cameras, fused per vessel by a Kalman filter.

Switches (all default off = the 11/8 pipeline of ``wx_env.Coarse._update_top``):
  WX_WFUSE=1    wrist cameras localise the vessels they see (needs WX_REAL=1 and WX_WSEGNET=<wrist segmenter>); turns
                on WX_TOP_FIT and WX_TOP_KF unless they are set to 0.
  WX_TOP_FIT=1  top camera: silhouette fit of the known vessel shape instead of mask centroid + calibrated height plane.
  WX_TOP_KF=1   per-vessel constant-position Kalman filter (variance grows with time and near an arm) instead of the hard
                freeze; measurements are applied with their own latency (top 2-4 steps, wrist 1-3 steps), so the policy
                view is not delayed again.
  WX_FUSE_STATS=1  error statistics of what the policy receives against the sim truth (works for both pipelines):
                ``WX_FUSE_STATS {json}`` every WX_FUSE_STATS_EVERY steps.

Silhouette fit (one camera, one vessel): the vessel is a frustum (known radii / height, upright) standing at base height
zb. Its image silhouette is predicted per sample pixel by testing the pixel's viewing ray against the solid at L heights.
Samples come from a square window around the vessel's mask (S x S points, segmenter logits sampled bilinearly =
sub-pixel class boundaries). Pixels showing an occluder (robot, another vessel) are excluded from both the observed and
the predicted mask, so partial occlusion and image borders do not bias the fit. Fixed-point iterations move the vessel
until the predicted mask centroid (as a ray through a plane) equals the observed one; with ``z`` free (wrist, well
visible) the area ratio also scales the distance along the viewing ray (known size -> depth).
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from . import cup_basket_env_cfg as cb
from . import wx_real as R
from .env_cfg import pad_center


def _f(name: str, default) -> float:
    return float(os.environ.get(name, default))


def _vec(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(v) for v in os.environ.get(name, default).split(","))


WFUSE = R.WFUSE
TOP_FIT = os.environ.get("WX_TOP_FIT", "1" if WFUSE else "0") == "1"
KF = os.environ.get("WX_TOP_KF", "1" if WFUSE else "0") == "1"
FUSED = WFUSE or TOP_FIT or KF
STATS = os.environ.get("WX_FUSE_STATS") == "1"
STATS_EVERY = int(_f("WX_FUSE_STATS_EVERY", 900))
STATS_STRIDE = int(_f("WX_FUSE_STATS_STRIDE", 3))  # sample every this many steps (the stats cost GPU syncs)

S_TOP, S_WR = int(_f("WX_FIT_S_TOP", 24)), int(_f("WX_FIT_S_WR", 24))  # window samples per side
L_FIT = int(_f("WX_FIT_L", 12))  # heights per ray in the silhouette test
IT_FIT = int(_f("WX_FIT_ITERS", 6))
NMIN_TOP, NMIN_WR = _f("WX_FIT_NMIN_TOP", 6), _f("WX_FIT_NMIN_WR", 30)  # min observed mask pixels (seg resolution)
IOU_MIN = _f("WX_FIT_IOU_MIN", 0.55)
VIS_MIN_FIT = _f("WX_FIT_VIS_MIN", 0.3)  # min visible fraction of the predicted silhouette (occlusion / border)
Z3_VIS = _f("WX_WR_Z3_VIS", 0.75)  # wrist: fit z too when this much of the silhouette is visible
WR_MAXD = _f("WX_WR_MAXD", 0.35)  # wrist: ignore vessels farther than this from the camera (m)
WR_EVERY = int(_f("WX_WFUSE_EVERY", 1))  # wrist perception every this many control steps
WR_CAL = _vec("WX_WFUSE_CAL", "0.003,1.5")  # perceived wrist camera pose error: FK + hand-eye calibration (m, deg)
PW, PH = (int(v) for v in os.environ.get("WX_WFUSE_RES", "320x240").split("x"))  # wrist perception image (seg input /2)
# measurement noise (1 sigma, m) and filter
SIG_TOP = _vec("WX_KF_SIG_TOP", "0.012,0.010")  # cup, bowl (top fit, fully visible)
SIG_TOP_STK = _f("WX_KF_SIG_TOP_STK", 0.018)
SIG_WR = _vec("WX_KF_SIG_WR", "0.002,0.02")  # wrist: a + b * distance
Q_FAR, Q_NEAR = _f("WX_KF_Q_FAR", 0.0003), _f("WX_KF_Q_NEAR", 0.004)  # process noise per control step (m)
NEAR_R = _f("WX_KF_NEAR_R", 0.08)
GATE = _f("WX_KF_GATE", 3.5)
WR_TRUST_P = _f("WX_KF_WR_TRUST_P", 0.03)  # estimate std (m) below which a far wrist fix is rejected
WR_JUMP = _f("WX_KF_WR_JUMP", 0.06)  # max plausible wrist correction of a confident estimate (m)
TIP_IOU = _f("WX_TIP_IOU", -1.0)  # top fit IoU below this (well visible) = lying; < 0: off (report upright)
PROF = os.environ.get("WX_FUSE_PROF") == "1"
_T: dict = {}


class _tic:
    """WX_FUSE_PROF=1: wall time per section (CUDA synchronised), printed with the stats."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROF:
            import time

            torch.cuda.synchronize()
            self.t = time.time()

    def __exit__(self, *a):
        if PROF:
            import time

            torch.cuda.synchronize()
            v = _T.get(self.name, (0.0, 0))
            _T[self.name] = (v[0] + time.time() - self.t, v[1] + 1)


def _geo(dev):
    g = cb._geo_t(dev)
    return g["r_top"], g["r_bot"], g["h"], g["pitch"]


# ---------------------------------------------------------------- silhouette fit


def silhouette(o, d, c, zb, rt, rb, h, L: int = L_FIT):
    """o (B, 3) camera centre, d (B, P, 3) rays, vessel axis c (B, 2), base z (B,), radii / height (B,) -> (B, P) hit."""
    lv = torch.linspace(0.0, 1.0, L, device=d.device)
    z = zb.unsqueeze(-1) + h.unsqueeze(-1) * lv  # (B, L)
    dz = d[..., 2:3]
    dz = torch.where(dz.abs() < 1e-6, torch.full_like(dz, -1e-6), dz)
    t = (z.unsqueeze(1) - o[:, None, None, 2]) / dz  # (B, P, L)
    px = o[:, None, None, 0] + t * d[..., 0:1] - c[:, None, None, 0]
    py = o[:, None, None, 1] + t * d[..., 1:2] - c[:, None, None, 1]
    r = rb.unsqueeze(-1) + (rt - rb).unsqueeze(-1) * lv  # (B, L)
    return ((px * px + py * py <= (r * r).unsqueeze(1)) & (t > 0)).any(-1)


def _centroid(o, du, w, zp):
    """Mean unit ray over the weighted pixels -> its point at height zp (B, 2)."""
    m = torch.einsum("bp,bpk->bk", w, du)
    m = m / torch.linalg.vector_norm(m, dim=-1, keepdim=True).clamp(min=1e-9)
    dz = m[:, 2].clamp(max=-0.05)
    return o[:, :2] + ((zp - o[:, 2]) / dz).unsqueeze(-1) * m[:, :2]


def fit(o, d, obs, valid, c0, zb0, typ, free_z=None, iters: int = IT_FIT):
    """Fixed-point silhouette fit. o (B, 3), d (B, P, 3), obs / valid (B, P) bool, c0 (B, 2), zb0 (B,), typ (B,) long,
    free_z (B,) bool. Returns c (B, 2), zb (B,), iou, vis (visible fraction of the predicted silhouette), nobs."""
    rt_, rb_, h_, _ = _geo(d.device)
    rt, rb, h = rt_[typ], rb_[typ], h_[typ]
    q = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp(min=1e-9)
    ok = obs & valid
    w_obs = ok.float()
    nobs = w_obs.sum(-1)
    c, zb = c0.clone(), zb0.clone()
    fz = torch.zeros_like(zb, dtype=torch.bool) if free_z is None else free_z
    for k in range(iters):
        pred = silhouette(o, d, c, zb, rt, rb, h)
        pv = pred & valid
        w_p = pv.float()
        npred = w_p.sum(-1)
        zp = zb + 0.5 * h
        co = _centroid(o, q, w_obs, zp)
        cp = _centroid(o, q, w_p, zp)
        step = torch.where((npred > 0).unsqueeze(-1), co - cp, co - c)
        c = c + step.clamp(-0.05, 0.05)
        if bool(fz.any()) and k >= 1:  # area ratio -> distance along the viewing ray of the vessel centre
            s = (npred / nobs.clamp(min=1)).clamp(min=1e-3).sqrt().clamp(0.85, 1.18)
            ctr = torch.cat((c, zp.unsqueeze(-1)), -1)
            nc = o + s.unsqueeze(-1) * (ctr - o)
            c = torch.where(fz.unsqueeze(-1), nc[:, :2], c)
            zb = torch.where(fz, nc[:, 2] - 0.5 * h, zb)
    pred = silhouette(o, d, c, zb, rt, rb, h)
    inter = (pred & ok).float().sum(-1)
    union = ((pred & valid) | ok).float().sum(-1)
    iou = inter / union.clamp(min=1)
    vis = (pred & valid).float().sum(-1) / pred.float().sum(-1).clamp(min=1)
    return c, zb, iou, vis, nobs


def boxes(cls: torch.Tensor, hw_img: tuple[float, float], nmin: float):
    """cls (NI, h, w) class map -> per image and vessel (NI, 6): count, centre u / v and half size (image units)."""
    ni, h, w = cls.shape
    H, W = hw_img
    su, sv = W / w, H / h
    vv, uu = torch.meshgrid(torch.arange(h, device=cls.device), torch.arange(w, device=cls.device), indexing="ij")
    out = []
    for i in range(cb.N_OBJ):
        m = cls == 2 + i
        cnt = m.flatten(1).sum(-1).float()
        big = torch.tensor(1e9, device=cls.device)
        u0 = torch.where(m, uu, big).flatten(1).amin(-1).float()
        u1 = torch.where(m, uu, -big).flatten(1).amax(-1).float()
        v0 = torch.where(m, vv, big).flatten(1).amin(-1).float()
        v1 = torch.where(m, vv, -big).flatten(1).amax(-1).float()
        out.append(torch.stack((cnt, (u0 + u1 + 1) / 2 * su, (v0 + v1 + 1) / 2 * sv,
                                torch.maximum((u1 - u0 + 1) * su, (v1 - v0 + 1) * sv) / 2), -1))
    b = torch.stack(out, 1)  # (NI, 6, 4)
    b[..., 0] = torch.where(b[..., 0] >= nmin, b[..., 0], torch.zeros_like(b[..., 0]))
    return b


def window_samples(logits: torch.Tensor, box: torch.Tensor, hw_img: tuple[float, float], S: int, half_min: torch.Tensor):
    """logits (NI, C, h, w); box (NI, 6, 4) -> sample pixel coords u, v (NI, 6, S*S) (image units), class (NI, 6, S*S),
    inside-image (NI, 6, S*S)."""
    ni = logits.shape[0]
    H, W = hw_img
    half = torch.maximum(box[..., 3] * 1.5 + 2 * W / logits.shape[-1], half_min)  # (NI, 6)
    g = (torch.arange(S, device=logits.device).float() + 0.5) / S * 2 - 1  # (S,)
    gv, gu = torch.meshgrid(g, g, indexing="ij")
    u = box[..., 1:2] + half.unsqueeze(-1) * gu.reshape(1, 1, -1)
    v = box[..., 2:3] + half.unsqueeze(-1) * gv.reshape(1, 1, -1)
    grid = torch.stack((u / W * 2 - 1, v / H * 2 - 1), -1).reshape(ni, cb.N_OBJ * S, S, 2)
    lg = F.grid_sample(logits, grid.to(logits.dtype), mode="bilinear", padding_mode="border", align_corners=False)
    cls = lg.argmax(1).reshape(ni, cb.N_OBJ, S * S)
    inside = (u >= 0) & (u <= W) & (v >= 0) & (v <= H)
    return u, v, cls, inside


def fit_image_objects(logits, cls_map, hw_img, ray_fn, zb0, c0, nmin, S, free_z_fn=None, half_px_fn=None):
    """All vessels in a batch of images (NI). ray_fn(u, v) -> o (NI, 3), d (NI, 6, P, 3). zb0 / c0 (NI, 6[, 2]) priors.
    Returns dict of (NI, 6) tensors: ok-candidate, c (.., 2), zb, iou, vis, nobs."""
    ni = logits.shape[0]
    dev = logits.device
    box = boxes(cls_map, hw_img, nmin)
    half_min = half_px_fn(box) if half_px_fn is not None else torch.zeros(ni, cb.N_OBJ, device=dev)
    u, v, cls, inside = window_samples(logits, box, hw_img, S, half_min)
    o, d = ray_fn(u, v)  # (NI, 3), (NI, 6, P, 3)
    tgt = (torch.arange(cb.N_OBJ, device=dev) + 2).view(1, -1, 1)
    obs = cls == tgt
    occ = (cls == 1) | ((cls >= 2) & (cls <= 7) & ~obs)
    valid = inside & ~occ
    typ = cb.obj_type(dev).view(1, -1).expand(ni, -1)
    act = box[..., 0] > 0
    # first guess: observed centroid ray at mid height
    oo = o.unsqueeze(1).expand(-1, cb.N_OBJ, -1).reshape(-1, 3)
    dd = d.reshape(ni * cb.N_OBJ, -1, 3)
    obs_f, val_f = obs.reshape(ni * cb.N_OBJ, -1), valid.reshape(ni * cb.N_OBJ, -1)
    _, _, h_, _ = _geo(dev)
    q = dd / torch.linalg.vector_norm(dd, dim=-1, keepdim=True).clamp(min=1e-9)
    zb = zb0.reshape(-1)
    cobs = _centroid(oo, q, (obs_f & val_f).float(), zb + 0.5 * h_[typ.reshape(-1)])
    c_init = torch.where((torch.linalg.vector_norm(c0.reshape(-1, 2) - cobs, dim=-1) < 0.03).unsqueeze(-1), c0.reshape(-1, 2), cobs)
    fz = free_z_fn(box).reshape(-1) if free_z_fn is not None else None
    sel = act.reshape(-1).nonzero().squeeze(-1)
    res = {k: torch.zeros(ni * cb.N_OBJ, device=dev) for k in ("zb", "iou", "vis", "nobs", "z3")}
    res["c"] = torch.zeros(ni * cb.N_OBJ, 2, device=dev)
    if len(sel):
        for ch in torch.split(sel, 8192):
            tc = typ.reshape(-1)[ch]
            c, z, iou, vis, nob = fit(oo[ch], dd[ch], obs_f[ch], val_f[ch], c_init[ch], zb[ch], tc)
            z3 = torch.zeros_like(vis, dtype=torch.bool)
            if fz is not None:  # known size -> depth only when the silhouette is well visible: refit with z free
                z3 = fz[ch] & (vis > Z3_VIS) & (iou > IOU_MIN)
                if bool(z3.any()):
                    j = z3.nonzero().squeeze(-1)
                    c2, z2, iou2, vis2, _ = fit(oo[ch][j], dd[ch][j], obs_f[ch][j], val_f[ch][j], c[j], z[j], tc[j], torch.ones_like(j, dtype=torch.bool))
                    c[j], z[j], iou[j], vis[j] = c2, z2, iou2, vis2
            res["c"][ch], res["zb"][ch], res["iou"][ch], res["vis"][ch], res["nobs"][ch] = c, z, iou, vis, nob
            res["z3"][ch] = z3.float()
    out = {k: v.view(ni, cb.N_OBJ, *v.shape[1:]) for k, v in res.items()}
    out["act"] = act
    out["box"] = box
    return out


# ---------------------------------------------------------------- wrist perception


def undistort(u, v, dfov, k1, pp, W, H):
    """Distorted pixel (u, v) of a W x H image of the lens -> undistorted normalised (x, y) (ROS camera frame)."""
    T = torch.tan(torch.deg2rad(dfov) / 2)
    r_d = 2 * T / (1 + torch.sqrt(1 - 4 * T * T * k1))
    f_out = math.hypot(W / 2, H / 2) / r_d
    s = W / R.WRIST_OUT_W
    xd = (u - W / 2 - s * pp[..., 0:1]) / f_out.unsqueeze(-1)
    yd = (v - H / 2 - s * pp[..., 1:2]) / f_out.unsqueeze(-1)
    den = 1 + k1.unsqueeze(-1) * (xd * xd + yd * yd)
    return xd / den, yd / den, f_out


class WristPerception:
    """Both wrist cameras: lens remap of the raw render to the native 320 x 240 distorted image (the student's lens
    model, shared via ``R.shared_wrist``), venue augmentation, wrist segmenter, per-vessel silhouette fit with the
    camera pose from the gripper FK (sim: the actual camera pose plus a per-episode FK / hand-eye error). Results are
    stored per capture step and delivered after the wrist latency."""

    def __init__(self, env, wreal):
        from .wx_seg import load

        self.env, self.wreal = env, wreal
        n, dev = env.num_envs, env.device
        self.n, self.dev = n, dev
        self.seg = load(os.environ["WX_WSEGNET"], dev)
        self.cal_p = torch.zeros(n, 2, 3, device=dev)
        self.cal_r = torch.zeros(n, 2, 3, device=dev)
        D = 4
        self.ring = torch.zeros(D, n, 2, cb.N_OBJ, 6, device=dev)  # x, y, zb, sigma, ok, dist
        self.cam_hist = None
        self.k = 0
        self.last = None
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        if len(ids) == 0:
            return
        k = len(ids)
        self.cal_p[ids] = torch.empty(k, 2, 3, device=self.dev).uniform_(-WR_CAL[0], WR_CAL[0])
        self.cal_r[ids] = torch.empty(k, 2, 3, device=self.dev).uniform_(-math.radians(WR_CAL[1]), math.radians(WR_CAL[1]))
        self.ring[:, ids] = 0.0

    def _cams(self):
        from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

        env = self.env
        P, Rm = [], []
        for k, name in enumerate(("wrist_left", "wrist_right")):
            cam = env.scene.sensors[name]
            p = cam.data.pos_w
            q = cam.data.quat_w_ros
            p = (p.torch if hasattr(p, "torch") else p) - env.scene.env_origins
            q = q.torch if hasattr(q, "torch") else q
            e = self.cal_r[:, k]
            Rm.append(matrix_from_quat(q) @ matrix_from_quat(quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2])))
            P.append(p + self.cal_p[:, k])
        return torch.stack(P, 1), torch.stack(Rm, 1)  # (n, 2, 3), (n, 2, 3, 3)

    @torch.no_grad()
    def step(self, raw: torch.Tensor, zb_prior: torch.Tensor, c_prior: torch.Tensor) -> None:
        """raw (n, 2, Hr, Wr, 3) uint8 current render; zb_prior / c_prior (n, 6[, 2]) current fused estimate."""
        from .wx_seg import segment

        n, dev = self.n, self.dev
        wr = self.wreal
        self.k += 1
        cam_p, cam_R = self._cams()
        self.cam_last = cam_p
        res = torch.zeros(n, 2, cb.N_OBJ, 6, device=dev)
        if (self.k - 1) % WR_EVERY == 0:
            x = raw.reshape(2 * n, *raw.shape[2:]).float().permute(0, 3, 1, 2) / 255.0
            if R.WRIST_POOL > 1:
                x = F.avg_pool2d(x, R.WRIST_POOL, R.WRIST_POOL)
            if getattr(self, "grid_key", None) != getattr(wr, "version", 0) or getattr(self, "grid", None) is None:
                self.grid = R.wrist_grid(2 * n, dev, wr.hfov.reshape(-1), wr.k1.reshape(-1), wr.pp.reshape(-1, 2), out_wh=(PW, PH))
                self.grid_key = getattr(wr, "version", 0)
            x = F.grid_sample(x, self.grid, mode="bilinear", padding_mode="border", align_corners=False).permute(0, 2, 3, 1)
            if wr.aug is not None:
                x = wr.aug(x)
            cls, lg = segment(self.seg, x, logits_out=True)
            self.last = (x, cls)
            hfov, k1, pp = wr.hfov.reshape(-1), wr.k1.reshape(-1), wr.pp.reshape(-1, 2)
            o_all = cam_p.reshape(2 * n, 3)
            R_all = cam_R.reshape(2 * n, 3, 3)

            def ray_fn(u, v):
                xu, yu, _ = undistort(u.flatten(1), v.flatten(1), hfov, k1, pp, PW, PH)
                dc = torch.stack((xu, yu, torch.ones_like(xu)), -1)  # (NI, 6*P, 3)
                dw = torch.einsum("bij,bpj->bpi", R_all, dc)
                return o_all, dw.view(2 * n, cb.N_OBJ, -1, 3)

            T = torch.tan(torch.deg2rad(hfov) / 2)
            r_d = 2 * T / (1 + torch.sqrt(1 - 4 * T * T * k1))
            f_px = math.hypot(PW / 2, PH / 2) / r_d  # (2n,)
            rt_, _, h_, _ = _geo(dev)
            ext = (rt_ + 0.5 * h_)[cb.obj_type(dev)]  # (6,)
            dist = torch.linalg.vector_norm(c_prior.unsqueeze(1) - cam_p[..., None, :2], dim=-1)  # (n, 2, 6) xy
            dist3 = torch.sqrt(dist ** 2 + (zb_prior.unsqueeze(1) - cam_p[..., None, 2]) ** 2).reshape(2 * n, cb.N_OBJ)

            def half_px(box):
                return (f_px.unsqueeze(-1) * ext / dist3.clamp(min=0.05) * 1.3).clamp(max=PW)

            def free_z(box):
                return torch.ones(2 * n, cb.N_OBJ, dtype=torch.bool, device=dev)

            zb0 = zb_prior.unsqueeze(1).expand(-1, 2, -1).reshape(2 * n, cb.N_OBJ)
            c0 = c_prior.unsqueeze(1).expand(-1, 2, -1, -1).reshape(2 * n, cb.N_OBJ, 2)
            out = fit_image_objects(lg, cls, (PH, PW), ray_fn, zb0, c0, NMIN_WR, S_WR, free_z, half_px)
            ctr = torch.cat((out["c"], (out["zb"]).unsqueeze(-1)), -1).view(n, 2, cb.N_OBJ, 3)
            dcam = torch.linalg.vector_norm(ctr - cam_p.unsqueeze(2), dim=-1)  # (n, 2, 6)
            iou, vis = out["iou"].view(n, 2, -1), out["vis"].view(n, 2, -1)
            ok = out["act"].view(n, 2, -1) & (iou > IOU_MIN) & (vis > VIS_MIN_FIT) & (dcam < WR_MAXD)
            sig = (SIG_WR[0] + SIG_WR[1] * dcam) / vis.clamp(min=0.3).sqrt()
            z3 = out["z3"].view(n, 2, -1)
            res = torch.cat((ctr, sig.unsqueeze(-1), ok.float().unsqueeze(-1), torch.where(z3 > 0.5, dcam, torch.full_like(dcam, 9.0)).unsqueeze(-1)), -1)
            self.raw_out = out
        self.ring = torch.cat((res.unsqueeze(0), self.ring[:-1]), 0)
        new = self.env.episode_length_buf <= 1
        if bool(new.any()):
            self.ring[:, new] = 0.0

    def delayed(self) -> torch.Tensor:
        """(n, 2, 6, 6) result captured lat steps ago (same latency as the student's wrist frames)."""
        lat = self.wreal.lat.clamp(max=self.ring.shape[0] - 1)
        return self.ring[lat, torch.arange(self.n, device=self.dev)]


# ---------------------------------------------------------------- Kalman fusion (replaces Coarse._update_top)


class Fuser:
    """Per vessel: xy (variance P), base z (variance Pz), age of the last top / wrist measurement (steps)."""

    def __init__(self, coarse, env):
        n, dev = env.num_envs, env.device
        self.c, self.n, self.dev = coarse, n, dev
        self.xy = torch.zeros(n, cb.N_OBJ, 2, device=dev)
        self.P = torch.full((n, cb.N_OBJ), 1.0, device=dev)
        self.zb = torch.full((n, cb.N_OBJ), cb.TABLE_TOP, device=dev)
        self.Pz = torch.full((n, cb.N_OBJ), 1e-4, device=dev)
        self.age_top = torch.full((n, cb.N_OBJ), 999.0, device=dev)
        self.age_wr = torch.full((n, cb.N_OBJ), 999.0, device=dev)
        self.wz = torch.zeros(n, cb.N_OBJ, dtype=torch.bool, device=dev)  # zb measured by a wrist camera
        self.top_ring = torch.zeros(5, n, cb.N_OBJ, 5, device=dev)  # x, y, sigma, ok, iou
        self.lat_top = torch.randint(2, 5, (n,), device=dev) if R.on("LAT") else torch.zeros(n, dtype=torch.long, device=dev)
        self.tip_iou = torch.ones(n, cb.N_OBJ, device=dev)
        self.wrist = None
        if WFUSE:
            self.wrist = WristPerception(env, R.shared_wrist(env))
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        if len(ids) == 0:
            return
        self.P[ids] = 1.0
        self.age_top[ids] = 999.0
        self.age_wr[ids] = 999.0
        self.wz[ids] = False
        self.top_ring[:, ids] = 0.0
        self.started[ids] = False
        if R.on("LAT"):
            self.lat_top[ids] = torch.randint(2, 5, (len(ids),), device=self.dev)
        if self.wrist is not None:
            self.wrist.reset(ids)

    def _surface(self, xy: torch.Tensor) -> torch.Tensor:
        uv = cb.to_basket(self.c.bk.unsqueeze(1), xy.unsqueeze(2))
        inside = (uv.abs() < uv.new_tensor((cb.IN_L / 2, cb.IN_W / 2))).all(-1).any(-1)
        return torch.where(inside, torch.full_like(xy[..., 0], cb.FLOOR_Z), torch.full_like(xy[..., 0], cb.TABLE_TOP)), inside

    def _kf(self, z, sig, ok, reset_on_gate: torch.Tensor):
        """xy measurement z (n, 6, 2), sigma (n, 6), ok (n, 6)."""
        Rv = sig * sig
        nu = z - self.xy
        d2 = (nu * nu).sum(-1)
        gate = d2 > GATE * GATE * (self.P + Rv)
        jump = ok & gate & reset_on_gate
        upd = ok & ~gate
        K = (self.P / (self.P + Rv)).unsqueeze(-1)
        self.xy = torch.where(upd.unsqueeze(-1), self.xy + K * nu, torch.where(jump.unsqueeze(-1), z, self.xy))
        self.P = torch.where(upd, (1 - K[..., 0]) * self.P, torch.where(jump, Rv, self.P))
        return upd | jump

    @torch.no_grad()
    def update(self, env, key: int) -> None:
        from .wx_top import VIS_MIN

        c, dev, n = self.c, self.dev, self.n
        top = c.top
        t = env.episode_length_buf
        valid = t >= 1
        if not hasattr(c, "held"):
            c.held = torch.zeros((n, cb.N_SLOT), dtype=torch.bool, device=dev)
        c.held &= valid.unsqueeze(-1)
        first = valid & ~self.started
        # ---- top camera (refresh every HOLD steps or the first post-reset frame)
        from .wx_env import HOLD

        meas = torch.zeros(n, cb.N_OBJ, 5, device=dev)
        if key % max(HOLD, 1) == 0 or bool(first.any()):
            with _tic("top_measure"):
                m = top.measure(valid, fit=TOP_FIT, prior=(self.xy, self.zb))
            bk, bk_ok = top.baskets(m, valid)
            c.bk = torch.where(bk_ok.unsqueeze(-1), bk, c.bk)
            typ = cb.obj_type(dev).view(1, -1)
            if TOP_FIT and "fit" in m:
                f = m["fit"]
                vis = f["vis"]
                fok = valid.unsqueeze(-1) & f["act"] & (f["iou"] > IOU_MIN) & (vis > VIS_MIN_FIT)
                stk = (self.zb > cb.FLOOR_Z + 0.5 * cb._geo_t(dev)["pitch"][typ])
                s0 = torch.where(typ == cb.CUP, SIG_TOP[0], SIG_TOP[1]) * torch.ones_like(vis)
                sig = torch.where(stk, torch.full_like(s0, SIG_TOP_STK), s0) / vis.clamp(min=0.3).sqrt()
                # fit failed (e.g. lying vessel): the plane estimate under the old acceptance rule, less trusted
                pok = valid.unsqueeze(-1) & ~fok & (m["vis"] > VIS_MIN)
                ok = fok | pok
                sig = torch.where(fok, sig, 1.5 * s0)
                xy = torch.where(fok.unsqueeze(-1), f["c"], m["xy"])
                self.tip_iou = torch.where(ok & (vis > 0.8), f["iou"], self.tip_iou)
            else:
                ok = valid.unsqueeze(-1) & (m["vis"] > VIS_MIN)
                xy = m["xy"]
                sig = torch.where(typ == cb.CUP, SIG_TOP[0], SIG_TOP[1]) * torch.ones_like(m["vis"])
                f = None
            fresh = first.unsqueeze(-1) & ~ok & (m["vis"] > 0.05)
            xy = torch.where(fresh.unsqueeze(-1), m["xy"], xy)
            ok = ok | fresh
            meas = torch.cat((xy, sig.unsqueeze(-1), ok.float().unsqueeze(-1),
                              (f["iou"] if f is not None else torch.ones_like(sig)).unsqueeze(-1)), -1)
            self.last_top = m
            self.dbg_top = meas
        self.top_ring = torch.cat((meas.unsqueeze(0), self.top_ring[:-1]), 0)
        newep = t <= 1
        if bool(newep.any()):
            self.top_ring[:, newep] = 0.0
            self.top_ring[0, newep] = meas[newep]
        # ---- predict: process noise, larger next to an arm (pushes)
        near = torch.zeros(n, cb.N_OBJ, dtype=torch.bool, device=dev)
        for arm in cb.ARMS:
            pad = pad_center(env, arm)
            near |= torch.linalg.vector_norm(pad[:, None, :2] - self.xy, dim=-1) < NEAR_R
        q = torch.where(near, Q_NEAR, Q_FAR)
        self.P = self.P + q * q
        self.age_top += 1
        self.age_wr += 1
        # ---- apply the top measurement captured lat_top steps ago (the first frame immediately)
        ar = torch.arange(n, device=dev)
        tm = torch.where(first.view(-1, 1, 1), self.top_ring[0], self.top_ring[self.lat_top, ar])
        tok = tm[..., 3] > 0.5
        held_obj = self._held_mask(env)
        tok &= ~held_obj
        init = first.unsqueeze(-1) & tok
        self.xy = torch.where(init.unsqueeze(-1), tm[..., :2], self.xy)
        self.P = torch.where(init, tm[..., 2] ** 2, self.P)
        recent_wr = self.age_wr < 15
        used = self._kf(tm[..., :2], tm[..., 2], tok & ~init, reset_on_gate=~recent_wr | near)
        self.age_top = torch.where(used | init, torch.where(first.unsqueeze(-1), 0.0, self.lat_top.float().unsqueeze(-1)), self.age_top)
        # ---- wrist cameras
        if self.wrist is not None:
            from .wx_student import wrist_images

            with _tic("wrist_step"):
                self.wrist.step(wrist_images(env), self.zb, self.xy)
            wm = self.wrist.delayed()  # (n, 2, 6, 6)
            for k in range(2):
                w = wm[:, k]
                wok = (w[..., 4] > 0.5) & valid.unsqueeze(-1) & ~held_obj & self.started.unsqueeze(-1)
                # plausibility: a wrist fix far from a confident estimate (cup / bowl or colour confusion of the wrist
                # segmenter) is not taken unless the vessel may have been pushed (arm near) by a plausible distance
                inn = torch.linalg.vector_norm(w[..., :2] - self.xy, dim=-1)
                conf = self.P.sqrt() < WR_TRUST_P
                wok &= ~(conf & (inn > WR_JUMP)) | (self.age_top > 900)
                u = self._kf(w[..., :2], w[..., 3], wok, reset_on_gate=near | ~conf)
                self.age_wr = torch.where(u, self.wreal_lat().float().unsqueeze(-1), self.age_wr)
                # base height from the wrist (known size -> depth), when it was fitted freely
                zok = u & (w[..., 5] < WR_MAXD)
                rz = (0.004 + 0.02 * w[..., 5]) ** 2
                Kz = self.Pz / (self.Pz + rz)
                self.zb = torch.where(zok, self.zb + Kz * (w[..., 2] - self.zb), self.zb)
                self.Pz = torch.where(zok, (1 - Kz) * self.Pz, self.Pz)
                self.wz |= zok
        # ---- base height prior where no wrist z: resting surface (+ the top's stack-level guess)
        surf, inside = self._surface(self.xy)
        lvl = getattr(top, "lvl_est", None)
        pitch = cb._geo_t(dev)["pitch"][cb.obj_type(dev)].view(1, -1)
        prior = surf + (lvl * pitch if lvl is not None else 0.0)
        self.zb = torch.where(self.wz & inside, self.zb, prior)
        self.Pz = torch.where(self.wz & inside, self.Pz + 1e-8, torch.full_like(self.Pz, 1e-4))
        self.wz &= inside
        self.started |= valid
        # ---- write the coarse estimate
        c.pos = torch.cat((self.xy, self.zb.unsqueeze(-1)), -1)
        typ = cb.obj_type(dev).view(1, -1)
        tipped = (self.tip_iou < TIP_IOU) if TIP_IOU > 0 else torch.zeros_like(near)
        upright = torch.zeros(n, cb.N_OBJ, 4, device=dev)
        upright[..., 3] = 1.0
        if TIP_IOU > 0:  # lying: 90 deg about the horizontal axis along y (direction unknown)
            s2 = math.sqrt(0.5)
            lying = torch.zeros_like(upright)
            lying[..., 0], lying[..., 3] = s2, s2
            c.quat = torch.where(tipped.unsqueeze(-1), lying, upright)
        else:
            c.quat = upright
        c.tipped = tipped
        # never seen this episode (hidden in a stack since the start): the zone centre of its start basket (bookkeeping)
        seen = (self.age_top < 900) | (self.age_wr < 900)
        loc = cb._state(env)["loc"]
        zc = cb.zone_center(c.bk.gather(1, loc.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)), cb.obj_type(dev).expand_as(loc))
        hid = ~seen & (loc >= 0) & valid.unsqueeze(-1)
        c.pos = torch.where(hid.unsqueeze(-1), torch.cat((zc, torch.full_like(zc[..., :1], cb.FLOOR_Z)), -1), c.pos)
        self.xy = torch.where(hid.unsqueeze(-1), zc, self.xy)
        c.seen = seen
        c._attach_held(env)
        # the filter follows the carried vessel (FK), with a small variance; on release it continues from there
        self.xy = torch.where(held_obj.unsqueeze(-1), c.pos[..., :2], self.xy)
        self.P = torch.where(held_obj, torch.full_like(self.P, 0.01 ** 2), self.P)
        c.init[:] = False

    def wreal_lat(self) -> torch.Tensor:
        return self.wrist.wreal.lat

    def _held_mask(self, env) -> torch.Tensor:
        c = self.c
        m = torch.zeros(self.n, cb.N_OBJ, dtype=torch.bool, device=self.dev)
        ar = torch.arange(self.n, device=self.dev)
        for s_ in range(cb.N_SLOT):
            k = cb.slot_obj(env, s_)
            m[ar, k] |= c.held[:, s_]
        return m

    def features(self, env) -> torch.Tensor:
        """Student inputs per vessel: sqrt(P) (cm), top age (s), wrist age (s, capped 3), held: (n, 24)."""
        dt = env.step_dt
        held = self._held_mask(env).float()
        return torch.cat((self.P.clamp(max=0.04).sqrt() * 100.0, (self.age_top * dt).clamp(max=3.0),
                          (self.age_wr * dt).clamp(max=3.0), held), -1)


# ---------------------------------------------------------------- statistics against the sim truth


class FuseStats:
    """Error of the vessel position the policy receives (after latency) against the true position now, by situation;
    and of the policy's relative inputs (rim - pads, goal - vessel) against the privileged ones."""

    def __init__(self, env):
        self.d: dict = {}
        self.k = 0

    def _add(self, key, x):
        if x.numel():
            self.d.setdefault(key, []).append(x.float().cpu())

    @torch.no_grad()
    def __call__(self, env, view, coarse) -> None:
        with _tic("stats"):
            self._call(env, view, coarse)
        if self.k % STATS_EVERY == 0:
            import time

            now = time.time()
            if hasattr(self, "t0"):
                _T["wall_per_step"] = (now - self.t0, STATS_EVERY)
            self.t0 = now
            self.print()

    def _call(self, env, view, coarse) -> None:
        self.k += 1
        if self.k % STATS_STRIDE:
            return
        dev = env.device
        p = cb.objs_pos(env)
        q = cb.objs_quat(env)
        valid = (env.episode_length_buf >= 2).unsqueeze(-1).expand(-1, cb.N_OBJ)
        err = torch.linalg.vector_norm(view.pos[..., :2] - p[..., :2], dim=-1)
        zerr = (view.pos[..., 2] - p[..., 2]).abs()
        st = cb._state(env)
        g = cb._geo_t(dev)
        typ = cb.obj_type(dev).view(1, -1).expand_as(err)
        pitch = g["pitch"][typ]
        rest = p[..., 2] < cb.TABLE_TOP + 0.012 + (st["level"] > 0).float() * 0.05
        held_true = torch.zeros_like(rest)
        for s_ in range(cb.N_SLOT):
            held_true[torch.arange(env.num_envs, device=dev), cb.slot_obj(env, s_)] |= cb.held(env, s_) & cb.slot_active(env, s_)
        lifted = ~rest | held_true
        upright = cb.tilt(q) < 0.6
        dmin = torch.full_like(err, 9.0)
        for arm in cb.ARMS:
            pad = pad_center(env, arm)
            dmin = torch.minimum(dmin, torch.linalg.vector_norm(pad[:, None, :2] - p[..., :2], dim=-1))
        uvt = cb.to_basket(cb.baskets(env).unsqueeze(1), p[..., :2].unsqueeze(2))
        in_bk = (uvt.abs() < uvt.new_tensor((cb.IN_L / 2, cb.IN_W / 2))).all(-1).any(-1)
        stacked = in_bk & (p[..., 2] > cb.FLOOR_Z + 0.5 * pitch) & ~lifted
        on = valid & (p[..., 2] > cb.TABLE_TOP - 0.03)
        # grasp approach: the commanded vessel of an active slot, its arm's pads within 6 cm, not yet lifted
        grasp = torch.zeros_like(rest)
        target = torch.zeros_like(rest)
        ar = torch.arange(env.num_envs, device=dev)
        for s_ in range(cb.N_SLOT):
            k = cb.slot_obj(env, s_)
            pad = pad_center(env, cb.ARMS[s_])
            act = cb.slot_active(env, s_)
            dk = torch.linalg.vector_norm(pad[:, :2] - p[ar, k, :2], dim=-1)
            grasp[ar, k] |= act & (dk < 0.06) & ~lifted[ar, k]
            mem = cb.stack_members(env, s_)
            depth = mem.sum(-1)
            topk = torch.where(mem, st["level"], -1).argmax(-1)
            dt_ = torch.linalg.vector_norm(pad[:, :2] - p[ar, topk, :2], dim=-1)
            target[ar, topk] |= act & (depth > 0) & (dt_ < 0.08) & held_true[ar, k]
        for tn, t in (("cup", 0), ("bowl", 1)):
            base = on & (typ == t) & upright
            for name, m in (("all", base & ~lifted), ("clear", base & ~lifted & (dmin > 0.10)),
                            ("arm_near", base & ~lifted & (dmin < 0.06)), ("grasp", base & grasp),
                            ("lifted", on & (typ == t) & held_true), ("stacked", base & stacked),
                            ("target_top", base & target)):
                self._add(f"{tn}/{name}", err[m])
                if name in ("stacked", "target_top", "grasp"):
                    self._add(f"{tn}/{name}/z", zerr[m])
            tip = on & (typ == t) & ~lifted & ~upright
            self._add(f"{tn}/tipped_xy", err[tip])
            if getattr(coarse, "tipped", None) is not None:
                te = coarse.tipped
                for a, b in ((True, True), (True, False), (False, True), (False, False)):
                    kk = f"{tn}/tip_true{int(a)}_est{int(b)}"
                    sel = on & (typ == t) & ~lifted & ((~upright) == a) & (te == b)
                    self.d[kk] = self.d.get(kk, 0) + int(sel.sum())
            fz = getattr(coarse, "fuser", None)
            if fz is not None:
                sel = on & (typ == t) & ~lifted & (fz.tip_iou < 1.0)
                self._add(f"{tn}/tipiou_up", fz.tip_iou[sel & upright])
                self._add(f"{tn}/tipiou_tip", fz.tip_iou[sel & ~upright])
                wr = fz.age_wr < 10
                self.d[f"{tn}/grasp_wrist_frac"] = (self.d.get(f"{tn}/grasp_wrist_frac", (0, 0))[0] + int((base & grasp & wr).sum()),
                                                    self.d.get(f"{tn}/grasp_wrist_frac", (0, 0))[1] + int((base & grasp).sum()))
        # relative inputs the policy acts on
        from .wx_env import coarse_relative_from

        rc = coarse_relative_from(env, view)
        rt = cb.target_relative(env)
        for s_ in range(cb.N_SLOT):
            act = cb.slot_active(env, s_) & valid[:, 0]
            k = cb.slot_obj(env, s_)
            hk = held_true[ar, k]
            pad = pad_center(env, cb.ARMS[s_])
            dk = torch.linalg.vector_norm(pad[:, :2] - p[ar, k, :2], dim=-1)
            e_rim = torch.linalg.vector_norm(rc[:, 9 * s_: 9 * s_ + 3] - rt[:, 9 * s_: 9 * s_ + 3], dim=-1)
            e_goal = torch.linalg.vector_norm(rc[:, 9 * s_ + 3: 9 * s_ + 5] - rt[:, 9 * s_ + 3: 9 * s_ + 5], dim=-1)
            self._add("rel/rim_pad/approach", e_rim[act & ~hk & (dk < 0.08)])
            self._add("rel/goal_xy/carry", e_goal[act & hk])
            self._add("rel/goal_xy/carry_near", e_goal[act & hk & (torch.linalg.vector_norm(
                pad[:, :2] - (rt[:, 9 * s_ + 3: 9 * s_ + 5] + p[ar, k, :2]), dim=-1) < 0.06)])
        # raw measurements at their capture (sources of the fused estimate), against the truth now
        fz = getattr(coarse, "fuser", None)
        if fz is not None:
            typ1 = cb.obj_type(dev)
            tm = getattr(fz, "dbg_top", None)
            if tm is not None:
                ok = (tm[..., 3] > 0.5) & valid & ~lifted
                e = torch.linalg.vector_norm(tm[..., :2] - p[..., :2], dim=-1)
                for tn, t in (("cup", 0), ("bowl", 1)):
                    self._add(f"src/top/{tn}", e[ok & (typ1 == t)])
                    self._add(f"src/top/{tn}/near", e[ok & (typ1 == t) & (dmin < 0.06)])
                fz.dbg_top = None
            w = getattr(fz.wrist, "ring", None) if fz.wrist is not None else None
            if w is not None and (fz.wrist.k - 1) % WR_EVERY == 0:
                w = w[0]  # captured this step
                for k in range(2):
                    ok = (w[:, k, :, 4] > 0.5) & valid & ~lifted
                    e = torch.linalg.vector_norm(w[:, k, :, :2] - p[..., :2], dim=-1)
                    ez = (w[:, k, :, 2] - p[..., 2]).abs()
                    d = w[:, k, :, 5]
                    for tn, t in (("cup", 0), ("bowl", 1)):
                        sel = ok & (typ1 == t)
                        self._add(f"src/wrist/{tn}", e[sel])
                        self._add(f"src/wrist/{tn}/z3_z", ez[sel & (d < 9)])
                        self._add(f"src/wrist/{tn}/far_gt5cm_frac", (e[sel] > 0.05).float() / 100.0)
                        ro = getattr(fz.wrist, "raw_out", None)
                        if ro is not None:
                            bad = sel & (e > 0.05)
                            gd = sel & (e < 0.02)
                            for nm in ("iou", "vis", "nobs"):
                                v = ro[nm].view(-1, 2, cb.N_OBJ)[:, k]
                                self._add(f"diag/wr_{tn}_{nm}_bad", v[bad] / 100.0)
                                self._add(f"diag/wr_{tn}_{nm}_good", v[gd] / 100.0)
                            dc = torch.linalg.vector_norm(w[:, k, :, :2] - fz.wrist.cam_last[:, k, None, :2], dim=-1) if hasattr(fz.wrist, "cam_last") else None
                            if dc is not None:
                                self._add(f"diag/wr_{tn}_dxy_bad", dc[bad])
                                self._add(f"diag/wr_{tn}_dxy_good", dc[gd])
                            # which true vessel is nearest to a bad fix (class confusion?)
                            dd = torch.linalg.vector_norm(w[:, k, :, None, :2] - p[:, None, :, :2], dim=-1)  # (n, 6, 6)
                            nn_ = dd.argmin(-1)
                            same = nn_ == torch.arange(cb.N_OBJ, device=dev).view(1, -1)
                            other_type = cb.obj_type(dev)[nn_] != typ1.view(1, -1)
                            self._add(f"diag/wr_{tn}_bad_near_other_vessel_frac", (~same[bad]).float() / 100.0)
                            self._add(f"diag/wr_{tn}_bad_near_other_type_frac", (other_type[bad] & ~same[bad]).float() / 100.0)
        # where the large errors of the policy view come from
        big = (err > 0.05) & valid & ~lifted & (p[..., 2] > cb.TABLE_TOP - 0.03)
        if fz is not None:
            self._add("big5cm/age_top_s", fz.age_top[big] * env.step_dt / 100.0)
            self._add("big5cm/age_wr_s", fz.age_wr[big].clamp(max=999) * env.step_dt / 100.0)
            self._add("big5cm/P_cm", fz.P[big].sqrt())
        self._add("big5cm/frac", big[valid].float() / 100.0)

    def print(self) -> None:
        out = {k: {"s_per_call": round(v[0] / max(v[1], 1), 4), "n": v[1]} for k, v in _T.items()}
        for k, v in self.d.items():
            if isinstance(v, int):
                out[k] = v
                continue
            if isinstance(v, tuple):
                out[k] = {"n": v[1], "frac": round(v[0] / max(v[1], 1), 3)}
                continue
            x = torch.cat(v)
            if len(x) > 3_000_000:
                x = x[torch.randperm(len(x))[:2_000_000]]
                self.d[k] = [x]
            if len(x) == 0:
                continue
            u = 1.0 if "tipiou" in k else 100.0
            out[k] = {"n": len(x), "mean": round(float(x.mean()) * u, 2), "p50": round(float(x.quantile(0.5)) * u, 2),
                      "p90": round(float(x.quantile(0.9)) * u, 2)}
        print("WX_FUSE_STATS " + json.dumps(out), flush=True)
