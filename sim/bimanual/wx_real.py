"""Realistic deployment model for the wrist student (``WX_REAL=1``): only signals the real robot has, with the real
cameras' geometry, timing and an unknown venue.

Factors (each can be switched off for the ablation, ``WX_R_<NAME>=0``; all on by default when ``WX_REAL=1``):
  CAM    C920 overhead top camera (70.4 x 43.3 deg, processed at 640 x 360 = 1280 x 720 / 2) straight down from
         0.95 m over (0.17, 0); wrist OV9732-like (rendered 120 deg / 320 x 180, resampled to the policy's 128 x 72
         with a per-episode HFOV in [90, 110] deg and barrel distortion (division model) k1 in [-0.08, 0]).
  VIS    venue appearance: colour temperature 3000-6500 K with partial auto white balance, exposure, light gradient,
         soft shadows, table colour (renderer material: white / wood / grey / black / dark mats) + table texture,
         distractor boxes at the table edge, sensor noise, MJPEG quality 60-95, defocus, motion blur (wrist),
         small filament hue jitter.
  LAT    top estimates at 10 Hz with 66-133 ms latency (2-4 control steps), wrist frames 33-100 ms (1-3 steps),
         action delay 0-2 steps.
  PROP   joint encoders: 4096-tick quantisation + noise + per-episode bias, velocity by finite differences, no
         privileged progress flags (zeroed for the actor), arm-base calibration error +-1 cm / +-2 deg.
  CALIB  top camera pose error +-3 cm / +-3 deg and focal +-3 % (perception keeps the nominal calibration), table
         height error +-5 mm.
  PHYS   PD gains +-30 %, joint friction / armature, servo deadband 0-1 deg, object mass +-30 %, friction 0.3-0.9.

This module has no Isaac Lab import at module level (``wx_seg.train`` uses ``TopAug`` standalone).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

REAL = os.environ.get("WX_REAL", "0") == "1"
# wrist-camera localisation fusion (wx_fuse / wx_wrist): the student / perception share one wrist lens model per env
WFUSE = os.environ.get("WX_WFUSE", "0") == "1"


def on(name: str) -> bool:
    return REAL and os.environ.get(f"WX_R_{name}", "1") == "1"


def _u(shape, lo, hi, dev):
    return torch.empty(shape, device=dev).uniform_(lo, hi)


# ---------------------------------------------------------------- camera models

# C920 poses (env frame; the table top is at z = 0.032). WX_TOP_POSE picks one; WX_TOP_EYE / WX_TOP_LOOK override.
#   A: oblique from behind the left arm base (the VLM planner camera pose)
#   B: oblique from the side of the basket row (user spec); B2: B moved 5 cm back / up, aimed 3 cm nearer (no clipping)
#   overhead: straight down 0.95 m above the table (dropped 11/8; kept for comparison)
TOP_POSES = {
    "A": ((-0.30, 0.0, 0.85), (0.17, 0.0, 0.0)),
    "B": ((0.17, -0.60, 0.80), (0.17, 0.0, 0.0)),
    "B2": ((0.17, -0.65, 0.85), (0.17, -0.03, 0.0)),
    "overhead": ((0.17, 0.0, 0.032 + float(os.environ.get("WX_TOP_HEIGHT", 0.95))), (0.17, 0.0, 0.0)),
    # G (user 11/10): the side camera between the two arms, in front of the basket row (the global view position)
    "G": ((0.17, -0.95, 0.80), (0.17, 0.0, 0.02)),
}
TOP_POSE = os.environ.get("WX_TOP_POSE", "A")  # A chosen 11/8 (lowest localisation error)
TOP_EYE, TOP_LOOK = TOP_POSES[TOP_POSE]
if os.environ.get("WX_TOP_EYE"):
    TOP_EYE = tuple(float(v) for v in os.environ["WX_TOP_EYE"].split(","))
if os.environ.get("WX_TOP_LOOK"):
    TOP_LOOK = tuple(float(v) for v in os.environ["WX_TOP_LOOK"].split(","))
TOP_OVERHEAD = TOP_POSE == "overhead" and not os.environ.get("WX_TOP_EYE")
TOP_HFOV = 70.4
TOP_W, TOP_H = 640, 360  # processing resolution (1280 x 720 / 2)
TOP_FOCAL_MM = 20.955 / 2 / math.tan(math.radians(TOP_HFOV / 2))  # on the 20.955 mm aperture
# OV9732 wrist camera (user spec 11/8): 320 x 240 (4:3) on the robot, 100 deg FOV taken as the DIAGONAL FOV of the
# distorted image (100 +- 4 per episode). 4:3 diagonal 100 deg -> pinhole-equivalent 87.3 x 71.1 deg (H x V) before
# distortion. Rendered natively at 320 x 240 with a 96 deg horizontal pinhole that covers the widest randomised lens
# (104 deg diagonal + barrel + principal point), remapped through the lens model to the 128 x 96 policy input.
WRIST_RENDER_W, WRIST_RENDER_H = (int(v) for v in os.environ.get("WX_WRIST_RENDER", "320x240").split("x"))
WRIST_RENDER_HFOV = float(os.environ.get("WX_WRIST_RENDER_HFOV", 96.0))
WRIST_POOL = max(1, WRIST_RENDER_W // 320)
WRIST_SRC_W, WRIST_SRC_H = WRIST_RENDER_W // WRIST_POOL, WRIST_RENDER_H // WRIST_POOL
WRIST_DFOV, WRIST_DFOV_J = float(os.environ.get("WX_WRIST_DFOV", 100.0)), float(os.environ.get("WX_WRIST_DFOV_J", 4.0))
WRIST_OUT_W, WRIST_OUT_H = (int(v) for v in os.environ.get("WX_WRIST_OUT", "128x96").split("x"))
WAUG = float(os.environ.get("WX_WAUG", 1.0))  # strength of both wrist degradation chains (TopAug + student ImageAug)


def kelvin_rgb(k: torch.Tensor) -> torch.Tensor:
    """Approximate white-point RGB of a black body at temperature k (Kelvin), normalised to max 1."""
    t = k / 100.0
    r = torch.where(t <= 66, torch.ones_like(t), (329.698727446 * (t - 60).clamp(min=1e-3) ** -0.1332047592) / 255)
    g = torch.where(t <= 66, (99.4708025861 * torch.log(t) - 161.1195681661) / 255,
                    (288.1221695283 * (t - 60).clamp(min=1e-3) ** -0.0755148492) / 255)
    b = torch.where(t >= 66, torch.ones_like(t), torch.where(t <= 19, torch.zeros_like(t),
                                                             (138.5177312231 * torch.log((t - 10).clamp(min=1e-3)) - 305.0447927307) / 255))
    rgb = torch.stack((r, g, b), -1).clamp(0.05, 1.0)
    return rgb / rgb.max(-1, keepdim=True).values


_QY = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_QC = torch.full((8, 8), 99.0)
_QC[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]], dtype=torch.float32)


def _dct_mat(dev):
    k = torch.arange(8, device=dev).float()
    m = torch.cos((2 * k.view(1, -1) + 1) * k.view(-1, 1) * math.pi / 16) * math.sqrt(2 / 8)
    m[0] /= math.sqrt(2)
    return m  # orthonormal DCT-II


def _jpeg_plane(p: torch.Tensor, q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """p (n, H, W) in [0, 255] (H, W multiples of 8); q (n, 8, 8) quantisation tables."""
    n, h, w = p.shape
    b = (p - 128).view(n, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)  # (n, by, bx, 8, 8)
    c = D @ b @ D.T
    c = torch.round(c / q[:, None, None]) * q[:, None, None]
    b = D.T @ c @ D
    return (b + 128).permute(0, 1, 3, 2, 4).reshape(n, h, w)


def jpeg(x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """MJPEG round trip on the GPU for (n, H, W, 3) float [0, 1] at per-image quality (libjpeg tables and scaling,
    4:2:0 chroma). Edges are padded to multiples of 16."""
    n, h, w, _ = x.shape
    dev = x.device
    ph, pw = (-h) % 16, (-w) % 16
    y = x.permute(0, 3, 1, 2) * 255
    if ph or pw:
        y = F.pad(y, (0, pw, 0, ph), mode="replicate")
    r, g, b = y[:, 0], y[:, 1], y[:, 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    qq = quality.view(-1).clamp(1, 100)
    sc = torch.where(qq < 50, 5000 / qq, 200 - 2 * qq).view(-1, 1, 1)
    qy = ((_QY.to(dev) * sc + 50) / 100).floor().clamp(1, 255)
    qc = ((_QC.to(dev) * sc + 50) / 100).floor().clamp(1, 255)
    D = _dct_mat(dev)
    Y = _jpeg_plane(Y, qy, D)
    cs = []
    for C in (Cb, Cr):
        c2 = F.avg_pool2d(C.unsqueeze(1), 2).squeeze(1)
        c2 = _jpeg_plane(c2, qc, D)
        cs.append(F.interpolate(c2.unsqueeze(1), scale_factor=2, mode="bilinear", align_corners=False).squeeze(1))
    Cb, Cr = cs
    r = Y + 1.402 * (Cr - 128)
    g = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    b = Y + 1.772 * (Cb - 128)
    out = torch.stack((r, g, b), -1)[:, :h, :w] / 255
    return out.clamp(0, 1)


class TopAug:
    """Per-episode venue / camera appearance for (n, H, W, 3) float images (top or wrist)."""

    def __init__(self, n: int, dev, motion_blur: bool = False, jpeg_on: bool = True, scale: float = 1.0):
        # scale: strength of every degradation (1 = the original ranges; the wrist chain uses WX_WAUG, user 11/10)
        self.n, self.dev, self.motion_blur, self.jpeg_on, self.s = n, dev, motion_blur, jpeg_on, scale
        z = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        self.cast, self.expo, self.off, self.grad, self.noise = z(3), z(1), z(1), z(2), z(1)
        self.blur, self.hue, self.q, self.tex_a = z(1), z(1), z(1), z(1)
        self.shadow = z(3, 5)  # up to 3 soft shadows: cx, cy, rx, ry, strength (image-normalised)
        self.tex_seed = torch.randint(0, 10_000, (n,), device=dev)
        self.mb = z(2)
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        k = len(ids)
        if k == 0:
            return
        d = self.dev
        c = self.s
        kel = _u((k,), 3000.0, 6500.0, d)
        awb = _u((k, 1), 0.6, 0.95, d)  # fraction of the cast removed by auto white balance
        self.cast[ids] = kelvin_rgb(kel).pow((1 - awb) * c) * (1 + c * _u((k, 3), -0.06, 0.06, d))
        self.expo[ids] = 1 + c * _u((k, 1), -0.4, 0.4, d)
        self.off[ids] = c * _u((k, 1), -0.04, 0.04, d)
        self.grad[ids] = c * _u((k, 2), -0.25, 0.25, d)
        self.noise[ids] = c * _u((k, 1), 0.006, 0.03, d)
        self.blur[ids] = c * _u((k, 1), 0.0, 1.0, d)
        self.hue[ids] = c * _u((k, 1), -0.1, 0.1, d)  # rad (+-6 deg): the three filament colours stay apart
        self.q[ids] = 95.0 - c * _u((k, 1), 0.0, 35.0, d)
        self.tex_a[ids] = c * _u((k, 1), 0.0, 0.1, d)
        sh = torch.stack((_u((k, 3), 0.0, 1.0, d), _u((k, 3), 0.0, 1.0, d), _u((k, 3), 0.05, 0.35, d), _u((k, 3), 0.05, 0.35, d),
                          c * _u((k, 3), 0.0, 0.45, d) * (torch.rand((k, 3), device=d) < 0.5)), -1)
        self.shadow[ids] = sh
        self.tex_seed[ids] = torch.randint(0, 10_000, (k,), device=d)
        self.mb[ids] = c * _u((k, 2), -3.0, 3.0, d) * (WRIST_OUT_W / 128.0)  # px of the output image

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """x (n, H, W, 3) float [0, 1] -> augmented, same shape."""
        n_, h, w, _ = x.shape
        sl = slice(0, n_)
        ys = torch.linspace(0, 1, h, device=x.device).view(1, h, 1, 1)
        xs = torch.linspace(0, 1, w, device=x.device).view(1, 1, w, 1)
        # hue rotation about the grey axis
        th = self.hue[sl].view(-1, 1, 1, 1)
        a = 1 / math.sqrt(3)
        cos, sin = torch.cos(th), torch.sin(th)
        mean = x.mean(-1, keepdim=True)
        cr = torch.cross(torch.full_like(x, a), x, dim=-1)
        x = x * cos + cr * sin + a * a * 3 * mean * (1 - cos)
        # shadows (soft ellipses) and light gradient
        light = 1 + (self.grad[sl, 0].view(-1, 1, 1, 1) * (xs - 0.5) + self.grad[sl, 1].view(-1, 1, 1, 1) * (ys - 0.5)) * 2
        for j in range(3):
            s = self.shadow[sl, j]
            dd = ((xs - s[:, 0].view(-1, 1, 1, 1)) / s[:, 2].view(-1, 1, 1, 1)) ** 2 + ((ys - s[:, 1].view(-1, 1, 1, 1)) / s[:, 3].view(-1, 1, 1, 1)) ** 2
            light = light * (1 - s[:, 4].view(-1, 1, 1, 1) * torch.exp(-dd))
        # low-frequency table texture (wood grain / mat): multiplicative noise
        tex = F.interpolate(torch.randn(n_, 1, 6, 10, device=x.device), size=(h, w), mode="bicubic", align_corners=False)
        light = light * (1 + self.tex_a[sl].view(-1, 1, 1, 1) * tex.permute(0, 2, 3, 1))
        x = x * light
        # exposure, colour cast (residual after auto white balance), offset
        x = x * self.expo[sl].view(-1, 1, 1, 1) * self.cast[sl].view(-1, 1, 1, 3) + self.off[sl].view(-1, 1, 1, 1)
        # defocus (and motion blur for the moving wrist cameras)
        xc = x.permute(0, 3, 1, 2)
        bl = self.blur[sl].view(-1, 1, 1, 1)
        g = F.avg_pool2d(xc, 3, 1, 1, count_include_pad=False)
        xc = xc + bl * (g - xc)
        if self.motion_blur:
            m = self.mb[sl]
            acc = xc.clone()
            for t in (0.33, 0.66, 1.0):
                theta = torch.zeros(n_, 2, 3, device=x.device)
                theta[:, 0, 0] = theta[:, 1, 1] = 1.0
                theta[:, 0, 2] = m[:, 0] * t * 2 / w
                theta[:, 1, 2] = m[:, 1] * t * 2 / h
                acc = acc + F.grid_sample(xc, F.affine_grid(theta, xc.shape, align_corners=False), padding_mode="border", align_corners=False)
            xc = acc / 4
        x = xc.permute(0, 2, 3, 1)
        x = x + self.noise[sl].view(-1, 1, 1, 1) * torch.randn_like(x) * (0.5 + x.clamp(0, 1).sqrt())
        x = x.clamp(0, 1)
        if self.jpeg_on:
            x = jpeg(x, self.q[sl, 0])
        return x


def wrist_rays(dev, dfov: torch.Tensor, k1: torch.Tensor, pp: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """Undistorted normalised image coordinates (n, out_h, out_w, 2) = (x / z, y / z) in the ROS camera frame (x right,
    y down, z forward) of every pixel centre of a distorted out_w x out_h image of the lens (dfov deg diagonal, division
    k1, principal point offset pp in px of a 128-wide image, scaled with the resolution)."""
    T = torch.tan(torch.deg2rad(dfov) / 2)  # undistorted tangent at the image corner
    # corner: r_u = r_d / (1 + k1 r_d^2) = T  ->  r_d = 2T / (1 + sqrt(1 - 4 T^2 k1))
    r_d = 2 * T / (1 + torch.sqrt(1 - 4 * T * T * k1))
    f_out = math.hypot(out_w / 2, out_h / 2) / r_d  # (n,)
    v, u = torch.meshgrid(torch.arange(out_h, device=dev) + 0.5, torch.arange(out_w, device=dev) + 0.5, indexing="ij")
    s = out_w / WRIST_OUT_W
    xd = (u.unsqueeze(0) - out_w / 2 - s * pp[:, 0, None, None]) / f_out.view(-1, 1, 1)
    yd = (v.unsqueeze(0) - out_h / 2 - s * pp[:, 1, None, None]) / f_out.view(-1, 1, 1)
    den = 1 + k1.view(-1, 1, 1) * (xd * xd + yd * yd)
    return torch.stack((xd / den, yd / den), -1)


def wrist_grid(n: int, dev, dfov: torch.Tensor, k1: torch.Tensor, pp: torch.Tensor | None = None,
               out_wh: tuple[int, int] | None = None) -> torch.Tensor:
    """grid_sample grid (n, OUT_H, OUT_W, 2) that turns the pinhole render (after the x WRIST_POOL area downscale) into a
    camera whose DIAGONAL field of view is ``dfov`` (deg, measured on the distorted image corners) with radial
    division-model distortion (barrel for k1 < 0); principal point jittered +-2 px (``pp``, drawn here if not given).
    ``out_wh`` = another output resolution of the same lens (wrist perception)."""
    ow, oh = out_wh if out_wh is not None else (WRIST_OUT_W, WRIST_OUT_H)
    f_src = (WRIST_SRC_W / 2) / math.tan(math.radians(WRIST_RENDER_HFOV / 2))
    if pp is None:
        pp = _u((n, 2), -2.0, 2.0, dev)
    r = wrist_rays(dev, dfov, k1, pp, ow, oh)
    sx = r[..., 0] * f_src / (WRIST_SRC_W / 2)
    sy = r[..., 1] * f_src / (WRIST_SRC_H / 2)
    return torch.stack((sx, sy), -1)


class WristReal:
    """Wrist images as the real robot would get them: optics (FOV / distortion), appearance, latency."""

    def __init__(self, n: int, dev):
        self.n, self.dev = n, dev
        self.aug = TopAug(2 * n, dev, motion_blur=True, jpeg_on=True, scale=WAUG) if on("VIS") else None
        self.hfov = torch.full((n, 2), WRIST_DFOV, device=dev)  # diagonal FOV (deg)
        self.k1 = torch.full((n, 2), -0.06, device=dev)
        self.grid = None
        self.lat = torch.ones(n, dtype=torch.long, device=dev)
        self.hist = None  # (4, n, 2, 3, H, W) uint8 ring
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        if len(ids) == 0:
            return
        if on("CAM"):
            self.hfov[ids] = _u((len(ids), 2), WRIST_DFOV - WRIST_DFOV_J, WRIST_DFOV + WRIST_DFOV_J, self.dev)
            self.k1[ids] = _u((len(ids), 2), -0.08, 0.0, self.dev)
        if not hasattr(self, "pp"):
            self.pp = _u((self.n, 2, 2), -2.0, 2.0, self.dev)
        if WFUSE:  # principal point per episode (the old path re-draws it for every env at any reset)
            self.pp[ids] = _u((len(ids), 2, 2), -2.0, 2.0, self.dev)
        else:
            self.pp = _u((self.n, 2, 2), -2.0, 2.0, self.dev)
        g = wrist_grid(2 * self.n, self.dev, self.hfov.reshape(-1), self.k1.reshape(-1), self.pp.reshape(-1, 2))
        self.grid = g
        self.version = getattr(self, "version", 0) + 1
        if self.aug is not None:
            self.aug.reset(torch.cat((2 * ids, 2 * ids + 1)))
        self.lat[ids] = torch.randint(1, 4, (len(ids),), device=self.dev) if on("LAT") else 0
        self.fresh = getattr(self, "fresh", torch.zeros(self.n, dtype=torch.bool, device=self.dev))
        self.fresh[ids] = True

    @torch.no_grad()
    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        """raw (n, 2, Hr, Wr, 3) uint8 render -> (n, 2, 3, OUT_H, OUT_W) uint8, delayed by the per-env latency."""
        n = raw.shape[0]
        x = raw.reshape(2 * n, *raw.shape[2:]).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        if WRIST_POOL > 1:  # larger render -> 320 wide (area), then the lens remap
            x = F.avg_pool2d(x, WRIST_POOL, WRIST_POOL)
        x = F.grid_sample(x, self.grid, mode="bilinear", padding_mode="border", align_corners=False)
        x = x.permute(0, 2, 3, 1)
        if self.aug is not None:
            x = self.aug(x)
        img = (x.clamp(0, 1) * 255).to(torch.uint8).permute(0, 3, 1, 2).reshape(n, 2, 3, WRIST_OUT_H, WRIST_OUT_W)
        if self.hist is None:
            self.hist = img.unsqueeze(0).repeat(4, 1, 1, 1, 1, 1)
        self.hist = torch.cat((img.unsqueeze(0), self.hist[:-1]), 0)
        # a reset env has no older frames of its new episode: repeat its current one
        if bool(self.fresh.any()):
            self.hist[:, self.fresh] = img[self.fresh].unsqueeze(0)
            self.fresh[:] = False
        return self.hist[self.lat, torch.arange(n, device=self.dev)]


def shared_wrist(env) -> "WristReal":
    """WX_WFUSE: one wrist lens / appearance / latency model per env shared by the student's images and the wrist
    perception; reset by the env's reset event (wx_env.coarse_reset), not by the scripts."""
    env = getattr(env, "unwrapped", env)
    if not hasattr(env, "_wx_wreal"):
        env._wx_wreal = WristReal(env.num_envs, env.device)
        env._wx_wreal.shared = True
    return env._wx_wreal


class ProprioReal:
    """Rebuilds the actor's 163-dim input from the coarse group with real encoders and no privileged flags."""

    TICK = 2 * math.pi / 4096

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.env = env
        self.n, self.dev, self.dt = n, dev, env.step_dt
        self.bias = torch.zeros(n, 12, device=dev)
        self.base_err = torch.zeros(n, 6, device=dev)
        self.prev = None
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        from .wx_student import _term_slice

        self.sl = {t: _term_slice(env, "coarse", t) for t in
                   ("left_joint_pos", "right_joint_pos", "left_joint_vel", "right_joint_vel", "progress", "base")}
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        k = len(ids)
        if k == 0:
            return
        self.bias[ids] = _u((k, 12), -0.01, 0.01, self.dev) if on("PROP") else 0.0
        e = torch.zeros(k, 6, device=self.dev)
        if on("PROP"):
            for a in range(2):
                e[:, 3 * a: 3 * a + 2] = torch.randn(k, 2, device=self.dev).clamp(-2, 2) * 0.005  # ~1 cm (2 sigma)
                e[:, 3 * a + 2] = _u((k,), -math.radians(2), math.radians(2), self.dev)
        self.base_err[ids] = e
        self.fresh[ids] = True

    @torch.no_grad()
    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        s = s.clone()
        lp, rp = self.sl["left_joint_pos"], self.sl["right_joint_pos"]
        lv, rv = self.sl["left_joint_vel"], self.sl["right_joint_vel"]
        q = torch.cat((s[:, lp[0]: lp[1]], s[:, rp[0]: rp[1]]), -1)
        if on("PROP"):
            q = torch.round((q + self.bias) / self.TICK) * self.TICK + 0.0008 * torch.randn_like(q)
            if self.prev is None:
                self.prev = q.clone()
            prev = torch.where(self.fresh.unsqueeze(-1), q, self.prev)
            v = (q - prev) / self.dt
            self.prev = q.clone()
            self.fresh[:] = False
            s[:, lp[0]: lp[1]], s[:, rp[0]: rp[1]] = q[:, :6], q[:, 6:]
            s[:, lv[0]: lv[1]], s[:, rv[0]: rv[1]] = v[:, :6], v[:, 6:]
        if REAL and os.environ.get("WX_KEEP_FLAGS") != "1":  # privileged milestone flags are never available to the actor
            # (WX_KEEP_FLAGS=1: teacher-(b) diagnostic only; the residual teacher cannot act without them).
            # WX_DEPLOY_FLAGS=1: replaced by flags inferred from deployable signals (wx_env.deploy_flags)
            p = self.sl["progress"]
            if os.environ.get("WX_DEPLOY_FLAGS") == "1":
                from .wx_env import deploy_flags

                s[:, p[0]: p[1]] = deploy_flags(self.env)
            else:
                s[:, p[0]: p[1]] = 0.0
        b = self.sl["base"]
        s[:, b[0]: b[1]] = s[:, b[0]: b[1]] + self.base_err
        return s


DEADBAND_DEG = float(os.environ.get("WX_DEADBAND_DEG", 1.0))  # max servo deadband (per episode U(0, max))


class ActionReal:
    """Executed-action model: per-episode delay 0-2 steps and a servo deadband 0-1 deg (arm joints)."""

    def __init__(self, n: int, dev, adim: int = 12):
        self.n, self.dev = n, dev
        self.q = torch.zeros(3, n, adim, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.dead = torch.zeros(n, 1, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        if len(ids) == 0:
            return
        self.delay[ids] = torch.randint(0, 3, (len(ids),), device=self.dev) if on("LAT") else 0
        # deadband in action units: the arm action is a joint delta of 0.033 rad per unit
        self.dead[ids] = _u((len(ids), 1), 0.0, math.radians(DEADBAND_DEG) / 0.033, self.dev) if on("PHYS") else 0.0
        self.fresh[ids] = True

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        arm = [i for i in range(a.shape[-1]) if i not in (5, 11)]
        small = a[:, arm].abs() < self.dead
        a[:, arm] = torch.where(small, torch.zeros_like(a[:, arm]), a[:, arm])
        if bool(self.fresh.any()):
            self.q[:, self.fresh] = a[self.fresh].unsqueeze(0)
            self.fresh[:] = False
        self.q = torch.cat((a.unsqueeze(0), self.q[:-1]), 0)
        return self.q[self.delay, torch.arange(self.n, device=self.dev)]
