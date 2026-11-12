"""Wrist-camera student for BiTidy-CupBasket: recurrent policy on coarse overhead poses + two wrist RGB images,
distilled online (DAgger) from the privileged teacher (frozen BC-RNN + residual PPO, ``residual.ResidualScript``).

Callbacks (``--external_callback isaaclab_tutorial.tasks.bimanual.wx_student.<name>``; each registers the Wrist tasks):
  install_dagger  online DAgger trainer (play loop drives the env; the policy's action is ignored)
  install_eval    first-episode evaluation through ``scripted.install_scripted`` with ``BITIDY_SCRIPT``:
                    residual:ResidualScript     (a) teacher, privileged state
                    wx_student:CoarseTeacher    (b) teacher fed the coarse overhead poses (no images)
                    wx_student:StudentScript    (c) student (``WX_STUDENT=<ckpt>``)
  install_video   zoomed third-person + both wrist views per env (task ...-Wrist-SO101-Render), PNG at each episode end

DAgger (env vars, defaults): every env runs whole episodes with either the teacher (prob. beta, drawn per episode) or
the student executing; the teacher labels every step either way (its BC-RNN memory follows the visited states). Each
``WX_T`` (32) steps the chunk (coarse obs, augmented wrist images, teacher action, auxiliary targets, episode starts,
student LSTM state at the chunk start) goes into a GPU ring buffer of ``WX_BUF`` chunks; then ``WX_K`` gradient steps
on ``WX_B`` random env-chunks (truncated BPTT from the stored state). beta = 1 for ``WX_WARM`` iterations, then linear
to ``WX_BETA_MIN`` at ``WX_DECAY``; ``WX_ITERS`` iterations in all. Loss: MSE on the 10 arm joints, BCE on the two
gripper signs, aux MSE (x10 m) on gripper->target rim and target->stack top/zone (true values, 12).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import wx_real as R

PKG = "isaaclab_tutorial.tasks.bimanual"
TASK = "BiTidy-CupBasket-Wrist-SO101"
GRIP = (5, 11)
GRIP_W = float(os.environ.get("WX_GRIP_W", "0.1"))  # DAgger: gripper BCE weight (arm MSE has weight 1)
GRIP_TW, GRIP_TK = float(os.environ.get("WX_GRIP_TW", "1")), int(os.environ.get("WX_GRIP_TK", "8"))  # x weight within +-TK steps of a switch
DAG_SHIELD = os.environ.get("WX_DAG_SHIELD", "0") == "1"  # DAgger: student episodes use gripper_shields
GRIP_BIAS = float(os.environ.get("WX_GRIP_BIAS", "0"))  # eval: pre-grasp close threshold on the gripper logit
REL_NEAR = tuple(float(v) for v in os.environ.get("WX_REL_NEAR", "0.03,0.05").split(","))  # "xy,z" (m): held vessel near its goal
REL_GATE = os.environ.get("WX_REL_GATE", "0") == "1"  # eval: a holding arm opens only near the goal
REL_LATCH = int(os.environ.get("WX_REL_LATCH", "0"))  # eval: after a release near the goal 1 = no grasp bias, 2 = stay open
REL_BIAS = float(os.environ.get("WX_REL_BIAS", "0"))  # eval: near the goal, open when the gripper logit > -REL_BIAS
ARM_DIMS = tuple(i for i in range(12) if i not in GRIP)


def _f(name: str, default) -> float:
    return float(os.environ.get(name, default))


def register() -> None:
    import gymnasium as gym

    for suffix, cfg in (("", "WristEnvCfg"), ("-Render", "WristRenderEnvCfg")):
        tid = f"{TASK}{suffix}"
        if tid in gym.registry:
            continue
        gym.register(id=tid, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{PKG}.wx_env:{cfg}",
                             "rsl_rl_cfg_entry_point": f"{PKG}.agents:BiCupBasketPPORunnerCfg", "default_agent": "rsl_rl"})


# ---------------------------------------------------------------- images


def wrist_images(env) -> torch.Tensor:
    """(n, 2, H, W, 3) uint8 from ``wrist_left`` / ``wrist_right``."""
    out = []
    for name in ("wrist_left", "wrist_right"):
        rgb = env.scene.sensors[name].data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        rgb = rgb[..., :3]
        if rgb.is_floating_point():
            rgb = (rgb.clamp(0, 1) * 255) if float(rgb.max()) <= 1.0 else rgb.clamp(0, 255)
        out.append(rgb.to(torch.uint8))
    return torch.stack(out, 1)


TOPIMG_W, TOPIMG_H = (int(v) for v in os.environ.get("WX_TOPIMG_WH", "160x90").split("x"))


def top_image(env) -> torch.Tensor:
    """(n, 3, TOPIMG_H, TOPIMG_W) uint8: the C920 overhead frame the perception last processed (10 Hz, held between
    refreshes; already carries the camera / appearance randomisation of wx_top), area-downsampled. Needs the top
    camera (WX_REAL=1 or WX_TOP=1)."""
    c = getattr(env, "_wx_coarse", None)
    top = getattr(c, "top", None) if c is not None else None
    if top is None:
        raise RuntimeError("WX_TOPIMG=1 needs the overhead camera (WX_REAL=1 or WX_TOP=1)")
    x = getattr(top, "last_full", None)
    if x is None:  # before the first refresh
        return torch.zeros(env.num_envs, 3, TOPIMG_H, TOPIMG_W, dtype=torch.uint8, device=env.device)
    if getattr(top, "_ti_src", None) is not x:  # recompute once per refresh
        y = F.interpolate(x.permute(0, 3, 1, 2).float(), size=(TOPIMG_H, TOPIMG_W), mode="area")
        top._ti = (y.clamp(0, 1) * 255).round().to(torch.uint8)
        top._ti_src = x
    return top._ti


def _shift(im: torch.Tensor, px: int = 4) -> torch.Tensor:
    """DrQ-style random shift (+-px), one per sequence and camera. im (B, T, K, 3, H, W) uint8 -> (T, B, K, 3, H, W)."""
    b, t, k, ch, hh, ww = im.shape
    flat = im.permute(0, 2, 1, 3, 4, 5).reshape(b * k, t * ch, hh, ww).float()
    shift = torch.randint(-px, px + 1, (b * k, 2), device=im.device).float() * 2 / torch.tensor([ww, hh], device=im.device).float()
    theta = torch.zeros(b * k, 2, 3, device=im.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, padding_mode="border", align_corners=False)
    return flat.view(b, k, t, ch, hh, ww).permute(2, 0, 1, 3, 4, 5).round().clamp(0, 255).to(torch.uint8)


class ImageAug:
    """Per-episode photometric randomisation of both wrist views (colour / hue jitter, gain, contrast, gamma, blur,
    a lighting gradient) plus per-frame pixel noise. Returns (n, 2, 3, H, W) uint8."""

    def __init__(self, n: int, dev, on: bool = True):
        self.n, self.dev, self.on = n, dev, on
        self.gain = torch.ones(n, 2, 3, 1, 1, device=dev)
        self.bright, self.contrast, self.gamma, self.blur = (torch.zeros(n, 2, 1, 1, 1, device=dev) for _ in range(4))
        self.hue = torch.eye(3, device=dev).repeat(n, 2, 1, 1)
        self.grad = torch.zeros(n, 2, 2, device=dev)
        self.reset(None)

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        k = len(ids)
        if k == 0:
            return

        def u(lo, hi, *shape):
            return torch.empty((k, 2, *shape), device=self.dev).uniform_(lo, hi)

        from . import wx_real as R_

        c = R_.WAUG  # strength (1 = the original ranges)
        self.gain[ids] = 1 + c * u(-0.18, 0.18, 3, 1, 1)
        self.bright[ids] = c * u(-0.08, 0.08, 1, 1, 1)
        self.contrast[ids] = 1 + c * u(-0.2, 0.2, 1, 1, 1)
        self.gamma[ids] = torch.exp(c * u(math.log(0.8), math.log(1.25), 1, 1, 1))
        self.blur[ids] = c * u(0.0, 1.0, 1, 1, 1) * (u(0, 1, 1, 1, 1) < 0.5)
        th = c * u(-0.2, 0.2)  # hue rotation about the grey axis (+-11 deg)
        a = torch.full((3,), 1 / math.sqrt(3), device=self.dev)
        K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], device=self.dev)
        eye = torch.eye(3, device=self.dev)
        self.hue[ids] = eye + torch.sin(th)[..., None, None] * K + (1 - torch.cos(th))[..., None, None] * (K @ K)
        self.grad[ids] = c * u(-0.18, 0.18, 2)
        self.noise_sd = 0.015 * c

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        x = img.permute(0, 1, 4, 2, 3).float() / 255.0
        if self.on:
            n, c, _, h, w = x.shape
            x = torch.einsum("ncij,ncjhw->ncihw", self.hue, x)
            blurred = F.avg_pool2d(x.flatten(0, 1), 3, 1, 1, count_include_pad=False).view_as(x)
            x = x.lerp(blurred, self.blur)
            x = x.clamp(1e-4, 1.0).pow(self.gamma)
            x = (x - 0.5) * self.contrast + 0.5
            x = x * self.gain + self.bright
            ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, 1, h, 1)
            xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, 1, w)
            x = x * (1 + self.grad[..., 0, None, None, None] * xs + self.grad[..., 1, None, None, None] * ys)
            x = x + getattr(self, 'noise_sd', 0.015) * torch.randn_like(x)
        return (x.clamp(0, 1) * 255).to(torch.uint8)


# ---------------------------------------------------------------- network


class SpatialSoftmax(nn.Module):
    def forward(self, x):
        n, c, h, w = x.shape
        att = torch.softmax(x.flatten(2).float(), -1).view(n, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1)
        xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w)
        return torch.cat(((att * xs).sum((2, 3)), (att * ys).sum((2, 3))), -1).to(x.dtype)


class Encoder(nn.Module):
    """Small conv net: spatial-softmax keypoints + pooled appearance -> 128."""

    def __init__(self, out: int = 128):
        super().__init__()

        def blk(i, o, k, s):
            return [nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ELU()]

        self.conv = nn.Sequential(*blk(3, 32, 5, 2), *blk(32, 64, 3, 2), *blk(64, 64, 3, 2), *blk(64, 64, 3, 1))
        self.kp = SpatialSoftmax()
        self.fc = nn.Sequential(nn.Linear(64 * 2 + 64, out), nn.ELU())

    def forward(self, x):
        f = self.conv(x)
        return self.fc(torch.cat((self.kp(f), f.mean((2, 3))), -1))


class Student(nn.Module):
    def __init__(self, sdim: int, adim: int = 12, auxdim: int = 12, hidden: int = 512, layers: int = 2,
                 use_img: bool = True, img_feat: int = 128, sx: int = 0, top_img: bool = False):
        super().__init__()
        # sx: number of extra inputs appended to the coarse group (WX_SX=1: student_extra, 50)
        # top_img: the C920 overhead frame as a third image input (own encoder), WX_TOPIMG=1
        self.cfg = dict(sdim=sdim, adim=adim, auxdim=auxdim, hidden=hidden, layers=layers, use_img=use_img, img_feat=img_feat)
        if sx:
            self.cfg["sx"] = sx
        if top_img:
            self.cfg["top_img"] = True
        self.use_img = use_img
        self.top_img = top_img
        self.enc = Encoder(img_feat) if use_img else None
        self.tenc = Encoder(img_feat) if top_img else None
        self.smlp = nn.Sequential(nn.Linear(sdim, 256), nn.ELU())
        self.lstm = nn.LSTM(256 + (2 * img_feat if use_img else 0) + (img_feat if top_img else 0), hidden, layers)
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ELU(), nn.Linear(256, adim))
        self.aux = nn.Sequential(nn.Linear(hidden, 128), nn.ELU(), nn.Linear(128, auxdim))
        self.register_buffer("mean", torch.zeros(sdim))
        self.register_buffer("std", torch.ones(sdim))
        self.register_buffer("keep", torch.ones(sdim))

    def features(self, s, img, timg=None):
        f = self.smlp(((s - self.mean) / self.std * self.keep).clamp(-10, 10))
        if self.use_img:
            b = img.shape[0]
            x = img.flatten(0, 1).float() / 255.0 - 0.5
            f = torch.cat((f, self.enc(x).view(b, -1)), -1)
        if self.top_img:
            f = torch.cat((f, self.tenc(timg.float() / 255.0 - 0.5)), -1)
        return f

    def step(self, s, img, state, timg=None):
        y, state = self.lstm(self.features(s, img, timg)[None], state)
        return self.head(y[0]), self.aux(y[0]), state

    def seq(self, s, img, first, state, timg=None):
        """s (T, B, sdim), img (T, B, 2, 3, H, W) or None, timg (T, B, 3, h, w) or None, first (T, B) bool: episode
        start (state zeroed)."""
        t, b = s.shape[:2]
        f = self.features(s.flatten(0, 1), None if img is None else img.flatten(0, 1),
                          None if timg is None else timg.flatten(0, 1)).view(t, b, -1)
        h, c = state
        ys = []
        for i in range(t):
            keep = (~first[i]).to(h.dtype).view(1, b, 1)
            h, c = h * keep, c * keep
            y, (h, c) = self.lstm(f[i: i + 1], (h, c))
            ys.append(y[0])
        y = torch.stack(ys)
        return self.head(y), self.aux(y)

    @staticmethod
    def to_action(pred):
        a = pred.clamp(-2.0, 2.0).clone()
        a[..., list(GRIP)] = torch.where(pred[..., list(GRIP)] > 0, 1.0, -1.0)
        return a


def _keep(spec: str, dim: int, dev) -> torch.Tensor:
    keep = torch.ones(dim, device=dev)
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        keep[int(lo): int(hi or lo) + 1] = 0.0
    return keep


def load_student(path: str, dev) -> Student:
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = Student(**ck["cfg"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def _term_slice(env, group: str, term: str) -> tuple[int, int]:
    om = env.observation_manager
    names, dims = om.active_terms[group], om.group_obs_term_dim[group]
    off = 0
    for nme, d in zip(names, dims):
        k = int(math.prod(d))
        if nme == term:
            return off, off + k
        off += k
    raise KeyError(term)


def aux_index(env) -> torch.Tensor:
    """critic 'relative' term: per slot rim-pad (3), goal-vessel (3) -> 12 indices."""
    lo, _ = _term_slice(env, "critic", "relative")
    return torch.tensor([lo + i for s in range(2) for i in range(9 * s, 9 * s + 6)], device=env.device)


SX_DIM = 50


def student_extra(env, prev_a: torch.Tensor) -> torch.Tensor:
    """WX_SX=1 extra student inputs (50): per vessel the fused estimate's uncertainty (cm), age of its last top /
    wrist measurement incl. latency (s) and the held flag (24, wx_fuse; zeros without the fuser); per arm slot the
    known depth of its target stack (task bookkeeping, 2); the last two commanded actions (24, latency compensation)."""
    from . import cup_basket_env_cfg as cb

    env = getattr(env, "unwrapped", env)
    c = getattr(env, "_wx_coarse", None)
    n = env.num_envs
    # WX_SX_NOFUSE=1: keep the fuser slots at zero (a student trained without the fuser, WX_SYNTH=1, evaluated with it)
    if c is not None and getattr(c, "fuser", None) is not None and os.environ.get("WX_SX_NOFUSE") != "1":
        f = c.fuser.features(env)
    else:
        f = torch.zeros(n, 24, device=env.device)
    act = torch.stack([cb.slot_active(env, s_) for s_ in range(cb.N_SLOT)], -1).float()
    depth = cb._state(env)["depth"].float() * act
    return torch.cat((f, depth, prev_a.reshape(n, -1)), -1)


def _wrist_model(env, n, dev):
    """Wrist image model of the realistic setup: shared with the wrist perception under WX_WFUSE (reset by the env)."""
    if R.WFUSE:
        return R.shared_wrist(env)
    return R.WristReal(n, dev)


def _reset_wrist(aug, ids) -> None:
    if aug is not None and not getattr(aug, "shared", False):
        aug.reset(ids)


# ---------------------------------------------------------------- executors (scripted-oracle interface)


def _teacher(env):
    from .scripted import _script_class

    return _script_class(os.environ.get("WX_TEACHER", "residual:ResidualScript"))(env)


class CoarseTeacher:
    """(b) The privileged teacher fed the coarse overhead estimate (``coarse`` group) in place of its inputs."""

    def __init__(self, env):
        self.env = env
        self.t = _teacher(env)
        self.preal = R.ProprioReal(env) if R.REAL else None
        self.areal = R.ActionReal(env.num_envs, env.device) if R.REAL else None

    def reset(self, ids):
        self.t.reset(ids)
        if self.preal is not None:
            self.preal.reset(ids)
            self.areal.reset(ids)

    @torch.no_grad()
    def act(self):
        ob = self.env.obs_buf
        saved = {k: ob[k] for k in ("policy", "critic")}
        c = ob["coarse"] if self.preal is None else self.preal(ob["coarse"])
        cp = os.environ.get("WX_DBG_COPY")  # diagnostics: these coarse terms replaced by the privileged critic's
        if cp:
            c = c.clone()
            for nme in cp.split(","):
                a0, a1 = _term_slice(self.env, "coarse", nme)
                b0, b1 = _term_slice(self.env, "critic", nme)
                c[:, a0:a1] = ob["critic"][:, b0:b1]
        if os.environ.get("WX_DBG") == "1":
            self.k = getattr(self, "k", 0) + 1
            if self.k % 150 == 0:
                om = self.env.observation_manager
                out = {}
                for nme in om.active_terms["coarse"]:
                    a0, a1 = _term_slice(self.env, "coarse", nme)
                    try:
                        b0, b1 = _term_slice(self.env, "policy", nme)
                    except KeyError:
                        continue
                    out[nme] = round(float((c[:, a0:a1] - ob["policy"][:, b0:b1]).abs().mean()), 4)
                print("WX_DBG " + json.dumps(out), flush=True)
        try:
            ob["policy"] = c
            ob["critic"] = c
            a = self.t.act()
        finally:
            ob.update(saved)
        return a if self.areal is None else self.areal(a)


def print_wrist_mount(env) -> None:
    """One-time: wrist camera pose in its gripper-link frame (offset in mm, optical axis), the pad centre in the camera
    frame, and the field of view, for the scene-setup sheet."""
    try:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

        from .env_cfg import pad_center

        for arm, cam_name in (("robot_left", "wrist_left"), ("robot_right", "wrist_right")):
            cam = env.scene.sensors[cam_name]
            robot = env.scene[arm]
            b = robot.find_bodies("gripper")[0][0]
            gp, gq = robot.data.body_pos_w.torch[:1, b], robot.data.body_quat_w.torch[:1, b]
            cp, cq = cam.data.pos_w.torch[:1], cam.data.quat_w_ros.torch[:1]
            rp, rq = subtract_frame_transforms(gp, gq, cp, cq)
            fwd = quat_apply(rq, rp.new_tensor([[0.0, 0.0, 1.0]]))  # ROS optical axis (z) in the gripper frame
            pad = pad_center(env, arm)[:1] + env.scene.env_origins[:1]
            pad_c = quat_apply_inverse(cq, pad - cp)  # pad centre in the camera (ROS) frame
            fwd_w = quat_apply(cq, cp.new_tensor([[0.0, 0.0, 1.0]]))
            print("WX_WRIST_MOUNT " + json.dumps({"arm": arm, "offset_mm_gripper_frame": [round(v * 1000, 1) for v in rp[0].tolist()],
                                                  "quat_xyzw_gripper_frame_ros": [round(v, 4) for v in rq[0].tolist()],
                                                  "optical_axis_gripper_frame": [round(v, 3) for v in fwd[0].tolist()],
                                                  "pad_centre_in_camera_mm": [round(v * 1000, 1) for v in pad_c[0].tolist()],
                                                  "optical_axis_world_now": [round(v, 3) for v in fwd_w[0].tolist()],
                                                  "res": [cam.cfg.width, cam.cfg.height],
                                                  "hfov_deg": round(2 * math.degrees(math.atan(20.955 / 2 / float(os.environ.get('WX_FOCAL', 11.0)))), 1)}),
                  flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"WX_WRIST_MOUNT failed: {error}", flush=True)


def gripper_shields(owner, env, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Deployable gripper rules on the student's action a (from logits pred): WX_GRIP_BIAS (pre-grasp close threshold),
    WX_REL_BIAS / WX_REL_GATE / WX_REL_LATCH (release near the estimated goal). Per-env state in owner.latch."""
    if GRIP_BIAS:  # deployable grasp trigger: while an arm holds nothing (gripper cmd + FK bookkeeping), close already
        from .wx_env import coarse  # when the gripper logit < GRIP_BIAS (default threshold 0)

        held = coarse(env).held if hasattr(coarse(env), "held") else torch.zeros(pred.shape[0], 2, dtype=torch.bool, device=pred.device)
        if not hasattr(owner, "latch"):
            owner.latch = torch.zeros(pred.shape[0], 2, dtype=torch.bool, device=pred.device)
        for s_, j in enumerate(GRIP):
            thr = torch.where(held[:, s_] | owner.latch[:, s_], 0.0, GRIP_BIAS)
            a[:, j] = torch.where(pred[:, j] > thr, 1.0, -1.0)
        owner.glog = pred[:, list(GRIP)]
    if REL_GATE or REL_BIAS:  # deployable release shaping with the estimated held vessel vs its goal
        from .wx_env import coarse, coarse_relative

        c = coarse(env)
        if hasattr(c, "held"):
            rel = coarse_relative(env)  # per slot: rim - pad (3), goal - vessel (3), 0 (3)
            for s_, j in enumerate(GRIP):
                gv = rel[:, 9 * s_ + 3: 9 * s_ + 6]
                near = (torch.linalg.vector_norm(gv[:, :2], dim=-1) < REL_NEAR[0]) & (-gv[:, 2] < REL_NEAR[1])
                h = c.held[:, s_]
                if REL_BIAS:  # near the goal: open already when the logit > -REL_BIAS
                    a[:, j] = torch.where(h & near, torch.where(pred[:, j] > -REL_BIAS, 1.0, -1.0), a[:, j])
                if REL_GATE:  # away from the goal: never open
                    a[:, j] = torch.where(h & ~near, -1.0, a[:, j])
                if REL_LATCH and hasattr(owner, "latch"):  # released near the goal: no grasp bias (1) / stay open (2)
                    if REL_LATCH == 2:
                        a[:, j] = torch.where(owner.latch[:, s_] & ~h, 1.0, a[:, j])
                    owner.latch[:, s_] |= h & near & (a[:, j] > 0)
    return a


class StudentScript:
    """(c) The trained student (``WX_STUDENT``), deterministic; wrist images randomised as in training
    (``WX_EVAL_AUG=0``: clean images)."""

    def __init__(self, env):
        self.env = env
        self.net = load_student(os.environ["WX_STUDENT"], env.device)
        hid, lay = self.net.cfg["hidden"], self.net.cfg["layers"]
        self.h = torch.zeros(lay, env.num_envs, hid, device=env.device)
        self.c = torch.zeros_like(self.h)
        self.aug = ImageAug(env.num_envs, env.device, on=os.environ.get("WX_EVAL_AUG", "1") == "1") if self.net.use_img else None
        if R.REAL and self.net.use_img:
            self.aug = _wrist_model(env, env.num_envs, env.device)
        self.sx = int(self.net.cfg.get("sx", 0))
        self.prev_a = torch.zeros(env.num_envs, 2, 12, device=env.device)
        self.preal = R.ProprioReal(env) if R.REAL else None
        self.areal = R.ActionReal(env.num_envs, env.device) if R.REAL else None
        self.last_img = None
        if self.net.use_img:
            print_wrist_mount(env)
        print(f"StudentScript {os.environ['WX_STUDENT']} img={self.net.use_img}", flush=True)

    def reset(self, ids):
        self.h[:, ids] = 0.0
        self.c[:, ids] = 0.0
        self.prev_a[ids] = 0.0
        if hasattr(self, "latch"):
            self.latch[ids] = False
        _reset_wrist(self.aug, ids)
        if self.preal is not None:
            self.preal.reset(ids)
            self.areal.reset(ids)

    @torch.no_grad()
    def propose(self) -> torch.Tensor:
        """The student's action (before the actuator delay / deadband model); keeps its inputs in self.s_in."""
        s = torch.nan_to_num(self.env.obs_buf["coarse"])
        if self.preal is not None:
            s = self.preal(s)
        if self.sx:
            s = torch.cat((s, student_extra(self.env, self.prev_a)), -1)
        self.s_in = s
        img = self.aug(wrist_images(self.env)) if self.aug is not None else None
        self.last_img = img
        timg = top_image(self.env) if self.net.top_img else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _, (self.h, self.c) = self.net.step(s, img, (self.h, self.c), timg)
        self.h, self.c = self.h.float(), self.c.float()
        pred = pred.float()
        a = Student.to_action(pred)
        return gripper_shields(self, self.env, pred, a)

    def commanded(self, a: torch.Tensor) -> torch.Tensor:
        """Record the commanded action (action history input) and pass it through the actuator model."""
        self.prev_a = torch.stack((a, self.prev_a[:, 0]), 1)
        return a if self.areal is None else self.areal(a)

    @torch.no_grad()
    def act(self):
        return self.commanded(self.propose())


class FreezeDiag(StudentScript):
    """Diagnostic eval of the student: the privileged teacher runs alongside (not executed). Per arm, a 'freeze' is a
    run of >= WX_FREEZE_STEPS steps in which every student arm command is inside the servo deadband (the arm does not
    move). Every 300 steps prints WX_FREEZE {...}: over frozen arm-steps, how often the teacher / student command the
    gripper closed, how often the teacher would command a move outside the deadband, and the mean |action|."""

    def __init__(self, env):
        super().__init__(env)
        self.teacher = _teacher(env)
        n = env.num_envs
        self.cnt = torch.zeros(n, 2, dtype=torch.long, device=env.device)
        self.qhist = []  # joint positions of the last WX_FREEZE_STEPS steps
        self.k = 0
        self.acc = {}

    def reset(self, ids):
        super().reset(ids)
        self.teacher.reset(ids)
        self.cnt[ids] = 0

    def _add(self, key, v):
        self.acc[key] = self.acc.get(key, 0.0) + float(v)

    @torch.no_grad()
    def act(self):
        a_s = self.propose()
        a_t = torch.nan_to_num(self.teacher.act().clone())
        dead = self.areal.dead if self.areal is not None else torch.zeros(a_s.shape[0], 1, device=a_s.device)
        L = int(_f("WX_FREEZE_STEPS", 60))
        q = torch.stack([self.env.scene[r].data.joint_pos[:, :6] for r in ("robot_left", "robot_right")], 1)  # (n, 2, 6)
        self.qhist.append(q.clone())
        if len(self.qhist) > L:
            self.qhist.pop(0)
        still = ((q[..., :5] - self.qhist[0][..., :5]).abs().amax(-1) < math.radians(0.5)) if len(self.qhist) == L else torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device)
        just = self.env.episode_length_buf < L
        for j, (arm, g) in enumerate(((list(range(5)), 5), (list(range(6, 11)), 11))):
            fr = still[:, j] & ~just
            self._add("arm_steps", a_s.shape[0])
            self._add("grip_q_all", q[:, j, 5].sum())
            self._add("stu_close_all", (a_s[:, g] > 0).sum())
            k = int(fr.sum())
            if k:
                self._add("frozen", k)
                self._add("tea_close", (a_t[fr, g] > 0).sum())
                self._add("stu_close", (a_s[fr, g] > 0).sum())
                self._add("tea_moves", (a_t[fr][:, arm].abs() >= dead[fr]).any(-1).sum())
                self._add("tea_absmax", a_t[fr][:, arm].abs().max(-1).values.sum())
                self._add("stu_absmax", a_s[fr][:, arm].abs().max(-1).values.sum())
                self._add("dead", dead[fr, 0].sum())
                self._add("grip_q", q[fr, j, 5].sum())
                ta, sa = a_t[fr][:, arm], a_s[fr][:, arm]
                self._add("cos_ts", F.cosine_similarity(ta, sa, -1, 1e-6).sum())
                self._add("tea_big", (ta.abs().amax(-1) > 1.0).sum())
                if os.environ.get("WX_FREEZE_PHYS", "0") == "1":
                    self._phys(j, fr, a_s[fr][:, arm])
            mv = ~still[:, j] & ~just
            if os.environ.get("WX_FREEZE_PHYS", "0") == "1" and bool(mv.any()):
                self._add("moving", int(mv.sum()))
                self._phys(j, mv, a_s[mv][:, arm], "mv_")
        self.k += 1
        if self.k % 300 == 0:
            f = max(1.0, self.acc.get("frozen", 0.0))
            out = {"step": self.k, "frozen_frac": self.acc.get("frozen", 0.0) / max(1.0, self.acc["arm_steps"]),
                   "grip_q_all": self.acc["grip_q_all"] / self.acc["arm_steps"], "stu_close_all": self.acc["stu_close_all"] / self.acc["arm_steps"]}
            for key in ("tea_close", "stu_close", "tea_moves", "tea_absmax", "stu_absmax", "dead", "grip_q", "cos_ts", "tea_big"):
                out[key] = round(self.acc.get(key, 0.0) / f, 4)
            for key in sorted(k for k in self.acc if k.startswith("ph_")):
                out[key] = round(self.acc[key] / f, 4)
            for key in sorted(k for k in self.acc if k.startswith("mv_")):
                out[key] = round(self.acc[key] / max(1.0, self.acc.get("moving", 0.0)), 4)
            print("WX_FREEZE " + json.dumps(out), flush=True)
        return self.commanded(a_s)

    def _phys(self, j: int, fr: torch.Tensor, a_arm: torch.Tensor, pre: str = "ph_") -> None:
        """Over frozen arm-steps of arm j: per joint |PD torque| = |kp (target - q)|, |gravity-compensation torque| G,
        |applied - G| (what is left for contact / friction), the commanded delta (rad), joint friction coefficient, how
        often the applied torque is at the effort limit, and how often a jaw is inside a vessel (contact proxy)."""
        from . import cup_basket_env_cfg as cb

        rob = self.env.scene[("robot_left", "robot_right")[j]]
        d = rob.data
        q, tgt = d.joint_pos[fr, :5], d.joint_pos_target[fr, :5]
        kp = d.joint_stiffness[fr, :5]
        tau = d.applied_torque[fr, :5]
        pd = kp * (tgt - q)
        try:
            G = d.gravity_compensation_forces
            G = (G.torch if hasattr(G, "torch") else G)
            G = (G if isinstance(G, torch.Tensor) else torch.as_tensor(G, device=tau.device))[fr][:, :5]
        except Exception as error:  # noqa: BLE001
            if not getattr(self, "_gwarn", False):
                print(f"[wx] no gravity compensation forces: {error}", flush=True)
                self._gwarn = True
            G = torch.zeros_like(tau)
        lim = d.joint_effort_limits[fr, :5] if hasattr(d, "joint_effort_limits") else None
        fc = d.joint_friction_coeff[fr, :5] if hasattr(d, "joint_friction_coeff") else torch.zeros_like(tau)
        for i in range(5):
            self._add(f"{pre}pd{i}", pd[:, i].abs().sum())
            self._add(f"{pre}G{i}", G[:, i].abs().sum())
            self._add(f"{pre}left{i}", (tau[:, i] - G[:, i]).abs().sum())
            self._add(f"{pre}cmd_deg{i}", (a_arm[:, i] * 0.033).abs().sum() * 57.2958)
            self._add(f"{pre}fric{i}", fc[:, i].sum())
            if lim is not None:
                self._add(f"{pre}sat{i}", (tau[:, i].abs() >= 0.98 * lim[:, i]).sum())
        # balance: frozen steps where every joint's applied torque is within 10 % (+0.01 N m) of pure gravity holding
        bal = ((tau - G).abs() <= 0.1 * G.abs() + 0.01).all(-1)
        self._add(pre + "gravity_only", bal.sum())
        try:
            self._add(pre + "jaw_in_own", (~cb.jaws_clear(self.env, j))[fr].sum())
            self._add(pre + "held_own", cb.held(self.env, j)[fr].sum())
        except Exception as error:  # noqa: BLE001
            if not getattr(self, "_jwarn", False):
                print(f"[wx] jaw proxy failed: {error}", flush=True)
                self._jwarn = True


class Watchdog:
    """Stall watchdog (deployable: measured joint angles + commanded actions only). Per arm, a stall is a window of
    WX_STALL_STEPS steps in which no arm joint moves more than WX_STALL_DEG while the policy keeps commanding a move
    (max |arm action| >= WX_STALL_CMD units). On a stall the arm backs off for WX_RETRY_STEPS steps toward the joint
    configuration (and gripper command) it had WX_BACK_STEPS steps before the window, then the policy has control
    again. At most WX_MAX_RETRY back-offs per arm and episode."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.env = env
        self.S = int(_f("WX_STALL_STEPS", 45))
        self.Rn = int(_f("WX_RETRY_STEPS", 30))
        self.Bk = int(_f("WX_BACK_STEPS", 60))
        self.tol = math.radians(_f("WX_STALL_DEG", 0.5))
        self.cmd_min = _f("WX_STALL_CMD", 0.5)
        self.max_retry = int(_f("WX_MAX_RETRY", 6))
        self.H = self.S + self.Bk + 1
        self.qh = torch.zeros(self.H, n, 2, 6, device=dev)
        self.gh = torch.zeros(self.H, n, 2, device=dev)
        self.ptr = 0
        self.left = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.goal = torch.zeros(n, 2, 6, device=dev)
        self.ggoal = torch.zeros(n, 2, device=dev)
        self.nret = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.stats = {"steps": 0, "backoff_steps": 0, "triggers": 0}

    def reset(self, ids):
        self.left[ids] = 0
        self.nret[ids] = 0

    def __call__(self, a: torch.Tensor, on: torch.Tensor | None = None) -> torch.Tensor:
        """a: (n, 12) policy actions -> actions with back-offs applied (arms of envs with on=False untouched)."""
        a = a.clone()
        q = torch.stack([self.env.scene[r].data.joint_pos[:, :6] for r in ("robot_left", "robot_right")], 1)
        t = self.env.episode_length_buf
        self.qh[self.ptr] = q
        for j, g in enumerate(GRIP):
            self.gh[self.ptr, :, j] = a[:, g]
        old = self.qh[(self.ptr - self.S) % self.H]
        moved = (q[..., :5] - old[..., :5]).abs().amax(-1) > self.tol
        on = torch.ones_like(t, dtype=torch.bool) if on is None else on
        for j, (arm, g) in enumerate(((list(range(5)), 5), (list(range(6, 11)), 11))):
            pushing = a[:, arm].abs().amax(-1) >= self.cmd_min
            stalled = on & ~moved[:, j] & pushing & (t > self.H) & (self.left[:, j] == 0) & (self.nret[:, j] < self.max_retry)
            fire = stalled.nonzero().squeeze(-1)
            if len(fire):
                back = (self.ptr - self.S - self.Bk) % self.H
                self.goal[fire, j] = self.qh[back, fire, j]
                self.ggoal[fire, j] = self.gh[back, fire, j]
                self.left[fire, j] = self.Rn
                self.nret[fire, j] += 1
                self.stats["triggers"] += len(fire)
            act_ = (self.left[:, j] > 0) & on
            if bool(act_.any()):
                sub = a[act_]
                sub[:, arm] = ((self.goal[act_, j, :5] - q[act_, j, :5]) / 0.033).clamp(-1.5, 1.5)
                sub[:, g] = self.ggoal[act_, j]
                a[act_] = sub
                self.left[act_, j] -= 1
                self.stats["backoff_steps"] += int(act_.sum())
        self.stats["steps"] += 2 * a.shape[0]
        self.ptr = (self.ptr + 1) % self.H
        return a

    def summary(self) -> dict:
        s = self.stats
        return {"triggers": s["triggers"], "backoff_frac": round(s["backoff_steps"] / max(1, s["steps"]), 4),
                "retries_per_arm_ep": round(float(self.nret.float().mean()), 3)}


class Retreat:
    """Deployable post-release retreat (WX_RETREAT=<back steps>): when an arm releases its vessel near the goal (the
    student's release latch), it backs off for WX_RETREAT_STEPS steps toward its joint configuration WX_RETREAT steps
    earlier (above the basket), gripper open; WX_RETREAT_HOLD=1 then keeps it there for the rest of the episode."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.env = env
        self.B = int(_f("WX_RETREAT", 30))
        self.Rn = int(_f("WX_RETREAT_STEPS", 30))
        self.hold = os.environ.get("WX_RETREAT_HOLD", "0") == "1"
        self.qh = torch.zeros(self.B + 1, n, 2, 6, device=dev)
        self.ptr = 0
        self.prev = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.left = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.done = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.goal = torch.zeros(n, 2, 6, device=dev)
        self.t0 = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, ids):
        self.prev[ids] = False
        self.left[ids] = 0
        self.done[ids] = False

    def __call__(self, a: torch.Tensor, latch: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        q = torch.stack([self.env.scene[r].data.joint_pos[:, :6] for r in ("robot_left", "robot_right")], 1)
        t = self.env.episode_length_buf
        self.qh[self.ptr] = q
        rise = latch & ~self.prev
        old = self.qh[(self.ptr - self.B) % (self.B + 1)]
        ok_hist = (t > self.B).unsqueeze(-1)
        for j, (arm, g) in enumerate(((list(range(5)), 5), (list(range(6, 11)), 11))):
            fire = rise[:, j] & ok_hist[:, 0]
            self.goal[fire, j] = old[fire, j]
            self.left[fire, j] = self.Rn
            act_ = self.left[:, j] > 0
            if self.hold:
                act_ = act_ | self.done[:, j]
            if bool(act_.any()):
                sub = a[act_]
                sub[:, arm] = ((self.goal[act_, j, :5] - q[act_, j, :5]) / 0.033).clamp(-1.5, 1.5)
                sub[:, g] = 1.0
                a[act_] = sub
            fin = (self.left[:, j] == 1)
            self.done[:, j] |= fin
            self.left[:, j] = (self.left[:, j] - 1).clamp(min=0)
        self.prev = latch.clone()
        self.ptr = (self.ptr + 1) % (self.B + 1)
        return a


class StallRetry(StudentScript):
    """Student + Watchdog (see there). Prints WX_STALLRETRY {...} every 300 steps."""

    def __init__(self, env):
        super().__init__(env)
        self.wd = Watchdog(env)
        self.k = 0

    def reset(self, ids):
        super().reset(ids)
        self.wd.reset(ids)

    @torch.no_grad()
    def act(self):
        a = self.wd(self.propose())
        self.k += 1
        if self.k % 300 == 0:
            print("WX_STALLRETRY " + json.dumps({"step": self.k, **self.wd.summary()}), flush=True)
        return self.commanded(a)


# ---------------------------------------------------------------- residual PPO on the frozen student


RRES_SCALE = _f("WX_RRES_SCALE", 0.5)
RRES_FIXED = [int(i) for i in os.environ.get("WX_RRES_FIXED", "5,11").split(",") if i]  # gripper dims stay with the student


def rres_features(st, base: torch.Tensor) -> torch.Tensor:
    """Actor input: the frozen student's top LSTM state (its memory of images / estimates, 512), its action (12) and its
    state inputs (coarse group after the encoder / flag model, + WX_SX extras): only signals the real robot has.
    Coarse base (``WX_RRES_BASE=coarse``): its BC-RNN memory, action, the last two commanded actions and the coarse
    inputs."""
    if isinstance(st, CoarseBase):
        return torch.cat((st.mem(), base, st.prev_a.flatten(1), st.s_in), -1)
    return torch.cat((st.h[-1], base, st.s_in), -1)


class CoarseBase:
    """Deployable-teacher base for residual PPO (``WX_RRES_BASE=coarse``): the frozen privileged teacher (BC-RNN +
    residual) fed only the coarse overhead estimate + realistic proprioception (as ``CoarseTeacher``), with the same
    propose / commanded / reset interface as ``StudentScript``."""

    def __init__(self, env):
        self.env = env
        self.t = _teacher(env)
        self.preal = R.ProprioReal(env) if R.REAL else None
        self.areal = R.ActionReal(env.num_envs, env.device) if R.REAL else None
        self.prev_a = torch.zeros(env.num_envs, 2, 12, device=env.device)
        self.s_in = None

    def reset(self, ids):
        self.t.reset(ids)
        self.prev_a[ids] = 0.0
        if self.preal is not None:
            self.preal.reset(ids)
            self.areal.reset(ids)

    def mem(self) -> torch.Tensor:
        """The base BC-RNN's top hidden state (its memory), zeros if not exposed."""
        b = getattr(self.t, "base", None) or getattr(self.t, "bc", None)
        h = getattr(b, "h", None) if b is not None else None
        if isinstance(h, tuple):
            h = h[0]
        if h is None or not torch.is_tensor(h):
            return torch.zeros(self.env.num_envs, 0, device=self.env.device)
        return torch.nan_to_num(h[-1] if h.dim() == 3 else h).float()

    @torch.no_grad()
    def propose(self) -> torch.Tensor:
        ob = self.env.obs_buf
        saved = {k: ob[k] for k in ("policy", "critic")}
        c = torch.nan_to_num(ob["coarse"])
        if self.preal is not None:
            c = self.preal(c)
        self.s_in = c
        try:
            ob["policy"] = c
            ob["critic"] = c
            a = self.t.act()
        finally:
            ob.update(saved)
        return torch.nan_to_num(a)

    def commanded(self, a: torch.Tensor) -> torch.Tensor:
        self.prev_a = torch.stack((a, self.prev_a[:, 0]), 1)
        return a if self.areal is None else self.areal(a)


class CoarseResScript(CoarseBase):
    """Deployable teacher for eval / distillation: coarse base + trained residual actor (``WX_RRES=<model_*.pt>``)."""

    def __init__(self, env):
        super().__init__(env)
        sd = torch.load(os.environ["WX_RRES"], map_location=env.device, weights_only=False)["actor_state_dict"]
        self.mean, self.std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
        layers = []
        for j, i in enumerate(idx):
            w = sd[f"mlp.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": sd[f"mlp.{i}.bias"]})
            layers += [lin] + ([nn.ELU()] if j < len(idx) - 1 else [])
        self.mlp = nn.Sequential(*layers).to(env.device).eval()
        self.mask = torch.ones(12, device=env.device)
        self.mask[RRES_FIXED] = 0.0

    @torch.no_grad()
    def act(self):
        base = self.propose()
        x = rres_features(self, base)
        res = self.mlp((x - self.mean) / (self.std + 1e-2))
        return self.commanded(base + RRES_SCALE * self.mask * res)


class StudentResScript(StudentScript):
    """Frozen student + trained residual actor (``WX_RRES=<rsl_rl model_*.pt>``), deterministic (actor mean)."""

    def __init__(self, env):
        super().__init__(env)
        sd = torch.load(os.environ["WX_RRES"], map_location=env.device, weights_only=False)["actor_state_dict"]
        self.mean, self.std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
        layers = []
        for j, i in enumerate(idx):
            w = sd[f"mlp.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": sd[f"mlp.{i}.bias"]})
            layers += [lin] + ([nn.ELU()] if j < len(idx) - 1 else [])
        self.mlp = nn.Sequential(*layers).to(env.device).eval()
        self.mask = torch.ones(12, device=env.device)
        self.mask[RRES_FIXED] = 0.0
        # WX_RRES_EVAL_STD: eval-time Gaussian noise on the executed action ("ckpt" = the PPO policy's own std,
        # i.e. the stochastic policy as in training; a number = fixed std, usable with WX_RRES_SCALE=0 for base+noise)
        ns = os.environ.get("WX_RRES_EVAL_STD", "0")
        if ns == "ckpt":
            self.noise = sd["distribution.log_std_param"].exp().to(env.device)
        elif "," in ns:  # "<arm std>,<gripper std>"
            a_, g_ = (float(v) for v in ns.split(","))
            self.noise = torch.tensor(([a_] * 5 + [g_]) * 2, device=env.device)
        else:
            self.noise = torch.full((12,), float(ns), device=env.device)
        self.wd = Watchdog(env) if os.environ.get("WX_RRES_WD", "0") == "1" else None
        self.ret = Retreat(env) if os.environ.get("WX_RETREAT") else None
        print(f"WX_RRES_EVAL noise={self.noise.tolist()} scale={RRES_SCALE} wd={self.wd is not None}", flush=True)

    def reset(self, ids):
        super().reset(ids)
        if self.wd is not None:
            self.wd.reset(ids)
        if self.ret is not None:
            self.ret.reset(ids)

    @torch.no_grad()
    def act(self):
        base = self.propose()
        if RRES_SCALE == 0:  # residual unused (base + noise): skip it, so students of any hidden size work
            res = self.noise * torch.randn_like(base)
        else:
            x = rres_features(self, base)
            res = self.mlp((x - self.mean) / (self.std + 1e-2))
            res = RRES_SCALE * res + self.noise * torch.randn_like(res)
        a = base + self.mask * res
        if self.ret is not None and hasattr(self, "latch"):
            a = self.ret(a, self.latch)
        if self.wd is not None:
            a = self.wd(a)
            self.n_act = getattr(self, "n_act", 0) + 1
            if self.n_act % 300 == 0:
                print("WX_STALLRETRY " + json.dumps({"step": self.n_act, **self.wd.summary()}), flush=True)
        return self.commanded(a)


def install_rres() -> list[str]:
    """``isaaclab train`` callback (wx_rres.sh): PPO learns a residual on the frozen student ``WX_STUDENT``. Executed
    action = student + WX_RRES_SCALE * residual (gripper dims masked), then the realistic actuator model; actor obs =
    ``rres_features`` (student inputs only), critic obs = privileged ``critic`` group + the student's action. The
    actor's output layer starts at zero (training starts from the student's own success rate). Sparse task reward
    (milestones + success); use gamma 0.998 (wx_rres.sh)."""
    register()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.models.mlp_model import MLPModel

    try:
        from .nanguard import install_guard

        install_guard()
    except Exception as error:  # noqa: BLE001
        print(f"WX_RRES no nan guard: {error}", flush=True)
    model_init = MLPModel.__init__

    def init(self, *args, **kwargs):
        model_init(self, *args, **kwargs)
        if self.distribution is not None:
            last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    MLPModel.__init__ = init
    inner_obs, inner_step = RslRlVecEnvWrapper.get_observations, RslRlVecEnvWrapper.step
    st: dict = {"ep": [], "k": 0}

    def ensure(env):
        if "s" not in st:
            with torch.inference_mode(False):
                st["s"] = CoarseBase(env) if os.environ.get("WX_RRES_BASE") == "coarse" else StudentScript(env)
            st["mask"] = torch.ones(12, device=env.device)
            st["mask"][RRES_FIXED] = 0.0
            st["base"] = st["s"].propose()
        return st["s"]

    def augment(obs, env):
        s_ = st["s"]
        obs["policy"] = rres_features(s_, st["base"])
        obs["critic"] = torch.cat((obs["critic"], st["base"]), -1)
        return obs

    def get_observations(self):
        env = self.unwrapped
        ensure(env)
        return augment(inner_obs(self), env)

    def step(self, actions):
        env = self.unwrapped
        s_ = ensure(env)
        executed = s_.commanded(st["base"] + RRES_SCALE * st["mask"] * actions)
        obs, rew, dones, extras = inner_step(self, executed)
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            s_.reset(ids)
            st["ep"] = (st["ep"] + env._bi_final[ids, -1].int().tolist())[-2000:]
        st["k"] += 1
        if st["k"] % 200 == 0:
            print(f"WX_RRES step={st['k']} succ_last{len(st['ep'])}={sum(st['ep']) / max(1, len(st['ep'])):.3f}", flush=True)
        st["base"] = torch.nan_to_num(s_.propose())
        return augment(obs, env), rew, dones, extras

    RslRlVecEnvWrapper.get_observations = get_observations
    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


# ---------------------------------------------------------------- DAgger


class DAgger:
    def __init__(self, env):
        self.env = env
        dev = self.dev = env.device
        n = self.n = env.num_envs
        self.T, self.C = int(_f("WX_T", 32)), int(_f("WX_BUF", 24))
        self.K, self.B = int(_f("WX_K", 24)), int(_f("WX_B", 64))
        self.iters, self.warm, self.decay = int(_f("WX_ITERS", 1500)), int(_f("WX_WARM", 40)), int(_f("WX_DECAY", 400))
        self.beta_min = _f("WX_BETA_MIN", 0.0)
        self.out = os.path.expanduser(os.environ.get("WX_OUT", "~/wx/student.pt"))
        self.use_img = os.environ.get("WX_IMG", "1") == "1"
        # multi-node data parallel (WX_WORLD > 1): every node rolls out its own envs into its own buffer; gradients are
        # averaged with NCCL all-reduce, rank 0 broadcasts the initial weights and alone writes checkpoints
        self.world, self.rank = int(_f("WX_WORLD", 1)), int(_f("WX_RANK", 0))
        if self.world > 1:
            import datetime

            import torch.distributed as dist

            dist.init_process_group("nccl", init_method=f"tcp://{os.environ['WX_MASTER']}:{os.environ.get('WX_PORT', '29531')}",
                                    rank=self.rank, world_size=self.world, timeout=datetime.timedelta(minutes=float(os.environ.get('WX_NCCL_TIMEOUT_MIN', 30))))
            self.dist = dist
            print(f"WX dist rank {self.rank}/{self.world} up", flush=True)
        self.teacher = _teacher(env)
        self.sx = SX_DIM if os.environ.get("WX_SX") == "1" else 0
        sdim = env.obs_buf["coarse"].shape[-1] + self.sx
        self.prev_a = torch.zeros(n, 2, 12, device=dev)
        self.aux_idx = aux_index(env)
        self.use_top = os.environ.get("WX_TOPIMG", "0") == "1"
        self.net = Student(sdim, hidden=int(_f("WX_HIDDEN", 512)), layers=int(_f("WX_LAYERS", 2)), use_img=self.use_img,
                           sx=self.sx, top_img=self.use_top).to(dev)
        self.net.keep.copy_(_keep(os.environ.get("WX_DROP", "36-47"), sdim, dev))
        init = os.environ.get("WX_INIT")
        self.keep_old_norm = None
        if init:  # warm start (e.g. re-distil from a new teacher): weights and normaliser of an earlier student
            sd = torch.load(init, map_location=dev, weights_only=False)["state_dict"]
            own = self.net.state_dict()
            if self.use_top and not any(k.startswith("tenc.") for k in sd) and sd["mean"].shape == own["mean"].shape:
                # new C920 image input on a student without it: the top encoder is fresh, the LSTM input weights of its
                # features start at 0 (the warm-started student initially acts as before); everything else is copied
                for k, v in sd.items():
                    if k == "lstm.weight_ih_l0":
                        own[k].zero_()
                        own[k][:, : v.shape[1]] = v
                    elif k in own and own[k].shape == v.shape:
                        own[k] = v
                self.net.load_state_dict(own)
                print(f"WX_INIT {init} + fresh top-image encoder (lstm in {sd['lstm.weight_ih_l0'].shape[1]} -> "
                      f"{own['lstm.weight_ih_l0'].shape[1]})", flush=True)
                self.normed = True
            elif sd["mean"].shape != own["mean"].shape:
                # new inputs (WX_SX): copy what fits; the first layer's new input columns start at 0, so the warm-started
                # student initially acts as before; the normaliser of the new columns comes from the first chunk
                for k, v in sd.items():
                    if k in own and own[k].shape == v.shape:
                        own[k] = v
                    elif k in ("mean", "std", "keep"):
                        own[k][: v.shape[0]] = v
                    elif k == "smlp.0.weight" and own[k].shape[0] == v.shape[0]:
                        own[k].zero_()
                        own[k][:, : v.shape[1]] = v
                self.net.load_state_dict(own)
                print(f"WX_INIT partial from {init} (sdim {sd['mean'].shape[0]} -> {own['mean'].shape[0]})", flush=True)
                self.keep_old_norm = sd["mean"].shape[0]
                self.normed = False
            elif any(k in own and own[k].shape != v.shape for k, v in sd.items()):
                # resized LSTM/head (WX_HIDDEN/WX_LAYERS): keep the image encoders, input MLP and normaliser, fresh core
                n_cp = 0
                for k, v in sd.items():
                    if k in own and own[k].shape == v.shape and not k.startswith(("lstm.", "head.", "aux.")):
                        own[k] = v
                        n_cp += 1
                self.net.load_state_dict(own)
                print(f"WX_INIT resized from {init}: copied {n_cp} tensors (encoders/smlp/norm), fresh LSTM+heads", flush=True)
                self.normed = True
            else:
                self.net.load_state_dict(sd)
                self.normed = True
        else:
            self.normed = False
        if self.world > 1:
            for t_ in self.net.state_dict().values():
                self.dist.broadcast(t_, 0)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=_f("WX_LR", 3e-4))
        self.lr = _f("WX_LR", 3e-4)
        L, Hd = self.net.cfg["layers"], self.net.cfg["hidden"]
        self.h = torch.zeros(L, n, Hd, device=dev)
        self.c = torch.zeros_like(self.h)
        self.aug = ImageAug(n, dev) if self.use_img else None
        if R.REAL and self.use_img:
            self.aug = _wrist_model(env, n, dev)
        self.preal = R.ProprioReal(env) if R.REAL else None
        self.areal = R.ActionReal(n, dev) if R.REAL else None
        self.img_cpu = os.environ.get("WX_BUF_CPU", "0") == "1"  # image ring buffers in host RAM (long replay at 320x240)
        # ring buffer, (chunk, env, time, ...)
        T, C = self.T, self.C
        self.bS = torch.zeros(C, n, T, sdim, device=dev)
        self.bA = torch.zeros(C, n, T, 12, device=dev)
        self.bX = torch.zeros(C, n, T, 12, device=dev)
        self.bF = torch.zeros(C, n, T, dtype=torch.bool, device=dev)
        self.bV = torch.zeros(C, n, T, dtype=torch.bool, device=dev)
        self.bH = torch.zeros(C, L, n, Hd, device=dev)
        self.bC = torch.zeros_like(self.bH)
        if self.use_img:
            r = int(_f("WX_RES", 88))
            ih, iw = (R.WRIST_OUT_H, R.WRIST_OUT_W) if R.REAL else (r, r)
            self.bI = (torch.empty(C, n, T, 2, 3, ih, iw, dtype=torch.uint8) if self.img_cpu else
                       torch.zeros(C, n, T, 2, 3, ih, iw, dtype=torch.uint8, device=dev))
            print(f"WX buffer images {self.bI.numel() / 1e9:.1f} GB on {self.bI.device}", flush=True)
        if self.use_top:
            self.bT = (torch.empty(C, n, T, 3, TOPIMG_H, TOPIMG_W, dtype=torch.uint8) if self.img_cpu else
                       torch.zeros(C, n, T, 3, TOPIMG_H, TOPIMG_W, dtype=torch.uint8, device=dev))
            print(f"WX buffer top images {self.bT.numel() / 1e9:.1f} GB ({TOPIMG_W}x{TOPIMG_H})", flush=True)
        # success-filtered self-imitation (WX_SI=1): student episodes run with OU exploration noise on the arm actions
        # (WX_SI_NOISE) and the stall Watchdog (WX_SI_WD=1); when such an episode succeeds, its steps still in the ring
        # buffer are relabelled with the executed actions (failed episodes keep the teacher labels)
        self.si = os.environ.get("WX_SI", "0") == "1"
        if self.si:
            self.wd = Watchdog(env) if os.environ.get("WX_SI_WD", "1") == "1" else None
            self.si_noise = _f("WX_SI_NOISE", 0.1)
            self.z = torch.zeros(n, 12, device=dev)
            self.bE = torch.zeros(C, n, T, 12, device=dev)
            self.bEp = torch.full((C, n, T), -1, dtype=torch.long, device=dev)
            self.si_stats = {"eps": 0, "steps": 0}
        self.filled, self.slot, self.t, self.it = 0, 0, 0, 0
        self.first = torch.ones(n, dtype=torch.bool, device=dev)
        self.use_t = torch.full((n,), os.environ.get("WX_USE_T0", "1") == "1", dtype=torch.bool, device=dev)
        self.epi = torch.zeros(n, dtype=torch.long, device=dev)  # finished episodes per env
        self.ep = {"stu": [], "tea": [], "stu0": [], "stu1": []}
        self.tstart, self.steps = time.time(), 0
        self.log = {}
        print(f"WX DAgger n={n} sdim={sdim} img={self.use_img} T={T} buf={C} K={self.K} B={self.B} iters={self.iters} "
              f"warm={self.warm} decay={self.decay} out={self.out} teacher={os.environ.get('WX_TEACHER', 'residual:ResidualScript')} "
              f"BCRNN={os.environ.get('BITIDY_BCRNN')} RES={os.environ.get('BITIDY_RESIDUAL')}", flush=True)

    def beta(self) -> float:
        if self.it < self.warm:
            return 1.0
        x = min(1.0, (self.it - self.warm) / max(1, self.decay - self.warm))
        return 1.0 + x * (self.beta_min - 1.0)

    def step(self, wrapper, inner_step):
        env, t = self.env, self.t
        ob = env.obs_buf
        s = torch.nan_to_num(ob["coarse"]).clone()
        if self.preal is not None:
            s = self.preal(s)
        if self.sx:
            s = torch.cat((s, student_extra(env, self.prev_a)), -1)
        img = self.aug(wrist_images(env).clone()) if self.use_img else None
        timg = top_image(env) if self.use_top else None
        a_t = self.teacher.act().clone()
        aux = ob["critic"][:, self.aux_idx].clone()
        valid = torch.isfinite(a_t).all(-1) & torch.isfinite(aux).all(-1) & torch.isfinite(ob["coarse"]).all(-1)
        a_t, aux = torch.nan_to_num(a_t), torch.nan_to_num(aux)
        if t == 0:
            self.bH[self.slot] = self.h
            self.bC[self.slot] = self.c
        if not self.normed and self.keep_old_norm is None:  # normaliser from the first rollout step's states
            self.net.mean.copy_(s.mean(0))  # (teacher-run); refined at chunk 1
            self.net.std.copy_(s.std(0).clamp(min=1e-2))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _, (h, c) = self.net.step(s, img, (self.h, self.c), timg)
        self.h, self.c = h.float(), c.float()
        a_s = Student.to_action(pred.float())
        if DAG_SHIELD:  # student episodes run with the deployable gripper rules (as deployed); labels stay the teacher's
            a_s = gripper_shields(self, env, pred.float(), a_s)
        if self.si:
            stu = ~self.use_t
            self.z = 0.9 * self.z + 0.4359 * self.si_noise * torch.randn_like(self.z)  # stationary std = WX_SI_NOISE
            noisy = a_s.clone()
            arm = [i for i in range(12) if i not in GRIP]
            noisy[:, arm] = noisy[:, arm] + self.z[:, arm]
            if self.wd is not None:
                noisy = self.wd(noisy, stu)
            a_s = torch.where(stu.unsqueeze(-1), noisy.clamp(-2, 2), a_s)
        executed = torch.where(self.use_t.unsqueeze(-1), a_t.clamp(-2, 2), a_s)
        self.prev_a = torch.stack((executed, self.prev_a[:, 0]), 1)
        policy_out = executed  # policy-level action (before ActionReal delay/deadband) -> SI label
        if self.areal is not None:
            executed = self.areal(executed)
        sl = self.slot
        self.bS[sl, :, t] = s
        self.bA[sl, :, t] = a_t.clamp(-2, 2)
        self.bX[sl, :, t] = aux
        self.bF[sl, :, t] = self.first & (t > 0)
        self.bV[sl, :, t] = valid
        if self.si:
            self.bE[sl, :, t] = policy_out
            self.bEp[sl, :, t] = self.epi
        if img is not None:
            self.bI[sl, :, t] = img.to(self.bI.device)
        if timg is not None:
            self.bT[sl, :, t] = timg.to(self.bT.device)
        result = inner_step(wrapper, executed)
        dones = result[2].bool()
        self.steps += 1
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            succ = env._bi_final[ids, -1]
            ut = self.use_t[ids]
            self.ep["tea"] += succ[ut].int().tolist()
            self.ep["stu"] += succ[~ut].int().tolist()
            self.ep["tea"], self.ep["stu"] = self.ep["tea"][-400:], self.ep["stu"][-400:]
            f0 = (self.epi[ids] == 0) & ~ut
            self.ep["stu0"] += succ[f0].int().tolist()
            self.ep["stu1"] += succ[(self.epi[ids] > 0) & ~ut].int().tolist()
            if self.si:
                for e in ids[(succ > 0.5) & ~ut].tolist():
                    m = self.bEp[:, e] == self.epi[e]
                    self.bA[:, e][m] = self.bE[:, e][m]
                    self.si_stats["eps"] += 1
                    self.si_stats["steps"] += int(m.sum())
                if self.wd is not None:
                    self.wd.reset(ids)
                self.z[ids] = 0.0
            self.epi[ids] += 1
            if hasattr(self, "latch"):
                self.latch[ids] = False
            self.teacher.reset(ids)
            self.h[:, ids] = 0.0
            self.c[:, ids] = 0.0
            self.prev_a[ids] = 0.0
            _reset_wrist(self.aug, ids)
            if self.preal is not None:
                self.preal.reset(ids)
                self.areal.reset(ids)
            self.use_t[ids] = torch.rand(len(ids), device=self.dev) < self.beta()
        self.first = dones.clone()
        self.t += 1
        if self.t == self.T:
            self.t = 0
            self.end_chunk()
        return result

    def end_chunk(self):
        if not self.normed:
            m = self.bV[self.slot]
            x = self.bS[self.slot][m]
            mean, std = x.mean(0), x.std(0).clamp(min=1e-2)
            if self.world > 1:
                self.dist.all_reduce(mean)
                self.dist.all_reduce(std)
                mean, std = mean / self.world, std / self.world
            k0 = self.keep_old_norm
            if k0:  # warm start with new input columns: the old columns keep the loaded normaliser
                mean[:k0], std[:k0] = self.net.mean[:k0], self.net.std[:k0]
            self.net.mean.copy_(mean)
            self.net.std.copy_(std)
            self.normed = True
        self.slot = (self.slot + 1) % self.C
        self.filled = min(self.filled + 1, self.C)
        t0 = time.time()
        frac = self.it / max(1, self.iters)
        lr = self.lr * (1.0 if frac < 0.7 else max(0.1, 1 - (frac - 0.7) / 0.3))
        for g in self.opt.param_groups:
            g["lr"] = lr
        with torch.enable_grad():
            stats = [self.train_step() for _ in range(self.K)]
        st = {k: sum(d[k] for d in stats) / len(stats) for k in stats[0]}
        self.it += 1
        if self.it % 10 == 0 or self.it == 1:
            stu, tea = self.ep["stu"], self.ep["tea"]
            el = time.time() - self.tstart
            print(f"WX_IT r={self.rank} it={self.it} beta={self.beta():.2f} lr={lr:.1e} loss={st['loss']:.4f} arm_mse={st['arm']:.4f} "
                  f"grip_acc={st['grip']:.3f} grip_sw={st['grip_sw']:.3f} aux_cm={st['aux_cm']:.2f} stu_succ={sum(stu) / max(1, len(stu)):.3f}(n={len(stu)}) "
                  f"tea_succ={sum(tea) / max(1, len(tea)):.3f}(n={len(tea)}) steps={self.steps} sps={self.steps * self.n / el:.0f} "
                  f"train_s={time.time() - t0:.2f} elapsed_min={el / 60:.1f}", flush=True)
        if self.si and self.it % 10 == 0:
            wd = self.wd.summary() if self.wd is not None else {}
            print(f"WX_SI r={self.rank} it={self.it} relabeled_eps={self.si_stats['eps']} relabeled_steps={self.si_stats['steps']} "
                  f"wd={json.dumps(wd)}", flush=True)
        if self.it % 10 == 0 and (self.ep["stu0"] or self.ep["stu1"]):
            a0, a1 = self.ep["stu0"], self.ep["stu1"]
            print(f"WX_IT0 r={self.rank} it={self.it} stu_first={sum(a0) / max(1, len(a0)):.3f}(n={len(a0)}) "
                  f"stu_later={sum(a1) / max(1, len(a1)):.3f}(n={len(a1)})", flush=True)
        if (self.it % 25 == 0 or self.it >= self.iters) and self.rank == 0 and os.environ.get("WX_NOSAVE") != "1":
            self.save(self.out)
            if self.it % 250 == 0:
                self.save(self.out.replace(".pt", f"_it{self.it}.pt"))
        if self.it >= self.iters:
            print(f"WX_DONE {self.out}", flush=True)
            os._exit(0)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "cfg": self.net.cfg, "it": self.it,
                    "env": {k: v for k, v in os.environ.items() if k.startswith(("WX_", "BITIDY_"))}}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _gather(self, buf, ci, ei):
        """buf[ci, ei] on the training device (the image ring buffer may live in host RAM, WX_BUF_CPU=1)."""
        if buf.device.type == "cpu":
            return buf[ci.cpu(), ei.cpu()].pin_memory().to(self.dev, non_blocking=True)
        return buf[ci, ei]

    def train_step(self) -> dict:
        net, dev, B = self.net, self.dev, self.B
        ci = torch.randint(self.filled, (B,), device=dev)
        # never the chunk being written (slot points at the next one to overwrite = oldest; all filled ones are complete)
        ei = torch.randint(self.n, (B,), device=dev)
        s = self.bS[ci, ei].transpose(0, 1)
        a = self.bA[ci, ei].transpose(0, 1)
        x = self.bX[ci, ei].transpose(0, 1)
        f = self.bF[ci, ei].transpose(0, 1)
        v = self.bV[ci, ei].transpose(0, 1).float()
        h0 = self.bH[ci, :, ei].transpose(0, 1).contiguous()
        c0 = self.bC[ci, :, ei].transpose(0, 1).contiguous()
        img = None
        if self.use_img:
            img = _shift(self._gather(self.bI, ci, ei))  # (B, T, 2, 3, H, W) -> (T, B, 2, 3, H, W), DrQ shift +-4 px
        timg = None
        if self.use_top:  # +-2 px on the 160 x 90 top frame (~ the residual calibration error)
            timg = _shift(self._gather(self.bT, ci, ei).unsqueeze(2), int(_f("WX_TOPIMG_SHIFT", 2)))[:, :, 0]
        net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, auxp = net.seq(s, img, f, (h0, c0), timg)
        pred, auxp = pred.float(), auxp.float()
        arm = ((pred[..., list(ARM_DIMS)] - a[..., list(ARM_DIMS)]) ** 2).mean(-1)
        lab = (a[..., list(GRIP)] > 0).float()  # (T, B, 2)
        gl = F.binary_cross_entropy_with_logits(pred[..., list(GRIP)] * 4.0, lab, reduction="none")
        # steps within +-GRIP_TK of a teacher open/close switch (rare, decisive): up-weighted by GRIP_TW, monitored
        ch = torch.zeros_like(lab)
        ch[1:] = ((lab[1:] != lab[:-1]) & ~f[1:].unsqueeze(-1)).float()
        dil = F.max_pool1d(ch.permute(1, 2, 0).reshape(-1, 1, ch.shape[0]), 2 * GRIP_TK + 1, stride=1, padding=GRIP_TK)
        dil = dil.reshape(ch.shape[1], ch.shape[2], ch.shape[0]).permute(2, 0, 1)
        if GRIP_TW != 1.0:
            gl = gl * (1.0 + (GRIP_TW - 1.0) * dil)
        gl = gl.mean(-1)
        ax = ((auxp - x * 10.0) ** 2).mean(-1)
        w = _f("WX_AUX_W", 0.5)
        den = v.sum().clamp(min=1)
        loss = ((arm + GRIP_W * gl + w * ax) * v).sum() / den
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.world > 1:
            ps = [p for p in net.parameters() if p.requires_grad]
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in ps])
            self.dist.all_reduce(flat)
            flat /= self.world
            off = 0
            for p in ps:
                k = p.numel()
                p.grad = flat[off: off + k].view_as(p).clone()
                off += k
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.opt.step()
        net.eval()
        with torch.no_grad():
            hit = ((pred[..., list(GRIP)] > 0) == (a[..., list(GRIP)] > 0)).float()
            grip = ((hit.mean(-1) * v).sum() / den)
            wsw = dil * v.unsqueeze(-1)
            grip_sw = (hit * wsw).sum() / wsw.sum().clamp(min=1)
            aux_cm = (((auxp / 10.0 - x).abs().mean(-1) * 100) * v).sum() / den
        return {"loss": float(loss), "arm": float((arm * v).sum() / den), "grip": float(grip), "grip_sw": float(grip_sw),
                "aux_cm": float(aux_cm)}


def install_dagger() -> list[str]:
    register()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    inner_step = RslRlVecEnvWrapper.step
    state: dict = {}

    def step(self, actions):
        # the play loop runs under torch.inference_mode: the trainer (parameters, buffers, autograd) must not
        with torch.inference_mode(False):
            if "d" not in state:
                torch.manual_seed(int(_f("WX_SEED", 0)) + 1000 * int(_f("WX_RANK", 0)))
                if os.environ.get("WX_EST_TRAIN") == "1":  # learned estimator of the teacher's privileged inputs
                    from .wx_est import EstDAgger

                    state["d"] = EstDAgger(self.unwrapped)
                else:
                    state["d"] = DAgger(self.unwrapped)
            return state["d"].step(self, inner_step)

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def _episode_counter_skip(skip: int) -> list[str]:
    """As evaluation.install_episode_counter, but each env's episode number ``skip`` (0-based) is the one counted, so
    the first ``skip`` episodes run as warm-up (``WX_EVAL_SKIP``)."""
    import json as _json

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original_step = RslRlVecEnvWrapper.step
    st = {}

    def step(self, actions):
        result = original_step(self, actions)
        env = self.unwrapped
        dones = result[2].bool()
        FLAGS = env._bi_flag_names
        if not st:
            st["k"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            st["final"] = torch.zeros((env.num_envs, len(FLAGS) + 1), dtype=torch.bool, device=env.device)
            st["steps"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        st["steps"] += (st["k"] == skip).long()
        hit = dones & (st["k"] == skip)
        st["final"][hit] = env._bi_final[hit]
        st["k"] += dones.long()
        if bool((st["k"] > skip).all()):
            rates = st["final"].float().mean(0).tolist()
            ok = st["final"][:, -1]
            res = {"episodes": env.num_envs, "skip": skip, "success_rate": rates[-1], "successes": int(ok.sum()),
                   **{f"{nm}_rate": r for nm, r in zip(FLAGS, rates[:-1], strict=True)},
                   "mean_success_time_s": float(st["steps"][ok].float().mean() * env.step_dt) if bool(ok.any()) else None}
            print("BITIDY_EVAL_RESULT=" + _json.dumps(res), flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def install_eval() -> list[str]:
    register()
    from .scripted import install_scripted

    skip = int(_f("WX_EVAL_SKIP", 0))
    if skip:
        from . import evaluation

        evaluation.install_episode_counter = lambda: _episode_counter_skip(skip)
    return install_scripted()


# ---------------------------------------------------------------- video


def install_video() -> list[str]:
    """Per env (first ``WX_VID_ENVS``): third-person view (``view_cam``, zoom via BITIDY_EYE / BITIDY_LOOK) | wrist L /
    wrist R (as the student sees them, x3). ``WX_VIDEO_DIR``/env<e>.mp4 after ``BITIDY_RENDER_STEPS`` steps; a PNG of
    the last frame of every finished episode (``ep<k>_env<e>_succ<0|1>.png``) and a JSON line per episode end."""
    register()
    import numpy as np

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import render
    from .scripted import _script_class

    inner_step = RslRlVecEnvWrapper.step
    out = os.path.expanduser(os.environ.get("WX_VIDEO_DIR", "~/wx/video"))
    os.makedirs(out, exist_ok=True)
    steps = int(os.environ.get("BITIDY_RENDER_STEPS", "1800"))
    nv = int(os.environ.get("WX_VID_ENVS", "4"))
    st: dict = {"frames": None, "k": 0, "ep": None}

    def up(img, s):  # (3, h, w) uint8 tensor -> (h*s, w*s, 3)
        return img.permute(1, 2, 0).repeat_interleave(s, 0).repeat_interleave(s, 1)

    def step(self, actions):
        env = self.unwrapped
        cam = env.scene["view_cam"]
        if st["frames"] is None:
            st["script"] = _script_class(os.environ["BITIDY_SCRIPT"])(env)
            n = env.num_envs
            o = env.scene.env_origins
            cam.set_world_poses_from_view(o + o.new_tensor(render.EYE).expand(n, -1), o + o.new_tensor(render.TARGET).expand(n, -1))
            st["frames"] = [[] for _ in range(nv)]
            st["ep"] = [0] * nv
        script = st["script"]
        a = script.act()
        wimg = getattr(script, "last_img", None)
        if wimg is None and "wrist_left" in env.scene.sensors:
            wimg = wrist_images(env).permute(0, 1, 4, 2, 3)
        result = inner_step(self, a)
        rgb = cam.data.output["rgb"]
        rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3]
        if rgb.is_floating_point():
            rgb = (rgb.clamp(0, 1) * 255)
        rgb = rgb.to(torch.uint8)
        dones = result[2].bool()
        for e in range(min(nv, env.num_envs)):
            fr = rgb[e]
            if wimg is not None:
                s = max(1, fr.shape[0] // (2 * wimg.shape[-2]))
                side = torch.cat((up(wimg[e, 0], s), up(wimg[e, 1], s)), 0)
                pad = torch.zeros((fr.shape[0] - side.shape[0], side.shape[1], 3), dtype=torch.uint8, device=fr.device)
                fr = torch.cat((fr, torch.cat((side, pad), 0)), 1)
            fr = fr.cpu().numpy()
            if bool(dones[e]) and st["frames"][e]:
                ok = int(env._bi_final[e, -1])
                last = st["frames"][e][-1]
                _png(os.path.join(out, f"ep{st['ep'][e]}_env{e}_succ{ok}.png"), last)
                print("WX_VID_EP " + json.dumps({"env": e, "ep": st["ep"][e], "step": st["k"], "success": ok,
                                                 "flags": env._bi_final[e].int().tolist(),
                                                 "cmd": env._cb["cmd"][e].tolist()}), flush=True)
                st["ep"][e] += 1
            st["frames"][e].append(fr)
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            script.reset(ids)
        st["k"] += 1
        if st["k"] % int(os.environ.get("WX_SNAP_EVERY", "300")) == 0:
            for e in range(min(nv, env.num_envs)):  # periodic snapshots too
                _png(os.path.join(out, f"t{st['k']:04d}_env{e}.png"), st["frames"][e][-1])
        if st["k"] >= steps:
            for e in range(min(nv, env.num_envs)):
                render._write(os.path.join(out, f"env{e}.mp4"), st["frames"][e], fps=round(1.0 / env.step_dt))
            print(f"WX_VIDEO_DONE {out}", flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def _png(path, arr) -> None:
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except Exception as error:  # noqa: BLE001
        import numpy as np

        np.save(path.replace(".png", ".npy"), arr)
        print(f"[wx] png failed {error}", flush=True)


def install_inputs() -> list[str]:
    """Composite 'what the robot sees' video (WX_REAL): per env (first WX_VID_ENVS) a 1280 x 720 mp4:
    top-left global view (view_cam), top-right the top C920 frame as perception receives it (640 x 360 after the
    augmentation chain incl. JPEG; segmenter mask outlines; x = the delayed coarse estimate the policy gets, o = sim truth),
    bottom: left / right wrist images exactly as the student receives them (128 x 72, latency, distortion, aug) x5
    nearest. Every WX_VID_SKIP-th control step, BITIDY_RENDER_STEPS steps."""
    register()
    import numpy as np
    from PIL import Image, ImageDraw

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import cup_basket_env_cfg as cb
    from . import render
    from . import wx_real as R
    from . import wx_top
    from .scripted import _script_class

    inner_step = RslRlVecEnvWrapper.step
    out = os.path.expanduser(os.environ.get("WX_VIDEO_DIR", "~/wx/inputs"))
    os.makedirs(out, exist_ok=True)
    steps = int(os.environ.get("BITIDY_RENDER_STEPS", "1350"))
    nv = int(os.environ.get("WX_VID_ENVS", "4"))
    skip = int(os.environ.get("WX_VID_SKIP", "2"))
    pal = torch.tensor([[0, 0, 0], [255, 140, 0], [255, 40, 40], [40, 220, 40], [60, 90, 255], [255, 150, 150], [150, 255, 150],
                        [150, 170, 255], [255, 255, 255], [200, 140, 70], [120, 120, 150]], dtype=torch.uint8)
    objc = [(255, 60, 60), (60, 230, 60), (80, 120, 255)] * 2
    st: dict = {"k": 0, "w": None}

    def project(top, p):  # (m, 3) env-frame points -> pixel (u, v) on the 640 x 360 image (nominal calibration)
        eye, f, r, up = (t.float().to(p.device) for t in wx_top._basis(wx_top.EYE, wx_top.LOOK))
        d = p - eye
        z = (d @ f).clamp(min=1e-3)
        fx = wx_top.W * wx_top.FOCAL / wx_top.APERTURE
        return wx_top.W / 2 + fx * (d @ r) / z, wx_top.H / 2 - fx * (d @ up) / z

    def step(self, actions):
        env = self.unwrapped
        n = env.num_envs
        if st["w"] is None:
            import imageio.v2 as imageio

            st["script"] = _script_class(os.environ["BITIDY_SCRIPT"])(env)
            o = env.scene.env_origins
            env.scene["view_cam"].set_world_poses_from_view(o + o.new_tensor(render.EYE).expand(n, -1),
                                                            o + o.new_tensor(render.TARGET).expand(n, -1))
            fps = round(1.0 / (env.step_dt * skip))
            st["w"] = [imageio.get_writer(os.path.join(out, f"env{e}.mp4"), fps=fps, codec="libx264", quality=8) for e in range(min(nv, n))]
            st["ep"], st["res"] = [0] * nv, ["-"] * nv
        script = st["script"]
        a = script.act()
        wimg = getattr(script, "last_img", None)
        result = inner_step(self, a)
        dones = result[2].bool()
        for e in range(min(nv, n)):
            if bool(dones[e]):
                ok = int(env._bi_final[e, -1])
                st["res"][e] = f"ep{st['ep'][e]} {'SUCCESS' if ok else 'FAIL'}"
                print("WX_VID_EP " + json.dumps({"env": e, "ep": st["ep"][e], "step": st["k"], "success": ok}), flush=True)
                st["ep"][e] += 1
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            script.reset(ids)
        st["k"] += 1
        if st["k"] % skip == 0:
            g = env.scene["view_cam"].data.output["rgb"]
            g = (g.torch if hasattr(g, "torch") else g)[..., :3]
            g = (g.clamp(0, 1) * 255 if g.is_floating_point() else g).to(torch.uint8)[:, 60:420]  # 640 x 360
            c = env._wx_coarse
            top = c.top
            tf = (top.last_full.clamp(0, 1) * 255).to(torch.uint8)  # (n, 360, 640, 3)
            cls = top.last_cls.reshape(n, top.gh, top.gw) if getattr(top, "last_cls", None) is not None else None
            if cls is not None:  # mask outlines (class boundaries) on the full-resolution frame
                cu = cls.repeat_interleave(2, 1).repeat_interleave(2, 2)
                edge = (cu != torch.roll(cu, 1, 1)) | (cu != torch.roll(cu, 1, 2))
                col = pal.to(cu.device)[torch.maximum(cu, torch.roll(cu, 1, 1))]
                tf = torch.where(edge.unsqueeze(-1) & (torch.maximum(cu, torch.roll(cu, 1, 1)) > 1).unsqueeze(-1), col, tf)
            est = c.hist_pos[c.lat, torch.arange(n, device=env.device)] if hasattr(c, "hist_pos") else c.pos
            gt = cb.objs_pos(env)
            for e in range(min(nv, n)):
                W_ = (wimg[e] if wimg is not None else torch.zeros(2, 3, R.WRIST_OUT_H, R.WRIST_OUT_W, dtype=torch.uint8, device=env.device))
                z_ = max(1, 640 // W_.shape[-1])  # nearest upscale to 640 wide (x5 for 128, x2 for 320)
                wl = W_[0].permute(1, 2, 0).repeat_interleave(z_, 0).repeat_interleave(z_, 1)
                wr = W_[1].permute(1, 2, 0).repeat_interleave(z_, 0).repeat_interleave(z_, 1)
                fr = torch.cat((torch.cat((g[e], tf[e]), 1), torch.cat((wl, wr), 1)), 0).cpu().numpy()
                im = Image.fromarray(fr)
                dr = ImageDraw.Draw(im)
                ue, ve = top.project_actual(e, est[e] + est.new_tensor([0, 0, 0.03]))
                ug, vg = top.project_actual(e, gt[e] + gt.new_tensor([0, 0, 0.03]))
                for i in range(cb.N_OBJ):
                    x0, y0 = 640 + float(ue[i]), float(ve[i])
                    dr.line((x0 - 6, y0 - 6, x0 + 6, y0 + 6), fill=objc[i], width=2)
                    dr.line((x0 - 6, y0 + 6, x0 + 6, y0 - 6), fill=objc[i], width=2)
                    x1, y1 = 640 + float(ug[i]), float(vg[i])
                    dr.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), outline=(255, 255, 255))
                cmd = env._cb["cmd"][e].tolist()
                cap = (f"step {st['k']}  t={st['k'] * env.step_dt:5.1f}s  cmd={cmd}  last: {st['res'][e]}   "
                       f"pose {R.TOP_POSE}  expo {float(top.vis_aug.expo[e]) if top.vis_aug is not None else 1:.2f}")
                dr.rectangle((0, 0, 1280, 18), fill=(0, 0, 0))
                dr.text((4, 3), cap, fill=(255, 255, 0))
                for x_, y_, t_ in ((4, 22, "global (render cam)"), (644, 22, f"side C920 pose {R.TOP_POSE} (x est / o truth)" + (f", policy image {TOPIMG_W}x{TOPIMG_H}" if os.environ.get("WX_TOPIMG") == "1" else ", NOT a policy input")),
                                   (4, 364, f"left wrist {R.WRIST_OUT_W}x{R.WRIST_OUT_H} as policy input"), (644, 364, f"right wrist {R.WRIST_OUT_W}x{R.WRIST_OUT_H}")):
                    dr.text((x_, y_), t_, fill=(255, 255, 255))
                st["w"][e].append_data(np.asarray(im))
        if st["k"] >= steps:
            for w in st["w"]:
                w.close()
            print(f"WX_VIDEO_DONE {out}", flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
