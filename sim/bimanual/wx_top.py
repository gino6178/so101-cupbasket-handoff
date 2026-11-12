"""Overhead RGB camera + classical perception for the coarse vessel / basket poses (``WX_TOP=1``).

The camera is the VLM planner's overhead camera (``cb_scene.CAM``): eye (-0.30, 0, 0.85), look (0.17, 0, 0) in the env
frame, 800 x 600, 18 mm on a 20.955 mm aperture (60 x 47 deg FOV). ``WX_TOP_PERTURB=dx,dy,dz,dyaw,dpitch`` (m, deg)
moves the *rendered* camera while the perception keeps the nominal calibration (camera misplacement study).

Pipeline per refresh (every ``HOLD`` control steps, i.e. 10 Hz), all on the GPU for every env at once, on a stride-2
pixel grid:
  1. colour classes per pixel (vessel red / green / blue, basket light grey / wood / dark grey) by chromaticity rules;
  2. each colour holds one cup and one bowl: 2-means on the pixels' table-plane positions, seeded from the previous
     estimates (first frame of an episode: split along the principal axis; the larger cluster is the bowl);
  3. vessel xy = the cluster's mean viewing ray intersected with a per-type calibrated height plane (``Z_CUP``,
     ``Z_BOWL`` above the table); z = the resting surface (table, or basket floor when inside an estimated basket);
  4. visibility = the cluster's ground-projected area / (pi r_top^2); below ``VIS_MIN`` (arm occlusion, merged or lost
     cluster) the previous value is kept (freeze);
  5. upright / lying from the cluster's elongation (eigenvalue ratio > ``TIP_RATIO``), lying direction = major axis;
  6. basket pose: per basket colour, principal axis -> yaw (in (-90, 90] deg), centre = midpoint of the 3 / 97 %
     quantiles along both axes, averaged over the episode's valid frames (baskets are static).
``WX_TOP_STATS=1`` accumulates the localisation error against the sim ground truth (per type, fresh / frozen, near an
arm or not) and the upright-flag confusion and prints ``WX_TOP_STATS {json}`` every 300 refreshes.
``WX_TOP_DUMP=<dir>`` saves the first envs' raw images + ground truth at a few steps (threshold tuning).
"""

from __future__ import annotations

import json
import math
import os

import torch

from . import cup_basket_env_cfg as cb
from .env_cfg import pad_center


def _vec(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(v) for v in os.environ.get(name, default).split(","))


def _f(name: str, default) -> float:
    return float(os.environ.get(name, default))


from . import wx_real as R  # noqa: E402

EYE = _vec("WX_TOP_EYE", "-0.30,0.0,0.85")
LOOK = _vec("WX_TOP_LOOK", "0.17,0.0,0.0")
W, H = int(_f("WX_TOP_W", 800)), int(_f("WX_TOP_H", 600))
FOCAL, APERTURE = _f("WX_TOP_FOCAL", 18.0), 20.955
PERTURB = _vec("WX_TOP_PERTURB", "0,0,0,0,0")
STRIDE = int(_f("WX_TOP_STRIDE", 2))
if R.REAL:  # C920 (pose R.TOP_POSE), processed at 640 x 360 (1280 x 720 / 2), segmenter grid 320 x 180
    EYE, LOOK = R.TOP_EYE, R.TOP_LOOK
    W, H, FOCAL, STRIDE = R.TOP_W, R.TOP_H, R.TOP_FOCAL_MM, 2
SEGNET = os.environ.get("WX_SEGNET")  # learned segmenter checkpoint (REAL perception)
SEG_COLLECT = os.environ.get("WX_SEG_COLLECT") == "1"
Z_TYPE = (_f("WX_TOP_Z_CUP", 0.040), _f("WX_TOP_Z_BOWL", 0.036))  # calibrated centroid plane above the vessel base (m)
if R.REAL:  # C920 oblique + segmenter: calibrated from WX_TOP_STATS zcal (pose A, 11/8): cup 3.5 cm, bowl 3.0 cm
    Z_TYPE = (_f("WX_TOP_Z_CUP", 0.035), _f("WX_TOP_Z_BOWL", 0.030))
Z_BASKET = _f("WX_TOP_Z_BASKET", 0.012)
NEST_R = _f("WX_TOP_NEST_R", 0.03)
# extrinsic / intrinsic calibration: the camera is placed within +-3 cm / +-3 deg / focal +-3 % (CALIB), then calibrated
# on site; perception uses the calibrated pose with this residual error (m, deg, focal fraction). WX_TOP_CALIBRATED=0:
# perception keeps the nominal pose (uncalibrated placement study).
CALIBRATED = os.environ.get("WX_TOP_CALIBRATED", "1") == "1"
CAL_RES = _vec("WX_TOP_CAL_RES", "0.005,0.5,0.005")
STACK = os.environ.get("WX_TOP_STACK", "1") == "1"  # stack-height correction of the intersection plane
ZCAL = [0.005 * k for k in range(15)]  # candidate centroid heights above the base (calibration stats)
VIS_MIN = _f("WX_TOP_VIS_MIN", 0.45)
TIP_RATIO = (_f("WX_TOP_TIP_CUP", 8.0), _f("WX_TOP_TIP_BOWL", 4.0))  # elongation (eigenvalue ratio) above = lying
if R.REAL and os.environ.get("WX_TOP_TIP", "0") != "1":  # mask elongation does not separate lying / upright in the
    TIP_RATIO = (float("inf"), float("inf"))  # oblique C920 view (11/8 stats): report everything upright
BK_EVERY = int(_f("WX_TOP_BK_EVERY", 10))  # basket pose every this many refreshes (+ the first valid frames)
BK_YAW_MAX = math.radians(_f("WX_TOP_BK_YAW_MAX_DEG", 20.0))  # baskets stand roughly along x (scene spec)
MERGE = _f("WX_TOP_MERGE", 0.045)
AUG = os.environ.get("WX_TOP_AUG", "1") == "1"
TABLE_RGB = _vec("WX_TOP_TABLE_RGB", "0.682,0.624,0.565")  # rendered desk colour = white-balance reference
ROI = (_f("WX_TOP_ROI_X0", -0.25), _f("WX_TOP_ROI_X1", 0.65), _f("WX_TOP_ROI_Y", 0.48))


def top_camera_cfg():
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import _camera_renderer_cfg

    kw = {}
    types = ["rgb"]
    if SEG_COLLECT:
        types.append("semantic_segmentation")
        kw["colorize_semantic_segmentation"] = False
    rcfg = _camera_renderer_cfg()
    if SEG_COLLECT:  # raw instance ids (+ idToLabels prim paths), not colours
        for name in ("newton_renderer", "default", "ovrtx"):
            r = getattr(rcfg, name, None)
            for flag in ("colorize_instance_segmentation", "colorize_semantic_segmentation"):
                if r is not None and hasattr(r, flag):
                    setattr(r, flag, False)
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCam",
        spawn=PinholeCameraCfg(focal_length=FOCAL, horizontal_aperture=APERTURE, clipping_range=(0.05, 5.0)),
        data_types=types,
        width=W,
        height=H,
        update_period=0.1 if R.REAL else 0.0,  # 10 Hz
        renderer_cfg=rcfg,
        background_color=(0.25, 0.25, 0.25),
        **kw,
    )


def _basis(eye, look):
    eye, look = torch.tensor(eye, dtype=torch.float64), torch.tensor(look, dtype=torch.float64)
    if R.REAL and R.TOP_OVERHEAD:  # straight down: forward -z, image right +y, image up -x
        return (eye, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
                torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64))
    f = torch.nn.functional.normalize(look - eye, dim=0)
    r = torch.nn.functional.normalize(torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)), dim=0)
    up = torch.linalg.cross(r, f)
    return eye, f, r, up


def actual_pose():
    """Rendered camera eye / look after the misplacement perturbation (yaw about z, pitch about the right axis)."""
    eye, f, r, up = _basis(EYE, LOOK)
    dx, dy, dz, dyaw, dpitch = PERTURB
    yaw, pitch = math.radians(dyaw), math.radians(dpitch)
    d = torch.tensor(LOOK, dtype=torch.float64) - eye
    c, s = math.cos(yaw), math.sin(yaw)
    d = torch.stack((c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]))
    # pitch: rotate d about the (yawed) right axis (Rodrigues)
    k = torch.nn.functional.normalize(torch.linalg.cross(d, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)), dim=0)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    d = d * cp + torch.linalg.cross(k, d) * sp + k * torch.dot(k, d) * (1 - cp)
    e = eye + torch.tensor((dx, dy, dz), dtype=torch.float64)
    return e.tolist(), (e + d).tolist()


class TopPerception:
    def __init__(self, env):
        self.env = env
        n, dev = env.num_envs, env.device
        self.n, self.dev = n, dev
        eye, f, r, up = _basis(EYE, LOOK)
        fx = W * FOCAL / APERTURE
        js = torch.arange(0, W, STRIDE, dtype=torch.float64) + 0.5
        iss = torch.arange(0, H, STRIDE, dtype=torch.float64) + 0.5
        self.gh, self.gw = len(iss), len(js)
        v, u = torch.meshgrid(iss, js, indexing="ij")
        d = f + ((u - W / 2) / fx)[..., None] * r - ((v - H / 2) / fx)[..., None] * up  # (gh, gw, 3)
        q = d[..., :2] / d[..., 2:3]  # ground xy per unit height below the eye
        self.eye = eye.to(dev).float()
        self.q = q.reshape(-1, 2).to(dev).float()  # (P, 2)
        self.un = ((u - W / 2) / fx).reshape(-1).to(dev).float()  # nominal normalised pixel coordinates
        self.vn = ((v - H / 2) / fx).reshape(-1).to(dev).float()
        self.fru = torch.stack((f, r, up)).to(dev).float()  # (3, 3) nominal forward / right / up
        # ground area per grid pixel on the table plane (finite differences of neighbouring grid points)
        z0 = cb.TABLE_TOP
        g = (eye[:2] + (z0 - eye[2]) * q)  # (gh, gw, 2)
        gx = torch.gradient(g, dim=1)[0]
        gy = torch.gradient(g, dim=0)[0]
        area = (gx[..., 0] * gy[..., 1] - gx[..., 1] * gy[..., 0]).abs()
        self.area0 = area.reshape(-1).to(dev).float()  # m^2 per grid pixel at the table plane
        self.ez = float(eye[2])
        P_ = self.q.shape[0]
        self.qe = self.q.unsqueeze(0).repeat(n, 1, 1)  # (n, P, 2) per-env perceived rays (calibrated pose)
        self.eye_e = self.eye.unsqueeze(0).repeat(n, 1)  # (n, 3)
        self.zoom_e = torch.ones(n, device=dev)
        self.res = torch.zeros(n, 7, device=dev)  # calibration residual: pos (3), rot (3), focal (1)
        X0 = self.ground(cb.TABLE_TOP + 0.03)
        self.roi = (X0[:, 0] > ROI[0]) & (X0[:, 0] < ROI[1]) & (X0[:, 1].abs() < ROI[2])
        Xt = self.ground(cb.TABLE_TOP)
        self.wb_roi = (Xt[:, 0] > -0.12) & (Xt[:, 0] < 0.42) & (Xt[:, 1].abs() < 0.37)  # on the desk (white balance)
        self.c0 = torch.zeros(n, cb.N_OBJ, 2, device=dev)  # cluster centres (plane z0 + 3 cm), per object
        self.seeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.bk_sum = torch.zeros(n, cb.N_BASKET, 4, device=dev)  # x, y, cos 2yaw, sin 2yaw sums
        self.bk_cnt = torch.zeros(n, cb.N_BASKET, device=dev)
        self.bk_max = torch.zeros(n, cb.N_BASKET, device=dev)
        self.bk_known = None  # (n, 3, 3) current basket estimate, for the vessels' resting height
        self.nmeas = 0
        self.gain = torch.ones(n, 1, 1, 3, device=dev)
        self.bright = torch.zeros(n, 1, 1, 1, device=dev)
        self.posed = False
        self.calls = 0
        self.seg = None
        if R.REAL and SEGNET:
            from .wx_seg import load

            self.seg = load(SEGNET, dev)
        self.vis_aug = R.TopAug(n, dev) if R.on("VIS") else None
        self.zoom = torch.ones(n, device=dev)  # focal error (image zoom about the centre)
        self.table_err = torch.zeros(n, device=dev)  # table height calibration error
        self.cam_pos_err = torch.zeros(n, 3, device=dev)
        self.cam_rot_err = torch.zeros(n, 3, device=dev)
        self.stats = {} if os.environ.get("WX_TOP_STATS") else None
        self.dump = os.environ.get("WX_TOP_DUMP")
        self.reset(None)
        print(f"WX_TOP eye={EYE} look={LOOK} {W}x{H} f={FOCAL} perturb={PERTURB} actual={actual_pose()} grid={self.gh}x{self.gw}",
              flush=True)

    def ground(self, z: float) -> torch.Tensor:
        return self.eye[:2] + (z - self.ez) * self.q

    def ground_e(self, z) -> torch.Tensor:  # (n, P, 2) per env, perceived calibration
        return self.eye_e[:, None, :2] + (z - self.eye_e[:, None, 2:3]) * self.qe

    def ray_xy(self, mq: torch.Tensor, z) -> torch.Tensor:
        """mq (n, ..., 2) mean ray (per unit height) -> xy where it meets height z (n, ...) (perceived calibration)."""
        sh = (-1,) + (1,) * (mq.dim() - 2)
        e = self.eye_e
        return e[:, :2].view(*sh, 2) + (z - e[:, 2].view(*sh)).unsqueeze(-1) * mq

    def _rot(self, e: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

        return matrix_from_quat(quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2]))

    def _recalib(self, ids) -> None:
        """Perceived per-env camera model = the actual placement (CALIB errors) + the calibration residual."""
        k = len(ids)
        if CALIBRATED:
            res = torch.cat((torch.empty(k, 3, device=self.dev).uniform_(-CAL_RES[0], CAL_RES[0]),
                             torch.empty(k, 3, device=self.dev).uniform_(-math.radians(CAL_RES[1]), math.radians(CAL_RES[1])),
                             torch.empty(k, 1, device=self.dev).uniform_(-CAL_RES[2], CAL_RES[2])), -1)
            pos, rot, zoom = self.cam_pos_err[ids] + res[:, :3], self.cam_rot_err[ids] + res[:, 3:6], self.zoom[ids] * (1 + res[:, 6])
        else:
            pos, rot, zoom = torch.zeros(k, 3, device=self.dev), torch.zeros(k, 3, device=self.dev), torch.ones(k, device=self.dev)
        Rm = self._rot(rot)  # (k, 3, 3)
        f, r, up = self.fru
        d = f + (self.un.unsqueeze(0) / zoom.view(-1, 1)).unsqueeze(-1) * r - (self.vn.unsqueeze(0) / zoom.view(-1, 1)).unsqueeze(-1) * up
        d = torch.einsum("kij,kpj->kpi", Rm, d)
        self.qe[ids] = d[..., :2] / d[..., 2:3]
        if not hasattr(self, "Rm_e"):
            self.Rm_e = torch.eye(3, device=self.dev).repeat(self.n, 1, 1)
        self.Rm_e[ids] = Rm
        self.eye_e[ids] = self.eye.unsqueeze(0) + pos
        self.zoom_e[ids] = zoom

    def project_actual(self, e: int, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Env-frame points (m, 3) -> pixel (u, v) in env e's rendered (and zoomed) 640 x 360 image (true camera)."""
        f, r, up = self.fru
        Rm = self._rot(self.cam_rot_err[e:e + 1])[0] if R.on("CALIB") else torch.eye(3, device=self.dev)
        eye = self.eye + (self.cam_pos_err[e] if R.on("CALIB") else 0.0)
        d = (pts - eye) @ Rm  # = Rm^T (p - eye): back into the nominal camera frame
        z = (d @ f).clamp(min=1e-3)
        fx = W * FOCAL / APERTURE * float(self.zoom[e])
        return W / 2 + fx * (d @ r) / z, H / 2 - fx * (d @ up) / z

    def reset(self, ids) -> None:
        ids = torch.arange(self.n, device=self.dev) if ids is None or isinstance(ids, slice) else torch.as_tensor(ids, device=self.dev).long()
        if len(ids) == 0:
            return
        self.seeded[ids] = False
        self.bk_sum[ids] = 0.0
        self.bk_cnt[ids] = 0.0
        self.bk_max[ids] = 0.0
        if AUG:
            self.gain[ids] = torch.empty(len(ids), 1, 1, 3, device=self.dev).uniform_(0.9, 1.1)
            self.bright[ids] = torch.empty(len(ids), 1, 1, 1, device=self.dev).uniform_(-0.04, 0.04)
        if self.vis_aug is not None:
            self.vis_aug.reset(ids)
        if R.on("CALIB"):
            k = len(ids)
            self.zoom[ids] = torch.empty(k, device=self.dev).uniform_(0.97, 1.03)
            self.table_err[ids] = torch.empty(k, device=self.dev).uniform_(-0.005, 0.005)
            self.cam_pos_err[ids] = torch.empty(k, 3, device=self.dev).uniform_(-0.03, 0.03)
            self.cam_rot_err[ids] = torch.empty(k, 3, device=self.dev).uniform_(-math.radians(3), math.radians(3))
            self.pose_dirty = getattr(self, "pose_dirty", torch.zeros(self.n, dtype=torch.bool, device=self.dev))
            self.pose_dirty[ids] = True
        if R.REAL:
            self._recalib(ids)

    def _place_real(self, cam) -> None:
        """Top camera pose per env: nominal pose composed with the per-episode calibration error."""
        from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix, quat_mul

        env = self.env
        dirty = getattr(self, "pose_dirty", None)
        if self.posed and (dirty is None or not bool(dirty.any())):
            return
        ids = torch.arange(self.n, device=self.dev) if not self.posed or dirty is None else dirty.nonzero().squeeze(-1)
        _, f, r, up = _basis(EYE, LOOK)
        rot = torch.stack((r, -up, f), 1).float().to(self.dev)  # ROS camera axes (columns): x right, y down, z forward
        q0 = quat_from_matrix(rot.unsqueeze(0)).expand(len(ids), -1)
        e = self.cam_rot_err[ids]
        q = quat_mul(quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2]), q0)
        pos = env.scene.env_origins[ids] + torch.tensor(EYE, device=self.dev) + self.cam_pos_err[ids]
        cam.set_world_poses(pos, q, env_ids=ids, convention="ros")
        self.posed = True
        if dirty is not None:
            dirty[:] = False

    def _image(self) -> torch.Tensor:
        env = self.env
        cam = env.scene.sensors["top_cam"]
        if R.REAL:
            self._place_real(cam)
            rgb = cam.data.output["rgb"]
            rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3]
            x = rgb.float() / 255.0 if not rgb.is_floating_point() else rgb.float()
            if R.on("CALIB"):  # focal error: zoom about the image centre
                th = torch.zeros(self.n, 2, 3, device=self.dev)
                th[:, 0, 0] = th[:, 1, 1] = 1.0 / self.zoom
                g = torch.nn.functional.affine_grid(th, (self.n, 3, H, W), align_corners=False)
                x = torch.nn.functional.grid_sample(x.permute(0, 3, 1, 2), g, padding_mode="border", align_corners=False).permute(0, 2, 3, 1)
            if self.vis_aug is not None:
                x = self.vis_aug(x)
            self.last_full = x
            return torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), STRIDE, STRIDE).permute(0, 2, 3, 1)
        if not self.posed:
            e, l_ = actual_pose()
            o = env.scene.env_origins
            cam.set_world_poses_from_view(o + o.new_tensor(e), o + o.new_tensor(l_))
            self.posed = True
        rgb = cam.data.output["rgb"]
        rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3]
        x = rgb.float() / 255.0 if not rgb.is_floating_point() else rgb.float()
        if AUG:  # per-episode colour cast / exposure + per-pixel sensor noise
            x = (x * self.gain + self.bright + 0.02 * torch.randn_like(x)).clamp(0, 1)
        self.last_full = x  # (n, H, W, 3) in [0, 1]: also a direct policy input (wx_student, WX_TOPIMG=1)
        # STRIDE x STRIDE average pooling (the full-resolution image denoised to the grid)
        x = torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), STRIDE, STRIDE).permute(0, 2, 3, 1)
        # white balance against the desk (the dominant colour inside the table ROI): per-channel gains
        flat = x.reshape(x.shape[0], -1, 3)
        med = torch.nanmedian(torch.where(self.wb_roi.reshape(1, -1, 1), flat, float("nan")), dim=1).values  # (n, 3)
        gain = (flat.new_tensor(TABLE_RGB) / med.clamp(min=0.05)).clamp(0.7, 1.4)
        return (x * gain.view(-1, 1, 1, 3)).clamp(0, 1)  # (n, gh, gw, 3)

    @staticmethod
    def classes(x: torch.Tensor) -> dict:
        """Colour classes (n, P) bool from rgb (n, P, 3) in [0, 1]."""
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        mx, mn = x.max(-1).values, x.min(-1).values
        s = r + g + b + 1e-6
        sat = (mx - mn) / (mx + 1e-6)
        # thresholds tuned on rendered frames against ground-truth footprints (see SCENE_SETUP.md for the colours)
        red = (r > 0.2) & (r / s > 0.48) & ((g - b).abs() < 0.15 * r)  # excludes the orange arm (g >> b)
        green = (g > 0.15) & (g / s > 0.42) & (g > 1.5 * r) & (g > 1.3 * b)
        blue = (b > 0.2) & (b / s > 0.45) & (b > 1.4 * g)
        A = (sat < 0.07) & (mx > 0.45) & (mx < 0.9)  # light grey
        C = (sat < 0.2) & (mx > 0.27) & (mx < 0.65) & ((b - r) > 0.025)  # dark (blue-tinted) grey; servos are neutral
        B = (sat > 0.21) & (sat < 0.45) & (r >= g) & (g >= b) & (mx > 0.25)  # wood; the desk is less saturated
        return {"red": red, "green": green, "blue": blue, "A": A, "B": B, "C": C}

    @torch.no_grad()
    def measure(self, valid: torch.Tensor, fit: bool = False, prior=None) -> dict:
        if self.seg is not None:
            return self.measure_seg(valid, fit=fit, prior=prior)
        env = self.env
        n, dev = self.n, self.dev
        x = self._image().reshape(n, -1, 3)
        cls = self.classes(x)
        roi = self.roi.unsqueeze(0)
        X0 = self.ground(cb.TABLE_TOP + 0.03)  # (P, 2) clustering plane
        g = cb._geo_t(dev)
        typ_r = (float(g["r_top"][0]), float(g["r_top"][1]))
        xy = torch.zeros(n, cb.N_OBJ, 2, device=dev)
        vis = torch.zeros(n, cb.N_OBJ, device=dev)
        ratio = torch.ones(n, cb.N_OBJ, device=dev)
        axis = torch.zeros(n, cb.N_OBJ, 2, device=dev)
        axis[..., 0] = 1.0
        new_seed = ~self.seeded & valid
        for ci, col in enumerate(("red", "green", "blue")):
            m = (cls[col] & roi).float()  # (n, P)
            ids = (ci, ci + 3)  # cup, bowl of this colour
            cnt = m.sum(-1)
            mean = (m @ X0) / cnt.clamp(min=1).unsqueeze(-1)
            # seed (episode start): split along the principal axis
            dx = X0.unsqueeze(0) - mean.unsqueeze(1)
            cov = torch.einsum("np,npi,npj->nij", m, dx, dx) / cnt.clamp(min=1).view(-1, 1, 1)
            ev, evec = torch.linalg.eigh(cov + 1e-9 * torch.eye(2, device=dev))
            pa = evec[..., 1]
            sd = ev[..., 1].clamp(min=0).sqrt().unsqueeze(-1)
            s0 = torch.stack((mean + pa * sd, mean - pa * sd), 1)
            c = torch.where(new_seed.view(-1, 1, 1), s0, self.c0[:, list(ids)])
            for _ in range(4):
                d0 = ((X0.unsqueeze(0) - c[:, 0:1]) ** 2).sum(-1)
                d1 = ((X0.unsqueeze(0) - c[:, 1:2]) ** 2).sum(-1)
                lab1 = d1 < d0
                w = torch.stack((m * (~lab1).float(), m * lab1.float()), 1)  # (n, 2, P)
                wc = w.sum(-1)
                cn = (w @ X0) / wc.clamp(min=1).unsqueeze(-1)
                c = torch.where((wc > 3).unsqueeze(-1), cn, c)
            # seeding: the cluster with the larger ground area is the bowl; a single blob (centres coincide) is a cup if
            # its area is below the mean of the cup and bowl rim discs
            A0 = w @ self.area0  # (n, 2) area at the table plane
            single = torch.linalg.vector_norm(c[:, 0] - c[:, 1], dim=-1) < MERGE
            seed_cup = single & (A0.sum(-1) < math.pi * (0.5 * (typ_r[0] + typ_r[1])) ** 2)
            swap = new_seed & ~single & (A0[:, 0] > A0[:, 1])
            w = torch.where(swap.view(-1, 1, 1), w.flip(1), w)
            c = torch.where(swap.view(-1, 1, 1), c.flip(1), c)
            wc = w.sum(-1)
            # merged / lost: the two centres coincide -> all pixels to the one whose previous centre is closer
            close = torch.linalg.vector_norm(c[:, 0] - c[:, 1], dim=-1) < MERGE
            if bool(close.any()):
                prev = torch.where(new_seed.view(-1, 1, 1), mean.unsqueeze(1).expand(-1, 2, -1), self.c0[:, list(ids)])
                mc = mean.unsqueeze(1)
                to1 = torch.linalg.vector_norm(prev[:, 1] - mean, dim=-1) < torch.linalg.vector_norm(prev[:, 0] - mean, dim=-1)
                to1 = torch.where(new_seed, ~seed_cup, to1)  # seeding: a single blob is a cup or a bowl by its area
                wm = torch.stack((m * (~to1).float().unsqueeze(-1), m * to1.float().unsqueeze(-1)), 1)
                w = torch.where(close.view(-1, 1, 1), wm, w)
                wc = w.sum(-1)
                recv = torch.stack((~to1, to1), 1).unsqueeze(-1)  # the receiving object takes the merged centre
                cm = torch.where(recv, mc.expand(-1, 2, -1), prev)
                c = torch.where(close.view(-1, 1, 1), cm, c)
            for k, i in enumerate(ids):
                t = 0 if k == 0 else 1
                wk = w[:, k]
                mq = (wk @ self.q) / wc[:, k].clamp(min=1).unsqueeze(-1)
                # base on the table, or on a basket floor when the estimate falls inside a known basket footprint
                z0 = cb.TABLE_TOP + Z_TYPE[t]
                zc = torch.full((n, 1), z0, device=dev)
                if self.bk_known is not None:
                    xy_t = self.eye[:2] + (z0 - self.ez) * mq
                    uv = cb.to_basket(self.bk_known, xy_t.unsqueeze(1))
                    ins = (uv.abs() < uv.new_tensor((cb.IN_L / 2, cb.IN_W / 2))).all(-1).any(-1)
                    zc = torch.where(ins, cb.FLOOR_Z + Z_TYPE[t], z0).unsqueeze(-1)
                xy[:, i] = self.eye[:2] + (zc - self.ez) * mq
                scale = ((z0 - self.ez) / (cb.TABLE_TOP - self.ez)) ** 2
                vis[:, i] = (wk @ self.area0) * scale / (math.pi * typ_r[t] ** 2)
                dxk = X0.unsqueeze(0) - c[:, k].unsqueeze(1)
                ck = torch.einsum("np,npi,npj->nij", wk, dxk, dxk) / wc[:, k].clamp(min=1).view(-1, 1, 1)
                evk, veck = torch.linalg.eigh(ck + 1e-9 * torch.eye(2, device=dev))
                ratio[:, i] = evk[:, 1] / evk[:, 0].clamp(min=1e-7)
                axis[:, i] = veck[..., 1]
            self.c0[:, ids[0]] = torch.where(valid.unsqueeze(-1), c[:, 0], self.c0[:, ids[0]])
            self.c0[:, ids[1]] = torch.where(valid.unsqueeze(-1), c[:, 1], self.c0[:, ids[1]])
        self.seeded |= valid
        # baskets: every BK_EVERY refreshes, or while some valid env has no basket estimate yet this episode
        self.nmeas += 1
        bk = torch.zeros(n, cb.N_BASKET, 3, device=dev)
        bcnt = torch.zeros(n, cb.N_BASKET, device=dev)
        if self.nmeas % BK_EVERY == 1 or bool(((self.bk_cnt == 0).any(-1) & valid).any()):
            Xb = self.ground(cb.TABLE_TOP + Z_BASKET)
            for j, col in enumerate(("A", "B", "C")):
                bk[:, j], bcnt[:, j] = self._basket_pose(cls[col] & roi, Xb)
        return {"xy": xy, "vis": vis, "ratio": ratio, "axis": axis, "bk": bk, "bcnt": bcnt, "rgb": x}

    def _basket_pose(self, m: torch.Tensor, Xb: torch.Tensor, kmax: int = 6000) -> tuple[torch.Tensor, torch.Tensor]:
        """Min-area rectangle (robust 2 / 98 % extents) over yaw in [-BK_YAW_MAX, BK_YAW_MAX] of the mask's table-plane
        points after dropping pixels > 16 cm from the median. Returns (n, 3) x, y, yaw and the pixel count."""
        n, dev = m.shape[0], m.device
        cnt = m.sum(-1).float()
        idx = torch.topk(m.float() + 0.5 * torch.rand(m.shape, device=dev), kmax, dim=-1).indices
        ok = m.gather(1, idx)
        pts = Xb[idx] if Xb.dim() == 2 else Xb.gather(1, idx.unsqueeze(-1).expand(-1, -1, 2))  # (n, K, 2)
        nanm = torch.where(ok, 0.0, float("nan")).unsqueeze(-1)
        med = torch.nan_to_num(torch.nanmedian(pts + nanm, dim=1).values)
        ok = ok & (torch.linalg.vector_norm(pts - med.unsqueeze(1), dim=-1) < 0.16)
        okf = ok.float()
        mean = (pts * okf.unsqueeze(-1)).sum(1) / okf.sum(1).clamp(min=1).unsqueeze(-1)
        d = pts - mean.unsqueeze(1)
        yaws = torch.linspace(-BK_YAW_MAX, BK_YAW_MAX, 41, device=dev)
        cy, sy = torch.cos(yaws), torch.sin(yaws)
        nn_ = torch.where(ok, 0.0, float("nan")).unsqueeze(-1)
        pu = d[..., 0:1] * cy + d[..., 1:2] * sy + nn_  # (n, K, Y)
        pv = -d[..., 0:1] * sy + d[..., 1:2] * cy + nn_
        qs = torch.tensor([0.02, 0.98], device=dev)
        qu = torch.nanquantile(pu, qs, dim=1)  # (2, n, Y)
        qv = torch.nanquantile(pv, qs, dim=1)
        area = torch.nan_to_num((qu[1] - qu[0]) * (qv[1] - qv[0]), nan=1e9)
        k = area.argmin(-1)
        ar = torch.arange(n, device=dev)
        yaw = yaws[k]
        mu, mv = torch.nan_to_num(qu[:, ar, k].mean(0)), torch.nan_to_num(qv[:, ar, k].mean(0))
        cx = mean[:, 0] + mu * torch.cos(yaw) - mv * torch.sin(yaw)
        cy_ = mean[:, 1] + mu * torch.sin(yaw) + mv * torch.cos(yaw)
        return torch.stack((cx, cy_, yaw), -1), torch.where(okf.sum(1) > 50, cnt, 0.0)

    @torch.no_grad()
    def measure_seg(self, valid: torch.Tensor, fit: bool = False, prior=None) -> dict:
        """Learned-segmenter perception: per-instance class masks -> table-plane centroid of each vessel (ray through
        the mask centroid intersected with the calibrated height above its resting surface), visibility, elongation,
        and basket min-area rectangles. ``fit`` (WX_TOP_FIT): also the silhouette fit of every vessel (wx_fuse), with
        ``prior`` = (xy (n, 6, 2), base z (n, 6)) of the fused estimate."""
        from .wx_seg import segment

        n, dev = self.n, self.dev
        x = self._image()  # (n, 180, 320, 3)
        lg = None
        if fit:
            cls, lg = segment(self.seg, x, logits_out=True)
            cls = cls.reshape(n, -1)
        else:
            cls = segment(self.seg, x).reshape(n, -1)
        self.last_cls = cls
        roi = self.roi.unsqueeze(0)
        g = cb._geo_t(dev)
        X0 = self.ground_e(cb.TABLE_TOP + 0.03)  # (n, P, 2)
        xy = torch.zeros(n, cb.N_OBJ, 2, device=dev)
        vis = torch.zeros(n, cb.N_OBJ, device=dev)
        ratio = torch.ones(n, cb.N_OBJ, device=dev)
        axis = torch.zeros(n, cb.N_OBJ, 2, device=dev)
        axis[..., 0] = 1.0
        tz = cb.TABLE_TOP  # perception's (nominal) table height; the true one differs by table_err in CALIB
        mqs = torch.zeros(n, cb.N_OBJ, 2, device=dev)
        zb = torch.full((n, cb.N_OBJ), tz, device=dev)  # resting surface (table or basket floor)
        ins_all = torch.zeros(n, cb.N_OBJ, dtype=torch.bool, device=dev)
        for i in range(cb.N_OBJ):
            t = cb.OBJ_TYPE[i]
            m = ((cls == 2 + i) & roi).float()
            wc = m.sum(-1)
            mq = torch.einsum("np,npk->nk", m, self.qe) / wc.clamp(min=1).unsqueeze(-1)
            mqs[:, i] = mq
            z0 = tz + Z_TYPE[t]
            if self.bk_known is not None:
                xy_t = self.ray_xy(mq, torch.full((n,), z0, device=dev))
                uv = cb.to_basket(self.bk_known, xy_t.unsqueeze(1))
                ins = (uv.abs() < uv.new_tensor((cb.IN_L / 2, cb.IN_W / 2))).all(-1).any(-1)
                ins_all[:, i] = ins
                zb[:, i] = torch.where(ins, tz + (cb.FLOOR_Z - cb.TABLE_TOP), zb[:, i])
            scale = ((z0 - self.ez) / (cb.TABLE_TOP - self.ez)) ** 2 / self.zoom_e ** 2
            vis[:, i] = (m @ self.area0) * scale / (math.pi * float(g["r_top"][t]) ** 2)
            c = torch.einsum("np,npk->nk", m, X0) / wc.clamp(min=1).unsqueeze(-1)
            dx = X0 - c.unsqueeze(1)
            cov = torch.einsum("np,npi,npj->nij", m, dx, dx) / wc.clamp(min=1).view(-1, 1, 1)
            ev, vec = torch.linalg.eigh(cov + 1e-9 * torch.eye(2, device=dev))
            ratio[:, i] = ev[:, 1] / ev[:, 0].clamp(min=1e-7)
            axis[:, i] = vec[..., 1]
        # stack height (known geometry): inside a basket, a vessel sits one nesting step above every same-type vessel
        # within NEST_R of it that is less visible (the lower one of a stack is mostly hidden by the upper one)
        typ = cb.obj_type(dev).view(-1)
        xy0 = self.ray_xy(mqs, zb + torch.tensor(Z_TYPE, device=dev)[typ].view(1, -1))
        dd = torch.linalg.vector_norm(xy0.unsqueeze(2) - xy0.unsqueeze(1), dim=-1)  # (n, 6, 6)
        same = (typ.view(-1, 1) == typ.view(1, -1)) & ~torch.eye(cb.N_OBJ, dtype=torch.bool, device=dev)
        below = same.unsqueeze(0) & (dd < NEST_R) & ins_all.unsqueeze(1) & ins_all.unsqueeze(2) & (vis.unsqueeze(1) < vis.unsqueeze(2))
        lvl = below.sum(-1).float() * (vis > 0.05).float()
        pitch = g["pitch"].to(dev).float()[typ].view(1, -1)
        zc = zb + torch.tensor(Z_TYPE, device=dev)[typ].view(1, -1) + (lvl * pitch if STACK else 0.0)
        xy = self.ray_xy(mqs, zc)
        self.lvl_est = lvl
        self.nmeas += 1
        bk = torch.zeros(n, cb.N_BASKET, 3, device=dev)
        bcnt = torch.zeros(n, cb.N_BASKET, device=dev)
        if self.nmeas % BK_EVERY == 1 or bool(((self.bk_cnt == 0).any(-1) & valid).any()):
            Xb = self.ground_e(cb.TABLE_TOP + Z_BASKET)
            for j in range(cb.N_BASKET):
                bk[:, j], bcnt[:, j] = self._basket_pose((cls == 8 + j) & roi, Xb)
        out = {"xy": xy, "vis": vis, "ratio": ratio, "axis": axis, "bk": bk, "bcnt": bcnt, "rgb": x, "mq": mqs, "lvl": lvl}
        if fit:
            out["fit"] = self._fit(lg, cls.reshape(n, self.gh, self.gw), xy, zb + (lvl * pitch if STACK else 0.0), prior)
        return out

    def _fit(self, lg, cls_map, xy_plane, zb_plane, prior) -> dict:
        """Silhouette fit of every vessel in the top image (perceived calibration); see wx_fuse.fit_image_objects."""
        from . import wx_fuse as FU

        n, dev = self.n, self.dev
        f, r, up = self.fru
        fx = W * FOCAL / APERTURE
        Rm = getattr(self, "Rm_e", torch.eye(3, device=dev).repeat(n, 1, 1))
        if prior is None:
            c0, zb0 = xy_plane, zb_plane
        else:
            c0, zb0 = prior
            # objects whose prior is far from the plane estimate (re-appeared / moved): start from the plane estimate
            far = torch.linalg.vector_norm(c0 - xy_plane, dim=-1) > 0.03
            c0 = torch.where(far.unsqueeze(-1), xy_plane, c0)

        def ray_fn(u, v):
            un = (u - W / 2) / fx / self.zoom_e.view(-1, 1, 1)
            vn = (v - H / 2) / fx / self.zoom_e.view(-1, 1, 1)
            d = f + un.unsqueeze(-1) * r - vn.unsqueeze(-1) * up  # (n, 6, P, 3)
            return self.eye_e, torch.einsum("nij,nkpj->nkpi", Rm, d)

        rt_, _, h_, _ = FU._geo(dev)
        ext = (rt_ + 0.5 * h_)[cb.obj_type(dev)]

        def half_px(box):
            ctr = torch.cat((c0, (zb0 + 0.03).unsqueeze(-1)), -1)
            dist = torch.linalg.vector_norm(ctr - self.eye_e.unsqueeze(1), dim=-1)
            return fx * ext / dist.clamp(min=0.2) * 1.3

        return FU.fit_image_objects(lg, cls_map, (H, W), ray_fn, zb0, c0, FU.NMIN_TOP, FU.S_TOP, None, half_px)

    def baskets(self, meas: dict, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode-averaged basket poses (static); frames with < 70 % of the episode's max pixel count are skipped."""
        cnt = meas["bcnt"]
        self.bk_max = torch.maximum(self.bk_max, torch.where(valid.unsqueeze(-1), cnt, 0.0))
        ok = valid.unsqueeze(-1) & (cnt > 0.7 * self.bk_max) & (cnt > 50)  # cnt is 0 on refreshes without baskets
        b = meas["bk"]
        add = torch.stack((b[..., 0], b[..., 1], torch.cos(2 * b[..., 2]), torch.sin(2 * b[..., 2])), -1)
        self.bk_sum += add * ok.unsqueeze(-1).float()
        self.bk_cnt += ok.float()
        mean = self.bk_sum / self.bk_cnt.clamp(min=1).unsqueeze(-1)
        yaw = 0.5 * torch.atan2(mean[..., 3], mean[..., 2])
        out = torch.stack((mean[..., 0], mean[..., 1], yaw), -1)
        self.bk_known = torch.where((self.bk_cnt > 0).unsqueeze(-1), out, torch.full_like(out, 99.0))  # (n, 3, 3)
        return out, self.bk_cnt > 0

    # ------------------------------------------------------------ evaluation against the sim ground truth

    def account(self, meas: dict, upd: torch.Tensor, est_xy: torch.Tensor, tipped_est: torch.Tensor, bk_est: torch.Tensor,
                valid: torch.Tensor) -> None:
        self.calls += 1
        if self.dump and self.calls in (5, 40, 120, 300):
            self._dump(meas)
        if self.stats is None:
            return
        env = self.env
        p = cb.objs_pos(env)
        q = cb.objs_quat(env)
        err = torch.linalg.vector_norm(est_xy - p[..., :2], dim=-1)  # (n, 6)
        near = torch.zeros_like(err, dtype=torch.bool)
        for arm in cb.ARMS:
            pad = pad_center(env, arm)
            near |= torch.linalg.vector_norm(pad[:, None, :2] - p[..., :2], dim=-1) < 0.06
        lifted = p[..., 2] > cb.TABLE_TOP + 0.012 + (cb._state(env)["level"] > 0).float() * 0.05
        typ = cb.obj_type(env.device).expand_as(err)
        tip_true = cb.tilt(q) > 0.6
        on = valid.unsqueeze(-1).expand_as(err) & (p[..., 2] > cb.TABLE_TOP - 0.03)
        st = self.stats
        uvt = cb.to_basket(cb.baskets(env).unsqueeze(1), p[..., :2].unsqueeze(2))
        in_bk = (uvt.abs() < uvt.new_tensor((cb.IN_L / 2, cb.IN_W / 2))).all(-1).any(-1)
        for tname, t in (("cup", 0), ("bowl", 1)):
            for sname, sm in (("fresh", upd), ("frozen", ~upd)):
                for oname, om in (("clear", ~near), ("arm_near", near & ~lifted), ("lifted", lifted)):
                    sel = on & (typ == t) & sm & om
                    if bool(sel.any()):
                        st.setdefault(f"{tname}/{sname}/{oname}", []).append(err[sel].float().cpu())
            sel = on & (typ == t)
            st.setdefault(f"{tname}/frozen_frac", []).append(torch.tensor([float((~upd)[sel].float().mean()) if bool(sel.any()) else 0.0]))
            for a, b_ in ((True, True), (True, False), (False, True), (False, False)):
                k = f"{tname}/tip_true{int(a)}_est{int(b_)}"
                st[k] = st.get(k, 0) + int((sel & (tip_true == a) & (tipped_est == b_)).sum())
            # occlusion (frozen = the camera estimate is not refreshed) next to an arm vs clear
            for oname, om in (("clear", ~near), ("arm_near", near & ~lifted)):
                sel2 = sel & om
                k = f"{tname}/occl_frac_{oname}"
                a0, b0 = st.get(k, (0, 0))
                st[k] = (a0 + int((~upd & sel2).sum()), b0 + int(sel2.sum()))
            # stacked (nested above the basket floor, not carried) and the elongation of upright vs lying vessels
            pitch = float(cb._geo_t(env.device)["pitch"][t])
            inbk = sel & in_bk & ~near & (p[..., 2] < cb.FLOOR_Z + 3.5 * pitch) & ~tip_true
            stk = inbk & (p[..., 2] > cb.FLOOR_Z + 0.5 * pitch)
            for sname, sm in (("fresh", upd), ("frozen", ~upd)):
                if bool((stk & sm).any()):
                    st.setdefault(f"{tname}/stacked/{sname}", []).append(err[stk & sm].float().cpu())
            k = f"{tname}/occl_frac_stacked"
            a0, b0 = st.get(k, (0, 0))
            st[k] = (a0 + int((~upd & stk).sum()), b0 + int(stk.sum()))
            if "lvl" in meas:
                lt = torch.round((p[..., 2] - cb.FLOOR_Z) / pitch).clamp(min=0)
                inb = inbk & upd
                k = f"{tname}/stack_level_acc"
                a0, b0 = st.get(k, (0, 0))
                st[k] = (a0 + int(((meas["lvl"] == lt) & inb).sum()), b0 + int(inb.sum()))
            rs = sel & upd & ~near & ~lifted
            if "ratio" in meas and bool(rs.any()):
                st.setdefault(f"{tname}/ratio_up", []).append(meas["ratio"][rs & ~tip_true].float().cpu())
                st.setdefault(f"{tname}/ratio_tip", []).append(meas["ratio"][rs & tip_true].float().cpu())
            if "mq" in meas and bool(rs.any()):  # centroid-plane calibration: error vs the assumed height above the base
                mq = meas["mq"][rs]
                pz = p[..., 2][rs]
                pxy = p[..., :2][rs]
                ee = self.eye_e.unsqueeze(1).expand(-1, cb.N_OBJ, -1)[rs]
                es = [float(torch.linalg.vector_norm(ee[:, :2] + (pz + h - ee[:, 2]).unsqueeze(-1) * mq - pxy, dim=-1).sum())
                      for h in ZCAL]
                a0, b0 = st.get(f"{tname}/zcal", ([0.0] * len(ZCAL), 0))
                st[f"{tname}/zcal"] = ([u + v for u, v in zip(a0, es)], b0 + int(rs.sum()))
        bt = cb.baskets(env)
        be = torch.linalg.vector_norm(bk_est[..., :2] - bt[..., :2], dim=-1)
        dy = torch.remainder(bk_est[..., 2] - bt[..., 2] + math.pi / 2, math.pi) - math.pi / 2
        vb = valid.unsqueeze(-1).expand_as(be)
        st.setdefault("basket/xy", []).append(be[vb].float().cpu())
        st.setdefault("basket/yaw_deg", []).append(dy[vb].abs().float().cpu() * 180 / math.pi)
        if self.calls % 300 == 0:
            self.print_stats()

    def print_stats(self) -> None:
        out = {}
        for k, v in self.stats.items():
            if isinstance(v, int):
                out[k] = v
                continue
            if isinstance(v, tuple):
                if k.endswith("zcal"):
                    e = [round(100 * u / max(v[1], 1), 2) for u in v[0]]
                    out[k] = {"n": v[1], "err_cm": e, "best_h_cm": round(100 * ZCAL[e.index(min(e))], 1)}
                else:
                    out[k] = {"n": v[1], "frac": round(v[0] / max(v[1], 1), 3)}
                continue
            x = torch.cat(v)
            if k.endswith("frozen_frac"):
                out[k] = round(float(x.mean()), 3)
                continue
            if len(x) == 0:
                continue
            u = 1.0 if k.endswith(("yaw_deg", "ratio_up", "ratio_tip")) else 100.0
            out[k] = {"n": len(x), "mean": round(float(x.mean()) * u, 2), "p50": round(float(x.quantile(0.5)) * u, 2) if len(x) < 16_000_000 else None,
                      "p90": round(float(x.quantile(0.9)) * u, 2) if len(x) < 16_000_000 else None, "max": round(float(x.max()) * u, 2)}
            if len(x) > 4_000_000:
                self.stats[k] = [x[torch.randperm(len(x))[:2_000_000]]]
        print("WX_TOP_STATS " + json.dumps(out), flush=True)

    def _dump(self, meas: dict) -> None:
        import numpy as np

        env = self.env
        os.makedirs(self.dump, exist_ok=True)
        k = min(4, self.n)
        cam = env.scene.sensors["top_cam"]
        rgb = cam.data.output["rgb"]
        rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[:k, ..., :3]
        np.savez_compressed(os.path.join(self.dump, f"top_{self.calls:04d}.npz"), rgb=rgb.cpu().numpy(),
                            objs=torch.cat((cb.objs_pos(env), cb.objs_quat(env)), -1)[:k].cpu().numpy(),
                            baskets=cb.baskets(env)[:k].cpu().numpy(), est_xy=meas["xy"][:k].cpu().numpy(),
                            vis=meas["vis"][:k].cpu().numpy(), ratio=meas["ratio"][:k].cpu().numpy(), bk=meas["bk"][:k].cpu().numpy())
