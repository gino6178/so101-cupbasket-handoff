"""Learned top-camera segmenter for the real deployment pipeline (replaces the colour thresholds of ``wx_top``).

Classes (per pixel): 0 table / background / distractor, 1 robot, 2-4 red / green / blue cup, 5-7 red / green / blue
bowl, 8-10 basket A (light grey) / B (wood) / C (dark grey). Per-instance classes (each vessel has a unique colour +
type) remove the cup / bowl clustering and the identity tracking of the threshold pipeline.

Network: a small U-Net (~0.6 M parameters) on the top image downscaled to 320 x 180 (the C920's 1280 x 720 / 4).

Data: ``install_segdata`` (callback) renders the overhead camera with ``instance_segmentation`` (raw ids + the
``idToLabels`` prim paths) while the privileged teacher works (plus random arm poses), and writes shards
``WX_SEG_DIR/seg_<rank>_<k>.npz`` (rgb 360 x 640 uint8, label uint8). ``python -m ...wx_seg train <dir> <out.pt>``
trains with the deployment image augmentation (``wx_real.TopAug``) applied on the fly.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLS = 11
IN_H, IN_W = 180, 320


def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.ELU(), nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.ELU())


class SegNet(nn.Module):
    def __init__(self, ch=(24, 48, 96, 128)):
        super().__init__()
        self.e0, self.e1, self.e2, self.e3 = _blk(3, ch[0]), _blk(ch[0], ch[1]), _blk(ch[1], ch[2]), _blk(ch[2], ch[3])
        self.d2, self.d1, self.d0 = _blk(ch[3] + ch[2], ch[2]), _blk(ch[2] + ch[1], ch[1]), _blk(ch[1] + ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], N_CLS, 1)

    def forward(self, x):  # x (n, 3, 180, 320) in [0, 1]
        x = x - 0.5
        e0 = self.e0(x)
        e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d2 = self.d2(torch.cat((up(e3, e2), e2), 1))
        d1 = self.d1(torch.cat((up(d2, e1), e1), 1))
        d0 = self.d0(torch.cat((up(d1, e0), e0), 1))
        return self.out(d0)


def label_lut(id_to_labels: dict) -> dict:
    """Instance id -> class from the prim paths (``.../Obj<k>/...``, ``.../Basket<j>/...``, ``.../robot_*``)."""
    out = {}
    for k, v in id_to_labels.items():
        if isinstance(k, tuple):  # (id,) style keys
            k = k[0] if len(k) == 1 else k
        if isinstance(k, str) and k.strip("()").replace(",", "").strip().isdigit():
            k = int(k.strip("()").split(",")[0])
        if not isinstance(k, int):
            continue
        path = v if isinstance(v, str) else str(v)
        m = re.search(r"obj(\d)", path, re.I)
        if m:
            i = int(m.group(1))
            out[int(k)] = 2 + i  # objects 0-2 cups -> 2-4, 3-5 bowls -> 5-7 (same order)
            continue
        m = re.search(r"basket(\d)", path, re.I)
        if m:
            out[int(k)] = 8 + int(m.group(1))
            continue
        if "robot" in path.lower():
            out[int(k)] = 1
    return out


@torch.no_grad()
def segment(net: SegNet, rgb: torch.Tensor, logits_out: bool = False):
    """rgb (n, H, W, 3) float [0, 1] at any resolution -> class map (n, h, w) long at the net's input size (180 x 320
    for the top segmenter). ``logits_out``: also the logits (n, N_CLS, h, w) bf16 (sub-pixel lookups)."""
    hw = getattr(net, "in_hw", (IN_H, IN_W))
    x = rgb.permute(0, 3, 1, 2)
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, size=hw, mode="area")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = net(x)
    cls = logits.float().argmax(1) if not logits_out else logits.argmax(1)
    return (cls, logits) if logits_out else cls


def load(path: str, dev) -> SegNet:
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = SegNet().to(dev)
    net.load_state_dict(ck["state_dict"])
    net.in_hw = tuple(ck.get("in_hw", (IN_H, IN_W)))
    return net.eval()


# ---------------------------------------------------------------- data collection (callback)


def install_segdata() -> list[str]:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import wx_student

    wx_student.register()
    inner_step = RslRlVecEnvWrapper.step
    out = os.path.expanduser(os.environ.get("WX_SEG_DIR", "~/wx/segdata"))
    os.makedirs(out, exist_ok=True)
    every, shards, per = int(os.environ.get("WX_SEG_EVERY", "12")), int(os.environ.get("WX_SEG_SHARDS", "8")), int(os.environ.get("WX_SEG_PER", "6"))
    tag = os.environ.get("WX_SEG_TAG", "0")
    st: dict = {"k": 0, "buf": [], "shard": 0}

    def step(self, actions):
        env = self.unwrapped
        if "script" not in st:
            from .scripted import _script_class

            st["script"] = _script_class(os.environ.get("BITIDY_SCRIPT", "residual:ResidualScript"))(env)
        a = st["script"].act()
        # some arms wander (random targets) for poses the teacher never shows
        if os.environ.get("WX_SEG_RANDOM", "1") == "1":
            wander = (torch.arange(env.num_envs, device=env.device) % 4 == 0).unsqueeze(-1)
            a = torch.where(wander, a + 0.6 * torch.randn_like(a), a)
        result = inner_step(self, a)
        dones = result[2].bool()
        ids = dones.nonzero().squeeze(-1)
        if len(ids):
            st["script"].reset(ids)
        st["k"] += 1
        if st["k"] % every == 0 and st["k"] > 3:
            cam = env.scene.sensors["top_cam"]
            if "lut" not in st:
                info = cam.data.info
                seginfo = info["semantic_segmentation"] if isinstance(info, dict) else info[0]["semantic_segmentation"]
                id2 = seginfo.get("idToLabels", seginfo.get("idToSemantics", seginfo)) if isinstance(seginfo, dict) else seginfo
                lut = label_lut(id2)
                print(f"WX_SEG raw keys sample={list(id2.items())[:4]}", flush=True)
                mx = max(list(lut.keys()) + [0]) + 1
                t = torch.zeros(mx + 1, dtype=torch.uint8, device=env.device)
                for k, v in lut.items():
                    t[k] = v
                st["lut"] = t
                print(f"WX_SEG ids={len(id2)} mapped={len(lut)} classes={sorted(set(lut.values()))} sample={list(id2.items())[:3]}", flush=True)
            rgb = cam.data.output["rgb"]
            rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3]
            seg = cam.data.output["semantic_segmentation"]
            seg = (seg.torch if hasattr(seg, "torch") else seg)[..., 0].long().clamp(0, len(st["lut"]) - 1)
            lab = st["lut"][seg]
            st["buf"].append((rgb.to(torch.uint8).cpu().numpy(), lab.cpu().numpy()))
            if len(st["buf"]) >= per:
                rgbs = np.concatenate([b[0] for b in st["buf"]])
                labs = np.concatenate([b[1] for b in st["buf"]])
                np.savez(os.path.join(out, f"seg_{tag}_{st['shard']:03d}.npz"), rgb=rgbs, label=labs)
                print(f"WX_SEG shard {st['shard']} {rgbs.shape} class_frac={np.bincount(labs.reshape(-1)[::97], minlength=N_CLS) / (labs.size / 97)}", flush=True)
                st["buf"] = []
                st["shard"] += 1
                if st["shard"] >= shards:
                    os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def install_wsegdata() -> list[str]:
    """Wrist-segmenter data (``WX_WSEG_COLLECT=1``, set by ``wx_run.sh wsegdata``): both wrist cameras' 320 x 240 renders
    remapped through a random OV9732 lens (diagonal FOV 100 +- 4 deg, barrel k1, principal point; labels nearest) while
    the privileged teacher works (a quarter of the envs with random action noise). Shards
    ``WX_SEG_DIR/wseg_<tag>_<k>.npz`` (rgb (m, 240, 320, 3) uint8, label (m, 240, 320) uint8); frames whose wrist view
    contains no vessel are subsampled (``WX_WSEG_EMPTY_P``)."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import wx_real as R
    from . import wx_student

    wx_student.register()
    inner_step = RslRlVecEnvWrapper.step
    out = os.path.expanduser(os.environ.get("WX_SEG_DIR", "~/wx/wsegdata"))
    os.makedirs(out, exist_ok=True)
    every, shards = int(os.environ.get("WX_SEG_EVERY", "15")), int(os.environ.get("WX_SEG_SHARDS", "24"))
    tag = os.environ.get("WX_SEG_TAG", "0")
    empty_p = float(os.environ.get("WX_WSEG_EMPTY_P", "0.25"))
    st: dict = {"k": 0, "shard": 0}

    def step(self, actions):
        env = self.unwrapped
        n, dev = env.num_envs, env.device
        if "script" not in st:
            from .scripted import _script_class

            st["script"] = _script_class(os.environ.get("BITIDY_SCRIPT", "residual:ResidualScript"))(env)
        a = st["script"].act()
        if os.environ.get("WX_SEG_RANDOM", "1") == "1":
            wander = (torch.arange(n, device=dev) % 4 == 0).unsqueeze(-1)
            a = torch.where(wander, a + 0.6 * torch.randn_like(a), a)
        result = inner_step(self, a)
        ids = result[2].bool().nonzero().squeeze(-1)
        if len(ids):
            st["script"].reset(ids)
        st["k"] += 1
        if st["k"] % every == 0 and st["k"] > 3:
            rgbs, labs = [], []
            for name in ("wrist_left", "wrist_right"):
                cam = env.scene.sensors[name]
                if name not in st:
                    info = cam.data.info
                    seginfo = info["semantic_segmentation"] if isinstance(info, dict) else info[0]["semantic_segmentation"]
                    id2 = seginfo.get("idToLabels", seginfo.get("idToSemantics", seginfo)) if isinstance(seginfo, dict) else seginfo
                    lut = label_lut(id2)
                    t = torch.zeros(max(list(lut.keys()) + [0]) + 2, dtype=torch.uint8, device=dev)
                    for k, v in lut.items():
                        t[k] = v
                    st[name] = t
                    print(f"WX_WSEG {name} ids={len(id2)} mapped={len(lut)} classes={sorted(set(lut.values()))}", flush=True)
                rgb = cam.data.output["rgb"]
                rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3]
                seg = cam.data.output["semantic_segmentation"]
                seg = (seg.torch if hasattr(seg, "torch") else seg)[..., 0].long().clamp(0, len(st[name]) - 1)
                rgbs.append(rgb.to(torch.uint8))
                labs.append(st[name][seg])
            rgb = torch.stack(rgbs, 1).reshape(2 * n, *rgbs[0].shape[1:])  # (2n, 240, 320, 3)
            lab = torch.stack(labs, 1).reshape(2 * n, *labs[0].shape[1:])
            dfov = R._u((2 * n,), R.WRIST_DFOV - R.WRIST_DFOV_J, R.WRIST_DFOV + R.WRIST_DFOV_J, dev)
            k1 = R._u((2 * n,), -0.08, 0.0, dev)
            pp = R._u((2 * n, 2), -2.0, 2.0, dev)
            g = R.wrist_grid(2 * n, dev, dfov, k1, pp, out_wh=(R.WRIST_SRC_W, R.WRIST_SRC_H))
            x = rgb.permute(0, 3, 1, 2).float()
            if R.WRIST_POOL > 1:
                x = F.avg_pool2d(x, R.WRIST_POOL, R.WRIST_POOL)
            x = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
            y = F.grid_sample(lab[:, None].float(), g, mode="nearest", padding_mode="border", align_corners=False)[:, 0]
            has = ((y >= 2) & (y <= 7)).flatten(1).any(-1)
            keep = has | (torch.rand(2 * n, device=dev) < empty_p)
            x = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)[keep].cpu().numpy()
            y = y.to(torch.uint8)[keep].cpu().numpy()
            np.savez(os.path.join(out, f"wseg_{tag}_{st['shard']:03d}.npz"), rgb=x, label=y)
            print(f"WX_WSEG shard {st['shard']} {x.shape} with_vessel={float(has.float().mean()):.2f} "
                  f"class_frac={np.round(np.bincount(y.reshape(-1)[::97], minlength=N_CLS) / (y.size / 97), 4).tolist()}", flush=True)
            st["shard"] += 1
            if st["shard"] >= shards:
                os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


# ---------------------------------------------------------------- training


def train(data_dir: str, out: str, epochs: int = 12, batch: int = 48, lr: float = 2e-3) -> None:
    """Top segmenter (default) or, with ``WX_SEG_IN=120x160 WX_SEG_GLOB=wseg_*.npz WX_SEG_MBLUR=1``, the wrist segmenter."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wx_real import TopAug  # noqa: E402  (standalone: no Isaac Lab import)

    IN_H, IN_W = (int(v) for v in os.environ.get("WX_SEG_IN", f"{globals()['IN_H']}x{globals()['IN_W']}").split("x"))
    dev = "cuda"
    files = sorted(glob.glob(os.path.join(data_dir, os.environ.get("WX_SEG_GLOB", "seg_*.npz"))))
    rgbs, labs = [], []
    for f in files:
        d = np.load(f)
        rgbs.append(torch.from_numpy(d["rgb"]))
        labs.append(torch.from_numpy(d["label"]))
    rgb = torch.cat(rgbs)  # (N, 360, 640, 3) uint8 (CPU)
    lab = torch.cat(labs)
    n = len(rgb)
    # labels at the network resolution (nearest)
    lab_s = F.interpolate(lab[:, None].float(), size=(IN_H, IN_W), mode="nearest")[:, 0].to(torch.uint8)
    perm = torch.randperm(n)
    val, tr = perm[: max(64, n // 20)], perm[max(64, n // 20):]
    freq = torch.bincount(lab_s[tr[:2000]].reshape(-1).long(), minlength=N_CLS).float()
    w = (freq.sum() / (freq + 1.0)).pow(0.5)
    w = (w / w.mean()).to(dev)
    print(f"SEG data n={n} files={len(files)} class_freq={(freq / freq.sum()).tolist()}", flush=True)
    net = SegNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    aug = TopAug(batch, dev, motion_blur=os.environ.get("WX_SEG_MBLUR") == "1")

    def prep(idx, augment):
        x = rgb[idx].to(dev, non_blocking=True).float() / 255.0
        if augment:
            aug.reset(None)
            x = aug(x[: batch] if len(x) == batch else x, n=len(x))
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(IN_H, IN_W), mode="area")
        return x, lab_s[idx].to(dev).long()

    it = 0
    for ep in range(epochs):
        net.train()
        p = tr[torch.randperm(len(tr))]
        tot = 0.0
        for b in range(0, len(p) - batch + 1, batch):
            x, y = prep(p[b: b + batch], True)
            # random flips keep the geometry consistent with the label (both flipped)
            if torch.rand(()) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(x).float(), y, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            it += 1
        net.eval()
        inter = torch.zeros(N_CLS, device=dev)
        union = torch.zeros(N_CLS, device=dev)
        with torch.no_grad():
            for b in range(0, len(val), batch):
                idx = val[b: b + batch]
                x, y = prep(idx, True)
                pr = net(x).argmax(1)
                for c in range(N_CLS):
                    inter[c] += ((pr == c) & (y == c)).sum()
                    union[c] += ((pr == c) | (y == c)).sum()
        iou = (inter / union.clamp(min=1)).tolist()
        print(f"SEG epoch {ep + 1}/{epochs} loss {tot / max(1, len(p) // batch):.4f} val_IoU(aug) {[round(v, 3) for v in iou]} mIoU {sum(iou) / N_CLS:.3f}", flush=True)
        torch.save({"state_dict": net.state_dict(), "iou": iou, "in_hw": (IN_H, IN_W)}, out)
    print(f"SEG_SAVED {out}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3], epochs=int(sys.argv[4]) if len(sys.argv) > 4 else 12)
