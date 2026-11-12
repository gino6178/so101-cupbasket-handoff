"""Render a trained bimanual expert to video.

The ``*-Render`` tasks add one third-person camera per environment. Play them with
``--external_callback isaaclab_tutorial.tasks.bimanual.render.install_recorder`` and ``presets=newton_mjwarp,newton_renderer``:
the recorder tiles the first four environments into a 2 x 2 frame, writes ``$BITIDY_VIDEO`` (default
``~/bitidy_render.mp4``) after ``$BITIDY_RENDER_STEPS`` control steps, and exits.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import _camera_renderer_cfg

EYE = tuple(float(v) for v in os.environ.get("BITIDY_EYE", "0.62,0.0,0.42").split(","))
TARGET = tuple(float(v) for v in os.environ.get("BITIDY_LOOK", "0.15,0.0,0.06").split(","))


def view_camera() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/ViewCam",
        spawn=PinholeCameraCfg(focal_length=16.0, horizontal_aperture=20.955),
        data_types=["rgb"],
        width=640,
        height=480,
        update_period=0.0,
        renderer_cfg=_camera_renderer_cfg(),
        background_color=(0.82, 0.84, 0.88),
    )


def with_camera(env_cfg_cls):
    """Subclass a task config with the view camera and a small, video-sized scene."""
    scene_cls = type(env_cfg_cls().scene)

    @configclass
    class RenderSceneCfg(scene_cls):
        view_cam = view_camera()

    @configclass
    class RenderEnvCfg(env_cfg_cls):
        scene: RenderSceneCfg = RenderSceneCfg(num_envs=4, env_spacing=1.2, replicate_physics=True)

    RenderEnvCfg.__name__ = env_cfg_cls.__name__ + "Render"
    return RenderEnvCfg


def install_recorder() -> list[str]:
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    steps = int(os.environ.get("BITIDY_RENDER_STEPS", "450"))
    path = os.path.expanduser(os.environ.get("BITIDY_VIDEO", "~/bitidy_render.mp4"))
    original_step = RslRlVecEnvWrapper.step
    frames: list[np.ndarray] = []

    def step(self, actions):
        env = self.unwrapped
        camera = env.scene["view_cam"]
        if not frames:
            n = env.num_envs
            origins = env.scene.env_origins
            eyes = origins + origins.new_tensor(EYE).expand(n, -1)
            targets = origins + origins.new_tensor(TARGET).expand(n, -1)
            camera.set_world_poses_from_view(eyes, targets)
        result = original_step(self, actions)
        rgb = camera.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        tiles = rgb[:4, ..., :3]
        if tiles.is_floating_point():
            tiles = (tiles.clamp(0, 1) * 255) if tiles.max() <= 1.0 else tiles.clamp(0, 255)
        tiles = tiles.to(torch.uint8).cpu().numpy()
        while len(tiles) < 4:
            tiles = np.concatenate((tiles, np.zeros_like(tiles[:1])))
        frames.append(np.concatenate((np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)), 0))
        if len(frames) >= steps:
            _write(path, frames, fps=round(1.0 / env.step_dt))
            print(f"BITIDY_VIDEO={path} frames={len(frames)}", flush=True)
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


def _write(path: str, frames: list[np.ndarray], fps: int) -> None:
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
    except Exception as error:  # no imageio/ffmpeg on the VM: keep raw frames for local encoding
        print(f"[render] mp4 failed ({error}); saving frames", flush=True)
        np.save(os.path.splitext(path)[0] + ".npy", np.stack(frames))


def _task_cfgs():
    from . import bar_env_cfg, bar_hard_env_cfg, env_cfg, lid3_env_cfg, lid_env_cfg, recv_env_cfg, stack_env_cfg, sort_env_cfg, stack_left_env_cfg, stack_recv_env_cfg

    return {
        "Handover": env_cfg.BiHandoverEnvCfg,
        "HandoverRecv": recv_env_cfg.BiHandoverRecvEnvCfg,
        "BarLift": bar_env_cfg.BiBarLiftEnvCfg,
        "BarLiftHard": bar_hard_env_cfg.BiBarLiftHardEnvCfg,
        "Stack": stack_env_cfg.BiStackEnvCfg,
        "StackLeft": stack_left_env_cfg.BiStackLeftEnvCfg,
        "StackRecv": stack_recv_env_cfg.BiStackRecvEnvCfg,
        "Lid": lid_env_cfg.BiLidEnvCfg,
        "SortLeft": sort_env_cfg.SortLeftEnvCfg,
        "SortRight": sort_env_cfg.SortRightEnvCfg,
        "SortHandover": sort_env_cfg.SortHandoverEnvCfg,
        "Lid3": lid3_env_cfg.BiLid3EnvCfg,
    }


HandoverRenderEnvCfg = with_camera(_task_cfgs()["Handover"])
HandoverRecvRenderEnvCfg = with_camera(_task_cfgs()["HandoverRecv"])
BarLiftRenderEnvCfg = with_camera(_task_cfgs()["BarLift"])
BarLiftHardRenderEnvCfg = with_camera(_task_cfgs()["BarLiftHard"])
StackRenderEnvCfg = with_camera(_task_cfgs()["Stack"])
StackLeftRenderEnvCfg = with_camera(_task_cfgs()["StackLeft"])
StackRecvRenderEnvCfg = with_camera(_task_cfgs()["StackRecv"])
LidRenderEnvCfg = with_camera(_task_cfgs()["Lid"])
SortLeftRenderEnvCfg = with_camera(_task_cfgs()["SortLeft"])
SortRightRenderEnvCfg = with_camera(_task_cfgs()["SortRight"])
SortHandoverRenderEnvCfg = with_camera(_task_cfgs()["SortHandover"])
Lid3RenderEnvCfg = with_camera(_task_cfgs()["Lid3"])


# Sub-task envs kept in their own modules, made into render variants on first use (module __getattr__).
_OPTIONAL = {"LidOpen": ("lid_open_env_cfg", "LidOpenEnvCfg"), "BoxPlace": ("box_place_env_cfg", "BoxPlaceEnvCfg"),
             "LidPlace": ("lid_place_env_cfg", "LidPlaceEnvCfg"), "LidSort": ("lid_sort_env_cfg", "LidSortEnvCfg"),
             "CupBasket": ("cup_basket_env_cfg", "CupBasketEnvCfg")}


def __getattr__(name: str):
    key = name[: -len("RenderEnvCfg")] if name.endswith("RenderEnvCfg") else None
    if key in _OPTIONAL:
        import importlib

        module, cls = _OPTIONAL[key]
        cfg = with_camera(getattr(importlib.import_module(f"{__package__}.{module}"), cls))
        globals()[name] = cfg
        return cfg
    raise AttributeError(name)
