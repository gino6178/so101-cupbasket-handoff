# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["lerobot[feetech]==0.6.1", "torch==2.10.0", "opencv-python>=4.10", "numpy", "faster-whisper>=1.1", "sounddevice"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Voice-commanded zone sorting on two SO-101 followers with the simulation-trained state policies.

Subcommands (run with ``uv run --script real/sort_cell.py <cmd> ...``):

  calib   Click the four reference marks in the overhead image -> ``table_homography.json`` (image px -> table m).
  detect  Show the cube detection (position in the table frame, folded yaw) live, nothing moves.
  run     Listen for "A"/"B"/"C" (or type it), pick the arm(s) and run the policy; dry run unless ``--execute``.

The table frame is the simulation's: x forward, y left, origin midway between the arm bases; the left base is at
(-0.05, +0.15), the right at (-0.05, -0.15), zones A/B/C (7 cm squares) centred at (0.20, +0.13), (0.13, 0), (0.20, -0.13).

The control loop matches simulation: policy at 30 Hz; every 1/120 s each arm joint target is
``measured + scale * action`` (gripper: open/close by the sign of its action), clipped to the joint map's limits.
Safety follows the tutorial's deploy.py: calibrated, verified joint maps only; start from the home pose; torque
off on exit or error unless ``--hold-at-end``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ZONES = {"A": (0.20, 0.13), "B": (0.13, 0.0), "C": (0.20, -0.13)}
ZONE_HALF = 0.035
TABLE_TOP = 0.032
CUBE_REST_Z = TABLE_TOP + 0.0125
PREGRASP_GRIPPER = math.radians(14.64)
GRIPPER_CLOSED = math.radians(-12.0)
ARM_FOR = {("A", "B"): "left", ("B", "A"): "left", ("B", "C"): "right", ("C", "B"): "right"}


# ---------------------------------------------------------------- camera: cube detection in the table frame


class CubeCamera:
    """Background capture + colour segmentation; reports cube (x, y) in the table frame and yaw mod 90 degrees."""

    def __init__(self, device, homography: np.ndarray | None, hsv_lo, hsv_hi):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(device) if str(device).isdigit() else device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.H = homography
        self.lo, self.hi = np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
        self.frame, self.detection, self.error = None, None, None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cv2 = self.cv2
        while True:
            ok, frame = self.cap.read()
            if not ok:
                self.error = "camera read failed"
                time.sleep(0.05)
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.morphologyEx(cv2.inRange(hsv, self.lo, self.hi), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            det = None
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 80:
                    (u, v), _, angle = cv2.minAreaRect(c)
                    det = {"uv": (u, v), "angle_deg": angle, "t": time.monotonic()}
                    if self.H is not None:
                        x, y = self.to_table(u, v)
                        # Image angle -> table yaw via the homography's local rotation.
                        x2, y2 = self.to_table(u + 10 * math.cos(math.radians(angle)), v + 10 * math.sin(math.radians(angle)))
                        det["xy"] = (x, y)
                        det["yaw"] = math.atan2(y2 - y, x2 - x)
            with self.lock:
                self.frame, self.detection = frame, det

    def to_table(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def latest(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.detection


def cube_feature(det) -> np.ndarray:
    x, y = det["xy"]
    yaw = det["yaw"]
    return np.array([x, y, CUBE_REST_Z, math.sin(4 * yaw), math.cos(4 * yaw)], dtype=np.float32)


def zone_of(xy) -> str | None:
    for name, (zx, zy) in ZONES.items():
        if abs(xy[0] - zx) < ZONE_HALF + 0.01 and abs(xy[1] - zy) < ZONE_HALF + 0.01:
            return name
    return None


# ---------------------------------------------------------------- voice


def listen_zone(model, seconds: float = 3.0) -> str | None:
    """Record a short utterance and return the first zone letter it names (A/B/C, also 'A區', 'a 區')."""
    import sounddevice as sd

    print(f"[voice] 請說目標區（例如「放到 C」），錄音 {seconds:.0f} 秒…", flush=True)
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    segments, _ = model.transcribe(audio[:, 0], language="zh", initial_prompt="把方塊放到 A 區、B 區或 C 區")
    text = "".join(s.text for s in segments)
    print(f"[voice] 聽到：{text}", flush=True)
    m = re.search(r"[ABCabc]|[欸诶誒][區区]|[比必逼][區区]|[西希][區区]", text)
    if not m:
        return None
    token = m.group(0).upper()[0]
    return {"欸": "A", "诶": "A", "誒": "A", "比": "B", "必": "B", "逼": "B", "西": "C", "希": "C"}.get(token, token)


# ---------------------------------------------------------------- arms and policy


class JointMap:
    """sim_rad = scale * lerobot_value + offset (the tutorial's joint map format)."""

    def __init__(self, data: dict):
        if data.get("units") != "lerobot_degrees_and_gripper_percent":
            raise ValueError("Joint map must declare LeRobot degrees and gripper percent")
        entries = [data["joints"][name] for name in JOINTS]
        self.scale = np.array([e["scale"] for e in entries])
        self.offset = np.array([e["offset"] for e in entries])
        self.limits = np.array([e["limits"] for e in entries])
        self.verified = data.get("verified") is True

    def positions(self, native: dict) -> np.ndarray:
        return np.array([native[n] for n in JOINTS]) * self.scale + self.offset

    def commands(self, target: np.ndarray) -> dict:
        native = (np.clip(target, self.limits[:, 0], self.limits[:, 1]) - self.offset) / self.scale
        return {f"{n}.pos": float(v) for n, v in zip(JOINTS, native, strict=True)}


class Arm:
    def __init__(self, name: str, port: str, robot_id: str, joint_map: Path, policy: Path, scales, execute: bool):
        import torch
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.name, self.execute = name, execute
        self.map = JointMap(json.loads(joint_map.read_text()))
        if execute and not self.map.verified:
            raise SystemExit(f"{name}: --execute needs a verified joint map")
        self.policy = torch.jit.load(str(policy)).eval()
        self.scale = np.asarray(scales, dtype=np.float64)
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True))
        self.robot.bus.connect()
        if not self.robot.is_calibrated:
            raise SystemExit(f"{name}: robot calibration must already match its motors")
        self.action = np.zeros(6)
        self.configured = False

    def measured(self) -> np.ndarray:
        return self.map.positions(self.robot.bus.sync_read("Present_Position"))

    def enable(self):
        if self.execute and not self.configured:
            native = self.robot.bus.sync_read("Present_Position")
            self.robot.bus.disable_torque()
            self.robot.bus.sync_write("Goal_Position", native)
            self.robot.configure()
            self.configured = True

    def infer(self, measured, target, cube, goal_xy):
        import torch

        obs = np.concatenate((measured, target, self.action, cube, goal_xy, cube[:2] - goal_xy)).astype(np.float32)
        with torch.inference_mode():
            self.action = self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy().astype(np.float64)

    def target(self, measured) -> np.ndarray:
        t = measured.copy()
        t[:5] = measured[:5] + self.scale * np.clip(self.action[:5], -1e3, 1e3)
        t[5] = GRIPPER_CLOSED if self.action[5] < 0 else PREGRASP_GRIPPER
        return np.clip(t, self.map.limits[:, 0], self.map.limits[:, 1])

    def send(self, target):
        if self.execute:
            self.robot.send_action(self.map.commands(target))

    def close(self, hold: bool):
        try:
            if hold and self.configured:
                pos = self.robot.bus.sync_read("Present_Position", normalize=False)
                self.robot.bus.sync_write("Goal_Position", pos, normalize=False)
        finally:
            self.robot.bus.disconnect(disable_torque=not (hold and self.configured))


def run_skill(arm: Arm, camera: CubeCamera, goal: str, timeout: float) -> bool:
    """One policy episode: carry the cube into ``goal``. Ends when the cube rests in the goal zone, unoccluded."""
    goal_xy = np.array(ZONES[goal], dtype=np.float64)
    _, det = camera.latest()
    if det is None or "xy" not in det:
        print("[run] cube not visible", flush=True)
        return False
    cube = cube_feature(det)
    arm.enable()
    arm.action[:] = 0
    measured = arm.measured()
    target = measured.copy()
    start = time.monotonic()
    next_policy = start
    lifted_once = False
    while time.monotonic() - start < timeout:
        tick = time.monotonic()
        measured = arm.measured()
        if tick >= next_policy:
            _, det = camera.latest()
            # Refresh the cube only when it is seen resting and clear of the jaws (as in training).
            if det is not None and "xy" in det and tick - det["t"] < 0.2:
                fresh = cube_feature(det)
                gripper_open = measured[5] > 0.1
                if np.linalg.norm(fresh[:2] - cube[:2]) < 0.01 or gripper_open:
                    cube = fresh
            arm.infer(measured, target, cube, goal_xy)
            next_policy += 1 / 30
            if arm.action[5] < 0:
                lifted_once = True
            if lifted_once and det is not None and "xy" in det and zone_of(det["xy"]) == goal and measured[5] > 0.1:
                print(f"[run] cube in {goal}", flush=True)
                return True
        target = arm.target(measured)
        arm.send(target)
        time.sleep(max(0.0, 1 / 120 - (time.monotonic() - tick)))
    print("[run] timeout", flush=True)
    return False


# ---------------------------------------------------------------- subcommands


def cmd_calib(args):
    import cv2

    cam = CubeCamera(args.camera, None, args.hsv_lo, args.hsv_hi)
    points_table = json.loads(args.points) if args.points else [
        [ZONES["A"][0], ZONES["A"][1]], [ZONES["C"][0], ZONES["C"][1]],
        [ZONES["C"][0] + 0.10, ZONES["C"][1]], [ZONES["A"][0] + 0.10, ZONES["A"][1]],
    ]
    print("Click, in order, the marks at table points (m):", points_table, "then press any key.")
    clicks = []
    cv2.namedWindow("calib")
    cv2.setMouseCallback("calib", lambda e, u, v, *_: clicks.append((u, v)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while len(clicks) < len(points_table):
        frame, _ = cam.latest()
        if frame is not None:
            for c in clicks:
                cv2.circle(frame, c, 5, (0, 0, 255), -1)
            cv2.imshow("calib", frame)
        cv2.waitKey(30)
    H, _ = cv2.findHomography(np.array(clicks, np.float32), np.array(points_table, np.float32))
    Path(args.homography).write_text(json.dumps({"H": H.tolist(), "clicks": clicks, "table": points_table}, indent=2))
    print("saved", args.homography)


def cmd_detect(args):
    import cv2

    H = np.array(json.loads(Path(args.homography).read_text())["H"])
    cam = CubeCamera(args.camera, H, args.hsv_lo, args.hsv_hi)
    while True:
        frame, det = cam.latest()
        if frame is not None:
            text = "no cube"
            if det and "xy" in det:
                text = f"x={det['xy'][0]:.3f} y={det['xy'][1]:.3f} yaw={math.degrees(det['yaw']):.0f} zone={zone_of(det['xy'])}"
                cv2.circle(frame, tuple(int(c) for c in det["uv"]), 6, (0, 255, 0), 2)
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("detect", frame)
        if cv2.waitKey(30) == 27:
            break


def cmd_run(args):
    H = np.array(json.loads(Path(args.homography).read_text())["H"])
    cam = CubeCamera(args.camera, H, args.hsv_lo, args.hsv_hi)
    arms = {
        "left": Arm("left", args.left_port, args.left_id, args.left_map, args.left_policy, args.scales, args.execute),
        "right": Arm("right", args.right_port, args.right_id, args.right_map, args.right_policy, args.scales, args.execute),
    }
    whisper = None
    if not args.keyboard:
        from faster_whisper import WhisperModel

        whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    try:
        while True:
            goal = (input("目標區 A/B/C（q 離開）> ").strip().upper()[:1] if args.keyboard else listen_zone(whisper))
            if goal == "Q":
                break
            if goal not in ZONES:
                print("沒聽到 A/B/C，再說一次")
                continue
            time.sleep(0.3)
            _, det = cam.latest()
            src = zone_of(det["xy"]) if det and "xy" in det else None
            if src is None or src == goal:
                print(f"方塊位置：{src}，不需要移動或看不到方塊")
                continue
            # A<->C goes through B: the near arm brings it to B, the other arm takes it on.
            legs = [(src, "B"), ("B", goal)] if {src, goal} == {"A", "C"} else [(src, goal)]
            for a, b in legs:
                arm = arms[ARM_FOR[(a, b)]]
                print(f"[run] {arm.name} arm: {a} -> {b}")
                if not run_skill(arm, cam, b, args.timeout):
                    break
    finally:
        for arm in arms.values():
            arm.close(hold=args.hold_at_end)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["calib", "detect", "run"])
    p.add_argument("--camera", default="0")
    p.add_argument("--homography", default="table_homography.json")
    p.add_argument("--points", help="JSON list of table points (m) to click, default: A, C, C+10cm, A+10cm")
    p.add_argument("--hsv-lo", type=int, nargs=3, default=[100, 120, 60], help="cube colour lower HSV (default blue)")
    p.add_argument("--hsv-hi", type=int, nargs=3, default=[130, 255, 255])
    for side in ("left", "right"):
        p.add_argument(f"--{side}-port")
        p.add_argument(f"--{side}-id", default=f"{side}_follower")
        p.add_argument(f"--{side}-map", type=Path)
        p.add_argument(f"--{side}-policy", type=Path, help="TorchScript policy.pt exported by isaaclab play")
    p.add_argument("--scales", type=float, nargs=5, default=[0.033] * 5, help="arm action scales used in training (rad)")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--keyboard", action="store_true", help="type A/B/C instead of speaking")
    p.add_argument("--whisper", default="small")
    p.add_argument("--execute", action="store_true", help="send motor commands (default: dry run)")
    p.add_argument("--hold-at-end", action="store_true")
    args = p.parse_args()
    {"calib": cmd_calib, "detect": cmd_detect, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
