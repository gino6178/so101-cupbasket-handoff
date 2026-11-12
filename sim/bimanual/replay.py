"""State-replay curriculum shared by the bimanual expert tasks.

When an episode first reaches ``record_flag`` (e.g. the cube at the handover point), the full physical state of
both arms and the task objects is stored in a ring buffer. Later resets restore a random stored state for a
fraction of environments, together with the milestone flags it had already earned, so the later stages of a long
task are practised without first replaying the whole approach. Every stored state was reached by the policy in
simulation, so it is physically consistent.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase

ARMS = ("robot_left", "robot_right")


class StateBuffer:
    def __init__(self, env: ManagerBasedRLEnv, objects: tuple[str, ...], num_flags: int, capacity: int = 16384):
        self.objects, self.capacity, self.size, self.head = objects, capacity, 0, 0
        dev = env.device
        nj = env.scene[ARMS[0]].data.joint_pos.torch.shape[1]
        self.joint = {a: torch.zeros((capacity, 3, nj), device=dev) for a in ARMS}
        self.root = {o: torch.zeros((capacity, 13), device=dev) for o in objects}
        self.flags = torch.zeros((capacity, num_flags), dtype=torch.bool, device=dev)
        # Per-episode task parameters (e.g. a sampled target) that a restored state must keep.
        extra = getattr(env, "_bi_extra", None)
        self.extra = None if extra is None else torch.zeros((capacity, *extra.shape[1:]), device=dev)

    def record(self, env: ManagerBasedRLEnv, ids: torch.Tensor, flags: torch.Tensor) -> None:
        if len(ids) == 0:
            return
        ids = ids[: self.capacity]
        slots = (self.head + torch.arange(len(ids), device=ids.device)) % self.capacity
        for a in ARMS:
            d = env.scene[a].data
            self.joint[a][slots] = torch.stack(
                (d.joint_pos.torch[ids], d.joint_vel.torch[ids], d.joint_pos_target.torch[ids]), dim=1
            )
        for o in self.objects:
            d = env.scene[o].data
            pos = d.root_pos_w.torch[ids] - env.scene.env_origins[ids]
            self.root[o][slots] = torch.cat(
                (pos, d.root_quat_w.torch[ids], d.root_lin_vel_w.torch[ids], d.root_ang_vel_w.torch[ids]), dim=-1
            )
        self.flags[slots] = flags[ids]
        if self.extra is not None:
            self.extra[slots] = env._bi_extra[ids]
        self.head = int((self.head + len(ids)) % self.capacity)
        self.size = min(self.capacity, self.size + len(ids))

    def restore(self, env: ManagerBasedRLEnv, ids: torch.Tensor) -> torch.Tensor:
        rows = torch.randint(0, self.size, (len(ids),), device=ids.device)
        for a in ARMS:
            robot, state = env.scene[a], self.joint[a][rows]
            robot.write_joint_position_to_sim_index(position=state[:, 0], env_ids=ids)
            robot.write_joint_velocity_to_sim_index(velocity=state[:, 1], env_ids=ids)
            robot.set_joint_position_target_index(target=state[:, 2], env_ids=ids)
        for o in self.objects:
            state = self.root[o][rows].clone()
            state[:, :3] += env.scene.env_origins[ids]
            env.scene[o].write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=ids)
            env.scene[o].write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=ids)
        if self.extra is not None:
            env._bi_extra[ids] = self.extra[rows]
        return self.flags[rows]


def buffer(env: ManagerBasedRLEnv, objects: tuple[str, ...], num_flags: int) -> StateBuffer:
    if not hasattr(env, "_bi_replay"):
        env._bi_replay = StateBuffer(env, objects, num_flags)
        env._bi_pending_flags = None
    return env._bi_replay


class replay_reset(ManagerTermBase):
    """Reset-mode event: restore stored states for a ``fraction`` of the resetting environments.

    Runs after the default scene reset. The restored milestone flags are handed to the success term, whose reset
    runs after the event manager and applies them once it has cleared the episode progress.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, objects: tuple, num_flags: int,
        fraction: float = 0.5, min_size: int = 256,
    ) -> None:
        buf = buffer(env, tuple(objects), num_flags)
        env._bi_pending_flags = None
        if env_ids is None or buf.size < min_size or getattr(env, "_bi_no_replay", False):
            return
        ids = env_ids[torch.rand(len(env_ids), device=env_ids.device) < fraction]
        if len(ids) == 0:
            return
        env._bi_pending_flags = (ids, buf.restore(env, ids))


def load_start_states(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, path: str, objects: tuple, num_flags: int
) -> None:
    """Reset-mode event: start every resetting environment from a stored state file (``collect.install_state_dump``).

    The file is held in its own buffer, separate from the replay curriculum's, so states recorded during training
    never displace the fixed start set. Restored flags are handed to the success term like replay resets.
    """
    if not hasattr(env, "_bi_start"):
        data = torch.load(path, map_location=env.device)
        start = StateBuffer(env, tuple(objects), num_flags, capacity=int(data["size"]))
        for a in ARMS:
            start.joint[a][:] = data["joint"][a]
        for o in objects:
            start.root[o][:] = data["root"][o]
        start.flags[:] = data["flags"]
        start.size = start.capacity
        env._bi_start = start
        print(f"[start] loaded {start.size} start states from {path}", flush=True)
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._bi_pending_flags = (env_ids, env._bi_start.restore(env, env_ids))
