"""Chained sorting from a voice / text command on LidSort: one episode executes several (cube, box) commands in order.

    BITIDY_VOICE_CMDS='[[0,0],[2,2],[1,1]]'      # from voice_command.py ("commands"), same chain for every env
    BITIDY_VOICE_CMDS='[[0,0],[1,1],[0,0,1]]'    # v2: a third entry 1 = take the cube out of that box
    BITIDY_VOICE_RANDOM=putall_takeall           # v2: per-env random cycle, 3 put-ins then 3 take-outs (see _cycles)
    BITIDY_SCRIPT=bcrnn:BCRNNScript BITIDY_BCRNN=<policy.pt>   # or the oracle, oracle_lid_sort:LidSortScript
    isaaclab play ... --task BiTidy-LidSort-SO101 --external_callback isaaclab_tutorial.tasks.bimanual.voice_sort.install_voice_sort

When the current command succeeds, the next one is issued to that env: the command is switched, the "other objects must
not move" baselines are refreshed to the current poses and the executor's memory is reset. Success-termination is off;
each env runs its first episode (``BITIDY_LID_SORT_EPISODE_S`` long) and the run prints ``BITIDY_VOICE_RESULT`` (share of
envs completing 1, 2, ... commands) and exits. With ``BITIDY_VIDEO`` set, the first 4 envs are recorded.
"""

from __future__ import annotations

import json
import os
import sys

import torch


def _t(x):
    """Isaac Lab 3 data arrays expose ``.torch``; plain tensors pass through."""
    return x.torch if hasattr(x, "torch") else x


def _cycles(n: int, dev) -> torch.Tensor:
    """(n, 6, 3) per-env put-all / take-all cycles: [cube, box, 0] x 3 in a random cube order with a random
    cube -> box permutation, then [cube, box, 1] x 3 in a new random cube order."""
    perm = torch.rand((n, 3), device=dev).argsort(-1)  # perm[e, i] = box of cube i
    put = torch.rand((n, 3), device=dev).argsort(-1)  # cube order of the put-ins
    take = torch.rand((n, 3), device=dev).argsort(-1)  # cube order of the take-outs
    box_of = lambda c: perm.gather(1, c)
    put_c = torch.stack((put, box_of(put), torch.zeros_like(put)), -1)
    take_c = torch.stack((take, box_of(take), torch.ones_like(take)), -1)
    return torch.cat((put_c, take_c), 1)


def install_voice_sort() -> list[str]:
    os.environ["BITIDY_NO_SUCCESS_END"] = "1"
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import lid_sort_env_cfg as ls
    from .scripted import _script_class

    if os.environ.get("BITIDY_VIDEO"):
        from . import render

        render.install_recorder()
    inner_step = RslRlVecEnvWrapper.step
    # [cube, box] (put cube into box) or [cube, box, mode] (mode 1: take cube out of box; v2 / BITIDY_LS_TAKEOUT)
    # BITIDY_VOICE_RANDOM=putall_takeall: every env draws its own 6-command cycle instead -- 3 put-ins (a random cube
    # order, each cube into its own random box: a random permutation, one cube per box) then 3 take-outs of those cubes
    # in a random order (mode 1, source box = the box the cube was put in).
    random_mode = os.environ.get("BITIDY_VOICE_RANDOM", "")
    cmds = [list(c) + [0] * (3 - len(c)) for c in json.loads(os.environ.get("BITIDY_VOICE_CMDS", "[]"))]
    if random_mode not in ("", "putall_takeall"):
        raise ValueError(f"BITIDY_VOICE_RANDOM={random_mode!r}: only 'putall_takeall'")
    n_cmds = 6 if random_mode else len(cmds)
    # Between commands both arms first return to their start pose (the executor was trained from there), for at most
    # BITIDY_VOICE_HOME steps; 0 issues the next command at once.
    home_steps = int(os.environ.get("BITIDY_VOICE_HOME", "120"))
    state: dict = {}

    # Homing mode. "target" (default): a homing controller ramps the arm joints' position targets straight to the
    # default pose (BITIDY_VOICE_HOME_RATE rad per step), as a real arm's homing routine would. The policy's relative
    # action cannot do it: it only leads the measured position by 0.033 rad, and against gravity the PD drive stalls
    # (measured: right elbow_flex stuck ~1.06 rad from home after 120 steps). "action": the old relative-action homing.
    home_mode = os.environ.get("BITIDY_VOICE_HOME_MODE", "target")
    home_rate = float(os.environ.get("BITIDY_VOICE_HOME_RATE", "0.03"))

    def _arm_ids(env, name):
        from .env_cfg import ARM_JOINTS

        if name not in state:
            state[name] = env.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0]
        return state[name]

    def home_action(env):
        """Relative joint action toward each arm's default pose, grippers open (action layout: arm 5 + gripper, x2).
        In "target" mode the arm part is overridden by the homing targets (see ``apply_actions``)."""
        parts = []
        for name in ("robot_left", "robot_right"):
            robot = env.scene[name]
            ids = _arm_ids(env, name)
            err = _t(robot.data.default_joint_pos)[:, ids] - _t(robot.data.joint_pos)[:, ids]
            parts += [(err / 0.033).clamp(-1.0, 1.0), torch.ones((env.num_envs, 1), device=env.device)]
        return torch.cat(parts, -1)

    def start_homing(env, ids):
        for name in ("robot_left", "robot_right"):
            robot = env.scene[name]
            jid = _arm_ids(env, name)
            if "home_tgt_" + name not in state:
                state["home_tgt_" + name] = _t(robot.data.joint_pos_target)[:, jid].clone()
            state["home_tgt_" + name][ids] = _t(robot.data.joint_pos_target)[ids][:, jid]

    def ramp_homing(env, mask):
        for name in ("robot_left", "robot_right"):
            robot = env.scene[name]
            jid = _arm_ids(env, name)
            h = state["home_tgt_" + name]
            step_ = (_t(robot.data.default_joint_pos)[:, jid] - h).clamp(-home_rate, home_rate)
            state["home_tgt_" + name] = torch.where(mask[:, None], h + step_, h)

    if home_mode == "target":
        from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction

        inner_apply = RelativeJointPositionAction.apply_actions

        def apply_actions(self):
            inner_apply(self)
            mask = state.get("hmask")
            key = "home_tgt_" + self.cfg.asset_name
            if mask is None or key not in state or not bool(mask.any()) or len(self._joint_ids) != state[key].shape[1]:
                return
            tgt = _t(self._asset.data.joint_pos_target)[:, self._joint_ids]
            tgt = torch.where(mask[:, None], state[key], tgt)
            self._asset.set_joint_position_target_index(target=tgt, joint_ids=self._joint_ids)

        RelativeJointPositionAction.apply_actions = apply_actions

    def issue(env, ids, k):
        """Give envs ``ids`` their k-th command and restart their progress and executor."""
        st = ls._state(env)
        if "diag" in state:
            _diag_issue(env, ids, k)
        st["cmd"][ids] = state["cmdt"][ids, k]
        st["lid_start"][ids] = ls._objs_pos(env, "lid")[ids]
        st["cube_start"][ids] = ls._objs_pos(env, "cube")[ids]
        ls.progress(env).reset(ids)
        state["script"].reset(ids)

    def _diag_issue(env, ids, k):
        """BITIDY_VOICE_DIAG: at each later command, how far the arms are from home and the scene from its start."""
        err = []
        for name in ("robot_left", "robot_right"):
            robot = env.scene[name]
            jid = _arm_ids(env, name)
            e = _t(robot.data.default_joint_pos)[:, jid] - _t(robot.data.joint_pos)[:, jid]
            te = _t(robot.data.default_joint_pos)[:, jid] - _t(robot.data.joint_pos_target)[:, jid]
            if int(k.min()) > 0 and state.setdefault("jprint", 0) < 4:
                state["jprint"] += 1
                print(f"BITIDY_VOICE_DIAG_J k={int(k[0])} {name} q0={[round(float(v), 3) for v in _t(robot.data.default_joint_pos)[ids[0], jid]]} "
                      f"mean(q0-q)={[round(float(v), 3) for v in e[ids].mean(0)]} mean(q0-target)={[round(float(v), 3) for v in te[ids].mean(0)]}", flush=True)
            err.append(e.abs().max(-1).values[ids])
        st = ls._state(env)
        lid_off = torch.linalg.vector_norm(ls._objs_pos(env, "lid")[ids, :, :2] - ls.box_centers(env)[ids], dim=-1)
        for r, e in enumerate(ids.tolist()):
            state["diag"].append((int(k[r]), float(err[0][r]), float(err[1][r]), [round(float(v), 4) for v in lid_off[r]]))

    def _diag_end(env, ids):
        fin = env._bi_final[ids].int().cpu()
        for r, e in enumerate(ids.tolist()):
            state["ends"].append((int(state["k"][e]), fin[r].tolist()))

    def _diag_report():
        import collections

        by_k = collections.defaultdict(list)
        for k, el, er, lo in state["diag"]:
            by_k[k].append((el, er, max(lo)))
        for k, v in sorted(by_k.items()):
            t = torch.tensor(v)
            print(f"BITIDY_VOICE_DIAG issue k={k} n={len(v)} home_err_left(max|q-q0|) p50={t[:, 0].median():.3f} p90={t[:, 0].quantile(0.9):.3f} "
                  f"right p50={t[:, 1].median():.3f} p90={t[:, 1].quantile(0.9):.3f} max_lid_offset p50={t[:, 2].median():.4f} p90={t[:, 2].quantile(0.9):.4f}", flush=True)
        names = list(ls.FLAGS) + ["success"]
        fails = collections.defaultdict(list)
        for k, f in state["ends"]:
            if k < n_cmds:
                fails[k].append(f)
        for k, v in sorted(fails.items()):
            t = torch.tensor(v).float()
            # furthest milestone reached (flags 0..8 are ordered stages)
            stage = torch.where(t[:, :9].bool(), torch.arange(1, 10).float(), torch.zeros(9)).max(-1).values.long()
            hist = torch.bincount(stage, minlength=10).tolist()
            print(f"BITIDY_VOICE_DIAG failed_at k={k} n={len(v)} flag_rates={dict(zip(names, [round(float(x), 3) for x in t.mean(0)]))} "
                  f"furthest_stage_hist(0=none,{','.join(str(i + 1) + '=' + n for i, n in enumerate(names[:9]))})={hist}", flush=True)

    def step(self, actions):
        env = self.unwrapped
        if "script" not in state:
            n = env.num_envs
            state["cmdt"] = _cycles(n, env.device) if random_mode else torch.tensor(cmds, device=env.device).expand(n, -1, -1)
            state.update(script=_script_class(os.environ["BITIDY_SCRIPT"])(env),
                         k=torch.zeros(n, dtype=torch.long, device=env.device),
                         finished=torch.zeros(n, dtype=torch.bool, device=env.device),
                         homing=torch.zeros(n, dtype=torch.long, device=env.device),
                         log=[])
            if os.environ.get("BITIDY_VOICE_DIAG"):
                state.update(diag=[], ends=[])
            issue(env, torch.arange(n, device=env.device), state["k"])
        act = state["script"].act()
        homing = state["homing"] > 0
        if bool(homing.any()):
            act = torch.where(homing[:, None], home_action(env), act)
            if home_mode == "target":
                ramp_homing(env, homing)
                state["hmask"] = homing
            state["homing"][homing] -= 1
            done_homing = homing & (state["homing"] == 0) & ~state["finished"]
            if bool(done_homing.any()):
                ids = done_homing.nonzero().squeeze(-1)
                issue(env, ids, state["k"][ids])
        result = inner_step(self, act)
        state["hmask"] = None
        dones = result[2].bool()
        p = ls.progress(env)
        running = ~state["finished"]
        # Command done: count it, then issue the next one (or stop this env's chain).
        won = p.success & running & ~dones & (state["homing"] == 0)
        if bool(won.any()):
            ids = won.nonzero().squeeze(-1)
            state["k"][ids] += 1
            last = state["k"][ids] >= n_cmds
            state["finished"][ids[last]] = True
            nxt = ids[~last]
            if len(nxt):
                if home_steps > 0:
                    # keep the finished command's success from re-triggering while homing
                    ls.progress(env).reset(nxt)
                    state["homing"][nxt] = home_steps
                    if home_mode == "target":
                        start_homing(env, nxt)
                else:
                    issue(env, nxt, state["k"][nxt])
            t = int(env.episode_length_buf[ids[0]])
            print(f"BITIDY_VOICE_STEP t={t} envs_done_cmd={[int(v) for v in state['k'][ids][:6]]}", flush=True)
        # Episode end (timeout / drop): the chain stops where it is.
        if "diag" in state and bool((dones & running).any()):
            _diag_end(env, (dones & running).nonzero().squeeze(-1))
        state["finished"] |= dones & running
        if bool(dones.any()):
            ids = dones.nonzero().squeeze(-1)
            state["script"].reset(ids)
        if bool(state["finished"].all()):
            k = state["k"]
            share = [round(float((k >= i).float().mean()), 4) for i in range(1, n_cmds + 1)]
            print("BITIDY_VOICE_RESULT " + json.dumps({"commands": random_mode or cmds, "envs": len(k), "completed_at_least": share,
                                                       "all_done": share[-1]}), flush=True)
            if random_mode:  # per command position: share of the envs that got command i that completed it
                got = [int((k >= i).sum()) for i in range(n_cmds)]
                print("BITIDY_VOICE_PER_CMD " + json.dumps({"issued": got, "done": [int((k >= i + 1).sum()) for i in range(n_cmds)],
                                                            "rate": [round(int((k >= i + 1).sum()) / max(g, 1), 4) for i, g in enumerate(got)]}), flush=True)
            if "diag" in state:
                _diag_report()
            if os.environ.get("BITIDY_VIDEO"):
                return result  # the recorder ends the run after its frame budget
            os._exit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
