"""Chained voice / text commands on CupBasket: one continuing scene executes N instructions in order (no reset between).

    # plan sources (one of them; the same plan for every env unless noted)
    BITIDY_VOICE_TEXT="紅杯放A，紅碗放B；綠杯放B，綠碗放C；藍杯放C，藍碗放A"   # rule parser (voice_command.py);
                                         # ；/;/。/newline/然後/接著/then separate instructions, "，" pairs within one
    BITIDY_VOICE_TEXT_FILE=cmd.txt       # the same text from a UTF-8 file
    BITIDY_VOICE_PLAN=plan.json          # {"steps": [[[obj,basket,1],[obj,basket,0|1]], ...]} (vlm_planner / voice_command
                                         # "commands" output, or a bare list); {"plans": [steps_env0, steps_env1, ...]}: per env
    BITIDY_VOICE_CMDS='[[[0,0,1],[3,1,1]], ...]'   # the same, inline
    BITIDY_VOICE_RANDOM=putall           # per env: the 6 objects in a random order, 3 two-pair instructions, random baskets
    BITIDY_VOICE_RANDOM=random           # per env: random 1-2 pair instructions over the table objects until all are away
                                         # (single with BITIDY_VOICE_P_SINGLE, default 0.15); 3..6 instructions

    BITIDY_SCRIPT=residual:ResidualScript BITIDY_BCRNN=<bcrnn.pt> BITIDY_RESIDUAL=<model_*.pt> BITIDY_CB_PRESORT=0 \
    BITIDY_CB_EPISODE_S=400 isaaclab play ... --task BiTidy-CupBasket-SO101 \
        --external_callback isaaclab_tutorial.tasks.bimanual.voice_cup.install_voice_cup

Per instruction: the (object, basket) pairs go to the arm slots (nearest arm, 2 x 2 min total distance, as the env's
reset), the env's chaining state is re-derived from the physics (``loc`` / ``level`` of every vessel: which basket, which
nesting level; ``ref`` = current poses, so "other vessels must not move" is measured from here), the progress flags and
the executor memory (``ResidualScript.reset`` -> BC-RNN hidden state) are reset for those envs. Success of the
instruction = the env's own joint success predicate. Then both arms home (``BITIDY_VOICE_HOME`` steps, homing
controller on the joint targets, grippers open) and the next instruction is issued. An instruction not done within
``BITIDY_VOICE_CMD_STEPS`` (default 1800 = 60 s, the single-command eval's episode) fails the chain there; a dropped
vessel (env termination) too. Success-termination is off (``BITIDY_NO_SUCCESS_END``). Two pairs aimed at the same zone
(same basket, same type) are split into two instructions (never in training).

Detection and recovery: every attempt (env success or timeout) is followed by homing and a status check of EVERY
vessel commanded so far (``obj_status``: ok / misplaced = upright but not in its basket zone or not nested / toppled /
dropped). All ok -> next instruction. Otherwise, with ``BITIDY_VOICE_RETRY=R`` (default 0 = no recovery), the bad
vessels (current instruction's first, at most two, never two into one zone) are re-commanded to their basket, at most R
times per vessel; from retry ``BITIDY_VOICE_RETRY_SWAP`` (default 2; 0 = never) the arm assignment is swapped (the
other arm tries). A toppled vessel is retried the same way (the policy was never trained to right one: counted
separately). No budget left -> the chain stops there. Any extra env var (task variants, e.g. a larger bowl zone) passes
through to the env unchanged.

After the last instruction the arms park at home for ``BITIDY_VOICE_CHECK`` steps and the whole scene is checked
("intact"): every commanded vessel in its last commanded basket, in its type's zone, upright, and every zone's stack
nested (xy on the bottom vessel's axis, z on its nesting level). Prints ``BITIDY_VOICE_CUP_RESULT`` (JSON) and exits;
with ``BITIDY_VIDEO`` the first 4 envs are recorded and the recorder ends the run.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

import torch


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# ---------------------------------------------------------------- plans

_SEP = r"[；;。\n]|然後|然后|接著|接着|\bthen\b"


def plan_from_text(text: str) -> list[list[list[int]]]:
    """Instructions separated by ；/;/。/newline/然後/接著/then; within one, up to two pairs ("紅杯放A，紅碗放B")."""
    try:
        from .voice_command import cup_basket_instruction
    except ImportError:  # run as a plain script
        from voice_command import cup_basket_instruction
    steps = []
    for seg in re.split(_SEP, text, flags=re.I):
        if seg.strip():
            steps += [s for s in cup_basket_instruction(seg) if any(p[2] for p in s)]
    return steps


def _load_plans(n: int) -> tuple[list, str, list[str]]:
    """Per env: list of instructions, each a list of 1-2 (object, basket) pairs; source; per-env group label (stats).
    ``BITIDY_VOICE_RANDOM=putall,random`` alternates the modes over the envs; a ``{"plans": [...]}`` file alternates its
    plans (each a slot-format step list or an instruction text)."""
    mode = os.environ.get("BITIDY_VOICE_RANDOM", "")
    if mode:
        rng = random.Random(int(os.environ.get("BITIDY_VOICE_SEED", "0")))
        modes = mode.split(",")
        groups = [modes[e % len(modes)] for e in range(n)]
        return [_random_plan(rng, g) for g in groups], f"random:{mode}", groups
    if os.environ.get("BITIDY_VOICE_TEXT_FILE"):  # UTF-8 text file (avoids shell / ssh code-page trouble)
        with open(os.path.expanduser(os.environ["BITIDY_VOICE_TEXT_FILE"]), encoding="utf-8") as f:
            text = f.read().strip()
        steps, src = plan_from_text(text), "text:" + text
    elif os.environ.get("BITIDY_VOICE_TEXT"):
        steps, src = plan_from_text(os.environ["BITIDY_VOICE_TEXT"]), "text:" + os.environ["BITIDY_VOICE_TEXT"]
    elif os.environ.get("BITIDY_VOICE_PLAN"):
        with open(os.path.expanduser(os.environ["BITIDY_VOICE_PLAN"]), encoding="utf-8") as f:
            steps, src = json.load(f), "plan:" + os.environ["BITIDY_VOICE_PLAN"]
    else:
        steps, src = json.loads(os.environ["BITIDY_VOICE_CMDS"]), "cmds"
    if isinstance(steps, dict):
        if "plans" in steps:
            per = [_pairs(plan_from_text(p) if isinstance(p, str) else p) for p in steps["plans"]]
            return [per[e % len(per)] for e in range(n)], src + " (per env)", [f"plan{e % len(per)}" for e in range(n)]
        steps = steps.get("steps", steps.get("commands"))
    one = _pairs(steps)
    if not one:
        raise ValueError(f"BITIDY_VOICE: empty plan from {src}: {steps}")
    return [one for _ in range(n)], src, ["all"] * n


def _pairs(steps) -> list[list[tuple[int, int]]]:
    """Slot-format steps -> pair lists; a step whose two pairs share a zone (same basket and type) is split in two."""
    from .cup_basket_env_cfg import OBJ_TYPE

    out = []
    for st in steps:
        pr = []
        for s in st:
            s = list(s) + [1] * (3 - len(s))
            if s[2] and all(int(s[0]) != k for k, _ in pr):
                pr.append((int(s[0]), int(s[1])))
        if len(pr) == 2 and OBJ_TYPE[pr[0][0]] == OBJ_TYPE[pr[1][0]] and pr[0][1] == pr[1][1]:
            out += [[pr[0]], [pr[1]]]
        elif pr:
            out.append(pr[:2])
    return out


def _random_plan(rng: random.Random, mode: str) -> list[list[tuple[int, int]]]:
    from .cup_basket_env_cfg import N_BASKET, N_OBJ, OBJ_TYPE

    objs = list(range(N_OBJ))
    rng.shuffle(objs)
    p_single = float(os.environ.get("BITIDY_VOICE_P_SINGLE", "0.15"))
    plan, i = [], 0
    while i < len(objs):
        m = 2 if mode == "putall" else (1 if rng.random() < p_single or i == len(objs) - 1 else 2)
        pr = []
        for k in objs[i:i + m]:
            j = rng.randrange(N_BASKET)
            if pr and OBJ_TYPE[k] == OBJ_TYPE[pr[0][0]] and j == pr[0][1]:
                j = rng.choice([b for b in range(N_BASKET) if b != j])
            pr.append((k, j))
        plan.append(pr)
        i += m
    return plan


# ---------------------------------------------------------------- driver


def install_voice_cup() -> list[str]:
    os.environ["BITIDY_NO_SUCCESS_END"] = "1"
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from . import cup_basket_env_cfg as cb
    from .scripted import _script_class

    cb.PRESORT = 0.0  # every vessel starts on the table (also pass BITIDY_CB_PRESORT=0: the first reset may come earlier)
    video = bool(os.environ.get("BITIDY_VIDEO"))
    if video:
        from . import render

        render.install_recorder()
    inner_step = RslRlVecEnvWrapper.step
    home_steps = int(os.environ.get("BITIDY_VOICE_HOME", "120"))
    home_rate = float(os.environ.get("BITIDY_VOICE_HOME_RATE", "0.03"))
    cmd_steps = int(os.environ.get("BITIDY_VOICE_CMD_STEPS", "1800"))
    check_steps = int(os.environ.get("BITIDY_VOICE_CHECK", "60"))
    trace_envs = int(os.environ.get("BITIDY_VOICE_TRACE", "4"))
    state: dict = {}

    # ---- homing (as voice_sort "target" mode: ramp the arm joints' position targets to the default pose)
    def _arm_ids(env, name):
        from .env_cfg import ARM_JOINTS

        if name not in state:
            state[name] = env.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0]
        return state[name]

    def home_action(env):
        parts = []
        for name in cb.ARMS:
            robot = env.scene[name]
            ids = _arm_ids(env, name)
            err = _t(robot.data.default_joint_pos)[:, ids] - _t(robot.data.joint_pos)[:, ids]
            parts += [(err / 0.033).clamp(-1.0, 1.0), torch.ones((env.num_envs, 1), device=env.device)]
        return torch.cat(parts, -1)

    def start_homing(env, ids):
        for name in cb.ARMS:
            robot = env.scene[name]
            jid = _arm_ids(env, name)
            key = "home_tgt_" + name
            if key not in state:
                state[key] = _t(robot.data.joint_pos_target)[:, jid].clone()
            state[key][ids] = _t(robot.data.joint_pos_target)[ids][:, jid]

    def ramp_homing(env, mask):
        for name in cb.ARMS:
            robot = env.scene[name]
            jid = _arm_ids(env, name)
            h = state["home_tgt_" + name]
            step_ = (_t(robot.data.default_joint_pos)[:, jid] - h).clamp(-home_rate, home_rate)
            state["home_tgt_" + name] = torch.where(mask[:, None], h + step_, h)

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

    # ---- scene state from the physics
    def derive(env):
        """loc (n, 6): basket whose inside holds the vessel's origin (-1 table); level: rank by height among the vessels
        of the same type in the same basket."""
        pos = cb.objs_pos(env)
        bk = cb.baskets(env)
        uv = cb.to_basket(bk.unsqueeze(1), pos[:, :, None, :2])  # (n, 6, 3, 2)
        half = torch.tensor((cb.IN_L / 2 + 0.005, cb.IN_W / 2 + 0.005), device=env.device)
        inside = (uv.abs() < half).all(-1) & (pos[..., 2:3] < cb.TABLE_TOP + 0.12)
        loc = torch.where(inside.any(-1), inside.float().argmax(-1), torch.full_like(pos[..., 0], -1).long())
        typ = cb.obj_type(env.device)
        same = (loc.unsqueeze(2) == loc.unsqueeze(1)) & (typ.view(1, -1, 1) == typ.view(1, 1, -1)) & (loc >= 0).unsqueeze(2)
        level = (same & (pos[..., 2].unsqueeze(1) < pos[..., 2].unsqueeze(2))).sum(-1)
        return loc, level, pos, uv

    retry_max = int(os.environ.get("BITIDY_VOICE_RETRY", "0"))
    # arm assignment: "nearest" (as the env: nearest arm, 2 x 2 min total distance) or "band" (reach-aware, see issue)
    assign = os.environ.get("BITIDY_VOICE_ASSIGN", "nearest")
    band_lo, band_hi = cb.OBJ_BAND[0] - 0.005, cb.OBJ_BAND[1] + 0.005
    swap_from = int(os.environ.get("BITIDY_VOICE_RETRY_SWAP", "2"))
    OK, MISPLACED, TOPPLED, DROPPED = 0, 1, 2, 3
    STATUS = ("ok", "misplaced", "toppled", "dropped")

    def obj_status(env, expect):
        """(n, 6) status of every vessel against ``expect`` (basket, -1 = not checked): ok = in that basket, in its type's
        zone, upright and nested (on the bottom vessel's axis, z on its nesting level); misplaced = upright but not so
        (table, wrong basket / zone, not nested, still in a gripper); toppled = tilt >= TILT_MAX; dropped = off the table."""
        loc, level, pos, uv = derive(env)
        typ = cb.obj_type(env.device)
        g = cb._geo_t(env.device)
        idx = expect.clamp(min=0)[..., None, None].expand(-1, -1, 1, 2)
        u = uv.gather(2, idx).squeeze(2)  # (n, 6, 2) in the expected basket's frame
        zone_u = torch.where(typ == cb.CUP, cb.ZONE_U[cb.CUP], cb.ZONE_U[cb.BOWL]).to(u.dtype)
        in_basket = loc == expect
        in_zone = torch.linalg.vector_norm(u - torch.stack((zone_u, torch.zeros_like(zone_u)), -1), dim=-1) < cb.ZONE_XY + 0.015
        upright = cb.tilt(cb.objs_quat(env)) < cb.TILT_MAX
        same = (loc.unsqueeze(2) == loc.unsqueeze(1)) & (typ.view(1, -1, 1) == typ.view(1, 1, -1)) & (loc >= 0).unsqueeze(2)
        bottom = same & (level.unsqueeze(1) == 0)
        bxy = (bottom.float().unsqueeze(-1) * pos[:, None, :, :2]).sum(2)
        axis_ok = (torch.linalg.vector_norm(pos[..., :2] - bxy, dim=-1) < 0.015) | (level == 0)
        zf = pos[..., 2] - cb.FLOOR_Z
        z_ok = (zf > level * g["pitch"][typ] - cb.Z_TOL) & (zf < level * g["pitch_max"][typ] + cb.Z_TOL + 0.004)
        parts = {"in_basket": in_basket, "in_zone": in_zone, "upright": upright, "nested": axis_ok & z_ok}
        ok = in_basket & in_zone & upright & axis_ok & z_ok
        code = torch.where(ok, OK, torch.where(pos[..., 2] < cb.TABLE_TOP - 0.05, DROPPED,
                                               torch.where(upright, MISPLACED, TOPPLED)))
        return code, parts

    def intact(env, expect):
        code, parts = obj_status(env, expect)
        want = expect >= 0
        out = {k: (v | ~want).all(-1) for k, v in parts.items()}
        out["all"] = ((code == OK) | ~want).all(-1)
        return out

    def expect_upto(e: int, k: int) -> dict:
        """Object -> basket for the plan instructions 0..k of env e (the last one naming an object wins)."""
        out = {}
        for pr in state["plans"][e][: k + 1]:
            for o, j in pr:
                out[o] = j
        return out

    # ---- per-instruction bookkeeping
    def issue(env, ids, pairs: list, swaps: list, metas: list):
        """Give envs ``ids`` an instruction (``pairs[r]``: 1-2 (object, basket)): arm slots (nearest arm / 2 x 2; ``swaps[r]``
        flips it), chaining state from the physics, progress and executor reset; opens an attempt record (``metas[r]``)."""
        st = cb._state(env)
        loc, level, pos, _ = derive(env)
        st["loc"][ids], st["level"][ids], st["ref"][ids] = loc[ids], level[ids], pos[ids]
        bases = torch.tensor(cb.BASES, device=env.device).unsqueeze(0) + st["base_off"].view(-1, 2, 3)[..., :2]
        dist = torch.linalg.vector_norm(pos[:, :, None, :2] - bases[:, None], dim=-1)  # (n, 6, 2)
        dl, kl, typ = dist[ids].tolist(), state["k"][ids].tolist(), cb.obj_type(env.device)
        nom = torch.linalg.vector_norm(pos[ids][:, :, None, :2] - torch.tensor(cb.BASES, device=env.device), dim=-1).tolist()
        cmds = []
        for r, e in enumerate(ids.tolist()):
            pr = pairs[r]
            d = [dl[r][k] for k, _ in pr]
            cmd = [[0, 0, 0], [0, 0, 0]]
            if assign == "band":
                # reach-aware: give each vessel an arm whose (nominal) base distance lies in the training reach band;
                # a pair that cannot go to two different in-band arms runs as two single instructions, one after the other
                dn = [nom[r][k] for k, _ in pr]
                inb = lambda o, a: band_lo <= dn[o][a] <= band_hi
                if len(pr) == 2:
                    opts = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
                    ok_opts = [op for op in opts if all(inb(p_, a) for a, p_ in op)]
                    if not ok_opts:
                        state["defer"][e].append(pr[1])
                        pr, d, dn = [pr[0]], d[:1], dn[:1]
                        metas[r] = dict(metas[r], split=1)
                    else:
                        op = min(ok_opts, key=lambda op: sum(d[p_][a] for a, p_ in op))
                        if swaps[r] and len(ok_opts) == 2:  # swap only to another in-band assignment
                            op = opts[1 - opts.index(op)]
                        elif swaps[r]:
                            swaps[r] = False
                        for a, p_ in op:
                            cmd[a] = [pr[p_][0], pr[p_][1], 1]
                if len(pr) == 1:
                    arms = [a for a in (0, 1) if inb(0, a)] or [0, 1]
                    a = min(arms, key=lambda a: d[0][a])
                    if swaps[r] and len(arms) == 2:  # swap only to the other arm if it is in band too
                        a = 1 - a
                    elif swaps[r]:
                        swaps[r] = False
                    cmd[a] = [pr[0][0], pr[0][1], 1]
            elif len(pr) == 1:
                a = 0 if d[0][0] <= d[0][1] else 1
                cmd[1 - a if swaps[r] else a] = [pr[0][0], pr[0][1], 1]
            else:
                keep = (d[0][0] + d[1][1] <= d[0][1] + d[1][0]) != bool(swaps[r])
                for a, p in ([(0, 0), (1, 1)] if keep else [(0, 1), (1, 0)]):
                    cmd[a] = [pr[p][0], pr[p][1], 1]
            cmds.append(cmd)
            state["att"][e] = len(state["rec"])
            state["rec"].append(dict(metas[r], e=e, k=kl[r], slots=[(a, s[0], s[1]) for a, s in enumerate(cmd) if s[2]],
                                     npair=len(pr), swap=bool(swaps[r])))
            if e < trace_envs:
                print(f"BITIDY_VOICE_ISSUE env={e} k={kl[r]} {metas[r]} swap={bool(swaps[r])} "
                      f"t={int(env.episode_length_buf[e])} cmd={cmd} loc={loc[e].tolist()} level={level[e].tolist()}", flush=True)
        c = torch.tensor(cmds, dtype=torch.long, device=env.device)
        st["cmd"][ids] = c
        lr = loc[ids]
        for s in range(cb.N_SLOT):  # stack depth per slot (same-type vessels already in that zone)
            same = (lr == c[:, s, 1:2]) & (typ.unsqueeze(0) == typ[c[:, s, 0]].unsqueeze(-1))
            same &= torch.arange(cb.N_OBJ, device=env.device).unsqueeze(0) != c[:, s, 0:1]
            st["depth"][ids, s] = same.sum(-1) * c[:, s, 2]
        dep = st["depth"][ids].tolist()
        for r, e in enumerate(ids.tolist()):
            rec = state["rec"][state["att"][e]]
            rec["slots"] = [(a, o, b, dep[r][a]) for a, o, b in rec["slots"]]
            rec["dist"] = {a: [round(nom[r][o][0], 3), round(nom[r][o][1], 3)] for a, o, b, _ in rec["slots"]}
        cb.progress(env).reset(ids)
        state["script"].reset(ids)
        state["t"][ids] = 0

    def close(env, ids, ok: bool, reason: str = "", after_reset: bool = False):
        """End of envs ``ids``' current attempt: the env's own joint-success predicate (``ok``), per-arm latches, flags."""
        p = cb.progress(env)
        so = p.slot_ok[ids].tolist() if not after_reset else [[False, False]] * len(ids)
        fl = (env._bi_final[ids, :-1] if after_reset else p.flags[ids]).int().tolist()
        tl = state["t"][ids].tolist()
        for r, e in enumerate(ids.tolist()):
            rec = state["rec"][state["att"][e]]
            rec.update(env_ok=ok, t=tl[r], reason=reason, flags=fl[r],
                       slot_ok={a: bool(so[r][a] or ok) for a, *_ in rec["slots"]})

    def finish(env, ids, ok: bool):
        state["finished"][ids] = True
        state["chain_ok"][ids] = ok
        state["homing"][ids] = 10 ** 9  # parked at home for good
        state["pending"][ids] = False

    def decide(env, ids):
        """After an attempt + homing: check every vessel commanded so far; advance, retry the bad ones, or stop."""
        n = env.num_envs
        exp = torch.full((n, cb.N_OBJ), -1, dtype=torch.long, device=env.device)
        el = ids.tolist()
        kl = state["k"][ids].tolist()
        for r, e in enumerate(el):
            for o, j in expect_upto(e, kl[r]).items():
                exp[e, o] = j
        code = obj_status(env, exp)[0][ids].tolist()
        go, go_pairs, go_swaps, go_meta, stop_ok, stop_fail = [], [], [], [], [], []
        for r, e in enumerate(el):
            k = kl[r]
            want = expect_upto(e, k)
            rec = state["rec"][state["att"][e]]
            rec["after"] = {s[1]: code[r][s[1]] for s in rec["slots"]}
            deferred = state["defer"][e]
            bad = [o for o in want if code[r][o] != OK and all(o != q for q, _ in deferred)]
            rec["bad_after"] = {o: code[r][o] for o in bad}
            if not bad and deferred:  # the split-off pair(s) of this instruction
                go.append(e)
                go_pairs.append(deferred[:2])
                go_swaps.append(False)
                go_meta.append({"retry": 0, "split": 2})
                state["defer"][e] = deferred[2:]
                continue
            if not bad:
                state["k"][e] = k + 1
                if k + 1 >= len(state["plans"][e]):
                    stop_ok.append(e)
                else:
                    go.append(e)
                    go_pairs.append(state["plans"][e][k + 1])
                    go_swaps.append(False)
                    go_meta.append({"retry": 0})
                continue
            cur = [s[1] for s in rec["slots"]]
            cand = [o for o in sorted(bad, key=lambda o: (o not in cur, o))
                    if state["used"][e][o] < retry_max and code[r][o] != DROPPED]
            if not cand:
                stop_fail.append(e)
                continue
            pick = [cand[0]]
            for o in cand[1:]:
                if not (cb.OBJ_TYPE[o] == cb.OBJ_TYPE[pick[0]] and want[o] == want[pick[0]]):
                    pick.append(o)
                    break
            for o in pick:
                state["used"][e][o] += 1
            go.append(e)
            go_pairs.append([(o, want[o]) for o in pick])
            go_swaps.append(swap_from > 0 and any(state["used"][e][o] >= swap_from for o in pick))
            go_meta.append({"retry": max(state["used"][e][o] for o in pick), "pre": {o: code[r][o] for o in pick}})
        dev = env.device
        if stop_ok:
            finish(env, torch.tensor(stop_ok, device=dev), True)
        if stop_fail:
            finish(env, torch.tensor(stop_fail, device=dev), False)
        if go:
            issue(env, torch.tensor(go, device=dev), go_pairs, go_swaps, go_meta)

    def step(self, actions):
        env = self.unwrapped
        n, dev = env.num_envs, env.device
        if "script" not in state:
            if os.environ.get("BITIDY_VOICE_RANDOM") == "own":  # driver self-test: the env's own sampled instruction
                c0 = cb._state(env)["cmd"].tolist()
                plans = [[[(o, b) for o, b, a in c0[e] if a]] for e in range(n)]
                src, groups = "own", ["own"] * n
            else:
                plans, src, groups = _load_plans(n)
            state.update(script=_script_class(os.environ["BITIDY_SCRIPT"])(env), plans=plans,
                         n_cmd=torch.tensor([len(p) for p in plans], device=dev),
                         k=torch.zeros(n, dtype=torch.long, device=dev),
                         t=torch.zeros(n, dtype=torch.long, device=dev),
                         finished=torch.zeros(n, dtype=torch.bool, device=dev),
                         chain_ok=torch.zeros(n, dtype=torch.bool, device=dev),
                         checked=torch.zeros(n, dtype=torch.bool, device=dev),
                         pending=torch.zeros(n, dtype=torch.bool, device=dev),
                         park=torch.zeros(n, dtype=torch.long, device=dev),
                         homing=torch.zeros(n, dtype=torch.long, device=dev),
                         used=[[0] * cb.N_OBJ for _ in range(n)], att=[0] * n, defer=[[] for _ in range(n)],
                         intact={}, rec=[], src=src, steps=0, groups=groups)
            n_att = int(state["n_cmd"].max()) + retry_max * cb.N_OBJ
            need = n_att * (cmd_steps + home_steps) + check_steps + 10
            print(f"BITIDY_VOICE_CUP source={src} envs={n} retry={retry_max} swap_from={swap_from} plan[0]={plans[0]} "
                  f"plan[1]={plans[1] if n > 1 else None} n_cmd_hist={torch.bincount(state['n_cmd']).tolist()} "
                  f"max_episode_length={env.max_episode_length} worst_case_steps={need} "
                  f"residual={os.environ.get('BITIDY_RESIDUAL')}", flush=True)
            if env.max_episode_length < need:
                print(f"BITIDY_VOICE_CUP note: episode ({env.max_episode_length} steps) shorter than the worst case "
                      f"({need}); chains still running then end as env_end", flush=True)
            ids = torch.arange(n, device=dev)
            issue(env, ids, [p[0] for p in plans], [False] * n, [{"retry": 0}] * n)
            loc0 = cb._state(env)["loc"]
            if bool((loc0 >= 0).any()):
                print(f"BITIDY_VOICE_CUP WARNING {int((loc0 >= 0).any(-1).sum())} envs start with a vessel in a basket "
                      f"(BITIDY_CB_PRESORT=0 missing?)", flush=True)
            try:  # the first action must see the new command (obs_buf was computed at reset with the env's own)
                env.obs_buf = env.observation_manager.compute()
            except Exception as error:  # noqa: BLE001
                print(f"BITIDY_VOICE_CUP obs recompute failed: {error}", flush=True)
        state["steps"] += 1
        act = state["script"].act()
        homing = state["homing"] > 0
        if bool(homing.any()):
            act = torch.where(homing[:, None], home_action(env), act)
            ramp_homing(env, homing)
            state["hmask"] = homing
            state["homing"][homing & ~state["finished"]] -= 1
            done_homing = homing & (state["homing"] == 0) & ~state["finished"] & state["pending"]
            if bool(done_homing.any()):
                ids = done_homing.nonzero().squeeze(-1)
                state["pending"][ids] = False
                decide(env, ids)
        result = inner_step(self, act)
        state["hmask"] = None
        dones = result[2].bool()
        p = cb.progress(env)
        running = ~state["finished"] & (state["homing"] == 0)
        state["t"] += running.long()
        won = p.success & running & ~dones
        late = running & ~won & ~dones & (state["t"] >= cmd_steps)
        for mask, ok, why in ((won, True, ""), (late, False, "timeout")):
            if bool(mask.any()):
                ids = mask.nonzero().squeeze(-1)
                close(env, ids, ok, why)
                p.reset(ids)  # keep the finished attempt's success from re-triggering while homing
                state["homing"][ids] = home_steps
                state["pending"][ids] = True
                start_homing(env, ids)
                for e in ids.tolist():
                    if e < trace_envs:
                        print(f"BITIDY_VOICE_END env={e} k={int(state['k'][e])} env_ok={ok} t_cmd={int(state['t'][e])} "
                              f"t={int(env.episode_length_buf[e])}", flush=True)
        # env termination (dropped vessel / episode timeout): the chain stops where it is
        cut = dones & ~state["finished"]
        if bool(cut.any()):
            ids = cut.nonzero().squeeze(-1)
            try:
                why = "dropped" if bool(env.termination_manager.get_term("dropped")[ids].any()) else "env_end"
            except Exception:  # noqa: BLE001
                why = "env_end"
            act_ids = ids[state["homing"][ids] == 0]
            if len(act_ids):
                close(env, act_ids, False, why, after_reset=True)
            for e in ids.tolist():
                state["rec"][state["att"][e]]["cut"] = why
            finish(env, ids, False)
            state["checked"][ids] = True
            start_homing(env, ids)
        if bool(dones.any()):
            state["script"].reset(dones.nonzero().squeeze(-1))
        # whole-scene check once a finished chain has parked
        state["park"] += (state["finished"] & ~state["checked"]).long()
        chk = state["finished"] & ~state["checked"] & (state["park"] >= check_steps)
        if bool(chk.any()):
            ids = chk.nonzero().squeeze(-1)
            exp = torch.full((n, cb.N_OBJ), -1, dtype=torch.long, device=dev)
            for e in ids.tolist():  # partial chains: the vessels of the completed instructions
                for o, j in expect_upto(e, int(state["k"][e]) - 1).items():
                    exp[e, o] = j
            res = intact(env, exp)
            for key, v in res.items():
                state["intact"].setdefault(key, torch.zeros(n, dtype=torch.bool, device=dev))[ids] = v[ids]
            state["checked"][ids] = True
        if state["steps"] % 500 == 0:
            print(f"BITIDY_VOICE_PROGRESS steps={state['steps']} finished={int(state['finished'].sum())}/{n} "
                  f"k_hist={torch.bincount(state['k'], minlength=int(state['n_cmd'].max()) + 1).tolist()}", flush=True)
        if bool(state["checked"].all()) and "reported" not in state:
            state["reported"] = True
            dump = os.environ.get("BITIDY_VOICE_DUMP")
            if dump:  # raw per-attempt records, to re-analyse offline
                with open(os.path.expanduser(dump), "w", encoding="utf-8") as f:
                    json.dump({"rec": state["rec"], "k": state["k"].tolist(), "n_cmd": state["n_cmd"].tolist(),
                               "plans": state["plans"], "intact": {a: v.tolist() for a, v in state["intact"].items()}}, f)
            try:
                report(env)
            except Exception as error:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"BITIDY_VOICE_CUP report failed: {error}", flush=True)
            if not video:
                os._exit(0)
        return result

    def report(env):
        from collections import Counter

        n = env.num_envs
        k, ncmd = state["k"].cpu(), state["n_cmd"].cpu()
        kmax = int(ncmd.max())
        recs = state["rec"]
        res = {"source": state["src"], "envs": n, "assign": assign, "retry": retry_max, "swap_from": swap_from, "cmd_steps": cmd_steps,
               "home_steps": home_steps, "residual": os.environ.get("BITIDY_RESIDUAL"), "plan_env0": state["plans"][0]}
        first = [r for r in recs if r.get("retry", 0) == 0]
        head = [r for r in first if r.get("split", 0) != 2]  # one per (env, instruction)
        reached = [sum(1 for r in head if r["k"] == i) for i in range(kmax)]
        done_k = [int((k > i).sum()) for i in range(kmax)]
        ff = lambda r: "after" in r and not r.get("bad_after")
        retried = {(r["e"], r["k"]) for r in recs if r.get("retry", 0) > 0}
        res["per_cmd_reached"] = reached
        res["per_cmd_completion"] = [round(d / max(a, 1), 4) for d, a in zip(done_k, reached)]
        res["per_cmd_first_try"] = [round(sum(1 for r in head if r["k"] == i and int(k[r["e"]]) > i and (r["e"], i) not in retried)
                                          / max(reached[i], 1), 4) for i in range(kmax)]
        res["split_instructions"] = sum(1 for r in head if r.get("split"))
        res["per_cmd_env_predicate_first_try"] = [
            round(sum(1 for r in head if r["k"] == i and r.get("env_ok")) / max(reached[i], 1), 4) for i in range(kmax)]
        res["completed_at_least"] = [round(float((k >= i).float().mean()), 4) for i in range(1, kmax + 1)]
        res["all_done"] = round(float((k >= ncmd).float().mean()), 4)
        done = k >= ncmd
        it = {key: v.cpu() for key, v in state["intact"].items()}
        if it:
            res["all_done_and_intact_after_park"] = round(float((done & it["all"]).float().mean()), 4)
            res["intact_after_park_given_all_done"] = {key: round(float(v[done].float().mean()), 4) for key, v in it.items()}
        if bool((ncmd != ncmd[0]).any()):
            res["all_done_by_n_cmd(rate,n)"] = {int(m): [round(float(done[ncmd == m].float().mean()), 4), int((ncmd == m).sum())]
                                               for m in ncmd.unique()}
        # per pair, first attempts: env per-arm latch and the post-homing status check; by basket / type / depth / arm
        tabs = {"basket_ABC": 3, "type_cup_bowl": 2, "depth_012": 3, "arm_LR": 2}
        cnt = {key: [[0, 0, 0] for _ in range(m)] for key, m in tabs.items()}
        for r in first:
            for a, obj, b, d in r["slots"]:
                ok_env = bool(r.get("slot_ok", {}).get(a, False))
                ok_chk = r.get("after", {}).get(obj, -1) == OK
                for key, i in (("basket_ABC", b), ("type_cup_bowl", cb.OBJ_TYPE[obj]), ("depth_012", min(d, 2)), ("arm_LR", a)):
                    cnt[key][i][0] += 1
                    cnt[key][i][1] += int(ok_env)
                    cnt[key][i][2] += int(ok_chk)
        for key, tab in cnt.items():
            res["first_try_pair_" + key + "(n,env_latch,status_ok)"] = [[a, round(s / max(a, 1), 4), round(c / max(a, 1), 4)]
                                                                      for a, s, c in tab]
        by = Counter()
        for r in first:
            by[(r["npair"], 0)] += 1
            by[(r["npair"], 1)] += int(ff(r))
        res["first_try_by_size(n,rate)"] = {"single": [by[(1, 0)], round(by[(1, 1)] / max(by[(1, 0)], 1), 4)],
                                            "pair": [by[(2, 0)], round(by[(2, 1)] / max(by[(2, 0)], 1), 4)]}
        ts = [r["t"] for r in first if r.get("env_ok")]
        if ts:
            ts_t = torch.tensor(ts, dtype=torch.float)
            res["cmd_steps_env_ok_p50_p90"] = [float(ts_t.median()), float(ts_t.quantile(0.9))]
        band = {}
        for r in first:
            for a, obj, b, d in r["slots"]:
                da = r.get("dist", {}).get(a)
                if da is None:
                    continue
                key = "in_band" if cb.OBJ_BAND[0] - 0.01 <= da[a] <= cb.OBJ_BAND[1] + 0.01 else ("near" if da[a] < cb.OBJ_BAND[0] else "far")
                v = band.setdefault(key, [0, 0])
                v[0] += 1
                v[1] += int(r.get("after", {}).get(obj, -1) == OK)
        res["first_try_pair_by_assigned_arm_dist(n,ok)"] = band
        stat = Counter()
        for r in first:
            for o, c in r.get("after", {}).items():
                stat[STATUS[c]] += 1
        res["first_try_pair_status_after"] = dict(stat)
        res["first_try_earlier_vessel_disturbed"] = sum(
            1 for r in first if any(o not in [s[1] for s in r["slots"]] for o in r.get("bad_after", {})))
        hist = [0] * 6
        for r in first:
            if r.get("env_ok") is False and "flags" in r:
                for a, obj, b, d in r["slots"]:
                    if not r["slot_ok"].get(a):
                        f = r["flags"][5 * a: 5 * a + 5]
                        hist[max([i + 1 for i, v in enumerate(f) if v] or [0])] += 1
        res["first_try_failed_slot_furthest_stage(none,grasp,lift,over,placed,released)"] = hist
        res["first_try_fail_other_moved"] = sum(1 for r in first if r.get("env_ok") is False and "flags" in r and r["flags"][-1])
        res["first_try_env_ok_but_status_bad"] = sum(1 for r in first if r.get("env_ok") and "after" in r and r.get("bad_after"))
        res["first_try_timeout_but_status_ok"] = sum(1 for r in first if r.get("env_ok") is False and ff(r))
        res["cut"] = dict(Counter(r["cut"] for r in recs if "cut" in r))
        rt = [r for r in recs if r.get("retry", 0) > 0]
        if rt:
            out = {}
            for r in rt:
                for o, pre in r["pre"].items():
                    key = f"{STATUS[pre]}_retry{r['retry']}{'_swap' if r['swap'] else ''}"
                    v = out.setdefault(key, [0, 0])
                    v[0] += 1
                    v[1] += int(r.get("after", {}).get(o, -1) == OK)
            res["retry_attempts"] = len(rt)
            res["retry_obj_fixed(pre_status_retryN[_swap]: n, fixed)"] = out
            used = {r["e"] for r in rt}
            res["envs_using_retry"] = len(used)
            res["all_done_among_envs_using_retry"] = sum(1 for e in used if bool(done[e]))
        gl = state["groups"]
        if len(set(gl)) > 1:
            res["by_group"] = {}
            for gname in sorted(set(gl)):
                m = torch.tensor([x == gname for x in gl])
                kg, ng = k[m], ncmd[m]
                e0 = gl.index(gname)
                res["by_group"][gname] = {
                    "n": int(m.sum()), "plan_first_env": state["plans"][e0],
                    "completed_at_least": [round(float((kg >= i).float().mean()), 4) for i in range(1, int(ng.max()) + 1)],
                    "all_done": round(float((kg >= ng).float().mean()), 4),
                    "all_done_and_intact": round(float(((kg >= ng) & it["all"][m]).float().mean()), 4) if it else None}
        stop = Counter(int(k[e]) for e in range(n) if not bool(done[e]))
        res["chain_stopped_at_cmd"] = {int(a): b for a, b in sorted(stop.items())}
        print("BITIDY_VOICE_CUP_RESULT " + json.dumps(res, ensure_ascii=False), flush=True)

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
