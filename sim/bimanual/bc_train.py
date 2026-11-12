"""Behaviour cloning of an RSL-RL actor from oracle demonstrations (``demos.py``).

    python -m isaaclab_tutorial.tasks.bimanual.bc_train <demos.npz[,more.npz]> <template model_*.pt> <out model.pt>
        [--epochs 30] [--std 0.1]

The template checkpoint (e.g. a 1-iteration run of the same task) fixes the network layout. The actor MLP is trained
by MSE on the oracle actions, the actor and critic observation normalisers are set from the demonstration data, and
the action std is set to ``--std``, so PPO can resume from the result with ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("demos")
    p.add_argument("template")
    p.add_argument("out")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--std", type=float, default=0.1)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Several datasets (comma-separated, e.g. oracle demos plus DAgger rounds) are concatenated.
    obs, act, episode, offset = [], [], [], 0
    for path in args.demos.split(","):
        data = np.load(path)
        obs.append(torch.from_numpy(data["obs"]))
        act.append(torch.from_numpy(data["act"]))
        episode.append(torch.from_numpy(data["episode"]).long() + offset)
        offset = int(episode[-1].max()) + 1
    obs, act, episode = torch.cat(obs).to(dev), torch.cat(act).to(dev), torch.cat(episode).to(dev)
    ck = torch.load(args.template, map_location=dev, weights_only=False)
    sd = ck["actor_state_dict"]

    # Normaliser statistics from the demonstrations (the same for actor and critic).
    mean, var = obs.mean(0, keepdim=True), obs.var(0, keepdim=True)
    for key in ("actor_state_dict", "critic_state_dict"):
        ck[key]["obs_normalizer._mean"] = mean.clone()
        ck[key]["obs_normalizer._var"] = var.clone()
        ck[key]["obs_normalizer._std"] = var.sqrt().clone()
        ck[key]["obs_normalizer.count"] = torch.tensor(len(obs), device=dev)
    std = var.sqrt().clamp(min=1e-2)

    # Actor MLP rebuilt from the state dict (Linear layers at even indices, ELU between them).
    linear = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for i, idx in enumerate(linear):
        w = sd[f"mlp.{idx}.weight"]
        layer = torch.nn.Linear(w.shape[1], w.shape[0]).to(dev)
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(sd[f"mlp.{idx}.bias"])
        layers.append(layer)
        if i < len(linear) - 1:
            layers.append(torch.nn.ELU())
    net = torch.nn.Sequential(*layers)

    # Hold out 5% of the episodes.
    n_ep = int(episode.max()) + 1
    val_ep = torch.randperm(n_ep, device=dev)[: max(1, n_ep // 20)]
    val = torch.isin(episode, val_ep)
    xtr, ytr, xva, yva = obs[~val], act[~val], obs[val], act[val]
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"BC data: {len(obs)} steps, {n_ep} episodes, obs {obs.shape[1]}, act {act.shape[1]}", flush=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(xtr), device=dev)
        total = 0.0
        for b in range(0, len(perm), args.batch):
            i = perm[b : b + args.batch]
            loss = torch.nn.functional.mse_loss(net((xtr[i] - mean) / std), ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(i)
        sched.step()
        with torch.no_grad():
            pred = net((xva - mean) / std)
            val_loss = torch.nn.functional.mse_loss(pred, yva).item()
            grip_ok = ((pred[:, [5, 11]] > 0) == (yva[:, [5, 11]] > 0)).float().mean().item()
        print(f"epoch {epoch + 1}/{args.epochs} train {total / len(xtr):.4f} val {val_loss:.4f} gripper-sign {grip_ok:.3f}", flush=True)

    for i, idx in enumerate(linear):
        layer = net[2 * i]
        sd[f"mlp.{idx}.weight"] = layer.weight.data.clone()
        sd[f"mlp.{idx}.bias"] = layer.bias.data.clone()
    sd["distribution.log_std_param"] = torch.full_like(sd["distribution.log_std_param"], math.log(args.std))
    ck["iter"] = 0
    torch.save(ck, args.out)
    print(f"BC_SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
