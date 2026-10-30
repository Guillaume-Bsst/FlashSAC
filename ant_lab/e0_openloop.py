"""E0 : rejeu en boucle ouverte d'une référence (PD qui suit q_ref, sans politique), départs RSI uniformes.

Variables fournies par l'appelant : ANT_LAB_REF, ANT_LAB_K. Écrit une ligne dans RESULTS/e0/openloop.jsonl.
usage : ANT_LAB_REF=... ANT_LAB_K=... ant_lab/run.sh -m ant_lab.e0_openloop
"""
import json
import os
from pathlib import Path

from ant_lab.sim import check_gpu_free, launch_app, run_main

NUM_ENVS = 512


def main():
    os.environ["ANT_LAB_ACTION_MODE"] = "residual"
    os.environ["ANT_LAB_RSI"] = "uniform"
    check_gpu_free()
    app = launch_app()
    import torch

    from ant_lab.config import FPS, RESULTS
    from ant_lab.registry import TRACK_ID
    from ant_lab.sim import make_env

    env = make_env(TRACK_ID, NUM_ENVS, seed=0)
    u = env.unwrapped
    env.reset()
    zeros = torch.zeros(NUM_ENVS, 8, device=u.device)
    length = torch.full((NUM_ENVS,), float("nan"), device=u.device)
    completed = torch.full((NUM_ENVS,), float("nan"), device=u.device)
    err = torch.full((NUM_ENVS,), float("nan"), device=u.device)
    for _ in range(u.ref_T + 5):
        u.last_episode_stats = {}
        env.step(zeros)
        s = u.last_episode_stats
        if s:
            ids = s["env_ids"]
            fresh = torch.isnan(length[ids])
            length[ids[fresh]] = s["length"][fresh]
            completed[ids[fresh]] = s["completed"][fresh]
            err[ids[fresh]] = s["mean_joint_err"][fresh]
        if not torch.isnan(length).any():
            break
    row = {"ref": Path(os.environ["ANT_LAB_REF"]).stem, "k": float(os.environ["ANT_LAB_K"]),
           "mean_survival_s": float(length.nanmean()) / FPS, "frac_complete": float(completed.nanmean()),
           "mean_joint_err": float(err.nanmean()), "n_episodes": int((~torch.isnan(length)).sum())}
    out = RESULTS / "e0"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "openloop.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("e0_openloop :", row, flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    run_main(main)
