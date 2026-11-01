"""Évaluateur commun : note chaque checkpoint d'un run (PPO ou FlashSAC) avec le même juge.

Départ au début du clip avec un petit bruit (ANT_LAB_RSI=start), politique déterministe, un épisode par env.
usage : ant_lab/run.sh -m ant_lab.evaluate_runs --run-dir D [--num-envs 64]
Ne jamais lancer pendant un run chronométré.
"""
import argparse
import json
import os
from pathlib import Path

from ant_lab.sim import check_gpu_free, launch_app, run_main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--num-envs", type=int, default=64)
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    os.environ.update(meta["env"])
    os.environ["ANT_LAB_RSI"] = "start"
    check_gpu_free()
    app = launch_app()
    import torch

    from ant_lab.metrics import EvalRow, list_checkpoints, read_rows, write_rows
    from ant_lab.registry import TRACK_ID
    from ant_lab.sim import make_env

    algo, N = meta["algo"], a.num_envs
    csv_path = run_dir / "eval.csv"
    done_names = {r.ckpt for r in read_rows(csv_path)} if csv_path.exists() else set()
    if algo == "ppo":
        ckpts = list_checkpoints(run_dir, "ppo", meta["num_envs"], meta["steps_per_env"])
    else:
        ckpts = list_checkpoints(run_dir, "flashsac", meta["num_envs"])
    todo = [c for c in ckpts if c[0].name not in done_names]
    print(f"evaluate_runs : {algo} {run_dir} : {len(todo)} checkpoints à évaluer", flush=True)
    if not todo:
        app.close()
        return

    gym_env = make_env(TRACK_ID, N, seed=1234)
    u = gym_env.unwrapped
    if algo == "ppo":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from ant_lab.policies import RslPolicy
        from ant_lab.ppo_cfg import train_cfg

        env = RslRlVecEnvWrapper(gym_env, clip_actions=None)
        policy = RslPolicy(env, train_cfg(meta["max_iterations"], meta["save_interval"], meta["seed"],
                                          meta["exp"], f"seed{meta['seed']}"))
        reset = lambda: env.reset()[0]  # noqa: E731
        step = lambda obs, act: env.step(act)[0]  # noqa: E731
    else:
        from ant_lab.policies import FlashSACPolicy

        policy = FlashSACPolicy(meta, u.cfg.observation_space, u.cfg.action_space)
        reset = lambda: gym_env.reset()[0]["policy"]  # noqa: E731
        step = lambda obs, act: gym_env.step(act)[0]["policy"]  # noqa: E731

    for path, mtime, env_steps in todo:
        policy.load(path)
        torch.manual_seed(0)
        obs = reset()
        completed = torch.full((N,), float("nan"), device=u.device)
        joint_err = torch.full((N,), float("nan"), device=u.device)
        vel_err = torch.full((N,), float("nan"), device=u.device)
        for _ in range(2 * u.ref_T + 10):
            u.last_episode_stats = {}
            obs = step(obs, policy(obs))
            s = u.last_episode_stats
            if s:
                ids = s["env_ids"]
                fresh = torch.isnan(completed[ids])  # seul le premier épisode de chaque env compte
                completed[ids[fresh]] = s["completed"][fresh]
                joint_err[ids[fresh]] = s["mean_joint_err"][fresh]
                vel_err[ids[fresh]] = s["mean_torso_vel_err"][fresh]
            if not torch.isnan(completed).any():
                break
        row = EvalRow(algo, meta["seed"], path.name, mtime - meta["start_time"], env_steps,
                      float(completed.nanmean()), float(joint_err.nanmean()), float(vel_err.nanmean()))
        write_rows(csv_path, [row])
        print(f"  {row}", flush=True)
    gym_env.close()
    app.close()


if __name__ == "__main__":
    run_main(main)
