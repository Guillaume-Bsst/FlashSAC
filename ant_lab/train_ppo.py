"""Lance PPO (rsl_rl 3.0.1, recette BeyondMimic) sur l'env de tracking.

usage : ant_lab/run.sh -m ant_lab.train_ppo --exp NAME --seed S [--num-envs 4096] [--max-iterations 100000]
        [--save-interval 50]
"""
import argparse
import json
import os
import time
from pathlib import Path

from ant_lab.sim import check_gpu_free, launch_app, run_main

ROOT = Path(__file__).resolve().parent.parent
STEPS_PER_ENV = 24


def main(start_time: float):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=100_000)
    p.add_argument("--save-interval", type=int, default=50)
    a = p.parse_args()

    from ant_lab.config import RESULTS
    from ant_lab.ppo_cfg import train_cfg

    run_dir = ROOT / RESULTS / "runs" / "ppo" / a.exp / f"seed{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"algo": "ppo", "env_id": "AntLab-Track-v0", "exp": a.exp, "seed": a.seed, "num_envs": a.num_envs,
            "start_time": start_time, "steps_per_env": STEPS_PER_ENV,
            "save_interval": a.save_interval, "max_iterations": a.max_iterations,
            "env": {k: v for k, v in os.environ.items() if k.startswith("ANT_LAB_")}}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))

    check_gpu_free()
    app = launch_app()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from ant_lab.registry import TRACK_ID
    from ant_lab.sim import make_env

    env = RslRlVecEnvWrapper(make_env(TRACK_ID, a.num_envs, seed=a.seed), clip_actions=None)
    cfg = train_cfg(a.max_iterations, a.save_interval, a.seed, a.exp, f"seed{a.seed}")
    runner = OnPolicyRunner(env, cfg, log_dir=str(run_dir), device="cuda:0")
    runner.learn(a.max_iterations, init_at_random_ep_len=True)
    env.close()
    app.close()


if __name__ == "__main__":
    t0 = time.time()
    run_main(lambda: main(t0))
