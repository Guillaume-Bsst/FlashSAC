"""Lance FlashSAC (train.run du dépôt) sur un env du labo, avec la recette des runs G1 (run_g1.sh).

usage : ant_lab/run.sh -m ant_lab.train_flashsac --env-id ID --exp NAME --seed S [--num-envs 2048]
        [--steps 1_000_000_000] [--buffer 2_000_000] [--ckpt-every 200]
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_overrides(env_id: str, exp: str, seed: int, num_envs: int, steps: int, buffer: int,
                    ckpt_every: int, save_path: str) -> list[str]:
    return [
        "group_name=ant_lab", f"exp_name={exp}", f"seed={seed}", "env=isaaclab", f"env.env_name={env_id}",
        f"num_env_steps={steps}", f"num_train_envs={num_envs}", "num_eval_envs=null", "num_record_envs=null",
        f"num_eval_episodes={num_envs}", "num_record_episodes=0", "agent=flashSAC",
        f"agent.buffer_max_length={buffer}", f"agent.buffer_min_length={num_envs * 20}",
        "agent.sample_batch_size=2048", "agent.use_amp=true", "updates_per_interaction_step=2",
        "agent.asymmetric_observation=false", "gamma=0.99", "n_step=3", "logger_type=tensorboard",
        "evaluation_per_interaction_step=null", "recording_per_interaction_step=null",
        f"save_checkpoint_per_interaction_step={ckpt_every}", f"save_path={save_path}",
    ]


def run_dir_for(exp: str, seed: int) -> Path:
    from ant_lab.config import RESULTS

    return RESULTS / "runs" / "flashsac" / exp / f"seed{seed}"


def main():
    start_time = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--steps", type=lambda s: int(s.replace("_", "")), default=1_000_000_000)
    p.add_argument("--buffer", type=lambda s: int(s.replace("_", "")), default=2_000_000)
    p.add_argument("--ckpt-every", type=int, default=200)
    a = p.parse_args()

    from ant_lab.registry import register
    from ant_lab.sim import check_gpu_free

    run_dir = ROOT / run_dir_for(a.exp, a.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.relpath(run_dir / "ckpt", start=ROOT / "ant_lab")
    overrides = build_overrides(a.env_id, a.exp, a.seed, a.num_envs, a.steps, a.buffer, a.ckpt_every, save_path)
    meta = {"algo": "flashsac", "env_id": a.env_id, "exp": a.exp, "seed": a.seed, "num_envs": a.num_envs,
            "start_time": start_time, "overrides": overrides,
            "env": {k: v for k, v in os.environ.items() if k.startswith("ANT_LAB_")}}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))

    check_gpu_free()
    register()
    import train

    train.run(argparse.Namespace(config_path="./configs", config_name="flashSAC_base", overrides=overrides))


if __name__ == "__main__":
    main()
