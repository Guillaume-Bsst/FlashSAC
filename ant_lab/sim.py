"""Démarrage d'IsaacSim et création des envs. ``launch_app`` doit précéder tout import d'isaaclab."""
import subprocess


def check_gpu_free(min_free_gb: float = 4.0) -> None:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    free_gb = int(out.strip().splitlines()[0]) / 1024
    if free_gb < min_free_gb:
        raise SystemExit(
            f"GPU : {free_gb:.1f} Go libres, il en faut au moins {min_free_gb:.1f}. "
            "D'autres entraînements occupent la carte : on attend qu'ils finissent (on ne les tue pas)."
        )


def launch_app(headless: bool = True):
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=headless).app


def make_env(env_id: str, num_envs: int, seed: int = 0, device: str = "cuda:0"):
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from ant_lab.registry import register

    register()
    cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
    cfg.seed = seed
    return gym.make(env_id, cfg=cfg)
