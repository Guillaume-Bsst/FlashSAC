"""Vérifie le critère de succès corrigé à E0 sur le simulateur (64 envs, départ au début du clip, ANT_LAB_REF).

- politique nulle (le robot agite les pattes sur place) : écart de vitesse du torse de l'ordre de la vitesse de la
  référence, donc échec ;
- robot rejoué sur la référence (état de la référence écrit à chaque pas) : écart proche de 0, donc succès.
usage : ANT_LAB_REF=results/ant_lab/refs/ref_s1.npz ant_lab/run.sh -m ant_lab.checks.check_success_criterion
"""
import os

from ant_lab.sim import check_gpu_free, launch_app, run_main

N = 64


def main():
    os.environ["ANT_LAB_ACTION_MODE"] = "residual"
    os.environ["ANT_LAB_RSI"] = "start"
    check_gpu_free()
    app = launch_app()
    import torch

    from ant_lab.config import SUCCESS
    from ant_lab.registry import TRACK_ID
    from ant_lab.sim import make_env

    env = make_env(TRACK_ID, N, seed=0)
    u = env.unwrapped
    for mode in ("politique nulle", "rejeu de la référence"):
        env.reset()
        vel = torch.full((N,), float("nan"), device=u.device)
        done = torch.full((N,), float("nan"), device=u.device)
        for _ in range(u.ref_T + 5):
            if mode == "rejeu de la référence":
                u.reset_to_frame(u.robot._ALL_INDICES, u.frame.clone())   # téléporte sur la référence
            u.last_episode_stats = {}
            env.step(torch.zeros(N, 8, device=u.device))
            s = u.last_episode_stats
            if s:
                ids = s["env_ids"]
                fresh = torch.isnan(vel[ids])
                vel[ids[fresh]] = s["mean_torso_vel_err"][fresh]
                done[ids[fresh]] = s["completed"][fresh]
            if not torch.isnan(vel).any():
                break
        v = float(vel.nanmean())
        print(f"check_success_criterion : {mode} → écart de vitesse du torse {v:.2f} m/s, "
              f"fin de clip {100 * float(done.nanmean()):.0f} %, critère vitesse "
              f"{'OK' if v < SUCCESS['max_torso_vel_err'] else 'échec'}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    run_main(main)
