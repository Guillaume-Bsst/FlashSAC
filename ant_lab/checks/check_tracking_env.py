"""Vérifie l'env de tracking : reward max sur la référence (env 0 et dernier env), fin de clip, formes, stabilité."""
import os

from ant_lab.sim import check_gpu_free, launch_app, run_main


def main():
    os.environ["ANT_LAB_REF"] = "results/ant_lab/refs/smoke_ref.npz"
    os.environ["ANT_LAB_K"] = "1.0"
    os.environ["ANT_LAB_ACTION_MODE"] = "residual"
    os.environ["ANT_LAB_RSI"] = "uniform"
    check_gpu_free()
    app = launch_app()
    import torch

    from ant_lab.registry import TRACK_ID
    from ant_lab.sim import make_env
    from ant_lab.tracking_math import total_reward

    N = 8
    env = make_env(TRACK_ID, N)
    env.reset()
    u = env.unwrapped
    T = u.ref_T
    all_ids = torch.arange(N, device=u.device)
    fails = []

    # 1. robot posé exactement sur la référence -> reward maximale, env 0 et dernier env
    u.reset_to_frame(all_ids, torch.zeros(N, dtype=torch.long, device=u.device))
    u.sim.forward()
    u.robot.update(u.physics_dt)
    terms = u.current_reward_terms()
    # la pénalité de limite articulaire dépend de la référence elle-même (rapportée à part) : on teste le suivi
    r = total_reward({**terms, "joint_limit": torch.zeros_like(terms["joint_limit"])})
    print("1. reward de suivi sur la référence :", r.tolist(), flush=True)
    print("   termes env 0 :", {k: round(float(v[0]), 4) for k, v in terms.items()}, flush=True)
    if not (r[0] >= 4.95 and r[N - 1] >= 4.95):
        fails.append(f"reward sur la référence {float(r[0]):.3f} / {float(r[N - 1]):.3f} < 4.95")

    # 2. fin de clip : départ à T-3 -> pas de time-out après 1 pas, terminé ou time-out après 2
    u.episode_length_buf[:] = torch.randint(0, 1000, (N,), device=u.device)
    u.reset_to_frame(all_ids, torch.full((N,), T - 3, dtype=torch.long, device=u.device))
    zeros = torch.zeros(N, 8, device=u.device)
    # IsaacLab renvoie les mêmes tenseurs à chaque pas : on les copie
    _, _, term1, trunc1, _ = env.step(zeros)
    term1, trunc1 = term1.clone(), trunc1.clone()
    obs, rew, term2, trunc2, _ = env.step(zeros)
    print("2. après 1 pas : time-out", trunc1.tolist(), "; après 2 pas : term", term2.tolist(),
          "time-out", trunc2.tolist(), flush=True)
    if trunc1.any():
        fails.append("time-out dès le 1er pas depuis T-3")
    # un env terminé au 1er pas a été remis à zéro : on ne regarde que ceux qui ont survécu
    alive = ~(term1 | trunc1)
    if not bool(((term2 | trunc2) | ~alive).all()):
        fails.append("après 2 pas depuis T-3, un env n'est ni terminé ni en time-out")

    # 3. formes
    if tuple(obs["policy"].shape) != (N, 55) or not torch.isfinite(rew).all():
        fails.append(f"observation {tuple(obs['policy'].shape)} ou reward non finie")

    # 4. stabilité et statistiques d'épisode
    for _ in range(300):
        obs, rew, _, _, _ = env.step(zeros)
        if not (torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()):
            fails.append("NaN pendant 300 pas")
            break
    s = u.last_episode_stats
    print("4. last_episode_stats :", {k: v.tolist()[:8] for k, v in s.items()}, flush=True)
    if len(s.get("env_ids", [])) == 0:
        fails.append("last_episode_stats vide")
    elif not (set(s["completed"].tolist()) <= {0.0, 1.0, 0, 1, True, False} and (s["mean_joint_err"] >= 0).all()):
        fails.append("last_episode_stats incohérent")

    env.close()
    if fails:
        print("check_tracking_env ÉCHEC :", *fails, sep="\n  ", flush=True)
        os._exit(1)
    print("check_tracking_env OK", flush=True)
    app.close()


if __name__ == "__main__":
    run_main(main)
