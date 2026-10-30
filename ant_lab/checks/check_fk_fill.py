"""Vérifie fill_bodies : sur une référence synthétique, le corps racine doit retomber sur la pose de la racine."""
import json
import os

from ant_lab.sim import check_gpu_free, launch_app, run_main


def main():
    check_gpu_free()
    app = launch_app()
    import numpy as np

    from ant_lab.config import FPS, RESULTS
    from ant_lab.fk_fill import fill_bodies
    from ant_lab.reference import from_kinematics
    from ant_lab.registry import EXPERT_ID
    from ant_lab.sim import make_env

    model = json.loads((RESULTS / "ant_model.json").read_text())
    T = 3
    root_pos = np.tile([0.0, 0.0, 0.6], (T, 1))
    root_pos[:, 0] = [0.0, 0.1, 0.2]
    root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    joint_pos = np.tile(model["default_joint_pos"], (T, 1))
    ref = from_kinematics(FPS, model["joint_names"], root_pos, root_quat, joint_pos)
    env = make_env(EXPERT_ID, T)
    env.reset()
    out = fill_bodies(env, ref)
    ok = out.has_bodies and out.body_pos.shape == (T, len(model["body_names"]), 3)
    ok = ok and np.allclose(out.body_pos[:, 0], root_pos, atol=1e-4)
    dots = np.abs((out.body_quat[:, 0] * root_quat).sum(-1))
    ok = ok and np.allclose(dots, 1.0, atol=1e-4)
    print("body_pos[:, 0] =", out.body_pos[:, 0].tolist(), flush=True)
    print("check_fk_fill OK" if ok else "check_fk_fill ÉCHEC", flush=True)
    env.close()
    if not ok:
        os._exit(1)
    app.close()


if __name__ == "__main__":
    run_main(main)
