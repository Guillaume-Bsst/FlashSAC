"""Cinématique directe d'une référence : pose de chaque corps, calculée par le simulateur lui-même.

usage : ant_lab/run.sh -m ant_lab.fk_fill --ref IN.npz --out OUT.npz
"""
import argparse

from ant_lab.sim import check_gpu_free, launch_app, run_main


def fill_bodies(env, ref):
    """``env`` : env expert brut avec ``num_envs == ref.num_frames`` ; l'env i reçoit la frame i."""
    import torch

    from ant_lab.reference import with_bodies

    u = env.unwrapped
    robot = u.robot
    assert u.num_envs == ref.num_frames, (u.num_envs, ref.num_frames)
    assert list(ref.joint_names) == list(robot.joint_names), (ref.joint_names, robot.joint_names)
    dev = u.device
    origins = u.scene.env_origins
    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    pose = torch.cat([f(ref.root_pos) + origins, f(ref.root_quat)], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_link_velocity_to_sim(torch.zeros(u.num_envs, 6, device=dev))
    robot.write_joint_state_to_sim(f(ref.joint_pos), torch.zeros_like(f(ref.joint_pos)))
    u.sim.forward()
    robot.update(u.physics_dt)
    d = robot.data
    o = origins[:, None, :]
    return with_bodies(
        ref, list(robot.body_names),
        (d.body_link_pos_w - o).double().cpu().numpy(), d.body_link_quat_w.double().cpu().numpy(),
        (d.body_com_pos_w - o).double().cpu().numpy(), d.body_com_quat_w.double().cpu().numpy(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    check_gpu_free()
    app = launch_app()
    from ant_lab.reference import load_reference, save_reference
    from ant_lab.registry import EXPERT_ID
    from ant_lab.sim import make_env

    ref = load_reference(a.ref)
    env = make_env(EXPERT_ID, ref.num_frames)
    env.reset()
    out = fill_bodies(env, ref)
    save_reference(a.out, out)
    print(f"fk_fill : {out.num_frames} frames, {len(out.body_names)} corps -> {a.out}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    run_main(main)
