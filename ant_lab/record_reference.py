"""Enregistre la cinématique (et les couples) de l'expert sur un clip de marche.

usage : ant_lab/run.sh -m ant_lab.record_reference --run-dir D [--ckpt C] --out F.npz
        [--warmup-s 2.0] [--duration-s 10.0] [--no-accept]
"""
import argparse
import json
import math
import os
from pathlib import Path

from ant_lab.sim import check_gpu_free, launch_app, run_main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--warmup-s", type=float, default=2.0)
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--no-accept", action="store_true")
    a = p.parse_args()
    check_gpu_free()
    app = launch_app()
    import numpy as np
    import torch

    from ant_lab.config import FPS
    from ant_lab.metrics import list_checkpoints
    from ant_lab.policies import FlashSACPolicy
    from ant_lab.reference import from_kinematics, save_reference
    from ant_lab.registry import EXPERT_ID
    from ant_lab.sim import make_env

    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    ckpt = Path(a.ckpt) if a.ckpt else list_checkpoints(run_dir, "flashsac", meta["num_envs"])[-1][0]
    env = make_env(EXPERT_ID, 1)
    u = env.unwrapped
    obs, _ = env.reset()
    policy = FlashSACPolicy(meta, obs["policy"].shape[-1], 8)
    policy.load(ckpt)
    robot = u.robot
    origin = u.scene.env_origins[0]
    warm = round(a.warmup_s * FPS)
    T = round(a.duration_s * FPS) + 1
    rec = {"root_pos": [], "root_quat": [], "joint_pos": [], "torque": []}
    terminated_any = False
    for i in range(warm + T):
        obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
        if i >= warm:
            terminated_any |= bool(term[0] or trunc[0])
            rec["root_pos"].append((robot.data.root_link_pos_w[0] - origin).cpu().double().numpy())
            rec["root_quat"].append(robot.data.root_link_quat_w[0].cpu().double().numpy())
            rec["joint_pos"].append(robot.data.joint_pos[0].cpu().double().numpy())
            rec["torque"].append(robot.data.applied_torque[0].cpu().double().numpy())
    arr = {k: np.stack(v) for k, v in rec.items()}
    ref = from_kinematics(FPS, list(robot.joint_names), arr["root_pos"], arr["root_quat"], arr["joint_pos"],
                          applied_torque=arr["torque"])
    save_reference(a.out, ref)
    speed = (arr["root_pos"][-1, 0] - arr["root_pos"][0, 0]) / a.duration_s
    q = torch.as_tensor(arr["root_quat"])
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)).numpy()
    yaw = np.unwrap(yaw)
    drift = float(abs(yaw[-1] - yaw[0]))
    print(f"record_reference : ckpt={ckpt} T={T} vitesse={speed:.2f} m/s dérive_lacet={drift:.2f} rad "
          f"terminaison={terminated_any} couple_max={np.abs(arr['torque']).max():.2f} N·m -> {a.out}", flush=True)
    env.close()
    if not a.no_accept and (terminated_any or speed < 1.0 or drift >= 0.5 or not math.isfinite(speed)):
        print("record_reference : clip REFUSÉ (terminaison, vitesse < 1,0 m/s ou dérive de lacet ≥ 0,5 rad)",
              flush=True)
        os._exit(1)
    app.close()


if __name__ == "__main__":
    run_main(main)
