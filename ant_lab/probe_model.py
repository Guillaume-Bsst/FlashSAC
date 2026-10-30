"""Sonde du modèle Ant dans IsaacLab : noms, masses, inerties, pieds, API PhysX. Écrit RESULTS/ant_model.json."""
import json
import os

from ant_lab.sim import check_gpu_free, launch_app, run_main

PHYSX_API = ["get_generalized_mass_matrices", "get_jacobians", "get_gravity_compensation_forces",
             "get_coriolis_and_centrifugal_compensation_forces"]


def main():
    check_gpu_free()
    app = launch_app()
    import torch

    from ant_lab.config import FOOT_RADIUS, FOOT_TIP_OFFSETS, RESULTS
    from ant_lab.feasibility import foot_tip_centers
    from ant_lab.registry import EXPERT_ID
    from ant_lab.sim import make_env

    env = make_env(EXPERT_ID, 4)
    env.reset()
    u = env.unwrapped
    r = u.robot
    zeros = torch.zeros(u.num_envs, 8, device=u.device)
    for _ in range(120):
        env.step(zeros)

    body_names = list(r.body_names)
    joint_names = list(r.joint_names)
    foot_names = [b for b in body_names if b.endswith("_foot")]
    foot_ids = [body_names.index(b) for b in foot_names]
    view = r.root_physx_view
    masses = view.get_masses()[0].cpu()
    inertias = view.get_inertias()[0].cpu().reshape(-1, 3, 3)
    api = {}
    for name in PHYSX_API:
        if not hasattr(view, name):
            api[name] = "missing"
            continue
        try:
            api[name] = list(getattr(view, name)().shape)
        except Exception as e:  # noqa: BLE001
            api[name] = f"error: {e!r}"[:300]

    origin = u.scene.env_origins[0].cpu()
    bp = (r.data.body_link_pos_w[0].cpu() - origin).numpy()[None]
    bq = r.data.body_link_quat_w[0].cpu().numpy()[None]
    offsets = [FOOT_TIP_OFFSETS[b] for b in foot_names]
    tips = foot_tip_centers(bp, bq, foot_ids, offsets)[0]
    clearance = (tips[:, 2] - FOOT_RADIUS).tolist()

    out = {
        "body_names": body_names, "joint_names": joint_names, "foot_body_names": foot_names, "foot_ids": foot_ids,
        "masses": masses.tolist(), "total_mass": float(masses.sum()), "inertias": inertias.tolist(),
        "default_joint_pos": r.data.default_joint_pos[0].cpu().tolist(),
        "joint_pos_limits": r.data.joint_pos_limits[0].cpu().tolist(),
        "foot_tip_offsets": offsets, "foot_radius": FOOT_RADIUS, "rest_tip_clearance_m": clearance,
        "rest_torso_z": float(r.data.root_link_pos_w[0, 2] - origin[2]),
        "physx_api": api,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ant_model.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "inertias"}, indent=1), flush=True)

    errors = []
    if len(foot_names) != 4:
        errors.append(f"{len(foot_names)} pieds trouvés au lieu de 4 : {body_names}")
    if len(joint_names) != 8:
        errors.append(f"{len(joint_names)} articulations au lieu de 8")
    if any(abs(c) > 0.015 for c in clearance):
        errors.append(f"bouts de pieds pas au sol au repos : {clearance} (offsets ou repères faux)")
    env.close()
    if errors:
        print("probe ÉCHEC :", *errors, sep="\n  ", flush=True)
        os._exit(1)  # app.close() termine le processus avec le code 0 : on sort avant
    print("probe OK", flush=True)
    app.close()


if __name__ == "__main__":
    run_main(main)
