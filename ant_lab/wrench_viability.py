"""Part des frames d'une référence qui sont dynamiquement viables, au sens du transport d'efforts (HoloV2).

Pour chaque frame : le torseur que la référence exige au centre de masse (Newton-Euler, ``required_wrench``)
peut-il être produit par des forces de contact dans les cônes de Coulomb, posées sur les seuls pieds au sol ?
Le solveur est celui du repo HoloV2 (``distribute_forces`` : force de norme minimale réalisant le torseur, par
projections de Dykstra sur le cône exact). Ce qu'il ne peut pas produire est le résidu.

Une frame est viable si le résidu de force est sous ``tol``·M·g ET le résidu de moment sous ``tol``·M·g·ℓ,
avec ℓ la distance horizontale médiane du CoM aux bouts de pattes (le moment est donc compté comme une force
appliquée au bout de la patte). Les couples articulaires ne sont pas regardés ici.

Deux versions :
- ``analyse`` (première version) : accélérations par double ``np.gradient``, frame par frame. Le schéma équivaut
  à ``(x[k+2] − 2x[k] + x[k−2]) / 4dt²`` et étale chaque impact sur ±2 frames : sur la référence σ = 0, physique
  par construction, seules 74 % des frames passent.
- ``analyse_windows`` (à utiliser) : impulsions. Vitesses aux demi-pas, Newton discret exact sur une fenêtre de
  n frames centrée sur la frame jugée, forces permises à chaque frame où le pied est au sol. Avec n = 5
  (±33 ms), σ = 0 passe à 98 % : un contact qui a lieu entre deux frames (physique à 120 Hz, enregistrement à
  60 Hz) est rattrapé par la frame voisine.

usage : ant_lab/run.sh -m ant_lab.wrench_viability [ref ...]     (par défaut ref_s0 ref_s1)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ant_lab.config import CONTACT_MARGIN, FRICTION_MU, RESULTS
from ant_lab.feasibility import contact_state, foot_tip_centers, required_wrench

HOLOV2 = "/home/ecarn/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2"
TOLS = (0.05, 0.10, 0.20)
TOL_MAIN = 0.10
HALVES = (0, 1, 2)                 # fenêtres d'impulsion de 1, 3 et 5 frames
SPEEDS = (1.25, 1.5, 2.0)          # la référence σ = 0 rejouée plus vite : accélérations × s²
G = 9.81


def _distribute():
    if HOLOV2 not in sys.path:
        sys.path.insert(0, HOLOV2)
    from src.solve.wrench_baselines import distribute_forces

    return distribute_forces


def frame_residual(wrench, com, contact_points, mu=FRICTION_MU):
    """(‖résidu de force‖, ‖résidu de moment‖) du torseur ``wrench`` (6,) avec ces points de contact (K, 3)."""
    wrench = np.asarray(wrench, dtype=np.float64)
    pts = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        res = -wrench
    else:
        k = len(pts)
        out = _distribute()(np.zeros((1, 3)), np.tile([0.0, 0, 1], (1, 1)), np.zeros(1), pts,
                            np.tile([0.0, 0, 1], (k, 1)), np.full(k, float(mu)), wrench, com, max_iter=2000)
        res = out.wrench_residual
    return float(np.linalg.norm(res[:3])), float(np.linalg.norm(res[3:]))


def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(s > 1e-12, v / np.maximum(s, 1e-12) * 2 * np.arctan2(s, q[..., :1]), 2 * v)


def _quat_mat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def half_step_momentum(masses, inertias, com_pos, com_quat, fps):
    """Quantité de mouvement P et moment cinétique L (au CoM) aux demi-pas k+½, chacun (T−1, 3), et le CoM (T, 3).

    Vitesses par différence avant ``(x[k+1] − x[k])/dt`` : Newton discret s'écrit alors exactement
    ``P[b] − P[a−1] = Σ_{k=a..b} F_k·dt − M·g·(b−a+1)·dt``, sans dériver deux fois.
    """
    m = np.asarray(masses, dtype=np.float64)
    I = np.asarray(inertias, dtype=np.float64)
    x = np.asarray(com_pos, dtype=np.float64)
    q = np.asarray(com_quat, dtype=np.float64)
    dt = 1.0 / fps
    M = m.sum()
    c = (m[None, :, None] * x).sum(1) / M
    v = (x[1:] - x[:-1]) / dt
    w = _rotvec(_qmul(q[1:], q[:-1] * np.array([1.0, -1, -1, -1]))) / dt
    R = _quat_mat(q[:-1])
    I_w = R @ I[None] @ np.swapaxes(R, -1, -2)
    xm, cm = 0.5 * (x[1:] + x[:-1]), 0.5 * (c[1:] + c[:-1])
    P = (m[None, :, None] * v).sum(1)
    L = (I_w @ w[..., None])[..., 0].sum(1) + np.cross(xm - cm[:, None], m[None, :, None] * v).sum(1)
    return P, L, c


def window_impulse(P, L, M, fps, a, b, g=G):
    """Impulsion (6,) que les contacts doivent fournir sur les frames a..b (1 ≤ a ≤ b ≤ T−2)."""
    n = b - a + 1
    return np.concatenate([P[b] - P[a - 1] + np.array([0.0, 0.0, M * g * n / fps]), L[b] - L[a - 1]])


def analyse_windows(ref: dict, model: dict, half: int, margin=CONTACT_MARGIN, mu=FRICTION_MU) -> dict:
    """Viabilité par fenêtres de 2·half+1 frames centrées sur chaque frame (impulsions, pas d'accélérations)."""
    masses = np.asarray(model["masses"])
    M = float(masses.sum())
    fps = float(ref["fps"])
    P, L, c = half_step_momentum(masses, np.asarray(model["inertias"]), ref["body_com_pos"], ref["body_com_quat"],
                                 fps)
    tips = foot_tip_centers(ref["body_pos"], ref["body_quat"], model["foot_ids"], model["foot_tip_offsets"])
    _, act = contact_state(tips, radius=model["foot_radius"], margin=margin)
    ground = tips.copy()
    ground[..., 2] = 0.0
    arms = ground - c[:, None]                                    # (T, 4, 3) bras de levier CoM → sol
    lever = float(np.median(np.linalg.norm(tips[..., :2] - c[:, None, :2], axis=-1)))
    n = 2 * half + 1
    T = len(c)
    fr, mr, none = [], [], []
    for t in range(1 + half, T - 1 - half):
        a, b = t - half, t + half
        J = window_impulse(P, L, M, fps, a, b)
        pts = np.concatenate([arms[k, act[k]] for k in range(a, b + 1)])
        f, m_ = frame_residual(J, np.zeros(3), pts, mu)
        fr.append(f / (M * G * n / fps))
        mr.append(m_ / (M * G * lever * n / fps))
        none.append(len(pts) == 0)
    fr, mr, none = np.array(fr), np.array(mr), np.array(none)
    bad = ~((fr < TOL_MAIN) & (mr < TOL_MAIN))
    return {"fenetre_frames": n, "fenetres": len(fr),
            "viable_pct": {f"{int(100 * k)}%": 100 * float(((fr < k) & (mr < k)).mean()) for k in TOLS},
            "causes_pct": {"aucun pied au sol": 100 * float((bad & none).mean()),
                           "force (frottement ou traction)": 100 * float((bad & ~none & (fr >= TOL_MAIN)).mean()),
                           "moment seul": 100 * float((bad & ~none & (fr < TOL_MAIN)).mean())},
            "_series": {"force_ratio": fr, "moment_ratio": mr}}


def time_scaled(ref: dict, s: float) -> dict:
    """La même trajectoire parcourue ``s`` fois plus vite (vitesses × s, fps × s)."""
    out = dict(ref)
    out["fps"] = float(ref["fps"]) * s
    for k in ("body_com_lin_vel", "body_ang_vel"):
        out[k] = np.asarray(ref[k]) * s
    return out


def analyse(ref: dict, model: dict, margin=CONTACT_MARGIN, mu=FRICTION_MU) -> dict:
    masses = np.asarray(model["masses"])
    M = float(masses.sum())
    W, c = required_wrench(masses, np.asarray(model["inertias"]), ref["body_com_pos"], ref["body_com_quat"],
                           ref["body_com_lin_vel"], ref["body_ang_vel"], float(ref["fps"]))
    tips = foot_tip_centers(ref["body_pos"], ref["body_quat"], model["foot_ids"], model["foot_tip_offsets"])
    pts, act = contact_state(tips, radius=model["foot_radius"], margin=margin)
    lever = float(np.median(np.linalg.norm(tips[..., :2] - c[:, None, :2], axis=-1)))
    T = len(W)
    f_res, m_res = np.zeros(T), np.zeros(T)
    for t in range(T):
        p = pts[t, act[t]].copy()
        p[:, 2] = 0.0                                   # la force s'applique au sol, sous le pied
        f_res[t], m_res[t] = frame_residual(W[t], c[t], p, mu)
    fr, mr = f_res / (M * G), m_res / (M * G * lever)
    holo = f_res / np.maximum(np.linalg.norm(W[:, :3], axis=1), 1e-6)     # métrique du script CMU de HoloV2
    viable = {tol: (fr < tol) & (mr < tol) for tol in TOLS}
    bad = ~viable[TOL_MAIN]
    none = act.sum(1) == 0
    return {
        "frames": T, "lever_m": lever, "contacts_moyens": float(act.sum(1).mean()),
        "viable_pct": {f"{int(100 * k)}%": 100 * float(v.mean()) for k, v in viable.items()},
        "holov2_force_seule_pct": 100 * float((holo < 0.2).mean()),
        "causes_pct": {"aucun pied au sol": 100 * float((bad & none).mean()),
                       "force (frottement ou traction)": 100 * float((bad & ~none & (fr >= TOL_MAIN)).mean()),
                       "moment seul": 100 * float((bad & ~none & (fr < TOL_MAIN)).mean())},
        "residu_force_median_pct_Mg": 100 * float(np.median(fr)),
        "residu_moment_median_pct": 100 * float(np.median(mr)),
        "_series": {"force_ratio": fr, "moment_ratio": mr, "contacts": act.sum(1)},
    }


def _resolve(r):
    p = Path(r)
    return p if p.suffix == ".npz" else RESULTS / "refs" / f"{r}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="*", default=["ref_s0", "ref_s1"])
    args = ap.parse_args()
    model = json.loads((RESULTS / "ant_model.json").read_text())
    refs = {Path(r).stem: dict(np.load(_resolve(r))) for r in args.refs}
    cases = dict(refs)
    if "ref_s0" in refs:
        for s in SPEEDS:
            cases[f"ref_s0 ×{s:g}"] = time_scaled(refs["ref_s0"], s)
    rows = {}
    for k, r in cases.items():
        rows[k] = {"par_frame": analyse(r, model),
                   **{f"fenetre_{2 * h + 1}": analyse_windows(r, model, h) for h in HALVES}}
        print(k, {kk: round(v["viable_pct"]["10%"], 1) for kk, v in rows[k].items()}, flush=True)
    margins = {(k, m): analyse_windows(refs[k], model, 1, margin=m)["viable_pct"]["10%"]
               for k in refs for m in (0.01, 0.04)}

    lines = ["# Frames dynamiquement viables (transport d'efforts HoloV2, μ = 1, tolérance 10 % du poids)", "",
             "Par frame : accélérations par double `np.gradient` (première version). Impulsions : Newton discret",
             "sur une fenêtre de n frames, vitesses aux demi-pas, forces permises à chaque frame où le pied est au sol.",
             "", "| Référence | Par frame (1re version) | Impulsion, 1 frame | **Impulsion, 3 frames** "
             "| Impulsion, 5 frames | Échecs (3 frames) : aucun pied au sol / force / moment |",
             "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        v = {kk: vv["viable_pct"]["10%"] for kk, vv in r.items()}
        cz = r["fenetre_3"]["causes_pct"]
        lines.append(f"| {k} | {v['par_frame']:.1f} % | {v['fenetre_1']:.1f} % | **{v['fenetre_3']:.1f} %** | "
                     f"{v['fenetre_5']:.1f} % | {cz['aucun pied au sol']:.1f} / "
                     f"{cz['force (frottement ou traction)']:.1f} / {cz['moment seul']:.1f} % |")
    lines += ["", "Sensibilité au seuil de contact (impulsion, 3 frames ; 2 cm dans le tableau) : "
              + ", ".join(f"{k} marge {100 * m:.0f} cm → {v:.1f} %" for (k, m), v in margins.items())]
    out = RESULTS / "e0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "wrench_viability.md").write_text("\n".join(lines) + "\n")
    (out / "wrench_viability.json").write_text(json.dumps(
        {k: {kk: {a: b for a, b in vv.items() if a != "_series"} for kk, vv in r.items()} for k, r in rows.items()},
        indent=1))
    np.savez(out / "wrench_viability_series.npz", **{f"{k}__{kk}__{s}": v for k, r in rows.items()
                                                     for kk, vv in r.items() for s, v in vv["_series"].items()})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
