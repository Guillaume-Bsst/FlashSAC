"""E0 : tableau de faisabilité (survie en boucle ouverte, demande de couple, résidu de contact) et démarche.

Pur : lit RESULTS/e0/openloop.jsonl, les références et RESULTS/ant_model.json.
usage : ant_lab/run.sh -m ant_lab.e0_report
"""
import json

import numpy as np

from ant_lab.config import K_VALUES, RESULTS, SOFT_LIMIT_FACTOR, TAU0, tracker_strength
from ant_lab.feasibility import contact_residual, contact_state, foot_tip_centers, torque_demand
from ant_lab.reference import load_reference

REFS = {"ref_s0": "σ = 0", "ref_s1": "σ plancher"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # palette catégorielle de référence (dataviz), ordre fixe


def contacts(ref, model):
    tips = foot_tip_centers(ref.body_pos, ref.body_quat, model["foot_ids"], model["foot_tip_offsets"])
    return contact_state(tips, radius=model["foot_radius"])


def residual(ref, model):
    pts, act = contacts(ref, model)
    return contact_residual(np.asarray(model["masses"]), np.asarray(model["inertias"]), ref.body_com_pos,
                            ref.body_com_quat, ref.body_com_lin_vel, ref.body_ang_vel, pts, act, ref.fps)


def soft_limit_frac(ref, model):
    lim = np.asarray(model["joint_pos_limits"])
    mid, half = lim.mean(1), 0.5 * (lim[:, 1] - lim[:, 0]) * SOFT_LIMIT_FACTOR
    out = (ref.joint_pos < mid - half) | (ref.joint_pos > mid + half)
    return float(out.any(1).mean())


def gait_plot(refs, model, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(refs), figsize=(10, 5), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    names = [n.replace("_foot", "") for n in model["foot_body_names"]]
    for c, (key, ref) in enumerate(refs.items()):
        t = np.arange(ref.num_frames) / ref.fps
        _, act = contacts(ref, model)
        ax = axes[0, c]
        for k in range(act.shape[1]):
            on = act[:, k]
            starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
            ends = np.flatnonzero(on & ~np.r_[on[1:], False])
            ax.broken_barh([(t[s], t[e] - t[s] + 1 / ref.fps) for s, e in zip(starts, ends)], (k - 0.35, 0.7),
                           color=COLORS[k])
        ax.set_yticks(range(act.shape[1]), names, fontsize=8)
        ax.set_title(f"{REFS[key]} : pieds au sol", fontsize=10)
        ax.invert_yaxis()
        ax = axes[1, c]
        ax.plot(t, ref.root_pos[:, 2], color="#52514e", lw=1.5)
        ax.set_ylabel("hauteur du torse (m)", fontsize=8)
        ax.set_xlabel("temps (s)")
        for a in axes[:, c]:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            a.grid(axis="x", color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(path)


def main():
    model = json.loads((RESULTS / "ant_model.json").read_text())
    out = RESULTS / "e0"
    rows = [json.loads(line) for line in (out / "openloop.jsonl").read_text().splitlines() if line.strip()]
    openloop = {(r["ref"], r["k"]): r for r in rows}
    core = load_reference(RESULTS / "refs" / "expert_core.npz")
    refs = {k: load_reference(RESULTS / "refs" / f"{k}.npz") for k in REFS}
    M_g = float(np.sum(model["masses"])) * 9.81
    res = {k: residual(r, model) for k, r in refs.items()}
    torque = {k: torque_demand(core.applied_torque, TAU0 * tracker_strength(k)) for k in K_VALUES}

    lines = ["# E0 — la référence est-elle presque faisable ?", "",
             "| Référence | k | Survie en boucle ouverte (s) | Fin de clip (%) | Frames au-delà de la limite de couple (%) "
             "| Résidu de contact médian (% M·g) | Résidu de contact p95 (% M·g) | Frames hors limite articulaire souple (%) |",
             "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for key, label in REFS.items():
        f = res[key]["force_res_N"] / M_g * 100
        soft = 100 * soft_limit_frac(refs[key], model)
        for k in K_VALUES:
            o = openloop.get((key, k), {})
            tq = torque[k]
            lines.append(f"| {label} | {k} | {o.get('mean_survival_s', float('nan')):.2f} | "
                         f"{100 * o.get('frac_complete', float('nan')):.0f} | {100 * tq['frac_frames_over']:.1f} | "
                         f"{np.median(f):.1f} | {np.percentile(f, 95):.1f} | {soft:.1f} |")
            summary[f"{key}_k{k}"] = {**o, "torque": tq, "res_median_pct": float(np.median(f)),
                                      "res_p95_pct": float(np.percentile(f, 95)), "soft_limit_pct": soft}
    lines += ["", f"Couple max enregistré de l'expert : {np.abs(core.applied_torque).max():.2f} N·m "
              f"(limite de l'expert {TAU0 * tracker_strength(1.0):.2f} N·m).",
              f"Vitesse moyenne de l'expert : {(core.root_pos[-1, 0] - core.root_pos[0, 0]) / ((core.num_frames - 1) / core.fps):.2f} m/s.",
              "Le résidu de contact ne dépend que de σ, la demande de couple que de k : ces colonnes se répètent."]
    (out / "e0_table.md").write_text("\n".join(lines) + "\n")
    (out / "e0_summary.json").write_text(json.dumps(summary, indent=1))
    gait_plot(refs, model, out / "gait.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
