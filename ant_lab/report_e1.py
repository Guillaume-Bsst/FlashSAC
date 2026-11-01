"""Rapport E1 : temps jusqu'au succès par run, médianes par algorithme, facteur d'accélération, courbes.

usage : ant_lab/run.sh -m ant_lab.report_e1 --runs D1 D2 ... --out results/ant_lab/e1
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from ant_lab.metrics import EvalRow, read_rows, steps_to_success, time_to_success

# palette catégorielle de référence (skill dataviz), créneaux 1 et 2, dans un ordre fixe
COLORS = {"ppo": "#2a78d6", "flashsac": "#eb6834"}
NAMES = {"ppo": "PPO", "flashsac": "FlashSAC"}


def summarize(rows_by_run: dict[str, list[EvalRow]]) -> dict:
    per_run, times = {}, {}
    for run, rows in rows_by_run.items():
        t = time_to_success(rows)
        algo = rows[0].algo
        per_run[run] = {"algo": algo, "seed": rows[0].seed, "time_to_success_s": t,
                        "steps_to_success": steps_to_success(rows)}
        times.setdefault(algo, []).append(math.inf if t is None else t)
    median_s = {}
    for algo in ("ppo", "flashsac"):
        m = statistics.median(times[algo]) if times.get(algo) else math.inf
        median_s[algo] = None if math.isinf(m) else m
    speedup = (median_s["ppo"] / median_s["flashsac"]
               if median_s["ppo"] is not None and median_s["flashsac"] is not None else None)
    return {"per_run": per_run, "median_s": median_s, "speedup": speedup}


def _fmt_s(t):
    return "> 3 h" if t is None else f"{t / 60:.1f} min"


def write_report(rows_by_run: dict[str, list[EvalRow]], out: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize(rows_by_run)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# E1 — PPO contre FlashSAC", "",
             "Succès : ≥ 90 % des épisodes d'évaluation vont au bout du clip, l'erreur articulaire moyenne < 0,1 rad "
             "et l'écart moyen de vitesse horizontale du torse (repère de cap) < 0,5 m/s.",
             "", "| Run | Algorithme | Graine | Temps jusqu'au succès | Pas d'env jusqu'au succès |",
             "|---|---|---|---|---|"]
    for run, r in s["per_run"].items():
        steps = "—" if r["steps_to_success"] is None else f"{r['steps_to_success']:,}".replace(",", " ")
        lines.append(f"| {run} | {NAMES[r['algo']]} | {r['seed']} | {_fmt_s(r['time_to_success_s'])} | {steps} |")
    lines += ["", f"- Médiane PPO : {_fmt_s(s['median_s']['ppo'])}",
              f"- Médiane FlashSAC : {_fmt_s(s['median_s']['flashsac'])}",
              "- Facteur d'accélération (médiane PPO / médiane FlashSAC) : "
              + ("non défini (un algorithme n'a pas de médiane finie)" if s["speedup"] is None
                 else f"**×{s['speedup']:.2f}**"), ""]
    (out / "report.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(s, indent=1))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    seen = set()
    for run, rows in rows_by_run.items():
        rows = sorted(rows, key=lambda r: r.elapsed_s)
        algo = rows[0].algo
        ax.plot([r.elapsed_s / 60 for r in rows], [100 * r.frac_complete for r in rows], color=COLORS[algo],
                lw=2, alpha=0.85, label=None if algo in seen else NAMES[algo])
        seen.add(algo)
    ax.axhline(90, color="#52514e", lw=1, ls="--")
    ax.text(0.01, 91, "seuil de succès (90 %)", color="#52514e", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("temps réel (min)")
    ax.set_ylabel("épisodes allant au bout du clip (%)")
    ax.set_ylim(0, 102)
    ax.grid(color="#e6e5e0", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows_by_run = {}
    for d in a.runs:
        d = Path(d)
        rows = read_rows(d / "eval.csv")
        rows_by_run[f"{rows[0].algo}/{d.parent.name}/{d.name}"] = rows
    s = write_report(rows_by_run, Path(a.out))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
