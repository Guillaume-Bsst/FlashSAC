"""Lignes d'évaluation, critère de succès, temps et pas jusqu'au succès, liste des checkpoints."""
import csv
import re
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from ant_lab.config import SUCCESS


@dataclass
class EvalRow:
    algo: str
    seed: int
    ckpt: str
    elapsed_s: float
    env_steps: int
    frac_complete: float
    mean_joint_err: float


def is_success(row: EvalRow) -> bool:
    return row.frac_complete >= SUCCESS["min_frac_complete"] and row.mean_joint_err < SUCCESS["max_joint_err"]


def _first_success(rows):
    for r in sorted(rows, key=lambda r: r.elapsed_s):
        if is_success(r):
            return r
    return None


def time_to_success(rows: list[EvalRow]) -> float | None:
    r = _first_success(rows)
    return None if r is None else r.elapsed_s


def steps_to_success(rows: list[EvalRow]) -> int | None:
    r = _first_success(rows)
    return None if r is None else r.env_steps


_FIELDS = [f.name for f in fields(EvalRow)]
_TYPES = {f.name: f.type for f in fields(EvalRow)}


def write_rows(path: Path, rows: list[EvalRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(_FIELDS)
        for r in rows:
            w.writerow(astuple(r))


def read_rows(path: Path) -> list[EvalRow]:
    with open(path, newline="") as f:
        return [EvalRow(**{k: _TYPES[k](v) for k, v in d.items()}) for d in csv.DictReader(f)]


def list_checkpoints(run_dir: Path, algo: str, num_envs: int, steps_per_iter: int = 24):
    """(chemin, mtime, pas d'env) triés par mtime.

    flashsac : dossiers ``run_dir/ckpt/step{N}``, mtime de ``actor.pt``, pas = N·num_envs.
    ppo : fichiers ``run_dir/model_{it}.pt``, pas = it·steps_per_iter·num_envs.
    """
    run_dir = Path(run_dir)
    out = []
    if algo == "flashsac":
        for d in (run_dir / "ckpt").glob("step*"):
            m = re.fullmatch(r"step(\d+)", d.name)
            a = d / "actor.pt"
            if m and a.exists():
                out.append((d, a.stat().st_mtime, int(m.group(1)) * num_envs))
    elif algo == "ppo":
        for f in run_dir.glob("model_*.pt"):
            m = re.fullmatch(r"model_(\d+)\.pt", f.name)
            if m:
                out.append((f, f.stat().st_mtime, int(m.group(1)) * steps_per_iter * num_envs))
    else:
        raise ValueError(f"algo inconnu : {algo}")
    return sorted(out, key=lambda x: x[1])
