"""Dégrade une référence avec le bruit lent du tube (pur, sans simulateur).

usage : ant_lab/run.sh -m ant_lab.make_noisy_ref --in IN.npz --out OUT.npz [--seed 0] [--sigma-joint ...]
"""
import argparse
import json

import numpy as np

from ant_lab.config import NOISE_DEFAULT, RESULTS
from ant_lab.reference import degrade, load_reference, save_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    for k, v in NOISE_DEFAULT.items():
        p.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    a = p.parse_args()
    model = json.loads((RESULTS / "ant_model.json").read_text())
    ref = load_reference(a.inp)
    out = degrade(ref, a.sigma_joint, a.sigma_root_pos, a.sigma_root_rot, a.ell_s, seed=a.seed,
                  joint_limits=np.asarray(model["joint_pos_limits"]))
    save_reference(a.out, out)
    print(f"make_noisy_ref : {out.num_frames} frames, sigma_joint={a.sigma_joint} -> {a.out}")


if __name__ == "__main__":
    main()
