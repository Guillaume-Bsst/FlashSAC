"""Mesures d'infaisabilité d'une référence : résidu de contact et demande de couple.

Le sol porte le robot : une grosse force sur le corps est normale (c'est le poids). Ce qui est infaisable, c'est la
part du torseur nécessaire au centre de masse que les pieds posés ne peuvent pas fournir (le sol pousse mais ne tire
pas, seulement aux points de contact, dans le cône de frottement).
"""
import numpy as np
import torch
from scipy.optimize import nnls

from ant_lab import quat as Q
from ant_lab.config import CONTACT_MARGIN, FOOT_RADIUS, FRICTION_MU


def foot_tip_centers(body_pos, body_quat, foot_ids, tip_offsets) -> np.ndarray:
    """(T, K, 3) : centres des sphères des bouts de pattes."""
    p = np.asarray(body_pos, dtype=np.float64)[:, foot_ids]
    q = torch.as_tensor(np.asarray(body_quat, dtype=np.float64)[:, foot_ids])
    off = torch.as_tensor(np.asarray(tip_offsets, dtype=np.float64)).expand(q.shape[:-1] + (3,))
    return p + Q.quat_apply(q, off).numpy()


def contact_state(tip_centers, radius=FOOT_RADIUS, margin=CONTACT_MARGIN):
    """Points de contact (bas des sphères) et contacts actifs (bas de la sphère à moins de ``margin`` du sol)."""
    c = np.asarray(tip_centers, dtype=np.float64)
    pts = c - np.array([0.0, 0.0, radius])
    return pts, pts[..., 2] < margin


def _cone_dirs(mu):
    ez = np.array([0.0, 0.0, 1.0])
    dirs = []
    for e in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
        d = ez + mu * np.asarray(e, dtype=np.float64)
        dirs.append(d / np.linalg.norm(d))
    return dirs


def contact_residual(masses, inertias, com_pos, com_quat, com_lin_vel, body_ang_vel, contact_points,
                     contact_active, fps, mu=FRICTION_MU, g=9.81):
    """Résidu, frame par frame, du torseur au centre de masse que les contacts ne peuvent pas fournir.

    Retourne ``force_res_N``, ``torque_res_Nm`` et ``force_ratio`` (= force_res / (M·g)), chacun (T,).
    """
    m = np.asarray(masses, dtype=np.float64)
    I = np.asarray(inertias, dtype=np.float64)
    x = np.asarray(com_pos, dtype=np.float64)
    v = np.asarray(com_lin_vel, dtype=np.float64)
    w = np.asarray(body_ang_vel, dtype=np.float64)
    R = Q.quat_to_mat(torch.as_tensor(np.asarray(com_quat, dtype=np.float64))).numpy()  # (T, B, 3, 3)
    M = m.sum()
    c = (m[None, :, None] * x).sum(1) / M  # (T, 3)
    p = (m[None, :, None] * v).sum(1)
    I_w = R @ I[None] @ np.swapaxes(R, -1, -2)
    L = (I_w @ w[..., None])[..., 0].sum(1) + np.cross(x - c[:, None], m[None, :, None] * v).sum(1)
    dt = 1.0 / fps
    pdot = np.gradient(p, dt, axis=0)
    Ldot = np.gradient(L, dt, axis=0)
    wrench = np.concatenate([pdot + np.array([0.0, 0.0, M * g]), Ldot], axis=-1)  # (T, 6)

    dirs = _cone_dirs(mu)
    pts = np.asarray(contact_points, dtype=np.float64)
    act = np.asarray(contact_active, dtype=bool)
    T = wrench.shape[0]
    f_res = np.zeros(T)
    t_res = np.zeros(T)
    for t in range(T):
        cols = []
        for k in np.nonzero(act[t])[0]:
            r = pts[t, k] - c[t]
            for d in dirs:
                cols.append(np.concatenate([d, np.cross(r, d)]))
        if cols:
            A = np.stack(cols, axis=1)
            lam, _ = nnls(A, wrench[t])
            res = A @ lam - wrench[t]
        else:
            res = -wrench[t]
        f_res[t] = np.linalg.norm(res[:3])
        t_res[t] = np.linalg.norm(res[3:])
    return {"force_res_N": f_res, "torque_res_Nm": t_res, "force_ratio": f_res / (M * g)}


def torque_demand(applied_torque, limit: float) -> dict:
    """Part des frames où au moins une articulation dépasse la limite de couple du suiveur."""
    ratio = np.abs(np.asarray(applied_torque, dtype=np.float64)).max(-1) / limit
    return {"frac_frames_over": float((ratio > 1.0).mean()),
            "p95_max_ratio": float(np.percentile(ratio, 95)),
            "max_ratio": float(ratio.max())}
