"""Référence cinématique : format, vitesses, dégradation, sauvegarde npz.

Positions en repère env-local, quaternions en (w, x, y, z), vitesses en repère monde.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from ant_lab import quat as Q
from ant_lab.noise import slow_noise


@dataclass
class Reference:
    fps: float
    joint_names: list[str]
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    joint_vel: np.ndarray
    applied_torque: np.ndarray | None = None
    body_names: list[str] | None = None
    body_pos: np.ndarray | None = None
    body_quat: np.ndarray | None = None
    body_lin_vel: np.ndarray | None = None
    body_ang_vel: np.ndarray | None = None
    body_com_pos: np.ndarray | None = None
    body_com_quat: np.ndarray | None = None
    body_com_lin_vel: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def has_bodies(self) -> bool:
        return self.body_pos is not None


def _ang_vel(q: np.ndarray, fps: float) -> np.ndarray:
    return Q.ang_vel_world(torch.as_tensor(np.asarray(q, dtype=np.float64)), 1.0 / fps).numpy()


def from_kinematics(fps, joint_names, root_pos, root_quat, joint_pos, applied_torque=None) -> Reference:
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    dt = 1.0 / fps
    return Reference(
        fps=float(fps), joint_names=list(joint_names), root_pos=root_pos, root_quat=root_quat,
        joint_pos=joint_pos,
        root_lin_vel=np.gradient(root_pos, dt, axis=0),
        root_ang_vel=_ang_vel(root_quat, fps),
        joint_vel=np.gradient(joint_pos, dt, axis=0),
        applied_torque=None if applied_torque is None else np.asarray(applied_torque, dtype=np.float64),
    )


def with_bodies(ref, body_names, body_pos, body_quat, body_com_pos, body_com_quat) -> Reference:
    dt = 1.0 / ref.fps
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_quat = np.asarray(body_quat, dtype=np.float64)
    body_com_pos = np.asarray(body_com_pos, dtype=np.float64)
    body_com_quat = np.asarray(body_com_quat, dtype=np.float64)
    return replace(
        ref, body_names=list(body_names), body_pos=body_pos, body_quat=body_quat,
        body_lin_vel=np.gradient(body_pos, dt, axis=0),
        body_ang_vel=_ang_vel(body_quat, ref.fps),
        body_com_pos=body_com_pos, body_com_quat=body_com_quat,
        body_com_lin_vel=np.gradient(body_com_pos, dt, axis=0),
    )


def degrade(ref, sigma_joint, sigma_root_pos, sigma_root_rot, ell_s, seed, joint_limits=None) -> Reference:
    """Ajoute le bruit lent du tube (articulations, position puis rotation de la racine)."""
    rng = np.random.default_rng(seed)
    T, J = ref.joint_pos.shape
    joint = ref.joint_pos + slow_noise(rng, T, J, sigma_joint, ell_s, ref.fps)
    root_pos = ref.root_pos + slow_noise(rng, T, 3, sigma_root_pos, ell_s, ref.fps)
    rot = slow_noise(rng, T, 3, sigma_root_rot, ell_s, ref.fps)
    root_quat = Q.quat_mul(Q.quat_from_rotvec(torch.as_tensor(rot)), torch.as_tensor(ref.root_quat)).numpy()
    if joint_limits is not None:
        lim = np.asarray(joint_limits, dtype=np.float64)
        joint = np.clip(joint, lim[:, 0], lim[:, 1])
    return from_kinematics(ref.fps, ref.joint_names, root_pos, root_quat, joint)


_NAME_FIELDS = ("joint_names", "body_names")


def save_reference(path, ref: Reference) -> None:
    out = {}
    for f in fields(ref):
        v = getattr(ref, f.name)
        if v is None:
            continue
        out[f.name] = np.array(v, dtype=str) if f.name in _NAME_FIELDS else np.asarray(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def load_reference(path) -> Reference:
    with np.load(path) as z:
        kw = {}
        for f in fields(Reference):
            if f.name not in z.files:
                continue
            v = z[f.name]
            if f.name in _NAME_FIELDS:
                kw[f.name] = [str(s) for s in v]
            elif f.name == "fps":
                kw[f.name] = float(v)
            else:
                kw[f.name] = v
    return Reference(**kw)
