import math

import numpy as np
import torch

from ant_lab.feasibility import contact_residual, contact_state, foot_tip_centers, torque_demand
from ant_lab.quat import quat_from_rotvec

G = 9.81
FPS = 60.0


def _single_body(z, T):
    masses = np.array([1.0])
    inertias = np.eye(3)[None] * 0.01
    com = np.zeros((T, 1, 3))
    com[:, 0, 2] = z
    quat = np.tile([1.0, 0, 0, 0], (T, 1, 1))
    vel = np.gradient(com, 1 / FPS, axis=0)
    ang = np.zeros((T, 1, 3))
    return masses, inertias, com, quat, vel, ang


def _contacts(pts, T):
    p = np.tile(np.asarray(pts, dtype=float)[None], (T, 1, 1))
    return p, np.ones(p.shape[:2], dtype=bool)


def test_static_four_feet():
    T = 10
    m, I, c, q, v, w = _single_body(np.full(T, 0.5), T)
    p, a = _contacts([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0]], T)
    r = contact_residual(m, I, c, q, v, w, p, a, FPS)
    assert np.all(r["force_ratio"] < 1e-6) and np.all(r["torque_res_Nm"] < 1e-6)


def test_no_contact():
    T = 10
    m, I, c, q, v, w = _single_body(np.full(T, 0.5), T)
    p, a = _contacts([[0.5, 0.5, 0]], T)
    r = contact_residual(m, I, c, q, v, w, p, np.zeros_like(a), FPS)
    assert np.allclose(r["force_ratio"], 1.0, atol=1e-9)


def test_cannot_pull():
    T = 10
    m, I, c, q, v, w = _single_body(np.zeros(T), T)
    p, a = _contacts([[1.0, 0.5, 0], [1.0, -0.5, 0]], T)
    r = contact_residual(m, I, c, q, v, w, p, a, FPS)
    assert np.all(r["torque_res_Nm"] > 0.1 * 1.0 * G)


def test_falling_faster_than_gravity():
    T = 30
    t = np.arange(T) / FPS
    m, I, c, q, v, w = _single_body(-G * t ** 2, T)
    p, a = _contacts([[0.5, 0.5, -5], [0.5, -0.5, -5], [-0.5, 0.5, -5], [-0.5, -0.5, -5]], T)
    r = contact_residual(m, I, c, q, v, w, p, a, FPS)
    assert np.allclose(r["force_ratio"][2:-2], 1.0, atol=0.05)


def test_tip_centers():
    bp = np.zeros((1, 2, 3))
    bq = np.tile([1.0, 0, 0, 0], (1, 2, 1))
    out = foot_tip_centers(bp, bq, [1], [(0.4, 0.4, 0.0)])
    assert out.shape == (1, 1, 3) and np.allclose(out[0, 0], [0.4, 0.4, 0])
    bq[0, 1] = quat_from_rotvec(torch.tensor([0, 0, math.pi / 2], dtype=torch.float64)).numpy()
    out = foot_tip_centers(bp, bq, [1], [(0.4, 0.4, 0.0)])
    assert np.allclose(out[0, 0], [-0.4, 0.4, 0], atol=1e-9)


def test_contact_state():
    centers = np.array([[[0.0, 0, 0.09], [0.0, 0, 0.2]]])
    pts, act = contact_state(centers)
    assert act.tolist() == [[True, False]]
    assert abs(pts[0, 0, 2] - 0.01) < 1e-9


def test_torque_demand():
    tau = np.array([[5.0], [9.0], [12.0]])
    d = torque_demand(tau, 9.0)
    assert abs(d["frac_frames_over"] - 1 / 3) < 1e-9
    assert abs(d["max_ratio"] - 12 / 9) < 1e-9
    assert "p95_max_ratio" in d
