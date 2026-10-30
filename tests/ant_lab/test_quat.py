import math

import torch

from ant_lab.quat import (
    ang_vel_world,
    quat_apply,
    quat_apply_inverse,
    quat_conj,
    quat_error_magnitude,
    quat_from_rotvec,
    quat_mul,
    quat_to_mat,
    yaw_quat,
)

ID = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _rand_quat(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _rv(x, y, z):
    return quat_from_rotvec(torch.tensor([x, y, z], dtype=torch.float64))


def test_mul_conj_identity():
    q = _rand_quat()
    out = quat_mul(q, quat_conj(q))
    assert torch.allclose(out, ID.double().expand_as(out), atol=1e-6)


def test_apply_yaw90():
    q = _rv(0, 0, math.pi / 2)
    assert torch.allclose(quat_apply(q, torch.tensor([1.0, 0, 0], dtype=torch.float64)),
                          torch.tensor([0.0, 1, 0], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(quat_apply_inverse(q, torch.tensor([0.0, 1, 0], dtype=torch.float64)),
                          torch.tensor([1.0, 0, 0], dtype=torch.float64), atol=1e-9)


def test_error_magnitude():
    q = _rand_quat()
    assert torch.allclose(quat_error_magnitude(q, q), torch.zeros(16, dtype=torch.float64), atol=1e-6)
    assert abs(float(quat_error_magnitude(ID.double(), _rv(0, 0, 0.3))) - 0.3) < 1e-6
    assert torch.allclose(quat_error_magnitude(q, -q), torch.zeros(16, dtype=torch.float64), atol=1e-6)


def test_yaw_quat():
    assert torch.allclose(yaw_quat(_rv(0, 0.4, 0)), ID.double(), atol=1e-9)
    q = quat_mul(_rv(0, 0, 0.7), _rv(0.3, 0, 0))
    assert torch.allclose(yaw_quat(q), _rv(0, 0, 0.7), atol=1e-9)


def test_to_mat_matches_apply():
    q = _rand_quat()
    v = torch.randn(16, 3, dtype=torch.float64)
    assert torch.allclose(quat_to_mat(q) @ v.unsqueeze(-1), quat_apply(q, v).unsqueeze(-1), atol=1e-9)


def test_ang_vel_constant_yaw_rate():
    t = torch.arange(121, dtype=torch.float64) / 60.0
    rv = torch.stack([torch.zeros_like(t), torch.zeros_like(t), t], dim=-1)
    q = quat_from_rotvec(rv)
    q[1::2] = -q[1::2]
    w = ang_vel_world(q, 1 / 60)
    assert torch.allclose(w[1:-1], torch.tensor([0.0, 0, 1], dtype=torch.float64).expand(119, 3), atol=1e-3)
