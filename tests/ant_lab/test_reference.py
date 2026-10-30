import numpy as np
import torch

from ant_lab.quat import quat_from_rotvec
from ant_lab.reference import degrade, from_kinematics, load_reference, save_reference, with_bodies

FPS = 60.0
NAMES = [f"j{i}" for i in range(8)]


def _clip(T=121):
    t = np.arange(T) / FPS
    root_pos = np.stack([2 * t, np.zeros(T), 0.5 * np.ones(T)], -1)
    rv = np.stack([np.zeros(T), np.zeros(T), t], -1)
    root_quat = quat_from_rotvec(torch.as_tensor(rv)).numpy()
    joint_pos = 0.5 * t[:, None] * np.ones((T, 8))
    return t, root_pos, root_quat, joint_pos


def test_from_kinematics_velocities():
    _, rp, rq, jp = _clip()
    ref = from_kinematics(FPS, NAMES, rp, rq, jp)
    assert np.allclose(ref.joint_vel, 0.5, atol=1e-9)
    assert np.allclose(ref.root_lin_vel[:, 0], 2.0, atol=1e-9)
    assert np.allclose(ref.root_ang_vel[1:-1], [0, 0, 1], atol=1e-3)
    assert ref.num_frames == 121 and not ref.has_bodies


def test_roundtrip(tmp_path):
    _, rp, rq, jp = _clip()
    ref = from_kinematics(FPS, NAMES, rp, rq, jp)
    save_reference(tmp_path / "r.npz", ref)
    back = load_reference(tmp_path / "r.npz")
    assert back.fps == FPS and back.joint_names == NAMES
    for f in ["root_pos", "root_quat", "joint_pos", "root_lin_vel", "root_ang_vel", "joint_vel"]:
        assert np.array_equal(getattr(back, f), getattr(ref, f))
    assert back.applied_torque is None and back.body_pos is None and back.body_names is None


def test_with_bodies_velocities():
    T = 60
    _, rp, rq, jp = _clip(T)
    ref = from_kinematics(FPS, NAMES, rp, rq, jp)
    t = np.arange(T) / FPS
    bp = np.stack([t, np.zeros(T), np.zeros(T)], -1)[:, None, :]
    bq = np.tile([1.0, 0, 0, 0], (T, 1, 1))
    ref2 = with_bodies(ref, ["torso"], bp, bq, bp.copy(), bq.copy())
    assert ref2.has_bodies
    assert np.allclose(ref2.body_lin_vel, [1, 0, 0], atol=1e-9)
    assert np.allclose(ref2.body_com_lin_vel, [1, 0, 0], atol=1e-9)
    assert np.allclose(ref2.body_ang_vel, 0, atol=1e-9)


def test_degrade_zero_sigma():
    _, rp, rq, jp = _clip()
    ref = from_kinematics(FPS, NAMES, rp, rq, jp, applied_torque=np.ones_like(jp))
    out = degrade(ref, 0.0, 0.0, 0.0, 0.5, seed=0)
    assert np.allclose(out.root_pos, ref.root_pos, atol=1e-12)
    assert np.allclose(out.root_quat, ref.root_quat, atol=1e-12)
    assert np.allclose(out.joint_pos, ref.joint_pos, atol=1e-12)
    assert out.applied_torque is None and out.body_pos is None


def test_degrade_stats():
    T = 6000
    ref = from_kinematics(FPS, NAMES, np.zeros((T, 3)), np.tile([1.0, 0, 0, 0], (T, 1)), np.zeros((T, 8)))
    out = degrade(ref, 0.035, 0.0, 0.0, 0.5, seed=0)
    assert abs(out.joint_pos.std() - 0.035) < 0.15 * 0.035
    lim = np.array([[-0.01, 0.01]] * 8)
    out2 = degrade(ref, 0.035, 0.0, 0.0, 0.5, seed=0, joint_limits=lim)
    assert out2.joint_pos.min() >= -0.01 and out2.joint_pos.max() <= 0.01


def test_degrade_seed():
    _, rp, rq, jp = _clip()
    ref = from_kinematics(FPS, NAMES, rp, rq, jp)
    a = degrade(ref, 0.035, 0.01, 0.035, 0.5, seed=1)
    b = degrade(ref, 0.035, 0.01, 0.035, 0.5, seed=1)
    c = degrade(ref, 0.035, 0.01, 0.035, 0.5, seed=2)
    assert np.array_equal(a.joint_pos, b.joint_pos) and np.array_equal(a.root_quat, b.root_quat)
    assert not np.array_equal(a.joint_pos, c.joint_pos)
