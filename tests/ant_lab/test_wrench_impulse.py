import numpy as np

from ant_lab.quat import quat_from_rotvec
from ant_lab.wrench_viability import frame_residual, half_step_momentum, window_impulse

G = 9.81
FPS = 60.0
M = 2.0
FEET = np.array([[0.3, 0.3, -0.5], [-0.3, 0.3, -0.5], [-0.3, -0.3, -0.5], [0.3, -0.3, -0.5]])  # bras, CoM → sol


def _one_body(x, q=None):
    T = len(x)
    q = np.tile([1.0, 0, 0, 0], (T, 1)) if q is None else q
    return dict(masses=np.array([M]), inertias=np.diag([0.01, 0.02, 0.03])[None], com_pos=x[:, None],
                com_quat=q[:, None], fps=FPS)


def test_ballistic_flight_needs_no_contact_impulse():
    t = np.arange(20) / FPS
    x = np.stack([1.0 + 3.0 * t, 0.5 * t, 1.0 + 2.0 * t - 0.5 * G * t**2], 1)
    P, L, _ = half_step_momentum(**_one_body(x))
    for a, b in [(1, 1), (3, 7), (5, 15)]:
        np.testing.assert_allclose(window_impulse(P, L, M, FPS, a, b), 0.0, atol=1e-9)


def test_resting_body_needs_its_weight_times_the_window_duration():
    x = np.tile([0.0, 0.0, 0.5], (10, 1))
    P, L, c = half_step_momentum(**_one_body(x))
    np.testing.assert_allclose(window_impulse(P, L, M, FPS, 2, 6), [0, 0, M * G * 5 / FPS, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(c, x, atol=1e-12)


def test_spinning_body_keeps_its_angular_momentum():
    w = np.array([0.0, 0.0, 4.0])
    q = np.stack([quat_from_rotvec(w * k / FPS).numpy() for k in range(10)])
    P, L, _ = half_step_momentum(**_one_body(np.zeros((10, 3)), q))
    np.testing.assert_allclose(L, np.tile([0.0, 0.0, 0.03 * 4.0], (9, 1)), atol=1e-9)
    np.testing.assert_allclose(window_impulse(P, L, M, FPS, 2, 6)[3:], 0.0, atol=1e-9)


def test_impulse_can_be_given_by_a_single_frame_of_the_window():
    # la chute est arrêtée par un impact : un seul instant de contact dans la fenêtre suffit
    J = np.array([0.0, 0.0, 3.0 * M * G * 5 / FPS, 0.0, 0.0, 0.0])
    f, m = frame_residual(J, np.zeros(3), FEET)
    assert f < 1e-3 and m < 1e-3
