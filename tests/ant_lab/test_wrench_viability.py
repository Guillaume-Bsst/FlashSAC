import numpy as np

from ant_lab.wrench_viability import frame_residual, required_wrench, time_scaled

G = 9.81
M = 2.0
UP = np.array([0.0, 0.0, 1.0])
FEET = np.array([[0.3, 0.3, 0.0], [-0.3, 0.3, 0.0], [-0.3, -0.3, 0.0], [0.3, -0.3, 0.0]])
COM = np.array([0.0, 0.0, 0.5])


def _static_wrench():
    return np.array([0.0, 0.0, M * G, 0.0, 0.0, 0.0])


def test_required_wrench_of_a_body_at_rest_is_its_weight():
    T = 5
    W, c = required_wrench(masses=np.array([M]), inertias=np.eye(3)[None] * 0.01,
                           com_pos=np.tile(COM, (T, 1, 1)), com_quat=np.tile([1.0, 0, 0, 0], (T, 1, 1)),
                           com_lin_vel=np.zeros((T, 1, 3)), body_ang_vel=np.zeros((T, 1, 3)), fps=60.0)
    np.testing.assert_allclose(W, np.tile(_static_wrench(), (T, 1)), atol=1e-9)
    np.testing.assert_allclose(c, np.tile(COM, (T, 1)), atol=1e-12)


def test_standing_on_four_feet_is_viable():
    f, m = frame_residual(_static_wrench(), COM, FEET, mu=1.0)
    assert f < 1e-3 and m < 1e-3


def test_free_fall_without_contact_is_viable_but_weight_without_contact_is_not():
    f, m = frame_residual(np.zeros(6), COM, FEET[:0], mu=1.0)
    assert f == 0.0 and m == 0.0
    f, m = frame_residual(_static_wrench(), COM, FEET[:0], mu=1.0)
    assert abs(f - M * G) < 1e-9


def test_lateral_push_beyond_friction_leaves_a_force_residual():
    W = _static_wrench()
    W[0] = 2.0 * M * G                                        # μ = 1 : au plus M·g à l'horizontale
    f, _ = frame_residual(W, np.array([0.0, 0.0, 0.0]), FEET, mu=1.0)
    assert f > 0.5 * M * G


def test_single_foot_cannot_hold_an_offset_weight():
    f, m = frame_residual(_static_wrench(), COM, FEET[:1], mu=1.0)   # un seul pied, décalé du CoM
    assert m > 0.1 * M * G * 0.4


def test_time_scaling_multiplies_velocities_and_fps():
    ref = {"fps": 60.0, "body_com_lin_vel": np.ones((3, 2, 3)), "body_ang_vel": 2 * np.ones((3, 2, 3)),
           "body_pos": np.zeros((3, 2, 3))}
    out = time_scaled(ref, 1.5)
    assert out["fps"] == 90.0
    np.testing.assert_allclose(out["body_com_lin_vel"], 1.5)
    np.testing.assert_allclose(out["body_ang_vel"], 3.0)
    np.testing.assert_allclose(out["body_pos"], 0.0)
