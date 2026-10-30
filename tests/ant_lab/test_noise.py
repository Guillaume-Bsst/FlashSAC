import numpy as np

from ant_lab.noise import slow_noise


def test_slow_noise_std():
    x = slow_noise(np.random.default_rng(0), 600, 4000, 0.02, 0.5)
    assert x.shape == (600, 4000)
    assert abs(x.std() - 0.02) < 0.002


def test_slow_noise_zero_and_seed():
    assert np.all(slow_noise(np.random.default_rng(0), 100, 3, 0.0, 0.5) == 0.0)
    a = slow_noise(np.random.default_rng(7), 100, 3, 0.02, 0.5)
    b = slow_noise(np.random.default_rng(7), 100, 3, 0.02, 0.5)
    assert np.array_equal(a, b)


def test_slow_noise_smooth():
    x = slow_noise(np.random.default_rng(1), 600, 200, 0.02, 0.5)
    x = x - x.mean(0)
    ac = (x[1:] * x[:-1]).sum() / (x * x).sum()
    assert ac > 0.95
