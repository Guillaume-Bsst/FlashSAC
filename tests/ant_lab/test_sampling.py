import torch

from ant_lab.sampling import AdaptiveSampler


def test_sampler_uniform_init():
    s = AdaptiveSampler(601, 60)
    assert s.num_bins == 11
    p = s.probabilities()
    assert torch.allclose(p, torch.full((11,), 1 / 11))


def test_sampler_focuses_on_failures():
    torch.manual_seed(0)
    s = AdaptiveSampler(601, 60)
    frames = torch.full((64,), 3 * 60 + 10)  # bin 3
    for _ in range(2000):
        s.record_failures(frames)
        s.step()
    p = s.probabilities()
    assert int(p.argmax()) == 3
    assert float(p.min()) > 0


def test_sampler_range():
    torch.manual_seed(0)
    s = AdaptiveSampler(601, 60)
    x = s.sample(10000)
    assert x.dtype == torch.long
    assert int(x.min()) >= 0 and int(x.max()) <= 599
