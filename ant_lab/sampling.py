"""Échantillonnage adaptatif des frames de départ (BeyondMimic).

Portage de ``AdaptiveTimestepsSampler`` de
``holosoma-otr-rl/src/holosoma/holosoma/managers/command/terms/wbt.py`` (l. 893–930).
"""
import torch

from ant_lab.config import FPS, SAMPLER


class AdaptiveSampler:
    def __init__(self, num_frames: int, fps: float = FPS, kernel_size: int = SAMPLER["kernel_size"],
                 lam: float = SAMPLER["lam"], uniform_ratio: float = SAMPLER["uniform_ratio"],
                 alpha: float = SAMPLER["alpha"], device: str = "cpu"):
        self.num_frames = int(num_frames)
        self.device = device
        self.kernel_size = int(kernel_size)
        self.uniform_ratio = float(uniform_ratio)
        self.alpha = float(alpha)
        self.num_bins = int(self.num_frames // max(fps, 1)) + 1  # bins d'environ 1 s, plus un bin de queue
        k = torch.tensor([lam ** i for i in range(self.kernel_size)], device=device, dtype=torch.float)
        self.kernel = k / k.sum()
        self.current = torch.zeros(self.num_bins, device=device)
        self.failed = torch.zeros(self.num_bins, device=device)

    def record_failures(self, frames: torch.Tensor) -> None:
        """Frames des épisodes terminés par échec (pas les time-outs)."""
        if frames.numel() == 0:
            return
        b = torch.clamp((frames.to(self.device).long() * self.num_bins) // max(self.num_frames, 1),
                        0, self.num_bins - 1)
        self.current += torch.bincount(b, minlength=self.num_bins).float()

    def step(self) -> None:
        self.failed = self.alpha * self.current + (1 - self.alpha) * self.failed
        self.current.zero_()

    def probabilities(self) -> torch.Tensor:
        p = self.failed + self.uniform_ratio / float(self.num_bins)
        p = torch.nn.functional.pad(p.view(1, 1, -1), (0, self.kernel_size - 1), mode="replicate")
        p = torch.nn.functional.conv1d(p, self.kernel.view(1, 1, -1)).view(-1)
        return p / p.sum()

    def sample(self, n: int) -> torch.Tensor:
        bins = torch.multinomial(self.probabilities(), n, replacement=True)
        phase = (bins + torch.rand(n, device=self.device)) / self.num_bins
        frame = torch.floor(phase * (self.num_frames - 1)).long()
        return torch.clamp(frame, 0, self.num_frames - 2)
