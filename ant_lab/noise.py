"""Bruit lent du tube."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ant_lab.config import FPS


def slow_noise(rng: np.random.Generator, n_frames: int, dim: int, sigma: float, ell_s: float,
               fps: float = FPS) -> np.ndarray:
    """(n_frames, dim) : bruit blanc lissé par une gaussienne d'écart-type ``ell_s`` secondes.

    Copie de ``slow_noise`` de ``holonew/OTRL/otrl/toys/tube.py``, généralisée de 2 à ``dim`` colonnes :
    tirage sur une séquence rembourrée puis recadrée (pas d'effet de bord), mise à l'échelle par la norme du
    noyau (pas par le lot) pour que chaque colonne ait l'écart-type ``sigma``.
    """
    k = ell_s * fps
    pad = int(np.ceil(4 * k))
    impulse = np.zeros(2 * pad + 1)
    impulse[pad] = 1.0
    norm = np.sqrt((gaussian_filter1d(impulse, k, mode="constant") ** 2).sum())
    w = rng.standard_normal((n_frames + 2 * pad, dim))
    return gaussian_filter1d(w, k, axis=0, mode="constant")[pad:pad + n_frames] * (sigma / norm)
