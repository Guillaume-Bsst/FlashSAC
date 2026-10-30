from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
try:  # jax is optional (only needed for MuJoCo Playground / jax-based envs)
    import jax.numpy as jnp

    Tensor = Union[NDArray, jnp.ndarray, torch.Tensor]
except ImportError:
    Tensor = Union[NDArray, torch.Tensor]  # type: ignore[misc]
