"""Politiques déterministes chargées depuis un checkpoint, pour l'enregistrement et l'évaluation."""
from pathlib import Path

import numpy as np
import torch


class FlashSACPolicy:
    """Acteur FlashSAC reconstruit avec la config du run (mêmes overrides que train.run)."""

    def __init__(self, run_meta: dict, obs_dim: int, act_dim: int, device: str = "cuda:0"):
        import hydra
        from gymnasium.spaces import Box
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        from flash_rl.agents import create_agent

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda s: eval(s))
        GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../configs")
        # buffer minuscule : on ne fait que de l'inférence
        extra = ["agent.use_compile=false", "agent.load_optimizer=false",
                 "agent.buffer_max_length=16", "agent.buffer_min_length=1"]
        keys = {e.split("=")[0] for e in extra}
        overrides = [o for o in run_meta["overrides"] if o.split("=")[0] not in keys] + extra
        cfg = hydra.compose("flashSAC_base", overrides=overrides)
        OmegaConf.resolve(cfg)
        self.agent = create_agent(
            Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32), Box(-1.0, 1.0, (act_dim,), dtype=np.float32),
            {"actor_observation_size": (obs_dim,), "asymmetric_obs": False}, cfg.agent,
        )

    def load(self, ckpt_dir: Path) -> None:
        """Charge l'acteur seul. L'entraînement compile le réseau : ses clés portent le préfixe ``_orig_mod.``."""
        net = self.agent._actor.network
        ckpt = torch.load(Path(ckpt_dir) / "actor.pt", map_location=next(net.parameters()).device)
        sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["network_state_dict"].items()}
        net.load_state_dict(sd)
        net.eval()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.agent.sample_actions(0, {"next_observation": obs.detach().cpu().numpy()}, training=False)
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=obs.device).clamp(-1.0, 1.0)


class RslPolicy:
    """Politique PPO rsl_rl, reconstruite avec la même config qu'à l'entraînement."""

    def __init__(self, wrapped_env, train_cfg: dict, device: str = "cuda:0"):
        from rsl_rl.runners import OnPolicyRunner

        self.device = device
        self.runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
        self._policy = None

    def load(self, ckpt_path: Path) -> None:
        self.runner.load(str(ckpt_path), load_optimizer=False)
        self._policy = self.runner.get_inference_policy(self.device)

    def __call__(self, obs_td) -> torch.Tensor:
        with torch.inference_mode():
            return self._policy(obs_td).clamp(-1.0, 1.0)
