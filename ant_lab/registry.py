"""Enregistrement gymnasium des envs du labo. Points d'entrée en chaînes : aucun import d'IsaacLab ici."""
import gymnasium as gym

EXPERT_ID = "AntLab-Expert-v0"
TRACK_ID = "AntLab-Track-v0"

_ENVS = {
    EXPERT_ID: ("ant_lab.expert_env:AntExpertEnv", "ant_lab.expert_env:AntExpertEnvCfg"),
    TRACK_ID: ("ant_lab.tracking_env:AntTrackingEnv", "ant_lab.tracking_env:AntTrackingEnvCfg"),
}


def register() -> None:
    for env_id, (entry, cfg) in _ENVS.items():
        if env_id in gym.registry:
            continue
        gym.register(id=env_id, entry_point=entry, disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": cfg})
