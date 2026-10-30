"""Recette PPO de BeyondMimic pour rsl_rl 3.0.1.

Copie de ``train_cfg`` de ``holonew/OTRL/otrl/ppo_dict.py`` avec deux changements : le critique lit le groupe
d'observations ``policy`` (pas d'observation privilégiée ici) et ``save_interval`` est un argument.
Un dict neuf à chaque appel : rsl_rl en retire des clés.
"""


def train_cfg(max_iterations: int, save_interval: int, seed: int, experiment: str, run_name: str,
              device: str = "cuda:0") -> dict:
    hidden = [512, 256, 128]
    return {
        "seed": seed,
        "device": device,
        "num_steps_per_env": 24,
        "max_iterations": max_iterations,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "save_interval": save_interval,
        "experiment_name": experiment,
        "run_name": run_name,
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": list(hidden),
            "critic_hidden_dims": list(hidden),
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }
