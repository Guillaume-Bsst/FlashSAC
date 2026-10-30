"""Toutes les constantes du labo Ant. Aucun autre module ne les redéfinit."""
from pathlib import Path

FPS = 60.0  # sim 1/120 s, décimation 2 : une frame de référence par pas d'env

# Force S = 1 : l'Ant d'IsaacLab (gear 15, action 0.5 -> 7,5 N·m max).
KP0, KD0, TAU0 = 15.0, 1.0, 7.5  # Kp (N·m/rad), Kd (N·m·s/rad), couple max (N·m)
ACTION_SCALE_RAD = 0.5  # cible PD = posture par défaut + 0.5 * action

S_EXPERT = 1.5
K_VALUES = (1.0, 1.25, 1.5)
K_DEFAULT = 1.25


def tracker_strength(k: float) -> float:
    """Force de l'Ant qui suit : S_EXPERT / k_cap."""
    return S_EXPERT / k


# Bruit lent du tube, niveau « plancher » de holonew (D26, r2p1t1).
NOISE_DEFAULT = {"sigma_joint": 0.035, "sigma_root_pos": 0.01, "sigma_root_rot": 0.035, "ell_s": 0.5}

# Reward BeyondMimic.
REWARD_SIGMA = {"anchor_pos": 0.3, "anchor_ori": 0.4, "body_pos": 0.3, "body_ori": 0.4,
                "body_lin_vel": 1.0, "body_ang_vel": 3.14}
REWARD_WEIGHT = {"anchor_pos": 0.5, "anchor_ori": 0.5, "body_pos": 1.0, "body_ori": 1.0, "body_lin_vel": 1.0,
                 "body_ang_vel": 1.0, "action_rate": -0.1, "joint_limit": -10.0}
SOFT_LIMIT_FACTOR = 0.9
TERM = {"anchor_z": 0.25, "anchor_grav_z": 0.8, "foot_z": 0.25}
SAMPLER = {"kernel_size": 1, "lam": 0.8, "uniform_ratio": 0.1, "alpha": 0.001}

# Évaluation et succès.
EVAL_INIT_NOISE = {"joint": 0.05, "root_xy": 0.02, "yaw": 0.05}
SUCCESS = {"min_frac_complete": 0.9, "max_joint_err": 0.1}
RUN_CAP_S = 3 * 3600

# Géométrie des pieds (nv_ant.xml) : centre de la sphère du bout de patte, en repère du corps « pied ».
FOOT_TIP_OFFSETS = {"front_left_foot": (0.4, 0.4, 0.0), "front_right_foot": (-0.4, 0.4, 0.0),
                    "left_back_foot": (-0.4, -0.4, 0.0), "right_back_foot": (0.4, -0.4, 0.0)}
FOOT_RADIUS = 0.08
FRICTION_MU = 1.0
CONTACT_MARGIN = 0.02

RESULTS = Path("results/ant_lab")

ENV_REF, ENV_K, ENV_ACTION_MODE, ENV_RSI = "ANT_LAB_REF", "ANT_LAB_K", "ANT_LAB_ACTION_MODE", "ANT_LAB_RSI"
