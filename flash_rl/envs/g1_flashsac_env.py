"""G1 flat-terrain velocity task with the FlashSAC sim-to-real rewards (paper Table 14, FlashSAC column).

Must be imported AFTER the IsaacSim app is launched (it imports isaaclab).
Registered as gym id `FlashSAC-Velocity-Flat-G1-v0`.

Deviations from the paper (not published / not reproducible here):
  * robot is IsaacLab's G1_MINIMAL (37 joints incl. fingers), not the 29-DoF G1 with Table-12 PD gains
  * policy obs = stock IsaacLab flat obs (no CENet / history, no privileged critic obs)
  * r_airtime (composite swing/stance reward) is approximated by IsaacLab's `feet_air_time_positive_biped` (0.4 s)
  * no large-scale domain randomization, flat terrain only
"""

from __future__ import annotations

import os

import gymnasium as gym
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

ENV_ID = "FlashSAC-Velocity-Flat-G1-v0"
CMD = "base_velocity"
FEET = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
FEET_ASSET = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")

# Table 14 prints -52.0 for the torso-orientation weight (PPO column: -2.0). Likely a typo, kept configurable.
TORSO_ORIENTATION_WEIGHT = float(os.environ.get("FLASHSAC_TORSO_W", "-52.0"))


# ----------------------------------------------------------------------------- reward functions
def _feet_forces(env, sensor_cfg):
    return env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]  # (N, 2, 3)


def _feet_contact(env, sensor_cfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0] > 1.0  # (N, 2)


def orthogonal_velocity_exp(env, command_name: str = CMD, scale: float = 1.5):
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    v = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)[:, :2]
    cnorm = cmd.norm(dim=1, keepdim=True)
    direction = cmd / cnorm.clamp(min=1e-6)
    v_perp = v - (v * direction).sum(1, keepdim=True) * direction
    v_perp = torch.where(cnorm < 0.1, v, v_perp)  # stop command: v_perp = v_xy
    return torch.exp(-scale * v_perp.square().sum(1))


def feet_yaw_drag(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    w_z = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2].abs()
    return (w_z * _feet_contact(env, sensor_cfg)).sum(1)


def feet_force(env, sensor_cfg: SceneEntityCfg, thresh: float = 700.0, max_over: float = 400.0):
    fz = _feet_forces(env, sensor_cfg)[..., 2].abs()
    return (fz - thresh).clamp(0.0, max_over).sum(1)


def feet_lateral_distance(env, asset_cfg: SceneEntityCfg, target: float = 0.3):
    asset = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (N, 2, 3)
    rel_b = quat_apply_inverse(asset.data.root_quat_w, pos[:, 0] - pos[:, 1])
    return (target - rel_b[:, 1].abs()).clamp(min=0.0)


def feet_stumble(env, sensor_cfg: SceneEntityCfg):
    f = _feet_forces(env, sensor_cfg)
    return (f[..., :2].norm(dim=-1) > 5.0 * f[..., 2].abs()).float().sum(1)


def air_time_variance(env, sensor_cfg: SceneEntityCfg):
    data = env.scene.sensors[sensor_cfg.name].data
    t_air = data.last_air_time[:, sensor_cfg.body_ids]
    t_contact = data.last_contact_time[:, sensor_cfg.body_ids]
    return t_air.var(dim=1, unbiased=False) + t_contact.var(dim=1, unbiased=False)


def impact_velocity_delta(env, asset_cfg: SceneEntityCfg, clamp_max: float = 1.0):
    vz = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    prev = getattr(env, "_flashsac_prev_feet_vz", None)
    env._flashsac_prev_feet_vz = vz.clone()
    if prev is None:
        return torch.zeros(vz.shape[0], device=vz.device)
    return (vz - prev).square().clamp(max=clamp_max).sum(1)


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return (asset.data.applied_torque * asset.data.joint_vel).norm(dim=1)


def torso_orientation_l2(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    q = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    g_b = quat_apply_inverse(q, asset.data.GRAVITY_VEC_W)
    return g_b[:, :2].square().sum(1)


def stand_still(env, command_name: str = CMD, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    dq2 = (asset.data.joint_pos - asset.data.default_joint_pos).square().sum(1)
    return dq2 * (env.command_manager.get_command(command_name).norm(dim=1) < 0.1)


def fly(env, sensor_cfg: SceneEntityCfg):
    return (~_feet_contact(env, sensor_cfg).any(dim=1)).float()


# ----------------------------------------------------------------------------- env cfg
@configclass
class G1FlashSACFlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        r = self.rewards

        # -- task
        r.track_lin_vel_xy_exp.weight = 2.0
        r.track_lin_vel_xy_exp.params["std"] = 0.25
        r.track_ang_vel_z_exp.weight = 1.5
        r.track_ang_vel_z_exp.params["std"] = 0.25
        r.orthogonal_velocity = RewTerm(func=orthogonal_velocity_exp, weight=1.0)

        # -- style
        r.feet_air_time.weight = 1.0
        r.feet_air_time.params["threshold"] = 0.4
        r.feet_slide.weight = -0.25
        r.feet_yaw_drag = RewTerm(func=feet_yaw_drag, weight=-0.5, params={"sensor_cfg": FEET, "asset_cfg": FEET_ASSET})
        r.feet_force = RewTerm(func=feet_force, weight=-3e-3, params={"sensor_cfg": FEET})
        r.feet_lateral_distance = RewTerm(func=feet_lateral_distance, weight=-5.0, params={"asset_cfg": FEET_ASSET})
        r.feet_stumble = RewTerm(func=feet_stumble, weight=-2.0, params={"sensor_cfg": FEET})
        r.air_time_variance = RewTerm(func=air_time_variance, weight=-1.0, params={"sensor_cfg": FEET})
        r.impact_velocity_delta = RewTerm(func=impact_velocity_delta, weight=-5.0, params={"asset_cfg": FEET_ASSET})

        # -- regularization
        r.lin_vel_z_l2.weight = -0.25
        r.ang_vel_xy_l2.weight = -1.0
        r.dof_torques_l2 = None
        r.energy = RewTerm(func=energy, weight=-1e-3)
        r.dof_acc_l2.weight = -2.5e-7
        r.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot")
        r.action_rate_l2.weight = -0.5
        r.flat_orientation_l2.weight = -5.0
        r.torso_orientation = RewTerm(
            func=torso_orientation_l2,
            weight=TORSO_ORIENTATION_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
        r.joint_deviation_hip.weight = -0.5  # hip yaw/roll
        r.joint_deviation_arms.weight = -1.0
        r.joint_deviation_fingers.weight = -1.0
        r.joint_deviation_torso.weight = -0.25
        r.joint_deviation_legs = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.25,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_.*_joint"])},
        )
        r.dof_pos_limits.weight = -5.0
        r.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot")
        r.stand_still = RewTerm(func=stand_still, weight=-5.0)

        # -- safety & termination
        r.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-5.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=r"(?!.*ankle_roll_link).*"), "threshold": 1.0},
        )
        r.fly = RewTerm(func=fly, weight=-1.0, params={"sensor_cfg": FEET})
        r.termination_penalty = None
        r.alive = RewTerm(func=mdp.is_alive, weight=1.0)

        # -- commands: omnidirectional + some standing envs (needed by the stand-still term)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


def register() -> None:
    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "flash_rl.envs.g1_flashsac_env:G1FlashSACFlatEnvCfg"},
    )
