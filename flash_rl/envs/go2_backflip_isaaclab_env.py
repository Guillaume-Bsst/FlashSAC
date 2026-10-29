"""Unitree Go2 backflip (reward-only, time-phased) in IsaacLab, ported from the Genesis `go2-backflip` env (run A9).

Must be imported AFTER the IsaacSim app is launched (it imports isaaclab).
Registered as gym id `FlashSAC-Backflip-Go2-v0`.

Same control setup as the Genesis env: 50 Hz policy (dt 0.005, decimation 4), explicit PD (kp 70, kd 3) with
torque limits 23.7 / 45.43 Nm, joint target = default + 3.0 * a (a in [-1, 1]; Genesis: 0.5 * clip(6a, +-6)),
2 s episodes, no early termination. Differences: no 1-step action latency, policy obs = stock-like proprio
(+ 6 episode-phase terms), symmetric (non-privileged) critic, joint friction not randomized.
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_inv, quat_mul
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

# joint resistance (viscous + Coulomb) added on top of the PD torque, randomized per env at construction.
# IsaacLab/PhysX joints have none by default, while the real Go2 (and MuJoCo Menagerie / Simple) do.
JOINT_VISC_RANGE = tuple(float(x) for x in os.environ.get("FLASHSAC_BF_VISC", "0.0,3.0").split(","))  # N.m.s/rad (Menagerie: 2.0)
JOINT_COUL_RANGE = tuple(float(x) for x in os.environ.get("FLASHSAC_BF_COUL", "0.0,0.3").split(","))  # N.m       (Menagerie: 0.2)
FOOT_SLIP_W = float(os.environ.get("FLASHSAC_BF_SLIP", "0.0"))


class FrictionIdealPDActuator(IdealPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        n, j = self._num_envs, len(self.joint_names)
        on = not os.environ.get("BF_PLAY")  # replay / scan: nominal (frictionless) joints, like IsaacLab by default
        self.visc = (torch.empty(n, j, device=self._device).uniform_(*cfg.visc_range) if on else torch.full((n, j), float(os.environ.get("BF_VISC", "0")), device=self._device))
        self.coul = (torch.empty(n, j, device=self._device).uniform_(*cfg.coul_range) if on else torch.full((n, j), float(os.environ.get("BF_COUL", "0")), device=self._device))

    def compute(self, control_action, joint_pos, joint_vel):
        ca = super().compute(control_action, joint_pos, joint_vel)
        ca.joint_efforts = ca.joint_efforts - self.visc * joint_vel - self.coul * torch.tanh(joint_vel / 0.05)
        return ca


@configclass
class FrictionIdealPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = FrictionIdealPDActuator
    visc_range: tuple = JOINT_VISC_RANGE
    coul_range: tuple = JOINT_COUL_RANGE


ENV_ID = "FlashSAC-Backflip-Go2-v0"
JUMP_CAP = float(os.environ.get("FLASHSAC_BF_JUMPCAP", "0.5"))  # jump_height reward saturates at 0.3 + JUMP_CAP
ACTION_SCALE = float(os.environ.get("FLASHSAC_BF_SCALE", "3.0"))  # joint target = default + ACTION_SCALE * a
ACTION_RATE_W = float(os.environ.get("FLASHSAC_BF_ACTRATE", "-0.001"))
TORQUE_W = float(os.environ.get("FLASHSAC_BF_TORQUE", "0.0"))
LANDING_W = float(os.environ.get("FLASHSAC_BF_LANDING", "1.0"))  # multiplier of the landing penalties (rewards are scaled by dt=0.02)
TERMINATION_W = float(os.environ.get("FLASHSAC_BF_TERM", "0.0"))  # one-off penalty when the episode is terminated by a head/base landing
FEET = SceneEntityCfg("robot", body_names=".*_foot")
# the Go2 head (lidar) is a separate body in the IsaacLab USD: it must be penalized too (nose-first landings)
BODY_CONTACT = SceneEntityCfg("contact_forces", body_names=["base", "Head_upper", "Head_lower", ".*_thigh", ".*_calf"])


def _t(env) -> torch.Tensor:
    return env.episode_length_buf * env.step_dt


def _asset(env):
    return env.scene["robot"]


# ----------------------------------------------------------------------------- observations
def episode_phase(env) -> torch.Tensor:
    ph = math.pi * _t(env)[:, None] / 2
    return torch.cat([ph.sin(), ph.cos(), (ph / 2).sin(), (ph / 2).cos(), (ph / 4).sin(), (ph / 4).cos()], dim=-1)


# ----------------------------------------------------------------------------- rewards
def ang_vel_y(env) -> torch.Tensor:
    a = _asset(env)
    t = _t(env)
    wy = a.data.root_ang_vel_b[:, 1]
    acc = getattr(env, "_bf_pitch_acc", None)
    if acc is None:
        acc = torch.zeros_like(t)
    acc = torch.where(env.episode_length_buf <= 1, torch.zeros_like(acc), acc)  # new episode
    r = (-wy).clamp(-14.0, 14.0) * ((t > 0.5) & (t < 1.0)) * (acc < 2 * math.pi)
    env._bf_pitch_acc = acc - wy * env.step_dt  # cumulative backflip rotation (rad)
    return r


def ang_vel_z_abs(env) -> torch.Tensor:
    return _asset(env).data.root_ang_vel_b[:, 2].abs()


def lin_vel_z_window(env) -> torch.Tensor:
    t = _t(env)
    return _asset(env).data.root_lin_vel_w[:, 2].clamp(max=3.5) * ((t > 0.5) & (t < 0.75))


def orientation_control(env) -> torch.Tensor:
    a = _asset(env)
    phase = (_t(env) - 0.5).clamp(0.0, 0.5)
    axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, 3)
    q = quat_mul(quat_from_angle_axis(4 * phase * math.pi, axis), a.data.default_root_state[:, 3:7])
    g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    desired = quat_apply(quat_inv(q), g)
    return (a.data.projected_gravity_b - desired).square().sum(1)


def feet_height_before_flip(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    a = _asset(env)
    z = a.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.02
    return z.clamp(min=0.0).sum(1) * (_t(env) < 0.5)


def height_control(env) -> torch.Tensor:
    t = _t(env)
    z = _asset(env).data.root_pos_w[:, 2]
    return (0.3 - z).square() * ((t < 0.4) | (t > 1.4))


def actions_symmetry(env) -> torch.Tensor:
    if not hasattr(env, "_bf_sym_ids"):
        _install_symmetry_ids(env)
    ids = env._bf_sym_ids  # dict: (hip, thigh, calf) x (FL, FR, RL, RR) joint ids in action order
    a = env.action_manager.action
    d = (a[:, ids["FL"][0]] + a[:, ids["FR"][0]]).square() + (a[:, ids["RL"][0]] + a[:, ids["RR"][0]]).square()
    for j in (1, 2):
        d += (a[:, ids["FL"][j]] - a[:, ids["FR"][j]]).square() + (a[:, ids["RL"][j]] - a[:, ids["RR"][j]]).square()
    return d


def gravity_y(env) -> torch.Tensor:
    return _asset(env).data.projected_gravity_b[:, 1].square()


def feet_distance(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    a = _asset(env)
    rel = a.data.body_pos_w[:, asset_cfg.body_ids] - a.data.root_pos_w[:, None]
    q = a.data.root_quat_w[:, None].expand(-1, rel.shape[1], -1)
    return quat_apply_inverse(q, rel)[..., 1].square().sum(1)


def landing_pose(env) -> torch.Tensor:
    a = _asset(env)
    return (a.data.joint_pos - a.data.default_joint_pos).square().sum(1) * (_t(env) > 1.1)


def landing_height(env) -> torch.Tensor:
    return (0.3 - _asset(env).data.root_pos_w[:, 2]).square() * (_t(env) > 1.1)


def landing_collision(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (f.norm(dim=-1) > 0.1).sum(1).float() * (_t(env) > 1.0)


def head_base_contact_after_flip(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Termination: base / head (lidar) touching the ground after the flip (nose-first or belly landing)."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (f.norm(dim=-1) > threshold).any(dim=1) & (_t(env) > 1.0)


def feet_slip(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Horizontal speed of the feet while they touch the ground (penalizes pushing by sliding)."""
    contact = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    v = _asset(env).data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (v * contact).sum(1)


def landing_ang_vel(env) -> torch.Tensor:
    return _asset(env).data.root_ang_vel_b[:, 1].abs() * (_t(env) > 1.0)


def landing_drift(env) -> torch.Tensor:
    return _asset(env).data.root_lin_vel_b[:, :2].square().sum(1) * (_t(env) > 1.1)


def jump_height(env) -> torch.Tensor:
    t = _t(env)
    return (_asset(env).data.root_pos_w[:, 2] - 0.3).clamp(0.0, JUMP_CAP) * ((t > 0.55) & (t < 1.0))


def straight_axis(env) -> torch.Tensor:
    # body y axis must stay parallel to world y during the flip (pure pitch: no roll / yaw twist)
    a = _asset(env)
    t = _t(env)
    y = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, 3)
    return (quat_apply(a.data.root_quat_w, y) - y).square().sum(1) * ((t > 0.5) & (t < 1.3))


def drift_x(env) -> torch.Tensor:
    a = _asset(env)
    return (a.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]).square()


def lateral_drift(env) -> torch.Tensor:
    a = _asset(env)
    return (a.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]).square()


# ----------------------------------------------------------------------------- config
@configclass
class BackflipObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        phase = ObsTerm(func=episode_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BackflipRewardsCfg:
    ang_vel_y = RewTerm(func=ang_vel_y, weight=5.0)
    ang_vel_z = RewTerm(func=ang_vel_z_abs, weight=-1.0)
    lin_vel_z = RewTerm(func=lin_vel_z_window, weight=20.0)
    orientation_control = RewTerm(func=orientation_control, weight=-1.0)
    feet_height_before_flip = RewTerm(func=feet_height_before_flip, weight=-30.0, params={"asset_cfg": FEET})
    height_control = RewTerm(func=height_control, weight=-10.0)
    actions_symmetry = RewTerm(func=actions_symmetry, weight=-0.1)
    gravity_y = RewTerm(func=gravity_y, weight=-10.0)
    feet_distance = RewTerm(func=feet_distance, weight=-1.0, params={"asset_cfg": FEET})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=ACTION_RATE_W)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=TORQUE_W)
    landing_pose = RewTerm(func=landing_pose, weight=LANDING_W * -2.0)
    landing_height = RewTerm(func=landing_height, weight=LANDING_W * -30.0)
    landing_collision = RewTerm(func=landing_collision, weight=LANDING_W * -10.0, params={"sensor_cfg": BODY_CONTACT})
    landing_ang_vel = RewTerm(func=landing_ang_vel, weight=-1.0)
    jump_height = RewTerm(func=jump_height, weight=20.0)
    straight_axis = RewTerm(func=straight_axis, weight=-5.0)
    landing_drift = RewTerm(func=landing_drift, weight=-2.0)
    lateral_drift = RewTerm(func=lateral_drift, weight=-1.0)
    drift_x = RewTerm(func=drift_x, weight=-5.0)
    feet_slip = RewTerm(func=feet_slip, weight=FOOT_SLIP_W, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "asset_cfg": FEET})
    termination = RewTerm(func=mdp.is_terminated, weight=TERMINATION_W)


@configclass
class BackflipEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 2.2),
            "dynamic_friction_range": (0.3, 2.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.5, 3.5),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": tuple(float(x) for x in os.environ.get("FLASHSAC_BF_KD", "0.7,2.2").split(",")),  # MuJoCo Menagerie has ~5 effective (PD 3 + joint 2)
            "operation": "scale",
        },
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.0, 0.03),
            "operation": "abs",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )


@configclass
class BackflipTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    landing_contact = DoneTerm(
        func=head_base_contact_after_flip,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_upper", "Head_lower"]), "threshold": 5.0},
    )


@configclass
class Go2BackflipEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.0
        self.commands = None
        self.curriculum = None
        self.observations = BackflipObsCfg()
        self.rewards = BackflipRewardsCfg()
        self.events = BackflipEventsCfg()
        self.terminations = BackflipTerminationsCfg()
        self.actions.joint_pos.scale = ACTION_SCALE
        self.actions.joint_pos.use_default_offset = True

        if os.environ.get("BF_PLAY"):  # evaluation / replay: no observation noise, no domain randomization
            self.observations.policy.enable_corruption = False
            for name in ("physics_material", "add_base_mass", "actuator_gains", "joint_parameters"):
                setattr(self.events, name, None)
            self.events.reset_robot_joints = None
            self.viewer.eye = (0.0, -3.0, 0.9)  # side view of the flip
            self.viewer.lookat = (0.0, 0.0, 0.4)

        r = self.scene.robot
        r.init_state.pos = (0.0, 0.0, 0.34)
        r.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }
        r.actuators = {
            "hip_thigh": FrictionIdealPDActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                effort_limit=23.7, velocity_limit=30.0, stiffness=70.0, damping=3.0,
            ),
            "calf": FrictionIdealPDActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit=45.43, velocity_limit=30.0, stiffness=70.0, damping=3.0,
            ),
        }


def _install_symmetry_ids(env) -> None:
    names = env.action_manager.get_term("joint_pos")._joint_names
    ids = {}
    for leg in ("FL", "FR", "RL", "RR"):
        ids[leg] = [names.index(f"{leg}_{j}_joint") for j in ("hip", "thigh", "calf")]
    env._bf_sym_ids = ids


def register() -> None:
    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "flash_rl.envs.go2_backflip_isaaclab_env:Go2BackflipEnvCfg"},
    )
