"""Unitree G1 (29 DoF) forearm stand / inverted balance in IsaacLab (reward only, no reference motion).

Must be imported AFTER the IsaacSim app is launched (it imports isaaclab).
Registered as gym id `FlashSAC-G1ForearmStand-v0`.

Spec: docs/superpowers/specs/2026-10-26-g1-forearm-stand-design.md
Phase 1+2 (kick-up + catch/balance) first: the robot starts crouched with forearms and feet on the ground
(`g1_stand/dolphin_pose.json`, solved with MuJoCo FK), the legs push the body over the forearms and it must hold
the inverted pose. Real torque limits (legs 88/139 Nm, ankles/waist 50, arms 25, wrist pitch/yaw 5).
Env vars: FLASHSAC_G1S_STAND (fraction of resets from the standing pose, default 0), FLASHSAC_G1S_SCALE.
"""

from __future__ import annotations

import json
import os

import gymnasium as gym
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

ENV_ID = "FlashSAC-G1ForearmStand-v0"
USD = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
POSE = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "g1_stand", "dolphin2_pose.json")))
TARGET = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "g1_stand", "pincha_pose.json")))["joint_pos"]
ACTION_SCALE = float(os.environ.get("FLASHSAC_G1S_SCALE", "2.0"))
HOLD_H = 0.85  # pelvis height saturation of the height reward
# bodies that must not touch the ground (the forearms - elbow / wrist_roll - are the support)
BAD = ["pelvis", "torso_link", ".*_knee_link", ".*_hip_.*_link", ".*_wrist_pitch_link", ".*_wrist_yaw_link"]
if os.environ.get("FLASHSAC_G1S_HANDOK"):  # palm (rubber_hand collision lives on wrist_yaw_link) may support: longer support base
    BAD = BAD[:-1]
# the USD has no collision on elbow_link (forearm): the elbow end of the support is the bottom of the upper arm
ELB = "shoulder_yaw_link" if os.environ.get("FLASHSAC_G1S_UPPERARM") else "elbow_link"
FORE = ["left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link"]
ANK = [".*_ankle_roll_link"]


def _a(env):
    return env.scene["robot"]


def _ids(env, key, names):
    if not hasattr(env, "_g1s_ids"):
        env._g1s_ids = {}
    if key not in env._g1s_ids:
        env._g1s_ids[key] = _a(env).find_bodies(names, preserve_order=True)[0]
    return env._g1s_ids[key]


def _torso_up(env):
    a = _a(env)
    ids = _ids(env, "torso", ["torso_link"])
    z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3)
    return quat_apply(a.data.body_quat_w[:, ids[0]], z)[:, 2]  # world-z component of the torso up axis; -1 = inverted


def _legs_dir_z(env):
    a = _a(env)
    ank = a.data.body_pos_w[:, _ids(env, "ank", ANK)].mean(1)
    v = ank - a.data.root_pos_w
    return v[:, 2] / v.norm(dim=1).clamp(min=1e-3)  # +1: feet straight above the pelvis


def _fore_contact(env):
    """(left, right) float flags: at least one forearm link (elbow / wrist_roll) of that side touches the ground."""
    sens = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_g1s_fc"):
        env._g1s_fc = [sens.find_bodies([f"{s}_elbow_link", f"{s}_wrist_roll_link"])[0] for s in ("left", "right")]
    f = sens.data.net_forces_w.norm(dim=-1)
    return [(f[:, ids] > 1.0).any(dim=1).float() for ids in env._g1s_fc]


def _fore4(env):
    """(T, 4) float flags: left_elbow, left_wrist_roll, right_elbow, right_wrist_roll each touch the ground."""
    sens = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_g1s_f4"):
        env._g1s_f4 = [sens.find_bodies([n])[0][0] for n in (f"left_{ELB}", "left_wrist_roll_link", f"right_{ELB}", "right_wrist_roll_link")]
    return (sens.data.net_forces_w.norm(dim=-1)[:, env._g1s_f4] > 1.0).float()


def fore_gate(env) -> torch.Tensor:
    """1 only when elbow AND wrist of BOTH forearms are on the ground (whole forearm section in contact)."""
    return _fore4(env).prod(1)


def fore_low(env) -> torch.Tensor:
    """Dense: elbow + wrist links of both forearms close to the ground (mean of exp(-z/5cm)); elbows must come down too."""
    a = _a(env)
    ids = _ids(env, "f4", ["left_elbow_link", "left_wrist_roll_link", "right_elbow_link", "right_wrist_roll_link"])
    return torch.exp(-a.data.body_pos_w[:, ids, 2] / 0.05).mean(1)


def fore_frac(env) -> torch.Tensor:
    return _fore4(env).mean(1)


def live_monitor(env) -> torch.Tensor:
    """Zero reward; every 50 env steps appends per-term weighted means + contacts / posture to $FLASHSAC_G1S_LIVE (jsonl)."""
    path = os.environ.get("FLASHSAC_G1S_LIVE")
    env._g1s_n = getattr(env, "_g1s_n", 0) + 1
    if path and env._g1s_n % 50 == 0:
        a = _a(env)
        d = {"step": int(env.common_step_counter), "ep_len_s": float(env.episode_length_buf.float().mean() * env.step_dt)}
        for name, cfg in zip(env.reward_manager.active_terms, env.reward_manager._term_cfgs):
            if cfg.func is not live_monitor:
                d["r_" + name] = float((cfg.func(env, **cfg.params) * cfg.weight).mean())
        d["up"] = float(_torso_up(env).mean()); d["legs_dir"] = float(_legs_dir_z(env).mean()); d["pelvis_z"] = float(a.data.root_pos_w[:, 2].mean())
        f4 = _fore4(env).mean(0)
        for nm, v in zip(("L_elbow", "L_wrist", "R_elbow", "R_wrist"), f4):
            d[nm] = float(v)
        d["gate"] = float(fore_gate(env).mean())
        with open(path, "a") as fh:
            fh.write(json.dumps(d) + "\n")
    return torch.zeros(env.num_envs, device=env.device)


def torso_vertical(env) -> torch.Tensor:
    """Torso axis along world -z (inverted AND vertical): narrow gaussian + wide one for gradient. theta = angle to vertical."""
    theta = torch.acos((-_torso_up(env)).clamp(-1.0, 1.0))
    return 0.7 * torch.exp(-(theta / 0.3) ** 2) + 0.3 * torch.exp(-(theta / 1.2) ** 2)


def fore_missing(env) -> torch.Tensor:
    """Number of the 4 contact points (elbow / wrist of each forearm) NOT touching the ground (0..4)."""
    return 4.0 - _fore4(env).sum(1)


def feet_on_ground(env) -> torch.Tensor:
    sens = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_g1s_ft"):
        env._g1s_ft = sens.find_bodies([".*_ankle_roll_link"])[0]
    return (sens.data.net_forces_w.norm(dim=-1)[:, env._g1s_ft] > 1.0).float().sum(1)


def vertical_legs(env) -> torch.Tensor:
    return _legs_dir_z(env).clamp(0.0, 1.0) * torso_vertical(env)


def vertical_pelvis(env) -> torch.Tensor:
    return (_a(env).data.root_pos_w[:, 2] / HOLD_H).clamp(0.0, 1.0) * torso_vertical(env)


LEG_J = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]


def leg_action_rate(env) -> torch.Tensor:
    if not hasattr(env, "_g1s_lj"):
        env._g1s_lj = _a(env).find_joints(LEG_J)[0]
    am = env.action_manager
    return (am.action[:, env._g1s_lj] - am.prev_action[:, env._g1s_lj]).square().sum(1)


def leg_joint_vel(env) -> torch.Tensor:
    if not hasattr(env, "_g1s_lj"):
        env._g1s_lj = _a(env).find_joints(LEG_J)[0]
    return _a(env).data.joint_vel[:, env._g1s_lj].square().sum(1)


def arm_slide(env) -> torch.Tensor:
    """Horizontal speed (m/s) of the elbow / wrist contact points while they touch the ground, summed over the 4 points."""
    a = _a(env)
    ids = _ids(env, "f4", ["left_elbow_link", "left_wrist_roll_link", "right_elbow_link", "right_wrist_roll_link"])
    return (a.data.body_lin_vel_w[:, ids, :2].norm(dim=-1) * _fore4(env)).sum(1)


def leg_straight(env) -> torch.Tensor:
    """Legs straight and parallel (all leg joints at 0), hip roll / yaw x3 against crossed / spread legs; only once the torso is nearly vertical-inverted (up < -0.9), not during the flip."""
    a = _a(env)
    if not hasattr(env, "_g1s_ls"):
        ids, names = a.find_joints(LEG_J)
        w = torch.tensor([3.0 if ("hip_roll" in n or "hip_yaw" in n) else 1.0 for n in names], device=env.device)
        env._g1s_ls = (ids, w)
    ids, w = env._g1s_ls
    return (a.data.joint_pos[:, ids].square() * w).sum(1) * (_torso_up(env) < -0.9) * _ramp(env)


def still_joints(env) -> torch.Tensor:
    return _a(env).data.joint_vel.square().sum(1) * (_torso_up(env) < -0.9) * _ramp(env)


def arm_lift(env) -> torch.Tensor:
    """Support points (elbow + wrist_pitch of both forearms) that are NOT on the ground, once the body tips toward inverted (up < -0.5)."""
    return fore_missing(env) * (_torso_up(env) < -0.5)


def _ramp(env) -> float:
    lo, hi = (float(x) for x in os.environ.get("FLASHSAC_G1S_RAMP", "0,1").split(","))
    return min(max((env.common_step_counter - lo) / max(hi - lo, 1.0), 0.0), 1.0)


def hands_lift(env) -> torch.Tensor:
    """Wrist supports (left / right wrist_roll) off the ground while holding inverted (up < -0.9): the hands stay planted,
    the balance has to come from the body (pelvis / shoulders / legs), not from re-placing the arms."""
    f4 = _fore4(env)
    return (2.0 - f4[:, 1] - f4[:, 3]) * (_torso_up(env) < -0.9)


def hand_anchor(env) -> torch.Tensor:
    """Horizontal distance (m, clamped at 0.3 so that drifting never costs more than holding inverted earns) of each wrist from where it first touched the ground once the body tipped
    (up < -0.3), summed over both hands. Catches both sliding and 'walking' on the hands (lift + re-place), which
    arm_slide misses since it only sees velocity during contact. Anchors are cleared in reset_rsi."""
    a = _a(env)
    if not hasattr(env, "_g1s_anc"):
        env._g1s_anc = torch.zeros(env.num_envs, 2, 2, device=env.device)
        env._g1s_anc_ok = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
    xy = a.data.body_pos_w[:, _ids(env, "wr2", ["left_wrist_roll_link", "right_wrist_roll_link"]), :2]
    f4 = _fore4(env)
    new = (f4[:, [1, 3]] > 0) & ~env._g1s_anc_ok & (_torso_up(env) < -0.3).unsqueeze(1)
    env._g1s_anc = torch.where(new.unsqueeze(-1), xy, env._g1s_anc)
    env._g1s_anc_ok |= new
    return ((xy - env._g1s_anc).norm(dim=-1).clamp(max=0.3) * env._g1s_anc_ok).sum(1) * _ramp(env)


def root_anchor(env) -> torch.Tensor:
    """Horizontal distance (m, clamped at 3) of the pelvis from where it was when the body first got inverted (up < -0.9);
    anchors cleared in reset_rsi; ramped in."""
    a = _a(env)
    if not hasattr(env, "_g1s_ranc"):
        env._g1s_ranc = torch.zeros(env.num_envs, 2, device=env.device)
        env._g1s_ranc_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    xy = a.data.root_pos_w[:, :2]
    new = ~env._g1s_ranc_ok & (_torso_up(env) < -0.9)
    env._g1s_ranc = torch.where(new.unsqueeze(-1), xy, env._g1s_ranc)
    env._g1s_ranc_ok |= new
    dz = float(os.environ.get("FLASHSAC_G1S_ROOTDZ", "0.0"))  # free radius: small balancing moves cost nothing
    return ((xy - env._g1s_ranc).norm(dim=-1) - dz).clamp(min=0.0, max=3.0) * env._g1s_ranc_ok * _ramp(env)


DRIFT_K = 50  # control steps (1 s at 50 Hz)


def root_drift(env) -> torch.Tensor:
    """Net horizontal pelvis displacement (m) over the last second, beyond 5 cm, while inverted (up < -0.9); ramped in.
    Stillness, not a return to a fixed point: it may settle wherever it balances best, it just must not keep moving."""
    a = _a(env)
    if not hasattr(env, "_g1s_hist"):
        env._g1s_hist = torch.zeros(env.num_envs, DRIFT_K, 2, device=env.device)
        env._g1s_hcnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._g1s_hptr = 0
    xy = a.data.root_pos_w[:, :2]
    k = env._g1s_hptr
    old = env._g1s_hist[:, (k + 1) % DRIFT_K].clone()
    env._g1s_hist[:, k] = xy
    env._g1s_hptr = (k + 1) % DRIFT_K
    env._g1s_hcnt += 1
    ok = (env._g1s_hcnt >= DRIFT_K) & (_torso_up(env) < -0.9)
    dz = float(os.environ.get("FLASHSAC_G1S_DRIFTDZ", "0.05"))  # free net displacement per second (corrective steps)
    return ((xy - old).norm(dim=-1) - dz).clamp(min=0.0) * ok * _ramp(env)


ARM_J = [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"]


def arm_joint_vel(env) -> torch.Tensor:
    """Arm joint speed^2 (shoulders, elbows, wrist roll) while inverted (up < -0.9): the arms stay planted, the balance
    comes from legs / pelvis / waist instead of re-placing the hands; ramped in."""
    a = _a(env)
    if not hasattr(env, "_g1s_aj"):
        env._g1s_aj = a.find_joints(ARM_J)[0]
    return a.data.joint_vel[:, env._g1s_aj].square().sum(1) * (_torso_up(env) < -0.9) * _ramp(env)


def hip_twist(env) -> torch.Tensor:
    """Leg rotation: hip_yaw^2 + 0.5 hip_roll^2 (rad^2, both legs), while tipped (up < -0.5); ramped in."""
    a = _a(env)
    if not hasattr(env, "_g1s_hip"):
        ids, names = a.find_joints([".*_hip_yaw_joint", ".*_hip_roll_joint"])
        env._g1s_hip = (ids, torch.tensor([1.0 if "yaw" in n else 0.5 for n in names], device=env.device))
    ids, w = env._g1s_hip
    return (a.data.joint_pos[:, ids].square() * w).sum(1) * (_torso_up(env) < -0.5) * _ramp(env)


def com_forward(env) -> torch.Tensor:
    """Whole-body CoM (xy) over a point FLASHSAC_G1S_COMFWD m ahead of the elbows along the forearms (elbow -> wrist_yaw,
    averaged over both arms), i.e. inside the support from the upper-arm bottom to the palm rather than at its elbow edge.
    exp(-(err / 5 cm)^2) while inverted (up < -0.9); ramped in."""
    a = _a(env)
    if not hasattr(env, "_g1s_mass"):
        env._g1s_mass = a.root_physx_view.get_masses().to(env.device).unsqueeze(-1)
    m = env._g1s_mass
    com = (a.data.body_com_pos_w[:, :, :2] * m).sum(1) / m.sum(1)
    e = a.data.body_pos_w[:, _ids(env, "elb2", ["left_elbow_link", "right_elbow_link"]), :2].mean(1)
    w = a.data.body_pos_w[:, _ids(env, "wy2", ["left_wrist_yaw_link", "right_wrist_yaw_link"]), :2].mean(1)
    u = (w - e) / (w - e).norm(dim=-1, keepdim=True).clamp(min=1e-3)
    tgt = e + float(os.environ.get("FLASHSAC_G1S_COMFWD", "0.10")) * u
    err = (com - tgt).norm(dim=-1)
    return torch.exp(-(err / 0.05).square()) * (_torso_up(env) < -0.9) * _ramp(env)


def leg_cross(env) -> torch.Tensor:
    """Legs crossing: lateral gap (pelvis frame, left - right) of knees and ankles below 15 cm (default pose: 23.7 cm),
    in m, clamped at 0.3 per pair, summed over the 2 pairs; while tipped (up < -0.5), ramped in like hand_anchor."""
    a = _a(env)
    ids = _ids(env, "kneeank", ["left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"])
    rel = a.data.body_pos_w[:, ids] - a.data.root_pos_w.unsqueeze(1)
    y = quat_apply_inverse(a.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1), rel)[..., 1]
    gap = torch.stack([y[:, 0] - y[:, 1], y[:, 2] - y[:, 3]], 1)
    return (0.15 - gap).clamp(0.0, 0.3).sum(1) * (_torso_up(env) < -0.5) * _ramp(env)


def both_forearms(env) -> torch.Tensor:
    l, r = _fore_contact(env)
    return l * r


def one_forearm(env) -> torch.Tensor:
    l, r = _fore_contact(env)
    return (l - r).abs()


def arm_symmetry(env) -> torch.Tensor:
    a = _a(env)
    if not hasattr(env, "_g1s_arm"):
        jn = a.joint_names
        sg = {"shoulder_pitch": 1.0, "elbow": 1.0, "shoulder_roll": -1.0, "shoulder_yaw": -1.0, "wrist_roll": -1.0}
        env._g1s_arm = [(jn.index(f"left_{k}_joint"), jn.index(f"right_{k}_joint"), v) for k, v in sg.items()]
    q = a.data.joint_pos
    return sum((q[:, i] - v * q[:, j]).square() for i, j, v in env._g1s_arm)


def pincha_pose(env) -> torch.Tensor:
    a = _a(env)
    if not hasattr(env, "_g1s_tgt"):
        env._g1s_tgt = torch.tensor([TARGET[n] for n in a.joint_names], device=env.device)
    err = (a.data.joint_pos - env._g1s_tgt).square().mean(1)
    return torch.exp(-4.0 * err) * (_torso_up(env) < -0.5)


def inverted(env) -> torch.Tensor:
    return (1.0 - _torso_up(env)) / 2.0 * fore_gate(env)


def legs_vertical(env) -> torch.Tensor:
    return _legs_dir_z(env).clamp(0.0, 1.0) * (_torso_up(env) < 0.0) * fore_gate(env)


def pelvis_height(env) -> torch.Tensor:
    return (_a(env).data.root_pos_w[:, 2] / HOLD_H).clamp(0.0, 1.0) * (_torso_up(env) < -0.3) * fore_gate(env)


def feet_above(env) -> torch.Tensor:
    a = _a(env)
    dz = a.data.body_pos_w[:, _ids(env, "ank", ANK), 2].mean(1) - a.data.root_pos_w[:, 2]
    return dz.clamp(-0.3, 0.7) * fore_gate(env)


def balance_xy(env) -> torch.Tensor:
    a = _a(env)
    sup = a.data.body_pos_w[:, _ids(env, "fore", FORE), :2].mean(1)
    return (a.data.root_pos_w[:, :2] - sup).square().sum(1) * (_torso_up(env) < -0.5)


def hold_still(env) -> torch.Tensor:
    a = _a(env)
    return (a.data.root_lin_vel_w.square().sum(1) + 0.1 * a.data.root_ang_vel_w.square().sum(1)) * (_torso_up(env) < -0.9) * _ramp(env)


def hold_bonus(env) -> torch.Tensor:
    a = _a(env)
    ok = (_torso_up(env) < -0.85) & (_legs_dir_z(env) > 0.85) & (a.data.root_pos_w[:, 2] > 0.55)
    return ok.float() * fore_gate(env)


def _near_ground(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N, nb) bool: sensor body is within 12 cm of the ground (the contact sensor also reports self-contacts, e.g. arm vs torso)."""
    sens = env.scene.sensors[sensor_cfg.name]
    key = "_g1s_ng_" + ",".join(sens.body_names[i] for i in sensor_cfg.body_ids)
    if not hasattr(env, key):
        a = _a(env)
        setattr(env, key, [a.body_names.index(sens.body_names[i]) for i in sensor_cfg.body_ids])
    return _a(env).data.body_pos_w[:, getattr(env, key), 2] < 0.12


def bad_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return ((f.norm(dim=-1) > 1.0) & True).sum(1).float()


def bad_contact_term(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return ((f.norm(dim=-1) > threshold) & True).any(dim=1) & (env.episode_length_buf * env.step_dt > 0.5)


def reset_rsi(env, env_ids) -> None:
    """Reference-state init: reset on a random frame of the retargeted MOYO clip (standing start .. inverted hold).
    The clip is only a source of start states; the reward is the task (stay inverted and balanced)."""
    path = os.environ.get("FLASHSAC_G1S_RSI")
    if not path:
        return
    a = _a(env)
    if not hasattr(env, "_g1s_bank"):
        import numpy as np
        b = np.load(path)
        dev = env.device
        idx = [list(b["joint_names"]).index(n) for n in a.joint_names]
        up = torch.tensor(b["torso_up"], device=dev)
        env._g1s_bank = dict(
            pos=torch.tensor(b["root_pos"], device=dev, dtype=torch.float32), quat=torch.tensor(b["root_quat"], device=dev, dtype=torch.float32),
            vel=torch.tensor(b["root_vel"], device=dev, dtype=torch.float32), jp=torch.tensor(b["joint_pos"][:, idx], device=dev, dtype=torch.float32),
            jv=torch.tensor(b["joint_vel"][:, idx], device=dev, dtype=torch.float32),
            stand=torch.nonzero((up > 0.9) & (torch.arange(len(up), device=dev) < 300))[:, 0], held=torch.nonzero(up < -0.85)[:, 0], n=len(up),
        )
    B = env._g1s_bank
    n = len(env_ids)
    u = torch.rand(n, device=env.device)
    f_stand, f_held = float(os.environ.get("FLASHSAC_G1S_FSTAND", "0.25")), float(os.environ.get("FLASHSAC_G1S_FHELD", "0.35"))
    fr = torch.randint(0, B["n"], (n,), device=env.device)
    fr = torch.where(u < f_stand, B["stand"][torch.randint(0, len(B["stand"]), (n,), device=env.device)], fr)
    fr = torch.where((u >= f_stand) & (u < f_stand + f_held), B["held"][torch.randint(0, len(B["held"]), (n,), device=env.device)], fr)
    if not hasattr(env, "_g1s_fr"):
        env._g1s_fr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._g1s_fr[env_ids] = fr
    if not hasattr(env, "_g1s_k"):
        env._g1s_k = torch.zeros(env.num_envs, device=env.device)
    env._g1s_k[env_ids] = 0.0
    if hasattr(env, "_g1s_anc_ok"):
        env._g1s_anc_ok[env_ids] = False
    if hasattr(env, "_g1s_ranc_ok"):
        env._g1s_ranc_ok[env_ids] = False
    if hasattr(env, "_g1s_hcnt"):
        env._g1s_hcnt[env_ids] = 0
    st =torch.cat([B["pos"][fr], B["quat"][fr]], 1)
    st[:, :2] += env.scene.env_origins[env_ids, :2]
    st[:, 2] += 0.01
    a.write_root_pose_to_sim(st, env_ids=env_ids)
    a.write_root_velocity_to_sim(torch.cat([B["vel"][fr], torch.zeros(n, 3, device=env.device)], 1), env_ids=env_ids)
    jp = B["jp"][fr] + 0.02 * torch.randn(n, a.num_joints, device=env.device)
    a.write_joint_state_to_sim(jp, B["jv"][fr], env_ids=env_ids)


CLIP_FPS = 30.0


def demo_actions(env, t_offset: float = 0.0) -> torch.Tensor:
    """Open-loop replay of the MOYO clip: per env, the action whose joint target is the clip pose one control step ahead
    of the env's time since its RSI start frame (own counter, advanced here: call once per env step).
    Returned in the wrapper's [-1, 1] units (clamped)."""
    B = env._g1s_bank
    term = env.action_manager._terms["joint_pos"]
    env._g1s_k += 1.0
    t = env._g1s_k * env.step_dt + t_offset
    f = (env._g1s_fr.float() + t * CLIP_FPS).clamp(0, B["n"] - 1.001)
    i0 = f.floor().long(); w = (f - i0).unsqueeze(1)
    q = B["jp"][i0] * (1 - w) + B["jp"][i0 + 1] * w
    return ((q - term._offset) / term._scale).clamp(-1.0, 1.0)


def demo_actions_raw(env) -> torch.Tensor:
    B = env._g1s_bank
    term = env.action_manager._terms["joint_pos"]
    t = (env._g1s_k + 1.0) * env.step_dt
    f = (env._g1s_fr.float() + t * CLIP_FPS).clamp(0, B["n"] - 1.001)
    i0 = f.floor().long(); w = (f - i0).unsqueeze(1)
    q = B["jp"][i0] * (1 - w) + B["jp"][i0 + 1] * w
    return (q - term._offset) / term._scale


def reset_stand_mix(env, env_ids) -> None:
    """Optionally reset a fraction of the envs from the standing pose (phase 3 curriculum)."""
    frac = float(os.environ.get("FLASHSAC_G1S_STAND", "0"))
    if frac <= 0.0:
        return
    a = _a(env)
    sel = env_ids[torch.rand(len(env_ids), device=env.device) < frac]
    if len(sel) == 0:
        return
    st = a.data.default_root_state[sel].clone()
    st[:, :3] += env.scene.env_origins[sel]
    st[:, 2] = 0.8
    st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    st[:, 7:] = 0.0
    a.write_root_pose_to_sim(st[:, :7], env_ids=sel)
    a.write_root_velocity_to_sim(st[:, 7:], env_ids=sel)
    jp = torch.zeros(len(sel), a.num_joints, device=env.device)
    a.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=sel)


FRONT = SceneEntityCfg("contact_forces", body_names=BAD)


@configclass
class StandObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.2, n_max=0.2))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StandRewardsCfg:
    torso_vertical = RewTerm(func=torso_vertical, weight=6.0)
    vertical_legs = RewTerm(func=vertical_legs, weight=3.0)
    vertical_pelvis = RewTerm(func=vertical_pelvis, weight=3.0)
    hold_bonus = RewTerm(func=hold_bonus, weight=5.0)
    fore_missing = RewTerm(func=fore_missing, weight=-1.0)  # per missing contact point (4 max)
    feet_on_ground = RewTerm(func=feet_on_ground, weight=-1.0)  # per foot on the ground
    arm_symmetry = RewTerm(func=arm_symmetry, weight=-1.0)
    balance_xy = RewTerm(func=balance_xy, weight=-4.0)
    bad_contact = RewTerm(func=bad_contact, weight=-1.0, params={"sensor_cfg": FRONT})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    leg_action_rate = RewTerm(func=leg_action_rate, weight=float(os.environ.get("FLASHSAC_G1S_LEGAR", "-0.2")))
    leg_joint_vel = RewTerm(func=leg_joint_vel, weight=float(os.environ.get("FLASHSAC_G1S_LEGVEL", "-0.005")))
    arm_slide = RewTerm(func=arm_slide, weight=float(os.environ.get("FLASHSAC_G1S_SLIDE", "-3.0")))
    hands_lift = RewTerm(func=hands_lift, weight=float(os.environ.get("FLASHSAC_G1S_HLIFT", "0.0")))
    hand_anchor = RewTerm(func=hand_anchor, weight=float(os.environ.get("FLASHSAC_G1S_ANCHOR", "0.0")))
    leg_cross = RewTerm(func=leg_cross, weight=float(os.environ.get("FLASHSAC_G1S_CROSS", "0.0")))
    root_anchor = RewTerm(func=root_anchor, weight=float(os.environ.get("FLASHSAC_G1S_ROOTANC", "0.0")))
    root_drift = RewTerm(func=root_drift, weight=float(os.environ.get("FLASHSAC_G1S_DRIFT", "0.0")))
    arm_joint_vel = RewTerm(func=arm_joint_vel, weight=float(os.environ.get("FLASHSAC_G1S_ARMVEL", "0.0")))
    hip_twist = RewTerm(func=hip_twist, weight=float(os.environ.get("FLASHSAC_G1S_HIPTW", "0.0")))
    com_forward = RewTerm(func=com_forward, weight=float(os.environ.get("FLASHSAC_G1S_COMW", "0.0")))
    terminated = RewTerm(func=mdp.is_terminated, weight=float(os.environ.get("FLASHSAC_G1S_TERMPEN", "0.0")))  # falls (not time-outs)
    live_monitor = RewTerm(func=live_monitor, weight=1.0e-9)


@configclass
class StandEventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.1, 0.1)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0)},
    )
    stand_mix = EventTerm(func=reset_stand_mix, mode="reset")
    rsi = EventTerm(func=reset_rsi, mode="reset")
    push_robot = None  # set from FLASHSAC_G1S_PUSH: random pelvis velocity kicks, to train reactive balance


@configclass
class StandTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_contact = DoneTerm(
        func=bad_contact_term,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 20.0},
    )


def _robot_cfg() -> ArticulationCfg:
    leg_kp = {".*_hip_.*": 100.0, ".*_knee_joint": 200.0}
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(POSE["root_pos"][0], POSE["root_pos"][1], POSE["root_pos"][2] + 0.02),
            rot=tuple(POSE["root_quat_wxyz"]),
            joint_pos=dict(POSE["joint_pos"]),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
                effort_limit={".*_hip_.*": 88.0, ".*_knee_joint": 139.0}, velocity_limit=30.0,
                stiffness=leg_kp, damping={".*_hip_.*": 2.5, ".*_knee_joint": 5.0}, armature=0.03,
            ),
            "feet": IdealPDActuatorCfg(
                joint_names_expr=[".*_ankle_.*"], effort_limit=50.0, velocity_limit=30.0,
                stiffness=40.0, damping=2.0, armature=0.03,
            ),
            "waist": IdealPDActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                effort_limit={"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0}, velocity_limit=30.0,
                stiffness=60.0, damping=3.0, armature=0.01,
            ),
            "arms": IdealPDActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"],
                effort_limit=25.0, velocity_limit=30.0, stiffness=60.0, damping=3.0, armature=0.01,
            ),
            "wrist": IdealPDActuatorCfg(
                joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
                effort_limit=5.0, velocity_limit=30.0, stiffness=15.0, damping=1.0, armature=0.005,
            ),
        },
    )


@configclass
class G1ForearmStandEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = float(os.environ.get("FLASHSAC_G1S_EPLEN", "6.0"))
        self.commands = None
        self.curriculum = None
        self.scene.robot = _robot_cfg()
        self.observations = StandObsCfg()
        self.rewards = StandRewardsCfg()
        self.events = StandEventsCfg()
        self.terminations = StandTerminationsCfg()
        if os.environ.get("G1S_NOTERM"):
            self.terminations.bad_contact = None
        self.actions.joint_pos.joint_names = [".*"]
        self.actions.joint_pos.scale = {
            ".*_hip_pitch_joint": ACTION_SCALE, ".*_knee_joint": ACTION_SCALE, ".*_shoulder_pitch_joint": ACTION_SCALE,
            ".*_wrist_.*": 0.5, ".*_hip_(roll|yaw)_joint": 1.0, ".*_ankle_.*": 1.0, "waist_.*_joint": 1.0,
            ".*_shoulder_(roll|yaw)_joint": 1.0, ".*_elbow_joint": 1.0,
        }
        if os.environ.get("FLASHSAC_G1S_WIDE"):
            # wider ranges so the MOYO clip targets are reachable (wrist roll / hip pitch / elbow saturated with the old scales)
            self.actions.joint_pos.scale = {
                ".*_hip_pitch_joint": 3.5, ".*_knee_joint": 2.0, ".*_shoulder_pitch_joint": 2.0,
                ".*_wrist_roll_joint": 3.0, ".*_wrist_pitch_joint": 0.5, ".*_wrist_yaw_joint": 1.5,
                ".*_hip_(roll|yaw)_joint": 1.0, ".*_ankle_.*": 1.5, "waist_.*_joint": 1.0,
                ".*_shoulder_(roll|yaw)_joint": 1.5, ".*_elbow_joint": 2.0,
            }
        self.actions.joint_pos.use_default_offset = True
        if os.environ.get("FLASHSAC_G1S_PUSH"):
            v = float(os.environ["FLASHSAC_G1S_PUSH"])
            self.events.push_robot = EventTerm(
                func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.5, 3.0),
                params={"velocity_range": {"x": (-v, v), "y": (-v, v), "roll": (-2 * v, 2 * v), "pitch": (-2 * v, 2 * v)}},
            )
        if os.environ.get("BF_PLAY"):
            self.observations.policy.enable_corruption = False
            self.events.reset_joints = None
            self.viewer.eye = (0.0, -3.0, 0.9)
            self.viewer.lookat = (0.0, 0.0, 0.5)


def register() -> None:
    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "flash_rl.envs.g1_forearm_stand_isaaclab_env:G1ForearmStandEnvCfg"},
    )
