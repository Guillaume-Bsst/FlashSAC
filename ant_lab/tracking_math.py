"""Reward, observation d'ancre et terminaison de BeyondMimic (torch pur, sans simulateur).

Sources : ``holosoma/managers/command/terms/wbt.py`` (cibles relatives, l. 1555–1577) et
``holosoma/managers/reward/terms/wbt.py`` (termes de reward).
"""
import torch

from ant_lab import quat as Q
from ant_lab.config import REWARD_SIGMA, REWARD_WEIGHT, SOFT_LIMIT_FACTOR, TERM


def relative_body_targets(ref_anchor_pos, ref_anchor_quat, ref_body_pos, ref_body_quat,
                          robot_anchor_pos, robot_anchor_quat):
    """Corps de la référence ré-ancrés sur l'ancre du robot (position xy et lacet du robot, hauteur de la réf)."""
    B = ref_body_pos.shape[-2]
    dq = Q.yaw_quat(Q.quat_mul(robot_anchor_quat, Q.quat_conj(ref_anchor_quat)))
    dq_b = dq.unsqueeze(-2).expand(dq.shape[:-1] + (B, 4))
    anchor = torch.cat([robot_anchor_pos[..., :2], ref_anchor_pos[..., 2:]], dim=-1)
    pos_rel = anchor.unsqueeze(-2) + Q.quat_apply(dq_b, ref_body_pos - ref_anchor_pos.unsqueeze(-2))
    quat_rel = Q.quat_mul(dq_b, ref_body_quat)
    return pos_rel, quat_rel


def anchor_obs(ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat):
    """(N, 9) : position de l'ancre de réf en repère du robot (3) + deux colonnes de la rotation relative (6)."""
    p = Q.quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    m = Q.quat_to_mat(Q.quat_mul(Q.quat_conj(robot_anchor_quat), ref_anchor_quat))[..., :, :2]
    return torch.cat([p, m.reshape(m.shape[:-2] + (6,))], dim=-1)


def _exp(err_sq, key):
    return torch.exp(-err_sq / REWARD_SIGMA[key] ** 2)


def reward_terms(*, ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat, body_pos_rel,
                 body_quat_rel, robot_body_pos, robot_body_quat, ref_body_lin_vel, ref_body_ang_vel,
                 robot_body_lin_vel, robot_body_ang_vel, actions, prev_actions, joint_pos, joint_limits):
    """Termes non pondérés, (N,) chacun, clés = REWARD_WEIGHT."""
    t = {}
    t["anchor_pos"] = _exp(((ref_anchor_pos - robot_anchor_pos) ** 2).sum(-1), "anchor_pos")
    t["anchor_ori"] = _exp(Q.quat_error_magnitude(ref_anchor_quat, robot_anchor_quat) ** 2, "anchor_ori")
    t["body_pos"] = _exp(((body_pos_rel - robot_body_pos) ** 2).sum(-1).mean(-1), "body_pos")
    t["body_ori"] = _exp((Q.quat_error_magnitude(body_quat_rel, robot_body_quat) ** 2).mean(-1), "body_ori")
    t["body_lin_vel"] = _exp(((ref_body_lin_vel - robot_body_lin_vel) ** 2).sum(-1).mean(-1), "body_lin_vel")
    t["body_ang_vel"] = _exp(((ref_body_ang_vel - robot_body_ang_vel) ** 2).sum(-1).mean(-1), "body_ang_vel")
    t["action_rate"] = ((actions - prev_actions) ** 2).sum(-1)
    mid = 0.5 * (joint_limits[..., 0] + joint_limits[..., 1])
    half = 0.5 * (joint_limits[..., 1] - joint_limits[..., 0]) * SOFT_LIMIT_FACTOR
    low, high = mid - half, mid + half
    t["joint_limit"] = ((low - joint_pos).clamp(min=0) + (joint_pos - high).clamp(min=0)).sum(-1)
    return t


def total_reward(terms):
    return sum(REWARD_WEIGHT[k] * terms[k] for k in REWARD_WEIGHT)


def termination_flags(*, ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat, body_pos_rel,
                      robot_body_pos, foot_ids):
    down = torch.tensor([0.0, 0.0, -1.0], dtype=ref_anchor_quat.dtype, device=ref_anchor_quat.device)
    g_ref = Q.quat_apply_inverse(ref_anchor_quat, down.expand(ref_anchor_quat.shape[:-1] + (3,)))
    g_rob = Q.quat_apply_inverse(robot_anchor_quat, down.expand(robot_anchor_quat.shape[:-1] + (3,)))
    bad_z = (ref_anchor_pos[..., 2] - robot_anchor_pos[..., 2]).abs() > TERM["anchor_z"]
    bad_g = (g_ref[..., 2] - g_rob[..., 2]).abs() > TERM["anchor_grav_z"]
    dz = (body_pos_rel[..., foot_ids, 2] - robot_body_pos[..., foot_ids, 2]).abs()
    bad_foot = (dz > TERM["foot_z"]).any(-1)
    return bad_z | bad_g | bad_foot
