import math

import torch

from ant_lab.quat import quat_apply, quat_from_rotvec, quat_mul
from ant_lab.tracking_math import (
    anchor_obs,
    heading_velocity,
    relative_body_targets,
    reward_terms,
    termination_flags,
    total_reward,
)

N, B, J = 4, 9, 8
FEET = [2, 4, 6, 8]


def _rand_pose(seed=0):
    g = torch.Generator().manual_seed(seed)
    anchor_p = torch.randn(N, 3, generator=g)
    anchor_q = torch.randn(N, 4, generator=g)
    anchor_q = anchor_q / anchor_q.norm(dim=-1, keepdim=True)
    body_p = torch.randn(N, B, 3, generator=g)
    body_q = torch.randn(N, B, 4, generator=g)
    body_q = body_q / body_q.norm(dim=-1, keepdim=True)
    lin = torch.randn(N, B, 3, generator=g)
    ang = torch.randn(N, B, 3, generator=g)
    return anchor_p, anchor_q, body_p, body_q, lin, ang


def _limits():
    return torch.tensor([[-1.0, 1.0]] * J).expand(N, J, 2)


def _terms(ap, aq, bp, bq, lin, ang, rap, raq, rbp, rbq, rlin, rang, joint_pos=None, actions=None, prev=None):
    pos_rel, quat_rel = relative_body_targets(ap, aq, bp, bq, rap, raq)
    return reward_terms(
        ref_anchor_pos=ap, ref_anchor_quat=aq, robot_anchor_pos=rap, robot_anchor_quat=raq,
        body_pos_rel=pos_rel, body_quat_rel=quat_rel, robot_body_pos=rbp, robot_body_quat=rbq,
        ref_body_lin_vel=lin, ref_body_ang_vel=ang, robot_body_lin_vel=rlin, robot_body_ang_vel=rang,
        actions=torch.zeros(N, J) if actions is None else actions,
        prev_actions=torch.zeros(N, J) if prev is None else prev,
        joint_pos=torch.zeros(N, J) if joint_pos is None else joint_pos, joint_limits=_limits(),
    )


def test_exact_match_reward():
    ap, aq, bp, bq, lin, ang = _rand_pose()
    t = _terms(ap, aq, bp, bq, lin, ang, ap, aq, bp, bq, lin, ang)
    for k in ["anchor_pos", "anchor_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel"]:
        assert torch.allclose(t[k], torch.ones(N), atol=1e-5), k
    assert torch.all(t["action_rate"] == 0) and torch.all(t["joint_limit"] == 0)
    assert torch.allclose(total_reward(t), torch.full((N,), 5.0), atol=1e-4)


def test_relative_targets_shift_yaw():
    ap, aq, bp, bq, _, _ = _rand_pose(1)
    ap = ap.clone()
    ap[:, 2] = 0.5
    aq = quat_from_rotvec(torch.tensor([0.0, 0.0, 0.2])).float().expand(N, 4)
    yaw90 = quat_from_rotvec(torch.tensor([0.0, 0.0, math.pi / 2])).float()
    rap = ap + torch.tensor([1.0, 2.0, 0.0])
    rap[:, 2] = 0.7
    raq = quat_mul(yaw90.expand(N, 4), aq)
    pos_rel, quat_rel = relative_body_targets(ap, aq, bp, bq, rap, raq)
    exp = quat_apply(yaw90.expand(N, B, 4), bp - ap[:, None])
    exp = exp + torch.cat([rap[:, :2], ap[:, 2:]], -1)[:, None]
    assert torch.allclose(pos_rel, exp, atol=1e-5)
    assert torch.allclose(quat_rel, quat_mul(yaw90.expand(N, B, 4), bq), atol=1e-5)


def test_anchor_pos_error():
    ap, aq, bp, bq, lin, ang = _rand_pose(2)
    rap = ap + torch.tensor([0.3, 0.0, 0.0])
    t = _terms(ap, aq, bp, bq, lin, ang, rap, aq, bp, bq, lin, ang)
    assert torch.allclose(t["anchor_pos"], torch.full((N,), math.exp(-1)), atol=1e-5)


def test_anchor_obs_identity():
    ap, aq, *_ = _rand_pose(3)
    o = anchor_obs(ap, aq, ap, aq)
    assert o.shape == (N, 9)
    assert torch.allclose(o, torch.tensor([0, 0, 0, 1, 0, 0, 1, 0, 0.0]).expand(N, 9), atol=1e-5)


def _term(ap, aq, bp, rap, raq, rbp):
    pos_rel, _ = relative_body_targets(ap, aq, bp, torch.zeros(N, B, 4) + torch.tensor([1.0, 0, 0, 0]), rap, raq)
    return termination_flags(ref_anchor_pos=ap, ref_anchor_quat=aq, robot_anchor_pos=rap, robot_anchor_quat=raq,
                             body_pos_rel=pos_rel, robot_body_pos=rbp, foot_ids=FEET)


def test_termination():
    ident = torch.tensor([1.0, 0, 0, 0]).expand(N, 4)
    ap = torch.tensor([0.0, 0, 0.5]).expand(N, 3)
    bp = torch.zeros(N, B, 3)
    bp[..., 2] = 0.1
    small = torch.tensor([0.0, 0, 0.1])
    # tous les écarts à 0,1 -> pas de terminaison
    rbp = bp + small
    assert not _term(ap, ident, bp, ap + small, ident, rbp).any()
    # écart de z de l'ancre à 0,3
    assert _term(ap, ident, bp, ap + torch.tensor([0.0, 0, 0.3]), ident, bp.clone()).all()
    # robot tourné de 90° autour de x
    rx = quat_from_rotvec(torch.tensor([math.pi / 2, 0.0, 0.0])).float().expand(N, 4)
    assert _term(ap, ident, bp, ap, rx, bp.clone()).all()
    # un pied à 0,3
    rbp = bp.clone()
    rbp[:, FEET[1], 2] += 0.3
    assert _term(ap, ident, bp, ap, ident, rbp).all()


def test_joint_limit_penalty():
    ap, aq, bp, bq, lin, ang = _rand_pose(4)
    q = torch.zeros(N, J)
    q[:, 0] = 1.0 + 0.1  # limites ±1 : pénalité au-delà de la vraie butée (facteur 1,0, décision E0)
    t = _terms(ap, aq, bp, bq, lin, ang, ap, aq, bp, bq, lin, ang, joint_pos=q)
    assert torch.allclose(t["joint_limit"], torch.full((N,), 0.1), atol=1e-5)
    q[:, 0] = 0.95       # dans les butées : suivre une référence collée à la butée ne coûte rien
    t = _terms(ap, aq, bp, bq, lin, ang, ap, aq, bp, bq, lin, ang, joint_pos=q)
    assert torch.allclose(t["joint_limit"], torch.zeros(N), atol=1e-6)


def test_heading_velocity_ignores_heading_drift_and_vertical_speed():
    yaw = torch.tensor([0.0, 0.7, -1.2, 3.0])
    q = quat_from_rotvec(torch.stack([torch.zeros(N), torch.zeros(N), yaw], -1))
    fwd = quat_apply(q, torch.tensor([4.0, 0.0, 0.3]).expand(N, 3))   # 4 m/s vers l'avant de chaque torse
    assert torch.allclose(heading_velocity(fwd, q), torch.tensor([4.0, 0.0]).expand(N, 2), atol=1e-5)
    assert torch.allclose(heading_velocity(torch.zeros(N, 3), q), torch.zeros(N, 2), atol=1e-6)
