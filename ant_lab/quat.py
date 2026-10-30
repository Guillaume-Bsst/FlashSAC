"""Quaternions torch en (w, x, y, z). Les dimensions de tête sont diffusées."""
import torch


def _as_tensor(x, like=None):
    if isinstance(x, torch.Tensor):
        return x
    dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float64
    device = like.device if isinstance(like, torch.Tensor) else None
    return torch.as_tensor(x, dtype=dtype, device=device)


def quat_mul(a, b):
    a, b = torch.broadcast_tensors(_as_tensor(a), _as_tensor(b, a))
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conj(q):
    q = _as_tensor(q)
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_apply(q, v):
    q = _as_tensor(q)
    v = _as_tensor(v, q)
    qv = q[..., 1:]
    w = q[..., :1]
    qv, v = torch.broadcast_tensors(qv, v)
    w = w.expand(qv.shape[:-1] + (1,))
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(quat_conj(q), v)


def yaw_quat(q):
    """Rotation de lacet seule (autour de z) contenue dans q."""
    q = _as_tensor(q)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)


def quat_error_magnitude(a, b):
    """Angle de la rotation relative, dans [0, π] (q et −q donnent 0)."""
    d = quat_mul(quat_conj(a), b)
    vn = d[..., 1:].norm(dim=-1)
    return 2.0 * torch.atan2(vn, d[..., 0].abs())


def quat_to_mat(q):
    q = _as_tensor(q)
    w, x, y, z = q.unbind(-1)
    m = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_from_rotvec(r):
    r = _as_tensor(r)
    angle = r.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(θ/2)/θ, avec sa limite 1/2 en 0
    small = angle < 1e-8
    k = torch.where(small, 0.5 - angle * angle / 48.0, torch.sin(half) / torch.where(small, torch.ones_like(angle), angle))
    return torch.cat([torch.cos(half), r * k], dim=-1)


def make_continuous(q):
    """Sur l'axe 0 : change le signe pour que deux frames consécutives aient un produit scalaire ≥ 0."""
    q = _as_tensor(q).clone()
    for i in range(1, q.shape[0]):
        flip = (q[i] * q[i - 1]).sum(-1, keepdim=True) < 0
        q[i] = torch.where(flip, -q[i], q[i])
    return q


def ang_vel_world(q, dt):
    """Vitesse angulaire en repère monde : 2·vec(q̇ ⊗ q*), q̇ par différences (axe 0)."""
    q = make_continuous(q)
    qd = torch.empty_like(q)
    qd[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    qd[0] = (q[1] - q[0]) / dt
    qd[-1] = (q[-1] - q[-2]) / dt
    return 2.0 * quat_mul(qd, quat_conj(q))[..., 1:]
