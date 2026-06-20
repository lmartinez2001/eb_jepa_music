"""Minimal axis-angle <-> rotation-matrix <-> 6D conversions (pytorch3d convention).

6D rep = first two rows of the rotation matrix, flattened (Zhou et al. 2019).
"""
import torch
import torch.nn.functional as F

_EPS = 1e-8


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(aa, dim=-1, keepdim=True)
    axis = aa / theta.clamp(min=_EPS)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    K = K.reshape(aa.shape[:-1] + (3, 3))
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1.0 - c) * (K @ K)


def matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
    return m[..., :2, :].clone().reshape(m.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_axis_angle(m: torch.Tensor) -> torch.Tensor:
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos).unsqueeze(-1)
    vx = m[..., 2, 1] - m[..., 1, 2]
    vy = m[..., 0, 2] - m[..., 2, 0]
    vz = m[..., 1, 0] - m[..., 0, 1]
    v = torch.stack([vx, vy, vz], dim=-1)
    axis = v / v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return axis * theta


def axis_angle_to_6d(aa: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))
