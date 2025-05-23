# satellite_util.py

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

import math

def sample_random_quaternion(target_device):
    u = torch.rand(3, device=target_device)
    w = torch.sqrt(1 - u[0]) * torch.sin(2 * math.pi * u[1])
    x = torch.sqrt(1 - u[0]) * torch.cos(2 * math.pi * u[1])
    y = torch.sqrt(u[0]) * torch.sin(2 * math.pi * u[2])
    z = torch.sqrt(u[0]) * torch.cos(2 * math.pi * u[2])
    return torch.tensor([x, y, z, w], dtype=torch.float32, device=target_device)

def sample_random_quaternion_batch(target_device, n):
    if n == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=target_device)
    quats = [sample_random_quaternion(target_device) for _ in range(n)]
    return torch.stack(quats)
 
def quat_diff(q1, q2):
    q2_inv = quat_conjugate(q2)
    return quat_mul(q1, q2_inv)

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    return 2.0 * torch.asin(
        torch.clamp(
            torch.norm(
                mul[:, 0:3],
                p=2, dim=-1), max=1.0)
    )


 
