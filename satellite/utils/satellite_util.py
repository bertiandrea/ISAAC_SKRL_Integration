# satellite_util.py

import math

import isaacgym #BugFix
import torch

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def sample_random_quaternion(target_device):
    u = torch.rand(3, device=target_device)
    w = torch.sqrt(1 - u[0]) * torch.sin(2 * math.pi * u[1])
    x = torch.sqrt(1 - u[0]) * torch.cos(2 * math.pi * u[1])
    y = torch.sqrt(u[0]) * torch.sin(2 * math.pi * u[2])
    z = torch.sqrt(u[0]) * torch.cos(2 * math.pi * u[2])
    return torch.tensor([w, x, y, z], dtype=torch.float32, device=target_device)

def sample_random_quaternion_batch(target_device, n):
    if n == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=target_device)
    quats = [sample_random_quaternion(target_device) for _ in range(n)]
    return torch.stack(quats)
