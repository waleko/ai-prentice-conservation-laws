import torch
from math import sqrt


def circular_dynamics(data):
    return (torch.arccos(data[..., 0] / (data ** 2).sum(axis=-1) ** 0.5) * torch.sign(data[..., 1]))[..., None]


pend_dynamics = ho_dynamics = circular_dynamics


def dp_le_dynamics(data):
    x1, x2 = data[..., [0, 1]], data[..., [2, 3]]
    sqrt2 = sqrt(2)
    return torch.cat((circular_dynamics(sqrt2 * x1 + x2), circular_dynamics(sqrt2 * x1 - x2)), axis=-1)


def co_dynamics(data):
    x1, x2 = data[..., [0, 2]], data[..., [1, 3]]
    return torch.cat((circular_dynamics(x1 + x2), circular_dynamics(x1 - x2)), axis=-1)


def kp_dynamics(data):
    return circular_dynamics(data[..., :2])