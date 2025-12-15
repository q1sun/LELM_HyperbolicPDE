import torch
import math
import numpy as np
import sodshock


def f_Exact(x):
    rho = torch.zeros(x.size())
    p = torch.zeros(x.size())
    u = torch.zeros(x.size())
    mask = x < 0.5
    not_mask = torch.logical_not(mask)
    
    rho[mask] = 1.0
    rho[not_mask] = 0.125
    
    p[mask] = 1.0
    p[not_mask] = 0.1

    u[mask] = 0.0
    u[not_mask] = 0.0

    f = torch.cat([rho.reshape(-1, 1), p.reshape(-1, 1), u.reshape(-1, 1)], dim=1)
    return torch.tensor(f)
    
head = -1.183215956619923
tail = -0.0702728125611825
def augmented_variable(x, t, s, c):
    temp = torch.ones(x.size())
    # return torch.heaviside(x - head * t - 0.5, torch.squeeze(temp)) + torch.heaviside(x - tail * t - 0.5, torch.squeeze(temp)) + torch.heaviside(x - s * t - 0.5, torch.squeeze(temp)) + torch.heaviside(x - c * t - 0.5, torch.squeeze(temp))
    return torch.heaviside(x - s * t - 0.5, torch.squeeze(temp)) + torch.heaviside(x - c * t - 0.5, torch.squeeze(temp))

