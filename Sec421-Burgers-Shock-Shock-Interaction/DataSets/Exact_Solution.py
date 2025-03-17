import torch
import math

def u_Exact_Solution(x, t):
    
    mask0 = t <= (5 - math.sqrt(13))/2 # find the point before intersection

    x0 = x[mask0]
    t0 = t[mask0]
    mask_left = x0 < torch.sqrt(1 + t0)
    u0 = torch.zeros(x0.size())
    u0[mask_left] = (x0[mask_left] / (1 + t0[mask_left]))
    mask_right = x0 > (2 - 0.000001 - t0) 
    u0[mask_right] = -2

    mask1 = t > (5 - math.sqrt(13))/2
    x1 = x[mask1]
    t1 = t[mask1]

    mask = x1 <= math.sqrt(13) * torch.sqrt(1 + t1) - 2 * (1 + t1)
    u1 = -2 * torch.ones(x1.size(0))
    u1[mask] = (x1[mask] / (1 + t1[mask]))

    u = torch.zeros(x.size(0))
    u[mask0] = u0
    u[mask1] = u1

    return u

def u0_Exact_Solution(x):
    
    f = torch.zeros(x.size())
    mask0 = x < 1
    f[mask0]= x[mask0]
    mask1 = x >= 2
    f[mask1] = -2
    return f

    

def augmented_variable(x,t):
    temp = torch.ones(x.size(0))
    mask0 = t <= (5 - math.sqrt(13))/2
    value0 = torch.heaviside(x[mask0] - torch.sqrt(1 + t[mask0]), torch.squeeze(temp[mask0])) + torch.heaviside(x[mask0] + t[mask0] - 2, torch.squeeze(temp[mask0]))
    mask1 = torch.logical_not(mask0)
    value1 = torch.heaviside(x[mask1] - (math.sqrt(13) * torch.sqrt(1 + t[mask1]) - 2 * (1 + t[mask1])), torch.squeeze(temp[mask1])) * 2
    value = torch.zeros(x.size(0))
    value[mask0] = value0
    value[mask1] = value1
    return value


 