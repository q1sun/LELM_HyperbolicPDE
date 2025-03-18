import torch

def u_Exact_Solution(x, y, t):
    u = - torch.ones(x.size(0))
    mask1 = (x-2 - 0.5*t) < 0
    u[mask1] = 2
    mask2 = (x-1 - 3 * t) < 0
    u[mask2] = 4
    return u

def u0val_Exact_Solution(x, y):
    
    f = -torch.ones(x.size(0))
    mask1 = x < 2
    f[mask1]= 2
    mask2 = x < 1
    f[mask2] = 4
    return f

    

def augmented_variable(x, y, t):
    temp = torch.ones(x.size(0))
    value = torch.heaviside(x-2 - 0.5*t, torch.squeeze(temp)) + torch.heaviside(x-1 - 3*t, torch.squeeze(temp))
    return value


