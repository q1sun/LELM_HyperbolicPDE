import torch


def u_Exact_Solution(x, t):
    mask = x <= torch.sqrt(1 + 4 * t)
    u = torch.zeros(x.size(0))
    u[mask] = (4 * x[mask] / (1 + 4 * t[mask]))
    return u

def u0_Exact_Solution(x):
    
    f = torch.zeros(x.size(0))
    mask = x <= 1
    f[mask]= 4 * x[mask]
    return f

    

def augmented_variable(x,t, shock):
    temp = torch.ones(x.size(0))
    value = torch.heaviside(torch.squeeze(x) - torch.squeeze(shock), torch.squeeze(temp))
    return value


