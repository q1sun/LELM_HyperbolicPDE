import torch
import math

def u_Exact_Solution(x, t):
    
    return u0_Exact_Solution(x - t)

def u0_Exact_Solution(x):
    
    f = torch.zeros(x.size(0))
    mask = x < 0
    f[mask]= 2
    return f

    

def augmented_variable(x,t, s):
    temp = torch.ones(x.size(0))
    value = torch.heaviside(x - s * t, torch.squeeze(temp))
    return value


