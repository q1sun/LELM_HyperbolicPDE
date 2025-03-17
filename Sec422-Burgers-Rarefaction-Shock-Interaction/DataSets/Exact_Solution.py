import torch
import math

def u_Exact_Solution(x, t):
    
    u = torch.zeros(x.size())

    # exact solution when t < 4
    ift1 = t < 4
    t1 = t[ift1]
    x1 = x[ift1]

    ## x <= t/2
    u1 = torch.ones(x1.size(0))
    mask = x1 <= t1/2
    u1[mask] = 2 * x1[mask] / t1[mask]
    
    ## x<=0
    mask = x1 <= 0
    u1[mask] = 0

    ## x>= t/4 + 1
    mask = x1 >= (t1/4 + 1)
    u1[mask] = 0

    # copy the value when t < 4 to the solution u
    u[ift1] = u1


    # exact solution when t >= 4
    ift2 = torch.logical_not(ift1)

    t2 = t[ift2]
    x2 = x[ift2]

    u2 = torch.zeros(x2.size(0))

    mask = x2 < torch.sqrt(t2)
    u2[mask] = 2 * x2[mask] / t2[mask]

    mask = x2 <= 0
    u2[mask] = 0

    # copy the value when t >=4 to the solution u
    u[ift2] = u2    

    return u



def u0_Exact_Solution(x):
    
    f = torch.ones(x.size(0))
    mask = x <= 0
    f[mask]= 0

    mask = x >= 1
    f[mask] = 0
    return f

    

def augmented_variable(x,t):
    value = torch.zeros(x.size())
    mask = t <= 4
    temp1 = torch.ones(x[mask].size(0))
    value1 = torch.heaviside(x[mask] - t[mask]/4 - 1, torch.squeeze(temp1))
    value[mask] = value1

    mask = torch.logical_not(mask)
    temp2 = torch.ones(x[mask].size(0))
    value2 = torch.heaviside(x[mask] - torch.sqrt(t[mask]), torch.squeeze(temp2))
    value[mask] = value2
    return value


