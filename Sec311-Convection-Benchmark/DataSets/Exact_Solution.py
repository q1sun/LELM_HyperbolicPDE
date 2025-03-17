import torch
import math
import numpy as np

def u_Exact_Solution(x, t, a):
    
    return u0_Exact_Solution(x - a * t)

def u0_Exact_Solution(x):

    i = 0
    f = []
    while (i < len(x)):
        while x[i] > 2*math.pi or x[i] < 0:
            if x[i] > 2 * math.pi:
                x[i] = x[i] - 2 * math.pi
            elif x[i] < 0:
                x[i] = x[i] + 2 * math.pi
            
        if x[i] < 2/3 * math.pi:
            f.append(0)
        elif x[i] >= 4/3 * math.pi:
            f.append(0)
        else:
            f.append(1)
        i += 1

    return torch.tensor(f)
    

def augmented_variable(x, t, a, T):

    n = 2 * math.floor(abs(a)*T/(2 * math.pi)) + math.floor((abs(a)*T % (2*math.pi) / (2*math.pi))*3)
    
    value = 0
    temp = torch.ones(x.size(0))
    for i in torch.arange(n+2):

        if i%2 == 0:
            xt = 2/3 * math.pi - np.sign(a) * 2 * math.pi * (i//2)
        else:
            xt = 4/3 * math.pi - np.sign(a) * 2 * math.pi * (i//2)
        
        value += torch.heaviside(x - a * t - xt, torch.squeeze(temp))

    return value
