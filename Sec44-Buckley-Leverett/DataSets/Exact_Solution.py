import torch
import numpy as np

def flux(S, M):
    return S**2 / (S**2 + M*(1 - S)**2)

def flux_prime(S, M):
    num = 2 * M * S * (1 - S)
    den = (S**2 + M*(1 - S)**2)**2
    return num / den

def u_Exact_Solution(x, t, M):
    u = torch.zeros_like(x)
    S_L, S_R = 1.0, 0.0
    S_star = np.sqrt(M / (M + 1))
    s = flux(S_star, M) / S_star  # shock speed
    
    xi = x / t
    xi_L = flux_prime(S_L, M)     # left char speed (0)
    xi_star = flux_prime(S_star, M)
    
    # 左侧常值区
    maskL = xi < xi_L
    u[maskL] = S_L
    
    # 稀疏波区
    maskM = (xi >= xi_L) & (xi <= xi_star)
    if maskM.any():
        xi_mid = xi[maskM]
        # 由 ξ = f'(S) 反解出 S(ξ)
        S_vals = []
        for val in xi_mid:
            # 二分求解 f'(S)=ξ
            a, b = S_star, S_L
            for _ in range(50):
                m = 0.5*(a+b)
                if flux_prime(m, M) > val:
                    a = m
                else:
                    b = m
            S_vals.append(0.5*(a+b))
        u[maskM] = torch.tensor(S_vals, dtype=u.dtype)
    
    # 激波后常值区
    maskR = xi > s
    u[maskR] = S_R
    
    # 激波前常值区（稀疏波结束到shock前）
    maskS = (xi > xi_star) & (xi <= s)
    u[maskS] = S_star
    
    return u

def u0_Exact_Solution(x):
    
    f = torch.zeros(x.size(0))
    mask = x < 0
    f[mask]= 1
    return f

    

def augmented_variable(x, t, s):
    temp = torch.ones(x.size(0))
    value = torch.heaviside(x - s * t, torch.squeeze(temp))
    return value
