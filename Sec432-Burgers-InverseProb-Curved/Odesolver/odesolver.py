import torch
from . import interpolate

def rk4(s, s_mid, t, x_0):
    # denote the time when using Rounge-Kutta method
    t0 = t[:-1] 
    t1 = t[1:] 
    dt = t1 - t0
    tmid = (t0 + t1) / 2

    # # get the function value in the correspond time
    # s0 = f(t0)
    # s1 = f(t1)
    # smid = f(tmid)

    # compute the difference between x(t_n) and x(t_n+1)
    s = torch.squeeze(s)
    s_mid = torch.squeeze(s_mid)
    dif = torch.squeeze(dt * (s[:-1] + 4 * s_mid + s[1:]))/6

    # the discrete value of x(t)
    M = torch.tril(torch.ones(len(dif), len(dif)))
    x = x_0 * torch.ones(t.size())
    x[1:] = x[1:] + torch.mv(M, dif)
    return torch.squeeze(x)    

def cubicintp(s, s_mid,t, x_0):
    x = rk4(s, s_mid, t, x_0).reshape(-1, 1)
    coeffs = interpolate.natural_cubic_spline_coeffs(t, x)
    spline = interpolate.NaturalCubicSpline(coeffs)
    return spline