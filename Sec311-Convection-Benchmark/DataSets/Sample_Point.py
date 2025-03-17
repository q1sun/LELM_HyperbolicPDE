import torch
import math
import numpy as np

def SmpPts_Interior(num_intrr_pts, dim_prob=2):    
    ''' num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain ''' 

    # entire domain                
    X = torch.rand(num_intrr_pts * 2, dim_prob-1) * 2 * math.pi
    T = torch.rand(num_intrr_pts * 2, dim_prob-1) * math.pi * 2 
    i = 0

    while i < len(X):
        if abs(X[i] + T[i] - 2/3 * math.pi) <= 0.001 or abs(X[i] + T[i] - 4/3 * math.pi) <= 0.0001:
            X[i] = 0
            T[i] = 0
        i += 1
    test = X + T
    X_test = torch.cat([X, T], dim=1)
    index = torch.nonzero(test)
    X_t = torch.index_select(X_test, 0, index[:,0], out=None)
    dex = torch.arange(0, num_intrr_pts)
    X_d = torch.index_select(X_t, 0, dex)
            
    return X_d

def SmpPts_Initial(num_initl_pts, dim_prob=2):
    ''' num_initl_pts = total number of sampling points for initial condition '''

    temp0 = torch.zeros(num_initl_pts, 1)

    X_initl = torch.cat([torch.rand(num_initl_pts, dim_prob-1) * 2 * math.pi, temp0], dim=1)

    return X_initl

def SmpPts_Boundary(num_bndry_pts, dim_prob, T):
    ''' num_bndry_pts = total number of sampling points for boundary condition '''

    temp0 = torch.zeros(num_bndry_pts, 1)
    temp1 = torch.ones(num_bndry_pts, 1) * 2 * math.pi
    x_rand = torch.rand(num_bndry_pts, dim_prob - 1) * T

    X_left = torch.cat([temp0, x_rand], dim = 1)
    X_right = torch.cat([temp1, x_rand], dim = 1)

    return X_left, X_right

def SmpPts_Shock(num_shock_pts, dim_prob, a, T):
    ''' num_shock_pts = total number of sampling points for shock curves '''

    if a == 0:
        raise ValueError('Error, constant a should not be zero')
    xt1 = 2/3 * math.pi
    xt2 = 4/3 * math.pi

    # the first two shock line
    # x = at + x_0
    t1 = torch.rand(num_shock_pts, dim_prob - 1) * (math.pi + np.sign(a)*math.pi - np.sign(a) * xt1) / abs(a)
    x1 = xt1 * torch.ones(num_shock_pts, dim_prob-1) + a * t1
    shock_line1 = torch.cat([x1, t1], dim=1)

    t2 = torch.rand(num_shock_pts, dim_prob - 1) * (math.pi + np.sign(a)*math.pi - np.sign(a) * xt2) / abs(a)
    x2 = xt2 * torch.ones(num_shock_pts, dim_prob-1) + a * t2
    shock_line2 = torch.cat([x2, t2], dim=1)

    # shock lines start at one boundary and end at the other boundary
    n = 2 * math.floor(abs(a)*T/(2 * math.pi)) + math.floor((abs(a)*T % (2*math.pi) / (2*math.pi))*3)
    temp = torch.ones(num_shock_pts, dim_prob-1)
    for i in torch.arange(2, n + 2):
        if i%2 == 0:
            xt = 2/3 * math.pi - np.sign(a) * 2 * math.pi * (i//2)
            x = torch.rand(num_shock_pts, dim_prob-1) * 2 * math.pi
            t = (x - xt*temp) / a
            # Delete points after time T
            if i >= n :
                mask = t <= 2*math.pi
                if t[mask].size(0) > 0:
                    t_min = min(t[mask])
                    t_max = max(t[mask])
                    t = torch.rand(num_shock_pts, dim_prob-1) * (t_max - t_min) + t_min
                    x = a * t + xt 
                else:
                    x, t = x[mask], t[mask]
            shock = torch.cat([x, t], dim=1)
            shock_line1 = torch.cat([shock_line1, shock], dim=0)
        else:
            xt = 4/3 * math.pi - np.sign(a) * 2 * math.pi * (i//2)
            x = torch.rand(num_shock_pts, dim_prob-1) * 2 * math.pi
            t = (x - xt*temp)/ a
            # Delete points after time T
            if i >= n:
                mask = t <= 2*math.pi
                if t[mask].size(0) > 0:
                    t_min = min(t[mask])
                    t_max = max(t[mask])
                    t = torch.rand(num_shock_pts, dim_prob-1) * (t_max - t_min) + t_min
                    x = a * t + xt 
            shock = torch.cat([x, t], dim=1)
            shock_line2 = torch.cat([shock_line2, shock], dim=0)
        
    return shock_line1, shock_line2
