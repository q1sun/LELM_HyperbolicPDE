import torch
import math

def SmpPts_Interior(num_intrr_pts, dim_prob=2):    
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """

    # entire domain (-1,1) * (-1,1)                
    X = torch.rand(num_intrr_pts * 2, dim_prob-1) * 2
    T = torch.rand(num_intrr_pts * 2, dim_prob-1) * 0.5
    X_d = torch.cat([X, T], dim=1)
    return X_d

def SmpPts_Initial(num_init_pts, dim_prob=2):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_init_pts, 1)

    X_init = torch.cat([torch.rand(num_init_pts, dim_prob-1) * 2, temp0], dim=1)

    return X_init

def SmpPts_Boundary(num_bndry_pts, dim_prob):

    temp0 = torch.zeros(num_bndry_pts, 1)
    temp1 = 2 * torch.ones(num_bndry_pts, 1)

    X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob - 1) * 0.5], dim = 1)
    X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob - 1) * 0.5], dim = 1)

    return X_left, X_right

def SmpPts_Shock(num_shock_pts, dim_prob, spline):
    t = torch.rand(num_shock_pts) * 0.5 
    x = spline.evaluate(torch.squeeze(t)).reshape(-1, 1)
    return torch.cat([x, t.reshape(-1, 1)], dim=1)
    