import torch
import math

def SmpPts_Interior(num_intrr_pts, dim_prob=2):    
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """

    # entire domain (-1,1) * (-1,1)                
    X = torch.rand(num_intrr_pts * 2, dim_prob-1) * 2 - 1 
    T = torch.rand(num_intrr_pts * 2, dim_prob-1)
    X_d = torch.cat([X, T], dim=1)
    return X_d

def SmpPts_Initial(num_initl_pts, dim_prob=2):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_initl_pts, 1)

    X_initl = torch.cat([torch.rand(num_initl_pts, dim_prob-1) * 2 -1, temp0], dim=1)

    return X_initl

def SmpPts_Boundary(num_bndry_pts, dim_prob):

    temp0 = - torch.ones(num_bndry_pts, 1)
    temp1 = torch.ones(num_bndry_pts, 1)

    X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob - 1)], dim = 1)
    X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob - 1)], dim = 1)

    return X_left, X_right

def SmpPts_Shock(num_shock_pts, dim_prob):
    x = torch.rand(num_shock_pts, dim_prob - 1)
    t = x
    return torch.cat([x, t], dim=1)
    