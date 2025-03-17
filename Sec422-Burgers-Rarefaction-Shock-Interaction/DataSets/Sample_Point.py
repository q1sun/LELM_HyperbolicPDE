import torch
import math
from pyDOE import lhs 

def SmpPts_Interior(num_intrr_pts, dim_prob=2):    
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """

    
    X_d = lhs(dim_prob, num_intrr_pts)
    X_d[:, 0] = X_d[:, 0] * 7 - 1
    X_d[:, 1] = X_d[:, 1] * 10

    return torch.Tensor(X_d)

def SmpPts_Initial(num_init_pts, dim_prob=2):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_init_pts, 1)

    X_init = torch.cat([torch.rand(num_init_pts, dim_prob-1) * 7 -1, temp0], dim=1)

    return X_init

def SmpPts_Boundary(num_bndry_pts, dim_prob):

    temp0 = - torch.ones(num_bndry_pts, 1)
    temp1 = torch.ones(num_bndry_pts, 1) * 6

    X_left = torch.cat([ temp0, torch.rand(num_bndry_pts, dim_prob - 1) * 10], dim = 1)
    X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob - 1) * 10], dim = 1)

    return X_left, X_right

def SmpPts_Shock(num_chrtc_pts, dim_prob):
    t1 = torch.rand(num_chrtc_pts, dim_prob - 1) * 4
    x1 = t1/4 + 1

    Shock_1 = torch.cat([x1, t1], dim=1)

    t2 = torch.rand(num_chrtc_pts, dim_prob - 1) * 6 + 4
    x2 = torch.sqrt(t2)

    Shock_2 = torch.cat([x2, t2], dim=1)

    return Shock_1, Shock_2
    

def SmpPts_Test(num_test_x, num_test_t):

    xs = torch.linspace(0, 1, steps = num_test_x) * 7 - 1
    ys = torch.linspace(0, 1, steps = num_test_t) * 10
    x, y = torch.meshgrid(xs, ys)

    return torch.squeeze(torch.stack([x.reshape(1, num_test_t * num_test_x), y.reshape(1, num_test_t * num_test_x)], dim=-1))
 
