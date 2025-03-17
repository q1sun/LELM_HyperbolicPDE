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

def SmpPts_Initial(num_init_pts, dim_prob=2):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_init_pts, 1)

    X_init = torch.cat([torch.rand(num_init_pts, dim_prob-1) * 2 -1, temp0], dim=1)

    return X_init

def SmpPts_Boundary(num_bndry_pts, dim_prob):

    temp0 = - torch.ones(num_bndry_pts, 1)
    temp1 = torch.ones(num_bndry_pts, 1)

    X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob - 1)], dim = 1)
    X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob - 1)], dim = 1)

    return X_left, X_right

def SmpPts_Shock(num_chrtc_pts, dim_prob, s):
    t = torch.rand(num_chrtc_pts, dim_prob-1)
    x = s * t
    return torch.cat([x, t], dim=1)
    
def SmpPts_Test(num_test_x, num_test_t):

    xs = torch.linspace(0, 1, steps = num_test_x) * 2 - 1
    ys = torch.linspace(0, 1, steps = num_test_t) 
    x, y = torch.meshgrid(xs, ys)

    return torch.squeeze(torch.stack([x.reshape(1, num_test_t * num_test_x), y.reshape(1, num_test_t * num_test_x)], dim=-1))
 
