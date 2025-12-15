import torch
import math
import numpy as np

def SmpPts_Interior(num_intrr_pts, dim_prob=2):    
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """

    # entire domain (-1,1) * (-1,1)                
    X = torch.rand(num_intrr_pts * 2, dim_prob-1) 
    T = torch.rand(num_intrr_pts * 2, dim_prob-1) * 0.25 
    X_d = torch.cat([X, T], dim=1)

    r = torch.rand(num_intrr_pts//5, dim_prob-1) * 0.1
    theta = torch.rand(num_intrr_pts//5, dim_prob-1) * math.pi
    X = 0.5 + r * torch.cos(theta)
    T = r * torch.sin(theta)
    X_semidisk = torch.cat([X, T], dim=1)
            
    return torch.cat([X_d, X_semidisk], dim = 0)

def SmpPts_Initial(num_init_pts, dim_prob=2):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_init_pts, 1)

    X_init = torch.cat([torch.rand(num_init_pts, dim_prob-1), temp0], dim=1)

    r = torch.rand(num_init_pts//5, dim_prob - 1) * 0.1
    temp = torch.zeros(num_init_pts//5, dim_prob - 1)
    x_init_r = torch.cat([0.5 + r, temp], dim = 1)
    x_init_l = torch.cat([0.5 - r, temp], dim = 1)


    return torch.cat([X_init, x_init_l, x_init_r], dim = 0)

# def SmpPts_Boundary(num_bndry_pts, dim_prob, T):

#     temp0 = torch.zeros(num_bndry_pts, 1)
#     temp1 = torch.ones(num_bndry_pts, 1) * 2 * math.pi
#     x_rand = torch.rand(num_bndry_pts, dim_prob - 1) * T

#     X_left = torch.cat([temp0, x_rand], dim = 1)
#     X_right = torch.cat([temp1, x_rand], dim = 1)

#     return X_left, X_right

def SmpPts_Characteristic(num_chrtc_pts, dim_prob, s):
    t = torch.rand(num_chrtc_pts, dim_prob-1) * 0.25
    x = 0.5 + s * t
    return torch.cat([x, t], dim=1)

def SmpPts_Test(num_test_x, num_test_t):

    xs = torch.linspace(0, 1, steps = num_test_x)
    ys = torch.linspace(0, 1, steps = num_test_t) * 0.25
    x, y = torch.meshgrid(xs, ys)

    return torch.squeeze(torch.stack([x.reshape(1, num_test_t * num_test_x), y.reshape(1, num_test_t * num_test_x)], dim=-1))
 
