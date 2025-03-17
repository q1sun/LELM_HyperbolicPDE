import torch
import math

def SmpPts_Interior(num_intrr_pts, dim_prob=2):    
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """
            
    X = torch.rand(num_intrr_pts * 2, dim_prob-1) * 3
    T = torch.rand(num_intrr_pts * 2, dim_prob-1) * 2
    X_d = torch.cat([X, T], dim=1)
    return X_d

def SmpPts_Initial(num_init_pts, dim_prob=2):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_init_pts, 1)

    X_init = torch.cat([torch.rand(num_init_pts, dim_prob-1) * 3, temp0], dim=1)

    return X_init

def SmpPts_Boundary(num_bndry_pts, dim_prob):

    temp0 = torch.zeros(num_bndry_pts, 1)
    temp1 = torch.ones(num_bndry_pts, 1) * 3

    X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob - 1) * 2], dim = 1)
    X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob - 1) * 2], dim = 1)

    return X_left, X_right

def SmpPts_Shock(num_chrtc_pts, dim_prob):
    t1 = torch.rand(num_chrtc_pts, 1) * (5 - math.sqrt(13)) / 2
    t2 = torch.rand(num_chrtc_pts, 1) * (5 - math.sqrt(13)) / 2
    x_left_c = torch.sqrt(1 + t1)
    x_right_c = 2 - t2
    t_line2 = torch.rand(num_chrtc_pts, 1) * (2 - (5 - math.sqrt(13))/2) + (5 - math.sqrt(13))/2
    x_line2 = math.sqrt(13) * torch.sqrt(1 + t_line2) - 2 *(1 + t_line2)

    Shock_left = torch.cat([x_left_c, t1], dim=1)
    Shock_right = torch.cat([x_right_c, t2], dim=1)
    Shock_line = torch.cat([x_line2, t_line2], dim=1)
    return Shock_left, Shock_right, Shock_line
    

def SmpPts_Test(num_test_x, num_test_t):

    xs = torch.linspace(0, 1, steps = num_test_x) * 3
    ys = torch.linspace(0, 1, steps = num_test_t) * 2
    x, y = torch.meshgrid(xs, ys)

    return torch.squeeze(torch.stack([x.reshape(1, num_test_t * num_test_x), y.reshape(1, num_test_t * num_test_x)], dim=-1))
 
