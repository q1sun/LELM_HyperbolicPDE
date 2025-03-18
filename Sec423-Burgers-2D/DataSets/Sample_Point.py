import torch

def SmpPts_Interior(num_intrr_pts, dim_prob=3):    
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """

    # entire domain (-1,1) * (-1,1)                
    X = torch.rand(num_intrr_pts * 2, 1) * 3 
    Y = torch.rand(num_intrr_pts * 2, 1)
    T = torch.rand(num_intrr_pts * 2, 1) * 0.4
    X_d = torch.cat([X, Y, T], dim=1)
    return X_d

def SmpPts_Initial(num_init_pts, dim_prob=3):
    ''' num_init_pts = total number of sampling points at initial time'''

    temp0 = torch.zeros(num_init_pts, 1)

    X = torch.rand(num_init_pts, 1) * 3 
    Y = torch.rand(num_init_pts, 1)

    X_init = torch.cat([X, Y, temp0], dim=1)

    return X_init

def SmpPts_Boundary(num_bndry_pts, dim_prob=3):

    temp0 = torch.zeros(num_bndry_pts, 1)
    temp1 = torch.ones(num_bndry_pts, 1) * 3
    temp = torch.cat([3 * torch.rand(num_bndry_pts, 1), 0.4 * torch.rand(num_bndry_pts, 1)], dim = 1)

    X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob - 2), torch.rand(num_bndry_pts, dim_prob - 2) * 0.4], dim = 1)
    X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob - 2), torch.rand(num_bndry_pts, dim_prob - 2) * 0.4], dim = 1)
    X_BF = torch.cat([temp[:,0].reshape(-1, 1), torch.heaviside(temp[:,0] - 0.5 * temp[:,1] - 2, torch.ones(temp0.size(0))).reshape(-1, 1), temp[:,1].reshape(-1, 1)], dim=1)

    return X_left, X_right, X_BF

# x = x_0 + st
def SmpPts_Shock(num_shock_pts, dim_prob=3):
    t = torch.rand(num_shock_pts, 1) * 0.4
    x_ch1 = 1 + 3 * t
    x_ch2 = 2 + 0.5 * t
    shock_line1 = torch.cat([x_ch1, torch.rand(num_shock_pts, dim_prob - 2), t], dim = 1)
    shock_line2 = torch.cat([x_ch2, torch.rand(num_shock_pts, dim_prob - 2), t], dim = 1)
    return shock_line1, shock_line2
    