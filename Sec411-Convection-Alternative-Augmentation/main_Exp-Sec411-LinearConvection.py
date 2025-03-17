import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import argparse
import scipy.io as io
import math
from torch import optim, autograd
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader
from DataSets  import Sample_Point, Exact_Solution

from Models.FcNet import FcNet
from itertools import cycle
from Utils import helper

print("pytorch version", torch.__version__, "\n")

##################################################################################################
## parser arguments
parser = argparse.ArgumentParser(description='Lift-and-Learning Methods for Solving Linear Convection Equation in Section 4-1-1')
# weights
parser.add_argument('-w', '--weights', default='Figures/Trained_Model/simulation_0', type=str, metavar='PATH', help='path to save model weights')
# figures 
parser.add_argument('-i', '--figures', default='Figures/Python/simulation_0', type=str, metavar='PATH', help='path to save figures')
args = parser.parse_args()

# hyperparameters
parser.add_argument('--num_epochs', default=12000, type=int, metavar='N', help='total number of training epochs')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint')
parser.add_argument('--milestones', type=int, nargs='+', default=[5000, 8000, 10000, 11000], help='decay of learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=2, help='network depth')
parser.add_argument('--width', type=int, default=40, help='network width')

# datasets options
parser.add_argument('--num_intrr_pts', type=int, default=10000, help='total number of sampling points at interior domain')
parser.add_argument('--num_initl_pts', type=int, default=1000, help='total number of sampling points for initial condition')
parser.add_argument('--num_bndry_pts', type=int, default=1000, help='total number of sampling points for boundary condition')
parser.add_argument('--num_shock_pts', type=int, default=1000, help='number of sampling points at characteristic lines')

args = parser.parse_known_args()[0]

# problem setting
dim_prob = 2        # dimension 
a = -1              # convection coeffient 
T = math.pi * 2     # maximun time of computational domain
x_min, x_max = 0, 2 * math.pi

batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
batchsize_initl_pts = 2 * args.num_initl_pts // args.num_batches
batchsize_bndry_pts = 2 * args.num_bndry_pts // args.num_batches
batchsize_shock_pts = 2 * args.num_shock_pts // args.num_batches
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> creating training datasets ...')

# training dataset for sample points inside the domain
class TraindataInterior(Dataset):    
    def __init__(self, num_intrr_pts, dim_prob): 
        
        self.SmpPts_Intrr = Sample_Point.SmpPts_Interior(num_intrr_pts, dim_prob)        
               
    def __len__(self):
        return len(self.SmpPts_Intrr)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Intrr[idx]

        return [SmpPt]

# training dataset for sample points at the Dirichlet boundary
class TraindataBoundary(Dataset):    
    def __init__(self, num_bndry_pts, dim_prob, T):         
        
        self.SmpPts_BndrL, self.SmpPts_BndrR = Sample_Point.SmpPts_Boundary(num_bndry_pts, dim_prob, T)
        
    def __len__(self):
        return len(self.SmpPts_BndrL)
    
    def __getitem__(self, idx):
        SmpPt_BndrL = self.SmpPts_BndrL[idx]
        SmpPt_BndrR = self.SmpPts_BndrR[idx]

        return [SmpPt_BndrL, SmpPt_BndrR]    
    
class TraindataInitial(Dataset):
    def __init__(self, num_initl_pts, dim_prob):

        self.SmpPts_Initl = Sample_Point.SmpPts_Initial(num_initl_pts, dim_prob)
        self.u0val_SmpPts = Exact_Solution.u0_Exact_Solution(self.SmpPts_Initl[:,0])

    def __len__(self):
        return len(self.SmpPts_Initl)
    
    def __getitem__(self, idx):
        SmpPts_Initl = self.SmpPts_Initl[idx]
        u0val_SmpPts = self.u0val_SmpPts[idx]

        return [SmpPts_Initl, u0val_SmpPts]
    
class TraindataShock(Dataset):
    def __init__(self, num_shock_pts, dim_prob, a , T):
        self.SmpPts_Shck1, self.SmpPts_Shck2 = Sample_Point.SmpPts_Shock(num_shock_pts, dim_prob, a, T)
    
    def __len__(self):
        return len(self.SmpPts_Shck1)
    
    def __getitem__(self, idx):
        SmpPt_Shck1 = self.SmpPts_Shck1[idx]
        SmpPt_Shck2 = self.SmpPts_Shck2[idx]

        return [SmpPt_Shck1, SmpPt_Shck2]

# create training datasets       
traindata_intrr = TraindataInterior(args.num_intrr_pts, dim_prob)
traindata_bndry = TraindataBoundary(args.num_bndry_pts, dim_prob, T)
traindata_initl = TraindataInitial(args.num_initl_pts, dim_prob)
traindata_shock = TraindataShock(args.num_shock_pts, dim_prob, a, T)

# define dataloader
dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
dataloader_bndry = DataLoader(traindata_bndry, batch_size=batchsize_bndry_pts, shuffle=True, num_workers=0)
dataloader_initl = DataLoader(traindata_initl, batch_size=batchsize_initl_pts, shuffle=True, num_workers=0)
dataloader_shock = DataLoader(traindata_shock, batch_size=batchsize_shock_pts, shuffle=True, num_workers=0)
####################################################################################################


##############################################################################################
def train_epoch(epoch, model, optimizer, device):
    
    # set model to training mode
    model.train()

    loss_epoch, loss_intrr_epoch, loss_bndry_epoch, loss_initl_epoch, loss_shock_epoch = 0, 0, 0, 0, 0

    for i, (data_intrr, data_bndry, data_initl, data_shock) in enumerate(zip(dataloader_intrr, cycle(dataloader_bndry), cycle(dataloader_initl), cycle(dataloader_shock))):

        # get mini-batch training data
        [smppts_intrr] = data_intrr
        smppts_bndrL, smppts_bndrR = data_bndry
        smppts_initl, u0val_smppts = data_initl
        smppts_shck1, smppts_shck2 = data_shock

        # add the agumented variable
        smppts_intrr = torch.cat([smppts_intrr, Exact_Solution.augmented_variable(smppts_intrr[:,0], smppts_intrr[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_bndrL = torch.cat([smppts_bndrL, Exact_Solution.augmented_variable(smppts_bndrL[:,0], smppts_bndrL[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_bndrR = torch.cat([smppts_bndrR, Exact_Solution.augmented_variable(smppts_bndrR[:,0], smppts_bndrR[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_initl = torch.cat([smppts_initl, Exact_Solution.augmented_variable(smppts_initl[:,0], smppts_initl[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_shk1L = torch.cat([smppts_shck1, Exact_Solution.augmented_variable(smppts_shck1[:,0] - 0.0001, smppts_shck1[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_shk1R = torch.cat([smppts_shck1, Exact_Solution.augmented_variable(smppts_shck1[:,0] + 0.0001, smppts_shck1[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_shk2L = torch.cat([smppts_shck2, Exact_Solution.augmented_variable(smppts_shck2[:,0] - 0.0001, smppts_shck2[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        smppts_shk2R = torch.cat([smppts_shck2, Exact_Solution.augmented_variable(smppts_shck2[:,0] + 0.0001, smppts_shck2[:,1], a, T).reshape(-1,1)], dim=1).to(device)
        
        u0val_smppts = u0val_smppts.to(device)

        smppts_intrr.requires_grad = True

        # forward pass to obtain NN prediction of u(x)
        u_vals_intrr = model(smppts_intrr)
        u_vals_bndrL = model(smppts_bndrL)
        u_vals_bndrR = model(smppts_bndrR)
        u_vals_initl = model(smppts_initl)

        u_vals_shk1L = model(smppts_shk1L)
        u_vals_shk1R = model(smppts_shk1R)
        u_vals_shk2L = model(smppts_shk2L)
        u_vals_shk2R = model(smppts_shk2R)        

        # zero parameter gradients and then compute NN prediction of gradient u(x)
        model.zero_grad()
        gradu_vals_intrr = torch.autograd.grad(outputs=u_vals_intrr, inputs=smppts_intrr, grad_outputs=torch.ones_like(u_vals_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]

        # construct mini-batch loss function and then perform backward pass
        loss_intrr = torch.mean(torch.pow(a * gradu_vals_intrr[:,0] + gradu_vals_intrr[:,1], 2))
        loss_bndry = torch.mean(torch.pow(u_vals_bndrL - u_vals_bndrR, 2))
        loss_initl = torch.mean(torch.pow(torch.squeeze(u_vals_initl) - u0val_smppts, 2))

        loss_shock1 = torch.mean(torch.pow(u_vals_shk1R - u_vals_shk1L - 1, 2))
        loss_shock2 = torch.mean(torch.pow(u_vals_shk2R - u_vals_shk2L + 1, 2))
        loss_shock = loss_shock1 + loss_shock2

        loss_minibatch = loss_intrr + loss_bndry + args.beta * (loss_initl + loss_shock)

        #zero parameter gradients
        optimizer.zero_grad()
        # backpropagation
        loss_minibatch.backward()
        # parameter update
        optimizer.step()

        # integrate loss over the entire training dataset
        loss_intrr_epoch += loss_intrr.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Intrr.shape[0]
        loss_bndry_epoch += loss_bndry.item() * smppts_bndrL.size(0) / traindata_bndry.SmpPts_BndrL.shape[0]
        loss_initl_epoch += loss_initl.item() * smppts_initl.size(0) / traindata_initl.SmpPts_Initl.shape[0]
        loss_shock_epoch += loss_shock.item() * smppts_shck1.size(0) / traindata_shock.SmpPts_Shck1.shape[0]
        loss_epoch += loss_intrr_epoch + loss_bndry_epoch + args.beta * (loss_initl_epoch + loss_shock_epoch) 
        
    return loss_epoch
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> training network solution ...')

# create model
model = FcNet.FcNet(dim_prob + 1, args.width, 1, args.depth)
model.Xavier_initi()

# create optimizer and learning rate schedular
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=False)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

# load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if not os.path.isdir(args.weights):
    helper.mkdir_p(args.weights)

train_loss = []
trainloss_best = 1e10
# training process
trainloss_best = 1e10
since = time.time()
for epoch in range(args.num_epochs):

    # execute training
    trainloss_epoch = train_epoch(epoch, model, optimizer, device)

    # save best model weights
    is_best = trainloss_epoch < trainloss_best
    trainloss_best = min(trainloss_epoch, trainloss_best)
    helper.save_checkpoint({'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                           }, is_best, checkpoint=args.weights)   
       
    # adjust learning rate according to predefined schedule
    schedular.step()

    train_loss.append(trainloss_epoch)
    
time_elapsed = time.time() - since

print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> loading trained model for inference ...')

# load trained model
checkpoint = torch.load(os.path.join(args.weights, 'model_best.pth.tar'), device)
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():  

    # trained network solution, evaluated at t = 1
    u_smppts = torch.cat([torch.linspace(0, 1, 1001).reshape(-1, 1) * (x_max - x_min) + x_min, torch.ones(1001, 1).reshape(-1,1)], dim=1)
    u_smppts = torch.cat([u_smppts, Exact_Solution.augmented_variable(u_smppts[:,0], u_smppts[:,1], a, T).reshape(-1,1)], dim=1)

    u_smppts = u_smppts.to(device)
    u_vals = model(u_smppts)

x = torch.squeeze(u_smppts[:,0]).cpu().detach().numpy().reshape(1001, 1)


if not os.path.isdir(args.figures):
    helper.mkdir_p(args.figures)
# plot u and its network prediction on testing dataset
fig=plt.figure()
u_Exact = Exact_Solution.u_Exact_Solution(u_smppts[:,0],u_smppts[:,1], a).cpu().detach().numpy().reshape(1001, 1)
plt.plot(x, u_Exact, ls = '-', lw = '2')
plt.title('Exact Solution u(x,1) on Test Dataset') 
fig.savefig(os.path.join(args.figures, 'Exact u(x,1).png'))
plt.close(fig)

fig=plt.figure()
u_vals = u_vals.cpu().detach().numpy().reshape(1001, 1)
plt.plot(x, u_vals, ls='-')
plt.title('Network Solution u_NN(x,1) on Test_data')
fig.savefig(os.path.join(args.figures, 'Predicted u(x,1).png'))
plt.close(fig)

fig = plt.figure()
plt.plot(torch.log10(torch.tensor(train_loss)), c = 'red', label = 'training loss' )
plt.title('Learning Curve during Training')
plt.legend(loc = 'upper right')
fig.savefig(os.path.join(args.figures, 'learning curve.png'))
plt.show()
##############################################################################################


                                           


