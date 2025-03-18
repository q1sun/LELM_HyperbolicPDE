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
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import Dataset, DataLoader
from DataSets  import Sample_Point, Exact_Solution

from Models.FcNet import FcNet

from itertools import cycle
from Utils import helper


print("pytorch version", torch.__version__, "\n")

## parser arguments
parser = argparse.ArgumentParser(description='Deep Residual Method for Solving 2D Linear Convection Equation in Section 4-2-3')
# weights
parser.add_argument('-w', '--weights', default='Figures/Trained_Model/simulation_0', type=str, metavar='PATH', help='path to save model weights')
# figures 
parser.add_argument('-i', '--figures', default='Figures/Python/simulation_0', type=str, metavar='PATH', help='path to save figures')
args = parser.parse_args()
##############################################################################################



##################################################################################################
# dataset setting
parser.add_argument('--num_epochs', default=15000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[4000, 8000, 11000, 14000], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=2, help='network depth')
parser.add_argument('--width', type=int, default=80, help='network width')

# datasets options
parser.add_argument('--num_intrr_pts', type=int, default=40000, help='total number of interior sampling points')
parser.add_argument('--num_initl_pts', type=int, default=10000, help='total number of sampling points of initial points')
parser.add_argument('--num_bndry_pts', type = int, default=10000, help='number of sampling points of boundary')
parser.add_argument('--num_shock_pts', type=int, default=10000, help='number of sampling points at characteristic lines')

args = parser.parse_known_args()[0]

# problem setting
dim_prob = 3

batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
batchsize_init_pts = 2 * args.num_initl_pts // args.num_batches
batchsize_bndry_pts = 2 * args.num_bndry_pts // args.num_batches
batchsize_shock_pts = 2 * args.num_shock_pts // args.num_batches
################################################################################################

################################################################################################
# training dataset for sample points inside the domain
class TraindataInterior(Dataset):    
    def __init__(self, num_intrr_pts, dim_prob): 
        
        self.SmpPts_Interior = Sample_Point.SmpPts_Interior(num_intrr_pts, dim_prob)        
               
    def __len__(self):
        return len(self.SmpPts_Interior)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Interior[idx]

        return [SmpPt]

# training dataset for sample points at the Dirichlet boundary
class TraindataBoundary(Dataset):    
    def __init__(self, num_bndry_pts, dim_prob):         
        
        self.SmpPts_bndrL, self.SmpPts_bndrR, self.SmpPts_bkaft = Sample_Point.SmpPts_Boundary(num_bndry_pts, dim_prob)
        
        
    def __len__(self):
        return len(self.SmpPts_bndrL)
    
    def __getitem__(self, idx):
        SmpPtl = self.SmpPts_bndrL[idx]
        SmpPtr = self.SmpPts_bndrR[idx]
        SmpPtbkaft = self.SmpPts_bkaft[idx]

        return [SmpPtl, SmpPtr, SmpPtbkaft]  
    
class TraindataInitial(Dataset):
    def __init__(self, num_initl_pts, dim_prob):

        self.SmpPts_Initl = Sample_Point.SmpPts_Initial(num_initl_pts, dim_prob)
        self.f_Exactu0val_smppts = Exact_Solution.u0val_Exact_Solution(self.SmpPts_Initl[:, 0], self.SmpPts_Initl[:, 1])

    def __len__(self):
        return len(self.SmpPts_Initl)
    
    def __getitem__(self, idx):
        SmpPts_Initl = self.SmpPts_Initl[idx]
        u0valu0val_smppts = self.f_Exactu0val_smppts[idx]

        return [SmpPts_Initl, u0valu0val_smppts]
    
class Traindatashock(Dataset):
    def __init__(self, num_shock_pts, dim_prob):
        self.SmpPts_Shock1, self.SmpPts_Shock2 = Sample_Point.SmpPts_Shock(num_shock_pts, dim_prob)
    
    def __len__(self):
        return len(self.SmpPts_Shock1)
    
    def __getitem__(self, idx):
        SmpPts_Shock1 = self.SmpPts_Shock1[idx]
        SmpPts_Shock2 = self.SmpPts_Shock2[idx]

        return [SmpPts_Shock1, SmpPts_Shock2]

################################################################################################



################################################################################################
# create training and testing datasets         
traindata_intrr = TraindataInterior(args.num_intrr_pts, dim_prob)
traindata_bndry = TraindataBoundary(args.num_bndry_pts, dim_prob)
traindata_initl = TraindataInitial(args.num_initl_pts, dim_prob)
traindata_shock = Traindatashock(args.num_shock_pts, dim_prob)

# define dataloader
dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
dataloader_bndry = DataLoader(traindata_bndry, batch_size=batchsize_bndry_pts, shuffle=True, num_workers=0)
dataloader_initl = DataLoader(traindata_initl, batch_size=batchsize_init_pts, shuffle=True, num_workers=0)
dataloader_shock = DataLoader(traindata_shock, batch_size=batchsize_shock_pts, shuffle=True, num_workers=0)
####################################################################################################


##############################################################################################

def train_epoch(epoch, model, optimizer, device):
    
    # set model to training mode
    model.train()

    loss_epoch, loss_intrr_epoch, loss_bndry_epoch, loss_initl_epoch, loss_shock_epoch = 0, 0, 0, 0, 0

    # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
    # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)

    for i, (data_intrr, data_bndry, data_initl, data_shock) in enumerate(zip(dataloader_intrr, cycle(dataloader_bndry), cycle(dataloader_initl), cycle(dataloader_shock))):

        # get mini-batch training data
        [smppts_intrr] = data_intrr
        smppts_bndrL, smppts_bndrR, smppts_bkaft = data_bndry
        smppts_initl, u0val_smppts = data_initl
        smppts_shock1, smppts_shock2= data_shock

        # add the third variable
        smppts_intrr = torch.cat([smppts_intrr, Exact_Solution.augmented_variable(smppts_intrr[:,0], smppts_intrr[:,1], smppts_intrr[:, 2]).reshape(-1,1)], dim=1)
        smppts_bndrL = torch.cat([smppts_bndrL, Exact_Solution.augmented_variable(smppts_bndrL[:,0], smppts_bndrL[:,1], smppts_bndrL[:, 2]).reshape(-1,1)], dim=1)
        smppts_bndrR = torch.cat([smppts_bndrR, Exact_Solution.augmented_variable(smppts_bndrR[:,0], smppts_bndrR[:,1], smppts_bndrR[:, 2]).reshape(-1,1)], dim=1)
        smppts_bkaft = torch.cat([smppts_bkaft, Exact_Solution.augmented_variable(smppts_bkaft[:,0], smppts_bkaft[:,1], smppts_bkaft[:,2]).reshape(-1, 1)], dim=1)
        smppts_initl = torch.cat([smppts_initl, Exact_Solution.augmented_variable(smppts_initl[:,0], smppts_initl[:,1], smppts_initl[:, 2]).reshape(-1,1)], dim=1)

        smppts_shock1l = torch.cat([smppts_shock1, Exact_Solution.augmented_variable(smppts_shock1[:,0] - 0.0001, smppts_shock1[:,1], smppts_shock1[:, 2]).reshape(-1,1)], dim=1)
        smppts_shock1r = torch.cat([smppts_shock1, Exact_Solution.augmented_variable(smppts_shock1[:,0] + 0.0001, smppts_shock1[:,1], smppts_shock1[:, 2]).reshape(-1,1)], dim=1)
        smppts_shock2l = torch.cat([smppts_shock1, Exact_Solution.augmented_variable(smppts_shock2[:,0] - 0.0001, smppts_shock2[:,1], smppts_shock2[:, 2]).reshape(-1,1)], dim=1)
        smppts_shock2r = torch.cat([smppts_shock1, Exact_Solution.augmented_variable(smppts_shock2[:,0] + 0.0001, smppts_shock2[:,1], smppts_shock2[:, 2]).reshape(-1,1)], dim=1)


        smppts_intrr = smppts_intrr.to(device)
        u0val_smppts = u0val_smppts.to(device)
        smppts_bndrL = smppts_bndrL.to(device)
        smppts_bndrR = smppts_bndrR.to(device)
        smppts_bkaft = smppts_bkaft.to(device)
        smppts_initl = smppts_initl.to(device)

        smppts_shock1l = smppts_shock1l.to(device)
        smppts_shock1r = smppts_shock1r.to(device)
        smppts_shock2l = smppts_shock2l.to(device)
        smppts_shock2r = smppts_shock2r.to(device)

        Prosp1 = 3 * torch.ones(smppts_shock1l.size(0)).reshape(-1,1)
        Prosp2 = 0.5 * torch.ones(smppts_shock2.size(0)).reshape(-1, 1)
        bndry_bkaft = Exact_Solution.u_Exact_Solution(smppts_bkaft[:,0], smppts_bkaft[:, 1], smppts_bkaft[:, 2]).to(device)

        Prosp1 = Prosp1.to(device)
        Prosp2 = Prosp2.to(device)

        smppts_intrr.requires_grad = True

        # forward pass to obtain NN prediction of u(x)
        u_NN_intrr = model(smppts_intrr)
        u_NN_bndrL = model(smppts_bndrL)
        u_NN_bndrR = model(smppts_bndrR)
        u_NN_bkaft = model(smppts_bkaft)
        u_NN_initl = model(smppts_initl)

        u_NN_shock1l = model(smppts_shock1l)
        u_NN_shock1r = model(smppts_shock1r)
        u_NN_shock2l = model(smppts_shock2l)
        u_NN_shock2r = model(smppts_shock2r)
        

        # zero parameter gradients and then compute NN prediction of gradient u(x)
        model.zero_grad()
        gradu_NN_intrr = torch.autograd.grad(outputs=u_NN_intrr, inputs=smppts_intrr, grad_outputs=torch.ones_like(u_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]

        # construct mini-batch loss function and then perform backward pass
        loss_intrr = torch.mean(torch.pow(gradu_NN_intrr[:,2] + torch.squeeze(u_NN_intrr) * gradu_NN_intrr[:,0] + torch.squeeze(u_NN_intrr) * gradu_NN_intrr[:, 1], 2))
        loss_bndry = torch.mean(torch.pow(u_NN_bndrL - 4, 2)) + torch.mean(torch.pow(u_NN_bndrR + 1, 2)) + torch.mean(torch.pow(torch.squeeze(u_NN_bkaft) - bndry_bkaft, 2))
        loss_initl = torch.mean(torch.pow(torch.squeeze(u_NN_initl) - u0val_smppts, 2))


        loss_shock = torch.mean(torch.pow((u_NN_shock1r + u_NN_shock1l) * 0.5 - torch.squeeze(Prosp1), 2)) + torch.mean(torch.pow((u_NN_shock2r + u_NN_shock2l) * 0.5 - torch.squeeze(Prosp2), 2))

        loss_minibatch = loss_intrr + loss_bndry + args.beta * (loss_initl + loss_shock)

        #zero parameter gradients
        optimizer.zero_grad()
        # backpropagation
        loss_minibatch.backward()
        # parameter update
        optimizer.step()

        # integrate loss over the entire training dataset
        loss_intrr_epoch += loss_intrr.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
        loss_bndry_epoch += loss_bndry.item() * smppts_bndrL.size(0) / traindata_bndry.SmpPts_bndrL.shape[0]
        loss_initl_epoch += loss_initl.item() * smppts_initl.size(0) / traindata_initl.SmpPts_Initl.shape[0]
        loss_shock_epoch += loss_shock.item() * smppts_shock1.size(0) / traindata_shock.SmpPts_Shock1.shape[0]
        loss_epoch += loss_intrr_epoch + loss_bndry_epoch + args.beta * (loss_initl_epoch + loss_shock_epoch) 
        
    return loss_epoch
        
################################################################################################


##############################################################################################
# create model
model = FcNet.FcNet(dim_prob + 1,args.width,1,args.depth)
model.Xavier_initi()

# create optimizer and learning rate schedular
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

# load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if not os.path.isdir(args.weights):
    helper.mkdir_p(args.weights)

# train and test 
train_loss = []
trainloss_best = 1e10
since = time.time()
for epoch in range(args.num_epochs):
      
        
    # execute training and testing
    trainloss_epoch = train_epoch(epoch, model, optimizer, device)

    # adjust learning rate according to predefined schedule
    schedular.step()

    # print results
    train_loss.append(trainloss_epoch)

    # save best model weights
    is_best = trainloss_epoch < trainloss_best
    trainloss_best = min(trainloss_epoch, trainloss_best)
    helper.save_checkpoint({'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                           }, is_best, checkpoint=args.weights)   

time_elapsed = time.time() - since

print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
##############################################################################################



##############################################################################################

# load trained model
checkpoint = torch.load(os.path.join(args.weights, 'model_best.pth.tar'), device)
model.load_state_dict(checkpoint['state_dict'])

if not os.path.isdir(args.figures):
    helper.mkdir_p(args.figures)
    
# compute NN predicution of u and gradu
with torch.no_grad():  

    xs = torch.linspace(0, 1, 1001) * 3
    ys = torch.linspace(0, 1, 1001)
    x, y = torch.meshgrid(xs, ys)


    test_smppts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), 0.2 * torch.ones(1001**2, 1).reshape(-1,1)], dim=1)
    test_smppts = torch.cat([test_smppts, Exact_Solution.augmented_variable(test_smppts[:,0], test_smppts[:,1], test_smppts[:,2]).reshape(-1,1)], dim=1)

    
    test_smppts = test_smppts.to(device)

    u_NN = model(test_smppts)

x = torch.squeeze(test_smppts[:,0]).cpu().detach().numpy().reshape(1001, 1001)
y = torch.squeeze(test_smppts[:,1]).cpu().detach().numpy().reshape(1001, 1001)



# plot u and its network prediction on testing dataset
fig=plt.figure()
u_Exact = Exact_Solution.u_Exact_Solution(test_smppts[:,0],test_smppts[:,1], test_smppts[:, 2]).cpu().detach().numpy().reshape(1001, 1001)
ax1 = plt.axes(projection='3d')
ax1.plot_surface(x, y, u_Exact, cmap = 'rainbow')
plt.title('Exact Solution u(x, y, 0.2) on Test Dataset')
#plt.show()  
fig.savefig(os.path.join(args.figures, 'Exact u(x,y,0.2).png'))
plt.close(fig)


fig=plt.figure()
u_NN = u_NN.cpu().detach().numpy().reshape(1001, 1001)
ax2 = plt.axes(projection='3d')
ax2.plot_surface(x, y, u_NN, cmap='rainbow')
plt.title('Predicted u_NN(x, y, 0.2) on Test_data')
#plt.show()
fig.savefig(os.path.join(args.figures, 'Predicted u(x,y,0.2).png'))
plt.close(fig)

    
# plot learning curves
fig = plt.figure()
plt.plot(torch.log10(torch.tensor(train_loss)), c = 'red', label = 'training loss' )
plt.title('Learning Curve during Training')
plt.legend(loc = 'upper right')
# plt.show()
fig.savefig(os.path.join(args.figures, 'learning curve.png'))
##############################################################################################


                                           


