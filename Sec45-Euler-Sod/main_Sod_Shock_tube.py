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
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import Dataset, DataLoader
from DataSets  import Sample_Point, Exact_Solution
from Utils import helper

from Models.FcNet import FcNet

from itertools import cycle

import sodshock


print("pytorch version", torch.__version__, "\n")

## parser arguments
parser = argparse.ArgumentParser(description='Deep Residual Method for discontinuous solution')
# checkpoints
parser.add_argument('-w', '--weights', default='Figures/Trained_Model/simulation_0', type=str, metavar='PATH', help='path to save model weights')
# figures 
parser.add_argument('-i', '--figures', default='Figures/Python/simulation_0', type=str, metavar='PATH', help='path to save figures')
parser.add_argument('--inits_contk', type=float, default=0.9, help='The initial guess of contack wace speed')
parser.add_argument('--inits_shock', type=float, default=1.8, help='The initial guess of shock speed s')
args = parser.parse_args()
##############################################################################################



##################################################################################################



# dataset setting
parser.add_argument('--num_epochs', default=22000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[8000, 15000, 19000, 21000], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=3, help='network depth')
parser.add_argument('--width', type=int, default=40, help='network width')

# datasets options
parser.add_argument('--num_intrr_pts', type=int, default=40000, help='total number of interior sampling points')
parser.add_argument('--num_init_pts', type=int, default=5000, help='total number of sampling points of initial points')
parser.add_argument('--num_test_t', type=int, default=37, help='number of time jet')
parser.add_argument('--num_test_x', type=int, default=1001, help='number of sampling points at a certain time')
parser.add_argument('--num_chrtc_pts', type=int, default=5000, help='number of sampling points at characteristic lines')


args = parser.parse_known_args()[0]

# problem setting
dim_prob = 2
gamma = 1.4
shock = 0.8504311464060357
contk = 0.6854905240097902
ref_shock = (shock - 0.5) * 5
ref_contk = (contk - 0.5) * 5

batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
batchsize_init_pts = 2 * args.num_init_pts // args.num_batches
batchsize_chrtc_pts = 2 * args.num_chrtc_pts // args.num_batches
################################################################################################

################################################################################################
print('*', '-' * 45, '*')
print('===> preparing training and testing datasets ...')
print('*', '-' * 45, '*')

# training dataset for sample points inside the domain
class TraindataInterior(Dataset):    
    def __init__(self, num_intrr_pts, dim_prob): 
        
        self.SmpPts_Interior = Sample_Point.SmpPts_Interior(num_intrr_pts, dim_prob)        
               
    def __len__(self):
        return len(self.SmpPts_Interior)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Interior[idx]

        return [SmpPt]

class TraindataInitial(Dataset):
    def __init__(self, num_init_pts, dim_prob):

        self.SmpPts_Init = Sample_Point.SmpPts_Initial(num_init_pts, dim_prob)
        self.f_Exact_SmpPts = Exact_Solution.f_Exact(self.SmpPts_Init[:,0])

    def __len__(self):
        return len(self.SmpPts_Init)
    
    def __getitem__(self, idx):
        SmpPts_Init = self.SmpPts_Init[idx]
        f_SmpPts = self.f_Exact_SmpPts[idx]

        return [SmpPts_Init, f_SmpPts]
    
class Traindatacharater(Dataset):
    def __init__(self, num_chrtc_pts, dim_prob, s):
        self.SmpPts_Chrtc = Sample_Point.SmpPts_Characteristic(num_chrtc_pts, dim_prob, s)
    
    def __len__(self):
        return len(self.SmpPts_Chrtc)
    
    def __getitem__(self, idx):
        SmpPts_chrtc = self.SmpPts_Chrtc[idx]

        return [SmpPts_chrtc]


class Testdata(Dataset):
    def __init__(self, num_test_t, num_test_x):
        self.SmpPts_Test = Sample_Point.SmpPts_Test(num_test_x, num_test_t)
        # self.u_Exact_Solution = Exact_Solution.u_Exact_Solution(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])

    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPts = self.SmpPts_Test[idx]
        # u_Exact_SmpPts = self.u_Exact_Solution[idx]

        return [SmpPts]
    

################################################################################################



#####################################################################################################
# create training and testing datasets         
traindata_intrr = TraindataInterior(args.num_intrr_pts, dim_prob)
traindata_init = TraindataInitial(args.num_init_pts, dim_prob)
traindata_tail = Traindatacharater(args.num_chrtc_pts, dim_prob, -0.0702728125611825)
traindata_head = Traindatacharater(args.num_chrtc_pts, dim_prob, -1.183215956619923)
testdata = Testdata(args.num_test_t, args.num_test_x)

# define dataloader
dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
dataloader_init = DataLoader(traindata_init, batch_size=batchsize_init_pts, shuffle=True, num_workers=0)
dataloader_test = DataLoader(testdata, batch_size=args.num_test_t*args.num_test_x, shuffle=True, num_workers=0)
dataloader_tail = DataLoader(traindata_tail, batch_size=batchsize_chrtc_pts, shuffle=True, num_workers=0)
dataloader_head = DataLoader(traindata_head, batch_size=batchsize_chrtc_pts, shuffle=True, num_workers=0)
####################################################################################################


##################################################################################################
if not os.path.isdir(args.figures):
    helper.mkdir_p(args.figures)
# draw sample points during testing
fig = plt.figure()
plt.scatter(testdata.SmpPts_Test[:,0], testdata.SmpPts_Test[:,1], c = 'black')
plt.title('Sample Points during Testing step')
# plt.show()
plt.savefig(os.path.join(args.figures,'TestSmpPts.png'))
plt.close(fig)
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> creating training model ...')
print('*', '-' * 45, '*', "\n", "\n")

def train_epoch(epoch, model, optimizer, device):

    # copy the value of shock speed s
    s = shock_param.detach()
    # copy the value of contack discontinuity speed
    c = contk_param.detach()
    # creating shock line by the value of shock speed
    traindata_shock = Traindatacharater(args.num_chrtc_pts, dim_prob, s)
    dataloader_shock = DataLoader(traindata_shock, batch_size=batchsize_chrtc_pts, shuffle=True, num_workers=0)
    traindata_contack = Traindatacharater(args.num_chrtc_pts, dim_prob, c)
    dataloader_contack = DataLoader(traindata_contack, batch_size=batchsize_chrtc_pts, shuffle=True, num_workers=0)

    # save the value of s
    hist_s.append(s.item())
    # save the value of c
    hist_c.append(c.item())
    # set model to training mode
    model.train()

    loss_epoch, loss_intrr_epoch_1, loss_intrr_epoch_2, loss_intrr_epoch_3, loss_init_epoch_rho, loss_init_epoch_p, loss_init_epoch_u, loss_shock_epoch, loss_contk_epoch = 0, 0, 0, 0, 0, 0, 0, 0, 0 

    

    # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
    # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)

    for i, (data_intrr, data_init, data_shock, data_contack, data_tail, data_head) in enumerate(zip(dataloader_intrr,  cycle(dataloader_init), cycle(dataloader_shock),cycle(dataloader_contack), cycle(dataloader_tail), cycle(dataloader_head))):

        # get mini-batch training data
        [smppts_intrr] = data_intrr
        smppts_init, f_smppts = data_init
        [smppts_shock] = data_shock
        [smppts_contk] = data_contack
        # add the third variable
        smppts_intrr = torch.cat([smppts_intrr, Exact_Solution.augmented_variable(smppts_intrr[:,0], smppts_intrr[:,1], s, c).reshape(-1,1)], dim=1)

        smppts_init = torch.cat([smppts_init, Exact_Solution.augmented_variable(smppts_init[:,0], smppts_init[:,1], s, c).reshape(-1,1)], dim=1)


        smppts_shockl = torch.cat([smppts_shock, 1 * torch.ones_like(smppts_shock[:, 0]).reshape(-1, 1)], dim=1)
        smppts_shockr = torch.cat([smppts_shock, 2 * torch.ones_like(smppts_shock[:, 0]).reshape(-1, 1)], dim=1)
        smppts_contkl = torch.cat([smppts_contk, 0 * torch.ones_like(smppts_shock[:, 0]).reshape(-1, 1)], dim=1)
        smppts_contkr = torch.cat([smppts_contk, 1 * torch.ones_like(smppts_shock[:, 0]).reshape(-1, 1)], dim=1)



        smppts_intrr = smppts_intrr.to(device)

        smppts_init = smppts_init.to(device)
        f_smppts = f_smppts.to(device) 

        smppts_shockl = smppts_shockl.to(device)
        smppts_shockr = smppts_shockr.to(device)
        smppts_contkl = smppts_contkl.to(device)
        smppts_contkr = smppts_contkr.to(device)

        Prosp_shock = shock_param * torch.ones(smppts_shockl.size(0)).reshape(-1,1)
        Prosp_shock = Prosp_shock.to(device)

        smppts_intrr.requires_grad = True

        # forward pass to obtain NN prediction of u(x)
        
        rho_NN_intrr, p_NN_intrr, u_NN_intrr = torch.split(model(smppts_intrr), split_size_or_sections=1, dim=1)

        rho_NN_init, p_NN_init, u_NN_init = torch.split(model(smppts_init), split_size_or_sections=1, dim=1)

        rho_NN_shockl, p_NN_shockl, u_NN_shockl = torch.split(model(smppts_shockl), split_size_or_sections=1, dim=1)
        rho_NN_shockr, p_NN_shockr, u_NN_shockr = torch.split(model(smppts_shockr), split_size_or_sections=1, dim=1)

        rho_NN_contkl, p_NN_contkl, u_NN_contkl = torch.split(model(smppts_contkl), split_size_or_sections=1, dim=1)
        rho_NN_contkr, p_NN_contkr, u_NN_contkr = torch.split(model(smppts_contkr), split_size_or_sections=1, dim=1)

        # zero parameter gradients and then compute NN prediction of gradient u(x)
        model.zero_grad()

        grad_NN_intrr_rho = torch.autograd.grad(outputs=rho_NN_intrr, inputs=smppts_intrr, grad_outputs=torch.ones_like(rho_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_NN_intrr_p = torch.autograd.grad(outputs=p_NN_intrr, inputs=smppts_intrr, grad_outputs=torch.ones_like(p_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_NN_intrr_u = torch.autograd.grad(outputs=u_NN_intrr, inputs=smppts_intrr, grad_outputs=torch.ones_like(u_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]

        # construct mini-batch loss function and then perform backward pass
        loss_intrr_1 = torch.mean(torch.pow(grad_NN_intrr_rho[:, 1] + torch.squeeze(u_NN_intrr) * grad_NN_intrr_rho[:,0] + torch.squeeze(rho_NN_intrr) * grad_NN_intrr_u[:,0], 2))
        loss_intrr_2 = torch.mean(torch.pow((grad_NN_intrr_u[:, 1] + torch.squeeze(u_NN_intrr) * grad_NN_intrr_u[:,0]) * torch.squeeze(rho_NN_intrr) + grad_NN_intrr_p[:,0], 2))
        loss_intrr_3 = torch.mean(torch.pow(grad_NN_intrr_p[:, 1] + gamma *  grad_NN_intrr_u[:,0] * torch.squeeze(p_NN_intrr) + torch.squeeze(u_NN_intrr) * grad_NN_intrr_p[:,0], 2))

        loss_init_rho = torch.mean(torch.pow(torch.squeeze(rho_NN_init) - f_smppts[:, 0], 2))
        loss_init_p = torch.mean(torch.pow(torch.squeeze(p_NN_init) - f_smppts[:, 1], 2))
        loss_init_u = torch.mean(torch.pow(torch.squeeze(u_NN_init) - f_smppts[:, 2], 2))

        loss_shock_1 = torch.mean(torch.pow(torch.squeeze(Prosp_shock) * (torch.squeeze(rho_NN_shockl - rho_NN_shockr)) - (torch.squeeze(rho_NN_shockl * u_NN_shockl - rho_NN_shockr * u_NN_shockr)), 2))
        loss_shock_2 = torch.mean(torch.pow(torch.squeeze(Prosp_shock) * (torch.squeeze(rho_NN_shockl * u_NN_shockl - rho_NN_shockr * u_NN_shockr)) - (torch.squeeze(rho_NN_shockl*u_NN_shockl**2 + p_NN_shockl - rho_NN_shockr*u_NN_shockr**2 - p_NN_shockr)), 2))
        loss_shock_3 = torch.mean(torch.pow(torch.squeeze(Prosp_shock) * (torch.squeeze(0.5 * u_NN_shockl**2 * rho_NN_shockl + p_NN_shockl/(gamma -1) - 0.5 * u_NN_shockr**2 * rho_NN_shockr - p_NN_shockr/(gamma-1))) - (torch.squeeze(0.5 * u_NN_shockl**3 * rho_NN_shockl + gamma * u_NN_shockl * p_NN_shockl/(gamma-1) - 0.5 * u_NN_shockr**3 * rho_NN_shockr - gamma * u_NN_shockr * p_NN_shockr/(gamma-1))), 2))
        loss_shock = (loss_shock_1 + loss_shock_2 + loss_shock_3) * 200

        loss_contk = torch.mean(torch.pow(torch.squeeze(u_NN_contkl - u_NN_contkr), 2) + torch.pow(torch.squeeze(p_NN_contkl - p_NN_contkr), 2)) * 400 + torch.mean(torch.pow(torch.squeeze(u_NN_contkl) - contk_param.to(device), 2)) * 5
        

        loss_minibatch = 10 * (loss_intrr_1 + loss_intrr_2 + loss_intrr_3) + args.beta * (loss_init_rho + loss_init_p) + 400 * (loss_init_u) + (loss_shock + loss_contk)

        #zero parameter gradients
        optimizer.zero_grad()
        # backpropagation
        loss_minibatch.backward()
        # parameter update
        optimizer.step()

        # integrate loss over the entire training dataset
        loss_intrr_epoch_1 += loss_intrr_1.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
        loss_intrr_epoch_2 += loss_intrr_2.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
        loss_intrr_epoch_3 += loss_intrr_3.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
        loss_init_epoch_rho += loss_init_rho.item() * smppts_init.size(0) / traindata_init.SmpPts_Init.shape[0]
        loss_init_epoch_p += loss_init_p.item() * smppts_init.size(0) / traindata_init.SmpPts_Init.shape[0]
        loss_init_epoch_u += loss_init_u.item() * smppts_init.size(0) / traindata_init.SmpPts_Init.shape[0]
        loss_shock_epoch += loss_shock.item() * smppts_shock.size(0) / traindata_shock.SmpPts_Chrtc.shape[0]
        loss_contk_epoch += loss_contk.item() * smppts_contk.size(0) / traindata_contack.SmpPts_Chrtc.shape[0]
        loss_epoch += (loss_intrr_epoch_1 + loss_intrr_epoch_2 + loss_intrr_epoch_3) * 5 + args.beta * (loss_init_epoch_rho + loss_init_epoch_p) + loss_init_epoch_u * 400 + (loss_shock_epoch+ loss_contk_epoch) 
        
    return loss_intrr_epoch_1, loss_intrr_epoch_2, loss_intrr_epoch_3, loss_init_epoch_rho, loss_init_epoch_p, loss_init_epoch_u, loss_shock_epoch, loss_contk_epoch, loss_epoch
        

##############################################################################################
print('*', '-' * 45, '*')
print('===> creating testing model ...')
print('*', '-' * 45, '*', "\n", "\n")



##############################################################################################
print('*', '-' * 45, '*')
print('===> neural network training ...')

if not os.path.isdir(args.weights):
    helper.mkdir_p(args.weights)

# load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create model
model = FcNet.FcNet(dim_prob + 1, args.width, 3, args.depth)

model.Xavier_initi()

shock_param = torch.nn.Parameter(args.inits_shock * torch.ones(1, requires_grad=True))
contk_param = torch.nn.Parameter(args.inits_contk * torch.ones(1, requires_grad=True))
hist_s = []
hist_c = []
param_groups = [
    {"params": model.parameters(), "lr": 0.01},
    {"params": [shock_param], "lr": 0.01},
    {"params": [contk_param], "lr": 0.01},
]
print('Network Architecture of Density:', "\n", model)
print('Total number of trainable parameters = ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# create optimizer and learning rate schedular
optimizer = torch.optim.AdamW(param_groups, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)


print('DEVICE: {}'.format(device), "\n")
model = model.to(device)
    
# train
train_loss = []
trainloss_best = 1e10
since = time.time()
for epoch in range(args.num_epochs):
       
    # execute training and testing
    trainloss_intrr_epoch_1, trainloss_intrr_epoch_2, trainloss_intrr_epoch_3, trainloss_init_epoch_rho, trainloss_init_epoch_p, trainloss_init_epoch_u, trainloss_shock_epoch, trainloss_contk_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device)
    
    # save current and best models to checkpoint
    is_best = trainloss_epoch < trainloss_best
    trainloss_best = min(trainloss_epoch, trainloss_best)
    helper.save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'shock_param': shock_param,
                            'contk_param': contk_param,
                            'optimizer': optimizer.state_dict(),
                           }, is_best, checkpoint=args.weights)   
    # save training process to log file
    # adjust learning rate according to predefined schedule
    schedular.step()

    # print results
    train_loss.append(trainloss_epoch)

time_elapsed = time.time() - since


print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################

##############################################################################################
# plot learning curves
fig = plt.figure()
plt.plot(torch.log10(torch.tensor(train_loss)), c = 'red', label = 'training loss' )
plt.title('Learning Curve during Training')
plt.legend(loc = 'upper right')
# plt.show()
fig.savefig(os.path.join(args.figures,'TrainCurve.png'))
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> loading trained model for inference ...')

# load trained model
checkpoint = torch.load(os.path.join(args.weights, 'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

# compute NN predicution of u and gradu
with torch.no_grad():  
    temp = torch.ones(args.num_test_x, 1) * 0.2
    init = torch.zeros(args.num_test_x, 1)
    test_smppts = torch.cat([torch.linspace(0, 1, steps=args.num_test_x).reshape(-1, 1), temp.reshape(-1,1)], dim=1)

    test_smppts_rho = torch.cat([test_smppts, Exact_Solution.augmented_variable(test_smppts[:,0], test_smppts[:,1], shock_param, contk_param).reshape(-1,1)], dim=1)
    test_smppts_p = torch.cat([test_smppts, Exact_Solution.augmented_variable(test_smppts[:,0], test_smppts[:,1], shock_param, contk_param).reshape(-1,1)], dim=1)
    test_smppts_u = torch.cat([test_smppts, Exact_Solution.augmented_variable(test_smppts[:,0], test_smppts[:,1], shock_param, contk_param).reshape(-1,1)], dim=1)

    init_smppts = torch.cat([torch.linspace(0, 1, steps=args.num_test_x).reshape(-1, 1), init.reshape(-1,1)], dim=1)

    init_smppts_rho = torch.cat([init_smppts, Exact_Solution.augmented_variable(init_smppts[:,0], init_smppts[:,1], shock_param, contk_param).reshape(-1,1)], dim=1)
    init_smppts_p = torch.cat([init_smppts, Exact_Solution.augmented_variable(init_smppts[:,0], init_smppts[:,1], shock_param, contk_param).reshape(-1,1)], dim=1)
    init_smppts_u = torch.cat([init_smppts, Exact_Solution.augmented_variable(init_smppts[:,0], init_smppts[:,1], shock_param, contk_param).reshape(-1,1)], dim=1)


    test_smppts_rho = test_smppts_rho.to(device)
    test_smppts_p = test_smppts_p.to(device)
    test_smppts_u = test_smppts_u.to(device)

    init_smppts_rho = init_smppts_rho.to(device)
    init_smppts_p = init_smppts_p.to(device)
    init_smppts_u = init_smppts_u.to(device)

    rho_NN, p_NN, u_NN = torch.split(model(test_smppts_rho), split_size_or_sections=1, dim=1)

    rho_init, p_init, u_init = torch.split(model(init_smppts_rho), split_size_or_sections=1, dim=1)

x = torch.squeeze(test_smppts[:,0]).cpu().detach().numpy().reshape(args.num_test_x, 1)




fig=plt.figure()
rho_NN = rho_NN.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, rho_NN, ls='-')
plt.title('Predicted density on Test_data')
#plt.show()
fig.savefig(os.path.join(args.figures, 'predited_rho_testdata_N.png'))
plt.close(fig)

fig=plt.figure()
p_NN = p_NN.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, p_NN, ls='-')
plt.title('Predicted pressure on Test_data')
#plt.show()
fig.savefig(os.path.join(args.figures, 'predited_p_testdata_N.png'))
plt.close(fig)

fig=plt.figure()
u_NN = u_NN.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, u_NN, ls='-')
plt.title('Predicted velocity on Test_data')
#plt.show()
fig.savefig(os.path.join(args.figures, 'predited_u_testdata_N.png'))
plt.close(fig)


fig = plt.figure()
rho_init = rho_init.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, rho_init, ls='-')
plt.title('Predicted density on initial condition')
fig.savefig(os.path.join(args.figures, 'predited_rho_init.png'))
plt.close(fig)

fig = plt.figure()
p_init = p_init.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, p_init, ls='-')
plt.title('Predicted pressure on initial condition')
fig.savefig(os.path.join(args.figures, 'predited_p_init.png'))
plt.close(fig)

fig = plt.figure()
u_init = u_init.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, u_init, ls='-')
plt.title('Predicted velocity on initial condition')
fig.savefig(os.path.join(args.figures, 'predited_u_init.png'))
plt.close(fig)

fig = plt.figure()
plt.title(r"s")
plt.plot(hist_s, label=r"s")
plt.hlines(ref_shock, 0, len(hist_s), label=r"s_ture", color="tab:green")
plt.ylim(0, 2) 
plt.yticks(ticks=np.arange(0, 2.1, 0.2))
plt.legend()
plt.xlabel("Training step")
fig.savefig(os.path.join(args.figures, 'Predited_s'))

fig = plt.figure()
plt.title("shock error")
plt.plot(np.log10(abs(np.array(hist_s) - ref_shock)), label='s_err')
fig.savefig(os.path.join(args.figures, 's_err'))
plt.close(fig)


fig = plt.figure()
plt.title(r"c")
plt.plot(hist_c, label=r"c")
plt.hlines(ref_contk, 0, len(hist_c), label=r"c_ture", color="tab:green")
plt.ylim(0, 1) 
plt.yticks(ticks=np.arange(0, 1.1, 0.2))
plt.legend()
plt.xlabel("Training step")
fig.savefig(os.path.join(args.figures, 'Predited_c'))
# plt.show()
plt.close(fig)
print('*', '-' * 45, '*', "\n", "\n")


hist_s = {"hist_s": np.array(hist_s)}
io.savemat(args.weights+"/hist_s.mat", hist_s)
hist_c = {"hist_c": np.array(hist_c)}
io.savemat(args.weights+"/hist_c.mat", hist_c)
##############################################################################################


                                           


