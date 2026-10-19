#import
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import argparse
import numpy as np
import numpy.random as rd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#lam, seed, device
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--dim',  type=int,   default=3)
parser.add_argument('--size', type=int,   default=100)
parser.add_argument('--alp',  type=str)
parser.add_argument('--lam',  type=str)
parser.add_argument('--Mnum', type=int,   default=1)
parser.add_argument('--act',  type=str)
parser.add_argument('--lay',  type=int,   default=2)
parser.add_argument('--seed', type=int,   default=0)
parser.add_argument('--cuda', type=int,   default=-1)
args = parser.parse_args()
#
TYPE = args.type
DIM  = args.dim
SIZE = args.size
ALP  = float(args.alp)
LAM  = float(args.lam)
M    = args.Mnum   #kept for CLI/folder-naming compatibility; unused by the gradient-based score
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/1MFGrad-%.10f/%s-%d-%d/%s/dim-%d/size-%d/lam-%.10f"%(ALP,ACT,LAY,M,TYPE,DIM,SIZE,LAM)
os.makedirs("%s/"%folder, exist_ok=True)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.zeros((SIZE+10000,DIM+1))
    for i in range(SIZE+10000):
        for j in range(DIM): XYful[i,j] = rd.normal(0,1)
        if TYPE=="non-additive-sq":      pass
        if TYPE=="quasi-multicolinear1": XYful[i,3] = XYful[i,2]/2 + rd.normal(0,1/2)
        if TYPE=="quasi-multicolinear2": XYful[i,3] = XYful[i,2]/2 + rd.normal(0,1)
        if TYPE=="quasi-multicolinear3": XYful[i,3] = XYful[i,2] + rd.normal(0,1)
        if TYPE=="quasi-multicolinear4": XYful[i,3] = 2*XYful[i,2] + rd.normal(0,1)
        XYful[i,-1] = np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])/5+rd.normal(0,1)
    XYful = rd.permutation(XYful)
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED) #train:val = 4:1
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.FloatTensor(XYtra[:,-1].reshape(-1,1)).to(device)
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.FloatTensor(XYval[:,-1].reshape(-1,1)).to(device)
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.FloatTensor(XYtes[:,-1].reshape(-1,1)).to(device)
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=20, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=20)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=20)

    #model, train, test
    class RegNet(nn.Module):
        def __init__(self, d, M):
            super(RegNet, self).__init__()
            if LAY>=2: self.fc1 = nn.Linear(d, M); torch.nn.init.normal_(self.fc1.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc1.bias, mean=.0, std=.1**2)
            if LAY>=3: self.fc2 = nn.Linear(M, M); torch.nn.init.normal_(self.fc2.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc2.bias, mean=.0, std=.1**2)
            if LAY>=4: self.fc3 = nn.Linear(M, M); torch.nn.init.normal_(self.fc3.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc3.bias, mean=.0, std=.1**2)
            self.fc4 = nn.Linear(M, 1);            torch.nn.init.normal_(self.fc4.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc4.bias, mean=.0, std=.1**2)
        def forward(self, x):
            if LAY>=2: x = eval("torch."+ACT)(self.fc1(x))
            if LAY>=3: x = eval("torch."+ACT)(self.fc2(x))
            if LAY>=4: x = eval("torch."+ACT)(self.fc3(x))
            x = self.fc4(x)
            return x

    #Gradient-based sensitivity regularization (a first-order Taylor approximation of the
    #permutation-based score of 1MFBridge.py): since
    #  f_theta([x,z,j]) - f_theta(x) ~= (d f_theta/d x_j)(x) * (z_j - x_j)
    #for z_j close to x_j, we replace the permutation-based score_j
    #(computing f_theta at f(x) and f([x,z,j]) and averaging their squared
    #difference over M draws of z) with a purely local, permutation-free score
    #that measures |d f_theta/d x_j| directly via autodiff (i.e. the input
    #gradient / saliency-map magnitude of Simonyan et al., 2013), instead of a
    #finite difference over the (possibly far-apart) values x_j and z_j:
    #  score_j = sqrt( E_x[(d f_theta/d x_j)(x)^2] + eps ).
    #This requires create_graph=True so the penalty (a function of this
    #gradient) can itself be backpropagated into the model parameters
    #("double backpropagation"; e.g. Drucker & Le Cun, 1992; Gu & Rigazio, 2014).
    def train(tra_loader):
        model.train()
        tra_loss = torch.zeros((), device=device)
        tra_pena = torch.zeros((), device=device)
        tra_obj  = torch.zeros((), device=device)
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            optimizer.zero_grad()
            #
            X1 = X1.clone().requires_grad_(True)
            FX1 = model(X1)
            #
            #d(sum_i FX1_i)/dX1 equals, per sample i, d FX1_i/d X1_i, since each
            #sample's output depends only on its own input in this feed-forward
            #model (no cross-sample interaction), so this single backward pass
            #yields the full (B, DIM) matrix of per-sample input gradients
            grad_x = torch.autograd.grad(FX1.sum(), X1, create_graph=True)[0]
            #
            loss = F.mse_loss(Y1, FX1)
            score = torch.sqrt(torch.mean(grad_x**2, dim=0)/2 + .1**6)
            pena = torch.sum(torch.pow(score, ALP))
            #
            obj = loss + LAM*pena
            obj.backward()
            optimizer.step()
            #
            with torch.no_grad():
                tra_loss  += loss
                tra_pena  += pena
                tra_obj   += obj
                tra_score += score
        n = batch_idx1 + 1
        result = torch.cat([torch.stack([tra_loss, tra_pena, tra_obj]), tra_score]) / n
        return result.cpu().numpy()

    def test(loader):
        model.eval()
        tes_loss = 0.0
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                FX = model(X)
                loss = F.mse_loss(FX, Y)
                #
                tes_loss += loss.item()
        return tes_loss/(batch_idx+1)#/2.

    #learning
    EPOCH = 200
    model = RegNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1**(1./EPOCH))
    best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 5+DIM))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
        # scheduler.step()
        # print(epoch,result[epoch,-2])
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), map_location=device, weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")