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
import itertools
import time
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--dim',  type=int,   default=3)
parser.add_argument('--size', type=int,   default=100)
parser.add_argument('--alp',  type=str)
parser.add_argument('--lam',  type=str)
parser.add_argument('--lam0', type=str, default='0.001')
parser.add_argument('--Mnum', type=int,   default=20)
parser.add_argument('--act',  type=str)
parser.add_argument('--lay',  type=int,   default=2)
parser.add_argument('--seed', type=int,   default=0)
parser.add_argument('--cuda', type=int,   default=-1)
args = parser.parse_args()
TYPE = args.type
DIM  = args.dim
SIZE = args.size
ALP  = float(args.alp)
LAM  = float(args.lam)
M    = args.Mnum
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/1SpAM"
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)


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
XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED)
XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED)
Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.FloatTensor(XYtra[:,-1].reshape(-1,1)).to(device)
Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.FloatTensor(XYval[:,-1].reshape(-1,1)).to(device)
Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.FloatTensor(XYtes[:,-1].reshape(-1,1)).to(device)
tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=M, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=M)
tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=M)

class RegNet(nn.Module):
    def __init__(self, d, M):
        super(RegNet, self).__init__()
        self.d = d
        if LAY>=2:
            self.W1 = nn.Parameter(torch.empty(d, 1, M)); nn.init.normal_(self.W1, mean=.0, std=.1**2)
            self.b1 = nn.Parameter(torch.empty(d, 1, M)); nn.init.normal_(self.b1, mean=.0, std=.1**2)
        if LAY>=3:
            self.W2 = nn.Parameter(torch.empty(d, M, M)); nn.init.normal_(self.W2, mean=.0, std=.1**2)
            self.b2 = nn.Parameter(torch.empty(d, 1, M)); nn.init.normal_(self.b2, mean=.0, std=.1**2)
        if LAY>=4:
            self.W3 = nn.Parameter(torch.empty(d, M, M)); nn.init.normal_(self.W3, mean=.0, std=.1**2)
            self.b3 = nn.Parameter(torch.empty(d, 1, M)); nn.init.normal_(self.b3, mean=.0, std=.1**2)
        self.W4 = nn.Parameter(torch.empty(d, M, 1)); nn.init.normal_(self.W4, mean=.0, std=.1**2)
        self.b4 = nn.Parameter(torch.empty(d, 1, 1)); nn.init.normal_(self.b4, mean=.0, std=.1**2)

    def forward(self, x):
        z = x.t().unsqueeze(-1)
        if LAY>=2: y = eval("torch."+ACT)(torch.baddbmm(self.b1, z, self.W1))
        if LAY>=3: y = eval("torch."+ACT)(torch.baddbmm(self.b2, y, self.W2))
        if LAY>=4: y = eval("torch."+ACT)(torch.baddbmm(self.b3, y, self.W3))
        y = torch.baddbmm(self.b4, y, self.W4)
        f = y.squeeze(-1).t()
        g = f.sum(dim=1, keepdim=True)
        return torch.cat([f, g], dim=1)


def train(tra_loader):
    model.train()
    for batch_idx1, (X1, Y1) in enumerate(tra_loader):
        optimizer.zero_grad()
        FX1 = model(X1)
        loss = F.mse_loss(Y1, FX1[:,-1].reshape(-1,1))
        score = torch.sqrt(torch.var(FX1[:,:-1], dim=0)+.1**6)
        pena = torch.sum(torch.pow(score, ALP))
        obj = loss + LAM*pena
        obj.backward()
        optimizer.step()

start = time.perf_counter()
EPOCH = int(1000*M/0.8/SIZE)
model = RegNet(DIM, 100).to(device)
optimizer = optim.Adam(model.parameters(), lr=.1**4.)
for epoch in range(EPOCH):
    train(tra_loader)
end = time.perf_counter()
np.savetxt(f"Results/1SpAM/{SEED}-{DIM}-{M}-{args.cuda}.csv",np.array([end - start]))