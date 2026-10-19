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

folder = "./Results/1STG"
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)

XYful = np.zeros((SIZE+10000,DIM+1))
for i in range(SIZE+10000):
    for j in range(DIM): XYful[i,j] = rd.normal(0,1)
    if TYPE=="non-additive-sq":      pass
    if TYPE=="quasi-multicolinear1": XYful[i,3] = XYful[i,2]/2 + rd.normal(0,1/2)
    if TYPE=="quasi-multicolinear2": XYful[i,3] = XYful[i,2]/2 + rd.normal(0,1)
    if TYPE=="quasi-multicolinear3": XYful[i,3] = XYful[i,2] + rd.normal(0,1)
    if TYPE=="quasi-multicolinear4": XYful[i,3] = 2*XYful[i,2] + rd.normal(0,1)
    XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-5.))) else 0
XYful = rd.permutation(XYful)
XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED)
XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED)
Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.LongTensor(XYtra[:,-1].reshape(-1,1)).to(device)
Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.LongTensor(XYval[:,-1].reshape(-1,1)).to(device)
Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.LongTensor(XYtes[:,-1].reshape(-1,1)).to(device)
tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=M, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=M)
tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=M)

class STGNet(nn.Module):
    def __init__(self, d, M):
        super(STGNet, self).__init__()
        self.mu = nn.Parameter(torch.full((d,), 0.5))
        if LAY>=2: self.fc1 = nn.Linear(d, M); torch.nn.init.normal_(self.fc1.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc1.bias, mean=.0, std=.1**2)
        if LAY>=3: self.fc2 = nn.Linear(M, M); torch.nn.init.normal_(self.fc2.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc2.bias, mean=.0, std=.1**2)
        if LAY>=4: self.fc3 = nn.Linear(M, M); torch.nn.init.normal_(self.fc3.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc3.bias, mean=.0, std=.1**2)
        self.fc4 = nn.Linear(M, 1);            torch.nn.init.normal_(self.fc4.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc4.bias, mean=.0, std=.1**2)
    def forward(self, x):
        if self.training: z = torch.clamp(self.mu+ALP*torch.randn_like(x), 0., 1.)
        else:             z = torch.clamp(self.mu, 0., 1.).expand_as(x)
        x = x*z
        if LAY>=2: x = eval("torch."+ACT)(self.fc1(x))
        if LAY>=3: x = eval("torch."+ACT)(self.fc2(x))
        if LAY>=4: x = eval("torch."+ACT)(self.fc3(x))
        x = self.fc4(x)
        return x
    def penalty(self):
        return torch.sum(.5*(1.+torch.erf((self.mu/ALP)/np.sqrt(2.))))

def train(tra_loader):
    model.train()
    for batch_idx1, (X1, Y1) in enumerate(tra_loader):
        optimizer.zero_grad()
        FX1 = model(X1)
        loss = -torch.mean(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
        pena = model.penalty()
        obj = loss + LAM*pena
        obj.backward()
        optimizer.step()

start = time.perf_counter()
EPOCH = int(1000*M/0.8/SIZE)
model = STGNet(DIM, 100).to(device)
optimizer = optim.Adam(model.parameters(), lr=.1**4.)
for epoch in range(EPOCH):
    train(tra_loader)
end = time.perf_counter()
np.savetxt(f"Results/1STG/{SEED}-{DIM}-{M}-{args.cuda}.csv",np.array([end - start]))
