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

folder = "./Results/1LassoNet"
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

class LassoNet(nn.Module):
    def __init__(self, d, M):
        super(LassoNet, self).__init__()
        self.skip = nn.Linear(d, 1, bias=False); torch.nn.init.zeros_(self.skip.weight)
        if LAY>=2: self.fc1 = nn.Linear(d, M); torch.nn.init.normal_(self.fc1.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc1.bias, mean=.0, std=.1**2)
        if LAY>=3: self.fc2 = nn.Linear(M, M); torch.nn.init.normal_(self.fc2.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc2.bias, mean=.0, std=.1**2)
        if LAY>=4: self.fc3 = nn.Linear(M, M); torch.nn.init.normal_(self.fc3.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc3.bias, mean=.0, std=.1**2)
        self.fc4 = nn.Linear(M, 1);            torch.nn.init.normal_(self.fc4.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc4.bias, mean=.0, std=.1**2)
    def forward(self, x):
        h = x
        if LAY>=2: h = eval("torch."+ACT)(self.fc1(h))
        if LAY>=3: h = eval("torch."+ACT)(self.fc2(h))
        if LAY>=4: h = eval("torch."+ACT)(self.fc3(h))
        h = self.fc4(h)
        return h + self.skip(x)

@torch.no_grad()
def hier_prox(theta, W, lam, M):
    d, K = theta.shape[0], W.shape[0]
    theta_abs, W_abs = theta.abs(), W.abs()
    W_sorted, _ = torch.sort(W_abs, dim=0, descending=True)
    cumsum = torch.cat([torch.zeros(1, d, device=theta.device),
                        torch.cumsum(W_sorted, dim=0)], dim=0)
    m = torch.arange(K+1, device=theta.device, dtype=theta.dtype).view(-1,1)
    w_m = (M/(1.+m*M*M))*torch.clamp(theta_abs.view(1,-1)+M*cumsum-lam, min=0.)
    upper = torch.cat([torch.full((1,d), float('inf'), device=theta.device), W_sorted], dim=0)
    lower = torch.cat([W_sorted, torch.zeros(1, d, device=theta.device)], dim=0)
    idx = torch.argmax(((lower<=w_m)&(w_m<=upper)).to(theta.dtype), dim=0)
    w_star = w_m.gather(0, idx.view(1,-1)).squeeze(0)
    theta.copy_(torch.sign(theta)*w_star/M)
    W.copy_(torch.sign(W)*torch.minimum(w_star.view(1,-1).expand_as(W), W_abs))

def train(tra_loader):
    model.train()
    lam = LAM
    for batch_idx1, (X1, Y1) in enumerate(tra_loader):
        optimizer.zero_grad()
        FX1 = model(X1)
        loss = -torch.mean(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
        loss.backward()
        optimizer.step()
        hier_prox(model.skip.weight.data.view(-1), model.fc1.weight.data,
                  lam*optimizer.param_groups[0]['lr'], ALP)

start = time.perf_counter()
EPOCH = int(1000*M/0.8/SIZE)
model = LassoNet(DIM, 100).to(device)
optimizer = optim.Adam(model.parameters(), lr=.1**4.)
for epoch in range(EPOCH):
    train(tra_loader)
end = time.perf_counter()
np.savetxt(f"Results/1LassoNet/{SEED}-{DIM}-{M}-{args.cuda}.csv",np.array([end - start]))
