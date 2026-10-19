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

folder = "./Results/1MFElim"
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
        self.sub_models = nn.ModuleList([
            nn.ModuleDict({
                'fc1': nn.Linear(d if i == 0 else d - 1, M),
                'fc2': nn.Linear(M, M) if LAY >= 3 else nn.Identity(),
                'fc3': nn.Linear(M, M) if LAY >= 4 else nn.Identity(),
                'fc4': nn.Linear(M, 1)
            }) for i in range(d + 1)
        ])
        for m in self.sub_models:
            for layer in m.values():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=.1**2)
                    torch.nn.init.normal_(layer.bias, mean=0.0, std=.1**2)
    def forward_sub(self, x, sub_idx):
        m = self.sub_models[sub_idx]
        x = eval("torch." + ACT)(m['fc1'](x))
        if LAY >= 3: x = eval("torch." + ACT)(m['fc2'](x))
        if LAY >= 4: x = eval("torch." + ACT)(m['fc3'](x))
        return m['fc4'](x)
    def forward(self, x):
        return self.forward_sub(x, 0)

def train(tra_loader):
    model.train()
    for batch_idx1, (X1, Y1) in enumerate(tra_loader):
        optimizer.zero_grad()
        FX1 = model.forward_sub(X1, 0)
        loss1 = F.mse_loss(Y1, FX1)
        FX2_list = []
        loss_list = [loss1]
        for j in range(DIM):
            mask = torch.ones(DIM, dtype=torch.bool, device=device)
            mask[j] = False
            X1_j = X1[:, mask]
            FX2_j = model.forward_sub(X1_j, j + 1)
            loss_j = F.mse_loss(Y1, FX2_j)
            FX2_list.append(FX2_j)
            loss_list.append(loss_j)
        FX2 = torch.stack(FX2_list)
        loss = torch.stack(loss_list).mean()
        Mlist = torch.randint(Xtra.shape[0], (M,), device=device)
        X3 = X1[:, None, None, :].expand(-1, DIM, M, -1).clone()
        for j in range(DIM): X3[:, j, :, j] = Xtra[Mlist, j].view(1, M)
        FX3 = model.forward_sub(X3.reshape(-1, DIM), 0)
        FX3 = FX3.view(Y1.shape[0], DIM, M, 1)
        FX3 = FX3.permute(1, 0, 2, 3)
        diff = FX3 - FX2.unsqueeze(2)
        score = torch.sqrt(torch.mean(diff**2, dim=(1, 2, 3)) + .1**6)
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
np.savetxt(f"Results/1MFElim/{SEED}-{DIM}-{M}-{args.cuda}.csv",np.array([end - start]))