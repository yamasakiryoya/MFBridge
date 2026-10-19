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
parser.add_argument('--dim',  type=int, default=3)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--alp',  type=str)
parser.add_argument('--lam',  type=str)
parser.add_argument('--Mnum', type=int, default=20)
parser.add_argument('--act',  type=str)
parser.add_argument('--lay',  type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()
#
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
folder = "./Results/1SpAM-%.10f/%s-%d/%s/dim-%d/size-%d/lam-%.10f"%(ALP,ACT,LAY,TYPE,DIM,SIZE,LAM)

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
            self.d, self.M = d, M
            self.W1 = nn.Parameter(torch.empty(d, 1, M)); self.b1 = nn.Parameter(torch.empty(d, 1, M))
            torch.nn.init.normal_(self.W1, mean=.0, std=.1**2); torch.nn.init.normal_(self.b1, mean=.0, std=.1**2)
            if LAY>=3:
                self.W2 = nn.Parameter(torch.empty(d, M, M)); self.b2 = nn.Parameter(torch.empty(d, 1, M))
                torch.nn.init.normal_(self.W2, mean=.0, std=.1**2); torch.nn.init.normal_(self.b2, mean=.0, std=.1**2)
            if LAY>=4:
                self.W3 = nn.Parameter(torch.empty(d, M, M)); self.b3 = nn.Parameter(torch.empty(d, 1, M))
                torch.nn.init.normal_(self.W3, mean=.0, std=.1**2); torch.nn.init.normal_(self.b3, mean=.0, std=.1**2)
            self.W4 = nn.Parameter(torch.empty(d, M, 1)); self.b4 = nn.Parameter(torch.empty(d, 1, 1))
            torch.nn.init.normal_(self.W4, mean=.0, std=.1**2); torch.nn.init.normal_(self.b4, mean=.0, std=.1**2)
            self.act = eval("torch."+ACT)
        def forward(self, x):
            z = x.t().unsqueeze(2)                       # (d, B, 1)
            y = self.act(torch.baddbmm(self.b1, z, self.W1))   # (d, B, M)
            if LAY>=3: y = self.act(torch.baddbmm(self.b2, y, self.W2))
            if LAY>=4: y = self.act(torch.baddbmm(self.b3, y, self.W3))
            y = torch.baddbmm(self.b4, y, self.W4)        # (d, B, 1)
            f = y.squeeze(-1).t()                          # (B, d)
            g = f.sum(dim=1, keepdim=True)                 # (B, 1)
            return torch.cat([f, g], dim=1)
    def train(tra_loader):
        model.train()
        tra_loss = torch.zeros((), device=device)
        tra_pena = torch.zeros((), device=device)
        tra_obj  = torch.zeros((), device=device)
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1[:,-1].reshape(-1,1))
            #
            score = torch.sqrt(torch.var(FX1[:,:-1], dim=0)+.1**6)
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
        tes_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                #
                FX = model(X)
                loss = F.mse_loss(FX[:,-1].reshape(-1,1), Y)
                #
                tes_loss += loss
        return (tes_loss/(batch_idx+1)).item()
    #learning
    EPOCH = 200
    model = RegNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 5+DIM))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), map_location=device, weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")