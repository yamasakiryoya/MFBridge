#import
import os
os.environ["OMP_NUM_THREADS"] = "1"
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
parser.add_argument('--Mnum', type=int, default=1)
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

folder = "./Results/ABridge-%.10f/%s-%d-%d/%s/dim-%d/size-%d/lam-%.10f"%(ALP,ACT,LAY,M,TYPE,DIM,SIZE,LAM)
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.zeros((SIZE+10100,DIM+1))
    for i in range(SIZE+10100):
        for j in range(DIM): XYful[i,j]  = rd.normal(0,1)
        if   TYPE=="linear":            XYful[i,-1] = XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2]+rd.normal(0,1)
        elif TYPE=="additive-sq":       XYful[i,-1] = np.square(XYful[i,0])+np.square(2.*XYful[i,1])+np.square(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sq1":  XYful[i,-1] = np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sq2":  XYful[i,-1] = np.square(XYful[i,0]+2.*XYful[i,1])+np.square(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sq3":  XYful[i,-1] = np.square(XYful[i,0])+np.square(2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="additive-sin":      XYful[i,-1] = np.sin(XYful[i,0])+np.sin(2.*XYful[i,1])+np.sin(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sin1": XYful[i,-1] = np.sin(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sin2": XYful[i,-1] = np.sin(XYful[i,0]+2.*XYful[i,1])+np.sin(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sin3": XYful[i,-1] = np.sin(XYful[i,0])+np.sin(2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
    XYful = rd.permutation(XYful)
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=SIZE, random_state=SEED) #train SIZE, valid 100
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]), torch.FloatTensor(XYtra[:,-1].reshape(-1,1))
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]), torch.FloatTensor(XYval[:,-1].reshape(-1,1))
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]), torch.FloatTensor(XYtes[:,-1].reshape(-1,1))
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=20, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=20)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=20)

    #model, train, test
    class RegNet(nn.Module):
        def __init__(self, d, M):
            super(RegNet, self).__init__()
            if LAY>=2:
                self.fc1 = nn.ModuleList([nn.Linear(d, M) for i in range(d)])
                for j in range(d): torch.nn.init.normal_(self.fc1[j].weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc1[j].bias, mean=.0, std=.1**2)
            if LAY>=3:
                self.fc2 = nn.ModuleList([nn.Linear(M, M) for i in range(d)])
                for j in range(d): torch.nn.init.normal_(self.fc2[j].weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc2[j].bias, mean=.0, std=.1**2)
            if LAY>=4:
                self.fc3 = nn.ModuleList([nn.Linear(M, M) for i in range(d)])
                for j in range(d): torch.nn.init.normal_(self.fc3[j].weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc3[j].bias, mean=.0, std=.1**2)
            self.fc4 = nn.ModuleList([nn.Linear(M, 1) for i in range(d)])
            for j in range(d): torch.nn.init.normal_(self.fc4[j].weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc4[j].bias, mean=.0, std=.1**2)
            self.d = d
        def forward(self, x):
            f = []
            z = x
            for j in range(self.d):
                if LAY>=2: x = eval("torch."+ACT)(self.fc1[j](z))
                if LAY>=3: x = eval("torch."+ACT)(self.fc2[j](x))
                if LAY>=4: x = eval("torch."+ACT)(self.fc3[j](x))
                x = self.fc4[j](x)
                if j==0: g = x
                else: g += x
                f.append(x)
            f.append(g)
            return f

    def train(tra_loader):
        model.train()
        tra_loss, tra_pena, tra_obj = 0.0, 0.0, 0.0
        tra_score = torch.zeros(DIM).to(device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1[-1])#/2.
            #
            score = torch.zeros(DIM).to(device)
            Mlist = rd.randint(SIZE, size=M).tolist()
            for j in range(DIM):
                score[j] = torch.sqrt(torch.var(FX1[j])+.1**8)
            pena = torch.sum(torch.pow(score, ALP))
            #
            obj = loss + LAM*pena
            obj.backward()
            optimizer.step()
            #
            tra_loss  += loss.item()
            tra_pena  += pena.item()
            tra_obj   += obj.item()
            tra_score += score
        return np.concatenate([torch.tensor([tra_loss, tra_pena, tra_obj]).cpu().detach().numpy(), tra_score.cpu().detach().numpy()])/(batch_idx1+1)

    def test(loader):
        model.eval()
        tes_loss = 0.0
        with torch.set_grad_enabled(False):
            for batch_idx, (X, Y) in enumerate(loader):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                #
                FX = model(X)
                loss = F.mse_loss(FX[-1], Y)
                #
                tes_loss += loss.item()
        return tes_loss/(batch_idx+1)#/2.

    #learning
    EPOCH = 2000
    model = RegNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1**(1./EPOCH))
    best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 5+DIM))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/seed-%d.pt"%(folder,SEED))
        scheduler.step()
    model.load_state_dict(torch.load("%s/seed-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/seed-%d.csv"%(folder,SEED), result, delimiter=",")
