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


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--alp',  type=str)
parser.add_argument('--lam',  type=str)
parser.add_argument('--Mnum', type=int, default=20)
parser.add_argument('--act',  type=str)
parser.add_argument('--lay',  type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()


DATA = args.data
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

folder = "./Results/1SpAM-%.10f/%s-%d/%s/size-%d/lam-%.10f"%(ALP,ACT,LAY,DATA,SIZE,LAM)
os.makedirs("%s/"%folder, exist_ok=True)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("../Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)
    XYful = rd.permutation(XYful)
    XYtrava, XYtes = train_test_split(XYful, train_size=SIZE, random_state=SEED)
    XYtra, XYval = train_test_split(XYtrava, train_size=0.8, random_state=SEED)
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.LongTensor(XYtra[:,-1].reshape(-1,1)).to(device)
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.LongTensor(XYval[:,-1].reshape(-1,1)).to(device)
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.LongTensor(XYtes[:,-1].reshape(-1,1)).to(device)
    #
    Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
    for j in range(Xmea.shape[0]):
        if Xstd[j]==0: Xtra[:,j] = 0.; Xval[:,j] = 0.; Xtes[:,j] = 0.
        else: Xtra[:,j] = (Xtra[:,j]-Xmea[j])/Xstd[j]; Xval[:,j] = (Xval[:,j]-Xmea[j])/Xstd[j]; Xtes[:,j] = (Xtes[:,j]-Xmea[j])/Xstd[j]
    #
    BS = 20
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=BS, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=BS)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=BS)
    DIM = len(XYful[0,:])-1


    class Net(nn.Module):
        def __init__(self, d, M):
            super(Net, self).__init__()
            self.d, self.M = d, M
            self.W1 = nn.Parameter(torch.empty(d, 1, M)); self.b1 = nn.Parameter(torch.empty(d, 1, M))
            torch.nn.init.normal_(self.W1, mean=.0, std=.1**2.); torch.nn.init.normal_(self.b1, mean=.0, std=.1**2.)
            if LAY>=3:
                self.W2 = nn.Parameter(torch.empty(d, M, M)); self.b2 = nn.Parameter(torch.empty(d, 1, M))
                torch.nn.init.normal_(self.W2, mean=.0, std=.1**2.); torch.nn.init.normal_(self.b2, mean=.0, std=.1**2.)
            if LAY>=4:
                self.W3 = nn.Parameter(torch.empty(d, M, M)); self.b3 = nn.Parameter(torch.empty(d, 1, M))
                torch.nn.init.normal_(self.W3, mean=.0, std=.1**2.); torch.nn.init.normal_(self.b3, mean=.0, std=.1**2.)
            self.W4 = nn.Parameter(torch.empty(d, M, 1)); self.b4 = nn.Parameter(torch.empty(d, 1, 1))
            torch.nn.init.normal_(self.W4, mean=.0, std=.1**2.); torch.nn.init.normal_(self.b4, mean=.0, std=.1**2.)
            self.act = eval("torch."+ACT)
        def forward(self, x):
            z = x.t().unsqueeze(2)
            y = self.act(torch.baddbmm(self.b1, z, self.W1))
            if LAY>=3: y = self.act(torch.baddbmm(self.b2, y, self.W2))
            if LAY>=4: y = self.act(torch.baddbmm(self.b3, y, self.W3))
            y = torch.baddbmm(self.b4, y, self.W4)
            f = y.squeeze(-1).t()
            g = f.sum(dim=1, keepdim=True)
            return torch.cat([f, g], dim=1)

    def train(tra_loader):
        model.train()
        tra_loss = torch.zeros((), device=device)
        tra_pena = torch.zeros((), device=device)
        tra_obj  = torch.zeros((), device=device)
        tra_acc  = torch.zeros((), device=device)
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            FX1_last = FX1[:,-1].reshape(-1,1)
            loss = -torch.mean(F.logsigmoid(FX1_last)*Y1+F.logsigmoid(-FX1_last)*(1-Y1))
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
                tra_acc   += (1.-torch.mean(torch.eq(FX1_last>0,Y1).float()))
                tra_score += score
        n = batch_idx1 + 1
        result = torch.cat([torch.stack([tra_loss, tra_pena, tra_obj, tra_acc]), tra_score]) / n
        return result.cpu().numpy()

    def test(loader):
        model.eval()
        tes_loss = torch.zeros((), device=device)
        tes_acc  = torch.zeros((), device=device)
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                X, Y = X.to(device), Y.to(device)
                #
                FX = model(X)
                FX_last = FX[:,-1].reshape(-1,1)
                loss = -torch.mean(F.logsigmoid(FX_last)*Y+F.logsigmoid(-FX_last)*(1-Y))
                #
                tes_loss += loss
                tes_acc  += (1.-torch.mean(torch.eq(FX_last>0,Y).float()))
        n = batch_idx + 1
        return (tes_loss/n).item(), (tes_acc/n).item()

    #learning
    EPOCH = 200
    model = Net(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    best_epoch1, best_loss1, best_epoch2, best_loss2, result = 0, 10.**30, 0, 10.**30, np.zeros((EPOCH, 8+DIM))
    torch.save(model.state_dict(), "%s/1-seed-1-%d.pt"%(folder,SEED))
    torch.save(model.state_dict(), "%s/1-seed-2-%d.pt"%(folder,SEED))
    for epoch in range(EPOCH):
        result[epoch,  :-4] = train(tra_loader)
        result[epoch,-4:-2] = test(val_loader)
        if result[epoch,-4]<=best_loss1:
            best_epoch1, best_loss1 = epoch, result[epoch,-4]
            torch.save(model.state_dict(), "%s/1-seed-1-%d.pt"%(folder,SEED))
        if result[epoch,-3]<best_loss2:
            best_epoch2, best_loss2 = epoch, result[epoch,-3]
            torch.save(model.state_dict(), "%s/1-seed-2-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-1-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch1,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch1,-2:],"@",best_epoch1)
    os.remove("%s/1-seed-1-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-2-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch2,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch2,-2:],"@",best_epoch2)
    os.remove("%s/1-seed-2-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
