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
M    = args.Mnum
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/NBridge1-%.10f/%s-%d-%d/%s/dim-%d/size-%d/lam-%.10f"%(ALP,ACT,LAY,M,TYPE,DIM,SIZE,LAM)
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.zeros((SIZE+10100,DIM+1))
    for i in range(SIZE+10100):
        for j in range(DIM): XYful[i,j]=rd.normal(0,1)
        if TYPE=="non-additive-sq1":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-6.))) else 0
        elif TYPE=="non-additive-sq2":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-5.))) else 0
        # if   TYPE=="linear":            XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2]))) else 0
        # elif TYPE=="additive-sq":       XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0])+np.square(2.*XYful[i,1])+np.square(3.*XYful[i,2])-10.))) else 0
        # elif TYPE=="non-additive-sq1":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-4.))) else 0
        # elif TYPE=="non-additive-sq2":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1])+np.square(3.*XYful[i,2])-10.))) else 0
        # elif TYPE=="non-additive-sq3":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0])+np.square(2.*XYful[i,1]+3.*XYful[i,2])-7.))) else 0
        # elif TYPE=="non-additive-sq4":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-6.))) else 0
        # elif TYPE=="additive-sin":      XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0])+np.sin(2.*XYful[i,1])+np.sin(3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin1": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin2": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0]+2.*XYful[i,1])+np.sin(3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin3": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0])+np.sin(2.*XYful[i,1]+3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin4": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.sin(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])))) else 0
    #print(np.sum(XYful[:,-1])/len(XYful[:,-1]))
    XYful = rd.permutation(XYful)
    XYtra, XYres = train_test_split(XYful, train_size=SIZE, random_state=SEED) #train SIZE
    XYval, XYtes = train_test_split(XYres, train_size=100,  random_state=SEED) #valid 100, test rest
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]), torch.LongTensor(XYtra[:,-1].reshape(-1,1))
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]), torch.LongTensor(XYval[:,-1].reshape(-1,1))
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]), torch.LongTensor(XYtes[:,-1].reshape(-1,1))
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=20, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=20)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=20)

    #set model, optimizer
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

    def train(tra_loader):
        model.train()
        tra_loss, tra_pena, tra_obj, tra_acc = 0.0, 0.0, 0.0, 0.0
        tra_score = torch.zeros(DIM).to(device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = -torch.mean(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
            #
            Mlist = torch.randint(SIZE, (M,), device=device)
            X2 = X1[:, None, None, :].expand(Y1.shape[0], DIM, M, DIM).clone()
            for j in range(DIM): X2[:, j, :, j] = Xtra[Mlist, j]
            #
            FX2 = model(X2.reshape(-1, DIM))
            FX2 = FX2.view(Y1.shape[0], DIM, M, -1)
            #
            diff = FX1[:, None, None, :] - FX2
            score = torch.sqrt(torch.mean(diff**2, dim=(0, 2, 3))/2./Y1.shape[0]+.1**6)
            """
            score = torch.zeros(DIM).to(device)
            Mlist = rd.randint(SIZE, size=M).tolist()
            for j in range(DIM):
                for i in range(Y1.shape[0]):
                    X12 = X1[i,:].repeat(M,1)
                    X12[:,j] = Xtra[Mlist,j]
                    FX12 = model(X12)
                    score[j] += torch.mean(torch.square(FX12-FX1[i,:]))
            score = torch.sqrt(score/2./Y1.shape[0]+.1**6)
            """
            pena = torch.sum(torch.pow(score, ALP))
            #
            obj = loss + LAM*pena
            obj.backward()
            optimizer.step()
            #
            tra_loss  += loss.item()
            tra_pena  += pena.item()
            tra_obj   += obj.item()
            tra_acc   += (1.-torch.mean(torch.eq(FX1>0,Y1).float()).item())
            tra_score += score
        return np.concatenate([torch.tensor([tra_loss, tra_pena, tra_obj, tra_acc]).cpu().detach().numpy(), tra_score.cpu().detach().numpy()])/(batch_idx1+1)

    def test(loader):
        model.eval()
        tes_loss, tes_acc = 0.0, 0.0
        with torch.set_grad_enabled(False):
            for batch_idx, (X, Y) in enumerate(loader):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                #
                FX = model(X)
                loss = -torch.mean(F.logsigmoid(FX)*Y+F.logsigmoid(-FX)*(1-Y))
                #
                tes_loss += loss.item()
                tes_acc  += (1.-torch.mean(torch.eq(FX>0,Y).float()).item())
        return tes_loss/(batch_idx+1), tes_acc/(batch_idx+1)

    #learning
    EPOCH = 500
    model = RegNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1**(1./EPOCH))
    best_epoch1, best_loss1, best_epoch2, best_loss2, result = 0, 10.**8, 0, 10.**8, np.zeros((EPOCH, 8+DIM))
    for epoch in range(EPOCH):
        result[epoch,  :-4] = train(tra_loader)
        result[epoch,-4:-2] = test(val_loader)
        if result[epoch,-4]<best_loss1:
            best_epoch1, best_loss1 = epoch, result[epoch,-4]
            torch.save(model.state_dict(), "%s/1-seed-1-%d.pt"%(folder,SEED))
        if result[epoch,-3]<best_loss2:
            best_epoch2, best_loss2 = epoch, result[epoch,-3]
            torch.save(model.state_dict(), "%s/1-seed-2-%d.pt"%(folder,SEED))
        scheduler.step()
    model.load_state_dict(torch.load("%s/1-seed-1-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch1,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch1,-2:], "@", best_epoch1)
    os.remove("%s/1-seed-1-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-2-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch2,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch2,-2:], "@", best_epoch2)
    os.remove("%s/1-seed-2-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
