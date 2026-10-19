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
parser.add_argument('--lam0', type=str, default='0.001')
parser.add_argument('--act',  type=str)
parser.add_argument('--lay',  type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()


DATA = args.data
SIZE = args.size
ALP  = float(args.alp)
LAM  = float(args.lam)
LAM0 = float(args.lam0)
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")


folder = "./Results/1SPINN-%.10f/%s-%d/%s/size-%d/lam-%.10f"%(ALP,ACT,LAY,DATA,SIZE,LAM)
os.makedirs("%s"%folder, exist_ok=True)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("../Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)
    XYful = rd.permutation(XYful)
    XYtrava, XYtes = train_test_split(XYful, train_size=SIZE, random_state=SEED)
    XYtra, XYval = train_test_split(XYtrava, train_size=0.8, random_state=SEED)
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.FloatTensor(XYtra[:,-1].reshape(-1,1)).to(device)
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.FloatTensor(XYval[:,-1].reshape(-1,1)).to(device)
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.FloatTensor(XYtes[:,-1].reshape(-1,1)).to(device)
    #
    Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
    for j in range(Xmea.shape[0]):
        if Xstd[j]==0: Xtra[:,j] = 0.; Xval[:,j] = 0.; Xtes[:,j] = 0.
        else: Xtra[:,j] = (Xtra[:,j]-Xmea[j])/Xstd[j]; Xval[:,j] = (Xval[:,j]-Xmea[j])/Xstd[j]; Xtes[:,j] = (Xtes[:,j]-Xmea[j])/Xstd[j]
    Ymea = np.mean(XYful[:,-1]); Ystd = np.std(XYful[:,-1])
    if Ystd==0.0: Ytra = 0.; Yval = 0.; Ytes = 0.
    else: Ytra = (Ytra-Ymea)/Ystd; Yval = (Yval-Ymea)/Ystd; Ytes = (Ytes-Ymea)/Ystd
    #
    BS = 20
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=BS, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=BS)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=BS)
    DIM = len(XYful[0,:])-1


    class Net(nn.Module):
        def __init__(self, d, M):
            super(Net, self).__init__()
            if LAY>=2: self.fc1 = nn.Linear(d, M); torch.nn.init.normal_(self.fc1.weight, mean=.0, std=.1**2.); torch.nn.init.normal_(self.fc1.bias, mean=.0, std=.1**2.)
            if LAY>=3: self.fc2 = nn.Linear(M, M); torch.nn.init.normal_(self.fc2.weight, mean=.0, std=.1**2.); torch.nn.init.normal_(self.fc2.bias, mean=.0, std=.1**2.)
            if LAY>=4: self.fc3 = nn.Linear(M, M); torch.nn.init.normal_(self.fc3.weight, mean=.0, std=.1**2.); torch.nn.init.normal_(self.fc3.bias, mean=.0, std=.1**2.)
            self.fc4 = nn.Linear(M, 1);            torch.nn.init.normal_(self.fc4.weight, mean=.0, std=.1**2.); torch.nn.init.normal_(self.fc4.bias, mean=.0, std=.1**2.)
        def forward(self, x):
            if LAY>=2: x = eval("torch."+ACT)(self.fc1(x))
            if LAY>=3: x = eval("torch."+ACT)(self.fc2(x))
            if LAY>=4: x = eval("torch."+ACT)(self.fc3(x))
            x = self.fc4(x)
            return x
        def upper(self):
            ws = []
            if LAY>=3: ws.append(self.fc2.weight)
            if LAY>=4: ws.append(self.fc3.weight)
            ws.append(self.fc4.weight)
            return ws

    @torch.no_grad()
    def sgl_prox(W, t):
        U = torch.sign(W)*torch.clamp(W.abs()-t*(1.-ALP), min=0.)
        nrm = torch.linalg.norm(U, dim=0, keepdim=True)
        W.copy_(U*torch.clamp(1.-t*ALP/torch.clamp(nrm, min=.1**12), min=0.))

    def train(tra_loader, epoch):
        model.train()
        tra_loss = torch.zeros((), device=device)
        tra_pena = torch.zeros((), device=device)
        tra_obj  = torch.zeros((), device=device)
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1)
            #
            ridge = LAM0*sum(torch.sum(w**2) for w in model.upper())
            (loss+ridge).backward()
            optimizer.step()
            #
            sgl_prox(model.fc1.weight.data, LAM*optimizer.param_groups[0]['lr'])
            #
            with torch.no_grad():
                W     = model.fc1.weight.data
                score = torch.linalg.norm(W, dim=0).clone()
                pena  = (1.-ALP)*W.abs().sum()+ALP*score.sum()
                obj   = loss + LAM*pena + ridge
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
                X, Y = X.to(device), Y.to(device)
                #
                FX = model(X)
                loss = F.mse_loss(FX, Y)
                #
                tes_loss += loss
        return (tes_loss/(batch_idx+1)).item()

    #learning
    EPOCH = 200
    model = Net(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    best_epoch, best_loss, result = 0, 10.**30, np.zeros((EPOCH, 5+DIM))
    torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader, epoch)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
