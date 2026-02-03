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
parser.add_argument('--data', type=str)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--alp',  type=str)
parser.add_argument('--Mnum', type=int, default=1)
parser.add_argument('--act',  type=str)
parser.add_argument('--lay',  type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()
#

DATA = args.data
SIZE = args.size
ALP  = float(args.alp)
LAM  = 0.
M    = args.Mnum
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

ALPs = [0.125,0.25,0.5,1,2]
LAMs = [10.**(i/4.-5.) for i in range(33)]
val = np.zeros((len(ALPs),len(LAMs)))
for al in range(len(ALPs)):
    for la in range(len(LAMs)):
        tmp = np.loadtxt("./Results/NBridge1-%.10f/relu-4-%d/%s/size-100/lam-%.10f/1-seed-%d.csv"%(ALPs[al],M,DATA,LAMs[la],SEED), delimiter=",")
        val[al,la] = np.min(tmp[:,-2])
idx = np.unravel_index(np.argmin(val), val.shape)
tmp = np.loadtxt("./Results/NBridge1-%.10f/relu-4-%d/%s/size-100/lam-%.10f/1-seed-%d.csv"%(ALPs[idx[0]],M,DATA,LAMs[idx[1]],SEED), delimiter=",")
FSs = tmp[np.argmin(tmp[:,-2]),3:-2]; FSs /= np.sum(FSs)
aso = np.argsort(FSs)[::-1]
CFS, RFs = 0., []
for i in aso:
    CFS += FSs[i]
    RFs.append(i)
    if CFS>ALP: break
RFs.append(len(aso))


folder = "./Results/NBridge2-%.10f/%s-%d-%d/%s/size-%d"%(ALP,ACT,LAY,M,DATA,SIZE)
if os.path.isdir("%s"%folder)==False: os.makedirs("%s"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("./Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)[:,RFs]
    XYful = rd.permutation(XYful)
    XYtra, XYres = train_test_split(XYful, train_size=SIZE, random_state=SEED) #train SIZE
    XYval, XYtes = train_test_split(XYres, train_size=100,  random_state=SEED) #valid 100, test rest
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]), torch.FloatTensor(XYtra[:,-1].reshape(-1,1))
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]), torch.FloatTensor(XYval[:,-1].reshape(-1,1))
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]), torch.FloatTensor(XYtes[:,-1].reshape(-1,1))
    #
    Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
    for j in range(Xmea.shape[0]):
        if Xstd[j]==0: Xtra[:,j] = 0.; Xval[:,j] = 0.; Xtes[:,j] = 0.
        else: Xtra[:,j] = (Xtra[:,j]-Xmea[j])/Xstd[j]; Xval[:,j] = (Xval[:,j]-Xmea[j])/Xstd[j]; Xtes[:,j] = (Xtes[:,j]-Xmea[j])/Xstd[j]
    Ymea = np.mean(XYful[:,-1]); Ystd = np.std(XYful[:,-1])
    if Ystd==0.0: Ytra = 0.; Yval = 0.; Ytes = 0.
    else: Ytra = (Ytra-Ymea)/Ystd; Yval = (Yval-Ymea)/Ystd; Ytes = (Ytes-Ymea)/Ystd
    #
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=20, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=20)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=20)
    DIM = len(XYful[0,:])-1

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

    def train(tra_loader):
        model.train()
        tra_loss, tra_pena, tra_obj = 0.0, 0.0, 0.0
        tra_score = torch.zeros(DIM).to(device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1)#/2.
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
                loss = F.mse_loss(FX, Y)
                #
                tes_loss += loss.item()
        return tes_loss/(batch_idx+1)#/2.

    #learning
    EPOCH = 2000
    model = RegNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1**(1./EPOCH))
    best_epoch, best_loss, result = 0, 10.**30, np.zeros((EPOCH, 5+DIM))
    torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
        scheduler.step()
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1])
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
