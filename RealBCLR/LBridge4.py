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

ALPs = [1.]
LAMs = [10.**(i/4.-5.) for i in range(33)]
val = np.zeros((len(ALPs),len(LAMs)))
for al in range(len(ALPs)):
    for la in range(len(LAMs)):
        tmp = np.loadtxt("./Results/LBridge1-%.10f/%s/size-100/lam-%.10f/1-seed-%d.csv"%(ALPs[al],DATA,LAMs[la],SEED), delimiter=",")
        val[al,la] = np.min(tmp[:,-4])
idx = np.unravel_index(np.argmin(val), val.shape)
tmp = np.loadtxt("./Results/LBridge1-%.10f/%s/size-100/lam-%.10f/1-seed-%d.csv"%(ALPs[idx[0]],DATA,LAMs[idx[1]],SEED), delimiter=",")
FSs = tmp[np.argmin(tmp[:,-4]),4:-4]; FSs /= np.sum(FSs)
aso = np.argsort(FSs)[::-1]
RFs = []
for i in range(int(np.ceil(ALP*len(aso)))):
    RFs.append(aso[i])
RFs.append(len(aso))

folder = "./Results/LBridge4-%.10f/%s/size-%d"%(ALP,DATA,SIZE)
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("./Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)[:,RFs]
    XYful = rd.permutation(XYful)
    XYtra, XYres = train_test_split(XYful, train_size=SIZE, random_state=SEED) #train SIZE
    XYval, XYtes = train_test_split(XYres, train_size=100,  random_state=SEED) #valid 100, test rest
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]), torch.LongTensor(XYtra[:,-1].reshape(-1,1))
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]), torch.LongTensor(XYval[:,-1].reshape(-1,1))
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]), torch.LongTensor(XYtes[:,-1].reshape(-1,1))
    #
    Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
    for j in range(Xmea.shape[0]):
        if Xstd[j]==0: Xtra[:,j] = 0.; Xval[:,j] = 0.; Xtes[:,j] = 0.
        else: Xtra[:,j] = (Xtra[:,j]-Xmea[j])/Xstd[j]; Xval[:,j] = (Xval[:,j]-Xmea[j])/Xstd[j]; Xtes[:,j] = (Xtes[:,j]-Xmea[j])/Xstd[j]
    #
    tra_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=20, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=20)
    tes_loader = DataLoader(TensorDataset(Xtes, Ytes), batch_size=20)
    DIM = len(XYful[0,:])-1

    #set model, optimizer
    class RegNet(nn.Module):
        def __init__(self, d, M):
            super(RegNet, self).__init__()
            self.fc = nn.Linear(d, 1); torch.nn.init.normal_(self.fc.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc.bias, mean=.0, std=.1**2)
        def forward(self, x):
            x = self.fc(x)
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
            score = torch.sqrt(torch.square(torch.flatten(model.fc.weight))+.1**6)
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
    optimizer = optim.Adam(model.parameters(), lr=.1**1.)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1**(1./EPOCH))
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
        scheduler.step()
    model.load_state_dict(torch.load("%s/1-seed-1-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch1,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch1,-2:])
    os.remove("%s/1-seed-1-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-2-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch2,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch2,-2:])
    os.remove("%s/1-seed-2-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")