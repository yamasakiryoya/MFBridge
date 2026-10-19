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
M    = args.Mnum
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")
#
ALPs = [.125,.25,.5,1.,2.]
LAMs = [10.**(i/8.-6.) for i in range(81)]
val = np.zeros((len(ALPs),len(LAMs)))
for i in range(len(ALPs)):
    for j in range(len(LAMs)):
        #columns: [loss,pena,obj,acc, DIM score cols, val_loss,val_acc,tes_loss,tes_acc]
        tmp = np.loadtxt("./Results/1MFBridge-%.10f/relu-4-20/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(ALPs[i],TYPE,DIM,SIZE,LAMs[j],SEED), delimiter=",")
        val[i,j] = np.min(tmp[:,-4])   #val_loss
B = np.unravel_index(np.argmin(val),val.shape)
#
tmp = np.loadtxt("./Results/1MFBridge-%.10f/relu-4-20/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(ALPs[B[0]],TYPE,DIM,SIZE,LAMs[B[1]],SEED), delimiter=",")
score = np.argsort(tmp[np.argmin(tmp[:,-4]),4:-4])[::-1].copy()
#
folder = "./Results/2MFBridge/%s-%d-%d/%s/dim-%d/size-%d"%(ACT,LAY,M,TYPE,DIM,SIZE)
os.makedirs("%s/"%folder, exist_ok=True)
print("%s/seed-%d.csv"%(folder,SEED))

if os.path.isfile("%s/2-seed-%d.csv"%(folder,SEED))==False:
    #dataset
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
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED) #train:val = 4:1
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.LongTensor(XYtra[:,-1].reshape(-1,1)).to(device)
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.LongTensor(XYval[:,-1].reshape(-1,1)).to(device)
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.LongTensor(XYtes[:,-1].reshape(-1,1)).to(device)

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
        tra_loss = torch.zeros((), device=device)
        tra_acc  = torch.zeros((), device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = -torch.mean(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
            #
            loss.backward()
            optimizer.step()
            #
            with torch.no_grad():
                tra_loss += loss
                tra_acc  += 1. - torch.mean(torch.eq(FX1>0, Y1).float())
        n = batch_idx1 + 1
        result = torch.stack([tra_loss, tra_acc]) / n
        return result.cpu().numpy()

    def test(loader):
        model.eval()
        tes_loss = torch.zeros((), device=device)
        tes_acc  = torch.zeros((), device=device)
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                #
                FX = model(X)
                loss = -torch.mean(F.logsigmoid(FX)*Y+F.logsigmoid(-FX)*(1-Y))
                #
                tes_loss += loss
                tes_acc  += 1. - torch.mean(torch.eq(FX>0,Y).float())
        n = batch_idx + 1
        return (tes_loss/n).item(), (tes_acc/n).item()

    #using the top 1, 2, ..., DIM features (ranked by score, most important first),
    #run ERM (no penalty) for each subset and record every one for later model selection
    summary = np.zeros((DIM, 7))  #nfeat, tra_loss, tra_acc, val_loss, val_acc, tes_loss, tes_acc
    for d in range(DIM):
        nfeat = d + 1
        feat_idx = score[:nfeat]
        #
        tra_loader = DataLoader(TensorDataset(Xtra[:,feat_idx], Ytra), batch_size=20, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(Xval[:,feat_idx], Yval), batch_size=20)
        tes_loader = DataLoader(TensorDataset(Xtes[:,feat_idx], Ytes), batch_size=20)
        #
        pt_path = "%s/1-seed-1-%d-%d.pt"%(folder,SEED,nfeat)
        #learning
        EPOCH = 200
        model = RegNet(nfeat, 100).to(device)
        optimizer = optim.Adam(model.parameters(), lr=.1**3.)
        best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 6))
        torch.save(model.state_dict(), pt_path)
        for epoch in range(EPOCH):
            result[epoch,  :2] = train(tra_loader)
            result[epoch, 2:4] = test(val_loader)
            if result[epoch,2]<best_loss:
                best_epoch, best_loss = epoch, result[epoch,2]
                torch.save(model.state_dict(), pt_path)
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        result[best_epoch,4:] = test(tes_loader)
        print('nfeat : ', nfeat, ' tes_loss : ', result[best_epoch,4:], "@", best_epoch)
        os.remove(pt_path)
        # np.savetxt("%s/1-seed-%d-%d.csv"%(folder,SEED,nfeat), result, delimiter=",")
        #
        summary[d] = [nfeat, result[best_epoch,0], result[best_epoch,1],result[best_epoch,2], result[best_epoch,3],result[best_epoch,4], result[best_epoch,5]]
    #
    np.savetxt("%s/2-seed-%d.csv"%(folder,SEED), summary, delimiter=",")
