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
import itertools

#lam, seed, device
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--dim',  type=int, default=3)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--alp',  type=str)   # sparse group lasso mixing parameter alpha (0=lasso, 1=group lasso)
parser.add_argument('--lam',  type=str)
parser.add_argument('--lam0', type=str, default='0.001')
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
LAM0 = float(args.lam0)   # ridge on the upper layers; fixed small, per the paper
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/1SPINN-%.10f/%s-%d/%s/dim-%d/size-%d/lam-%.10f"%(ALP,ACT,LAY,TYPE,DIM,SIZE,LAM)
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
    #SPINN (Feng & Simon, 2017), objective (4):
    #  (1/n) sum_i l(y_i,f(x_i)) + LAM0 * sum_{a=2}^{L+1} ||theta_a||_2^2
    #                            + LAM  * sum_j Omega_ALP(theta_{1,.,j})
    #with the sparse group lasso  Omega_a(t) = (1-a)||t||_1 + a||t||_2.
    #Thus ALP=0 is the pure lasso and ALP=1 the pure group lasso.
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
        def upper(self):
            #theta_a for a=2,...,L+1, i.e. every weight matrix above the first layer
            ws = []
            if LAY>=3: ws.append(self.fc2.weight)
            if LAY>=4: ws.append(self.fc3.weight)
            ws.append(self.fc4.weight)
            return ws

    #Proximal steps (9)-(10) of Feng & Simon (2017), Algorithm 1:
    #coordinate-wise soft-threshold at LAM*(1-ALP), then group soft-scaling at LAM*ALP.
    @torch.no_grad()
    def sgl_prox(W, t):
        U = torch.sign(W)*torch.clamp(W.abs()-t*(1.-ALP), min=0.)
        nrm = torch.linalg.norm(U, dim=0, keepdim=True)
        W.copy_(U*torch.clamp(1.-t*ALP/torch.clamp(nrm, min=.1**12), min=0.))

    def train(tra_loader, epoch):
        model.train()
        tra_loss, tra_pena, tra_obj = 0.0, 0.0, 0.0
        tra_score = torch.zeros(DIM).to(device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1)#/2.
            #
            #smooth part (5): loss + ridge on the upper layers; the sparse group
            #lasso on the first layer is handled by the proximal step below
            ridge = LAM0*sum(torch.sum(w**2) for w in model.upper())
            (loss+ridge).backward()
            optimizer.step()
            #
            sgl_prox(model.fc1.weight.data, LAM*optimizer.param_groups[0]['lr'])
            #
            with torch.no_grad():
                W     = model.fc1.weight.data
                score = (1.-ALP)*W.abs().sum(dim=0)+ALP*torch.linalg.norm(W, dim=0)  #Omega_ALP(theta_{1,.,j})
                pena  = score.sum()                                     #sum_j Omega_ALP
                obj   = loss + LAM*pena + ridge
            #
            tra_loss  += loss.item()
            tra_pena  += pena.item()
            tra_obj   += obj.item()
            tra_score += score
        return np.concatenate([torch.tensor([tra_loss, tra_pena, tra_obj]).cpu().detach().numpy(), tra_score.cpu().detach().numpy()])/(batch_idx1+1)

    def test(loader):
        model.eval()
        tes_loss = 0.0
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                FX = model(X)
                loss = F.mse_loss(FX, Y)
                #
                tes_loss += loss.item()
        return tes_loss/(batch_idx+1)#/2.

    #learning
    EPOCH = 200
    model = RegNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 5+DIM))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader, epoch)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), map_location=device, weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
