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
parser.add_argument('--alp',  type=str)   # hierarchy coefficient M of LassoNet
parser.add_argument('--lam',  type=str)
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
ACT  = args.act
LAY  = args.lay
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/1LassoNet-%.10f/%s-%d/%s/dim-%d/size-%d/lam-%.10f"%(ALP,ACT,LAY,TYPE,DIM,SIZE,LAM)
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
    #LassoNet (Lemhadri et al., 2021), objective (2):
    #  min_{theta,W} L(theta,W) + LAM*||theta||_1
    #  s.t. ||W^(1)_j||_inf <= ALP*|theta_j| for every feature j,
    #where ALP is the hierarchy coefficient M of the paper (default M=10).
    #f(x) = theta^T x + g_W(x) is a residual feed-forward network.
    class LassoNet(nn.Module):
        def __init__(self, d, M):
            super(LassoNet, self).__init__()
            self.skip = nn.Linear(d, 1, bias=False); torch.nn.init.zeros_(self.skip.weight)
            if LAY>=2: self.fc1 = nn.Linear(d, M); torch.nn.init.normal_(self.fc1.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc1.bias, mean=.0, std=.1**2)
            if LAY>=3: self.fc2 = nn.Linear(M, M); torch.nn.init.normal_(self.fc2.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc2.bias, mean=.0, std=.1**2)
            if LAY>=4: self.fc3 = nn.Linear(M, M); torch.nn.init.normal_(self.fc3.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc3.bias, mean=.0, std=.1**2)
            self.fc4 = nn.Linear(M, 1);            torch.nn.init.normal_(self.fc4.weight, mean=.0, std=.1**2); torch.nn.init.normal_(self.fc4.bias, mean=.0, std=.1**2)
        def forward(self, x):
            h = x
            if LAY>=2: h = eval("torch."+ACT)(self.fc1(h))
            if LAY>=3: h = eval("torch."+ACT)(self.fc2(h))
            if LAY>=4: h = eval("torch."+ACT)(self.fc3(h))
            h = self.fc4(h)
            return h + self.skip(x)

    #Hier-Prox (Lemhadri et al., 2021, Algorithm 2), vectorized over features.
    #theta: (d,) skip weights; W: (K,d) first-layer weights, column j = feature j.
    @torch.no_grad()
    def hier_prox(theta, W, lam, M):
        d, K = theta.shape[0], W.shape[0]
        theta_abs, W_abs = theta.abs(), W.abs()
        W_sorted, _ = torch.sort(W_abs, dim=0, descending=True)               #|W_(j,1)|>=...>=|W_(j,K)|
        cumsum = torch.cat([torch.zeros(1, d, device=theta.device),
                            torch.cumsum(W_sorted, dim=0)], dim=0)            #(K+1,d): sum_{i<=m}
        m = torch.arange(K+1, device=theta.device, dtype=theta.dtype).view(-1,1)
        w_m = (M/(1.+m*M*M))*torch.clamp(theta_abs.view(1,-1)+M*cumsum-lam, min=0.)
        #conventions of Alg.2: |W_(j,0)|=+inf, |W_(j,K+1)|=0
        upper = torch.cat([torch.full((1,d), float('inf'), device=theta.device), W_sorted], dim=0)
        lower = torch.cat([W_sorted, torch.zeros(1, d, device=theta.device)], dim=0)
        idx = torch.argmax(((lower<=w_m)&(w_m<=upper)).to(theta.dtype), dim=0)  #first feasible m
        w_star = w_m.gather(0, idx.view(1,-1)).squeeze(0)
        theta.copy_(torch.sign(theta)*w_star/M)
        W.copy_(torch.sign(W)*torch.minimum(w_star.view(1,-1).expand_as(W), W_abs))

    def train(tra_loader, epoch):
        model.train()
        #Alg.1 line 2: initialize by training the dense network on the loss alone.
        #The paper's dense-to-sparse strategy is essential (its Fig.4); since each
        #lambda is run independently here, we emulate it with a dense warm-up.
        lam = LAM
        tra_loss, tra_pena, tra_obj = 0.0, 0.0, 0.0
        tra_score = torch.zeros(DIM).to(device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1)#/2.
            #
            #the l1 penalty on theta is handled by Hier-Prox, so only L is differentiated
            loss.backward()
            optimizer.step()
            #
            hier_prox(model.skip.weight.data.view(-1), model.fc1.weight.data,
                      lam*optimizer.param_groups[0]['lr'], ALP)
            #
            with torch.no_grad():
                score = model.skip.weight.data.view(-1).abs().clone()   #|theta_j|
                pena  = torch.sum(score)                                #||theta||_1
                obj   = loss + lam*pena
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
    model = LassoNet(DIM, 100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3.)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1**(1./EPOCH))
    best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 5+DIM))
    for epoch in range(EPOCH):
        result[epoch,:-2] = train(tra_loader, epoch)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
        # scheduler.step()
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), map_location=device, weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
