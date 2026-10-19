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
parser.add_argument('--lay',  type=int, default=2)
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


folder = "./Results/1MFElim-%.10f/%s-%d-%d/%s/size-%d/lam-%.10f"%(ALP,ACT,LAY,M,DATA,SIZE,LAM)
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
            self.d = d
            self.sub_models = nn.ModuleList([
                nn.ModuleDict({
                    'fc1': nn.Linear(d if i == 0 else d - 1, M),
                    'fc2': nn.Linear(M, M) if LAY >= 3 else nn.Identity(),
                    'fc3': nn.Linear(M, M) if LAY >= 4 else nn.Identity(),
                    'fc4': nn.Linear(M, 1)
                }) for i in range(d + 1)
            ])
            for m in self.sub_models:
                for layer in m.values():
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.normal_(layer.weight, mean=0.0, std=.1**2.)
                        torch.nn.init.normal_(layer.bias, mean=0.0, std=.1**2.)
        def forward_sub(self, x, sub_idx):
            m = self.sub_models[sub_idx]
            x = eval("torch." + ACT)(m['fc1'](x))
            if LAY >= 3: x = eval("torch." + ACT)(m['fc2'](x))
            if LAY >= 4: x = eval("torch." + ACT)(m['fc3'](x))
            return m['fc4'](x)
        def forward(self, x):
            return self.forward_sub(x, 0)

    def train_per_feature(tra_loader):
        model.train()
        tra_loss = 0.0
        tra_pena = 0.0
        tra_obj  = 0.0
        tra_score = torch.zeros(DIM, device=device)
        mask_mat = ~torch.eye(DIM, dtype=torch.bool, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            B = X1.shape[0]
            #
            Mlist = torch.randint(Xtra.shape[0], (M,), device=device)
            #
            FX1 = model.forward_sub(X1, 0)
            loss1 = F.mse_loss(Y1, FX1)
            (loss1/(DIM+1)).backward()
            #
            score = torch.zeros(DIM, device=device)
            loss_sum = loss1.item()
            pena_sum = 0.0
            for j in range(DIM):
                X1_j = X1[:, mask_mat[j]]
                FX2_j = model.forward_sub(X1_j, j + 1)
                loss_j = F.mse_loss(Y1, FX2_j)
                #
                X2j = X1[:, None, :].expand(B, M, DIM).clone()
                X2j[:, :, j] = Xtra[Mlist, j]
                FX3_j = model.forward_sub(X2j.reshape(-1, DIM), 0).view(B, M, 1)
                #
                diff_j = FX3_j - FX2_j.unsqueeze(1)
                score_j = torch.sqrt(torch.mean(diff_j**2)/2. + .1**6)
                pena_j = torch.pow(score_j, ALP)
                #
                obj_j = loss_j/(DIM+1) + LAM*pena_j
                obj_j.backward()
                #
                score[j] = score_j.item()
                loss_sum += loss_j.item()
                pena_sum += pena_j.item()
            optimizer.step()
            #
            loss_mean = loss_sum / (DIM+1)
            tra_loss += loss_mean
            tra_pena += pena_sum
            tra_obj  += loss_mean + LAM*pena_sum
            tra_score += score
        n = batch_idx1 + 1
        result = torch.cat([torch.tensor([tra_loss, tra_pena, tra_obj], device=device), tra_score]) / n
        return result.cpu().numpy()

    def train_batched(tra_loader):
        model.train()
        tra_loss = torch.zeros((), device=device)
        tra_pena = torch.zeros((), device=device)
        tra_obj  = torch.zeros((), device=device)
        tra_score = torch.zeros(DIM, device=device)
        mask_mat = ~torch.eye(DIM, dtype=torch.bool, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            B = X1.shape[0]
            #
            Mlist = torch.randint(Xtra.shape[0], (M,), device=device)
            X3 = X1[:, None, None, :].expand(-1, DIM, M, -1).clone()
            for j in range(DIM): X3[:, j, :, j] = Xtra[Mlist, j].view(1, M)
            X3 = X3.reshape(-1, DIM)
            #
            Xcat0 = torch.cat([X1, X3], dim=0)
            Fcat0 = model.forward_sub(Xcat0, 0)
            FX1 = Fcat0[:B]
            FX3 = Fcat0[B:].view(B, DIM, M, 1).permute(1, 0, 2, 3)
            loss1 = F.mse_loss(Y1, FX1)
            #
            FX2_list = []
            loss_list = [loss1]
            for j in range(DIM):
                X1_j = X1[:, mask_mat[j]]
                FX2_j = model.forward_sub(X1_j, j + 1)
                loss_j = F.mse_loss(Y1, FX2_j)
                FX2_list.append(FX2_j)
                loss_list.append(loss_j)
            FX2 = torch.stack(FX2_list)
            loss = torch.stack(loss_list).mean()
            #
            diff = FX3 - FX2.unsqueeze(2)
            score = torch.sqrt(torch.mean(diff**2, dim=(1, 2, 3))/2. + .1**6)
            #
            pena = torch.sum(torch.pow(score, ALP))
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

    train = train_batched

    def test(loader):
        model.eval()
        tes_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                X, Y = X.to(device), Y.to(device)
                #
                FX = model.forward_sub(X, 0)
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
        result[epoch,:-2] = train(tra_loader)
        result[epoch, -2] = test(val_loader)
        if result[epoch,-2]<best_loss:
            best_epoch, best_loss = epoch, result[epoch,-2]
            torch.save(model.state_dict(), "%s/1-seed-%d.pt"%(folder,SEED))
        print(epoch,result[epoch,0],result[epoch,-2])
    model.load_state_dict(torch.load("%s/1-seed-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch,-1] = test(tes_loader)
    print('tes_loss : ', result[best_epoch,-1], "@", best_epoch)
    os.remove("%s/1-seed-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
