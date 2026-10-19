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
CHUNK = 300
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")


folder = "./Results/1MFLoss-%.10f/%s-%d-%d/%s/size-%d/lam-%.10f"%(ALP,ACT,LAY,M,DATA,SIZE,LAM)
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


    def train_per_feature(tra_loader):
        model.train()
        tra_loss = 0.0
        tra_pena = 0.0
        tra_obj  = 0.0
        tra_acc  = 0.0
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            B = X1.shape[0]
            #
            Mlist = torch.randint(Xtra.shape[0], (M,), device=device)
            #
            FX1 = model(X1)
            loss1 = -(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
            loss = torch.mean(loss1)
            #
            loss.backward(retain_graph=True)
            #
            score = torch.zeros(DIM, device=device)
            pena_sum = 0.0
            for j in range(DIM):
                X2j = X1[:, None, :].expand(B, M, DIM).clone()
                X2j[:, :, j] = Xtra[Mlist, j]
                FX2j = model(X2j.reshape(-1, DIM)).view(B, M, -1)
                Y2 = Y1.view(-1, 1, 1)
                loss2_j = -(F.logsigmoid(FX2j)*Y2+F.logsigmoid(-FX2j)*(1-Y2))
                diff_j = loss1.view(-1, 1, 1) - loss2_j
                score_j = torch.sqrt(torch.mean(diff_j**2) + .1**6)
                pena_j = torch.pow(score_j, ALP)
                (LAM*pena_j).backward(retain_graph=(j < DIM-1))
                score[j] = score_j.item()
                pena_sum += pena_j.item()
            optimizer.step()
            #
            tra_loss  += loss.item()
            tra_pena  += pena_sum
            tra_obj   += loss.item() + LAM*pena_sum
            tra_acc   += (1.-torch.mean(torch.eq(FX1>0,Y1).float())).item()
            tra_score += score
        n = batch_idx1 + 1
        result = torch.cat([torch.tensor([tra_loss, tra_pena, tra_obj, tra_acc], device=device), tra_score]) / n
        return result.cpu().numpy()

    def train_batched(tra_loader):
        model.train()
        tra_loss = torch.zeros((), device=device)
        tra_pena = torch.zeros((), device=device)
        tra_obj  = torch.zeros((), device=device)
        tra_acc  = torch.zeros((), device=device)
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            B = X1.shape[0]
            #
            Mlist = torch.randint(Xtra.shape[0], (M,), device=device)
            X2 = X1[:, None, None, :].expand(B, DIM, M, DIM).clone()
            for j in range(DIM): X2[:, j, :, j] = Xtra[Mlist, j]
            X2 = X2.reshape(-1, DIM)
            #
            Xcat = torch.cat([X1, X2], dim=0)
            Fcat = model(Xcat)
            FX1 = Fcat[:B]
            FX2 = Fcat[B:].view(B, DIM, M, -1)
            #
            loss1 = -(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
            loss = torch.mean(loss1)
            #
            Y2 = Y1.view(-1, 1, 1, 1)
            loss2 = -(F.logsigmoid(FX2)*Y2+F.logsigmoid(-FX2)*(1-Y2))
            #
            diff = loss1.view(-1, 1, 1, 1) - loss2
            score = torch.sqrt(torch.mean(diff**2, dim=(0, 2, 3))+.1**6)
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
                tra_acc   += (1.-torch.mean(torch.eq(FX1>0,Y1).float()))
                tra_score += score
        n = batch_idx1 + 1
        result = torch.cat([torch.stack([tra_loss, tra_pena, tra_obj, tra_acc]), tra_score]) / n
        return result.cpu().numpy()

    def train_chunked(tra_loader):
        model.train()
        tra_loss = 0.0
        tra_pena = 0.0
        tra_obj  = 0.0
        tra_acc  = 0.0
        tra_score = torch.zeros(DIM, device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            B = X1.shape[0]
            #
            Mlist = torch.randint(Xtra.shape[0], (M,), device=device)
            #
            FX1 = model(X1)
            loss1 = -(F.logsigmoid(FX1)*Y1+F.logsigmoid(-FX1)*(1-Y1))
            loss = torch.mean(loss1)
            loss.backward(retain_graph=True)
            #
            score = torch.zeros(DIM, device=device)
            pena_sum = 0.0
            for j0 in range(0, DIM, CHUNK):
                j1 = min(j0+CHUNK, DIM)
                K = j1 - j0
                X2 = X1[:, None, None, :].expand(B, K, M, DIM).clone()
                for k in range(K): X2[:, k, :, j0+k] = Xtra[Mlist, j0+k]
                FX2 = model(X2.reshape(-1, DIM)).view(B, K, M, -1)
                Y2 = Y1.view(-1, 1, 1, 1)
                loss2 = -(F.logsigmoid(FX2)*Y2+F.logsigmoid(-FX2)*(1-Y2))
                diff = loss1.view(-1, 1, 1, 1) - loss2
                score_c = torch.sqrt(torch.mean(diff**2, dim=(0, 2, 3))+.1**6)
                pena_c = torch.sum(torch.pow(score_c, ALP))
                (LAM*pena_c).backward(retain_graph=(j1 < DIM))
                score[j0:j1] = score_c.detach()
                pena_sum += pena_c.item()
            optimizer.step()
            #
            tra_loss  += loss.item()
            tra_pena  += pena_sum
            tra_obj   += loss.item() + LAM*pena_sum
            tra_acc   += (1.-torch.mean(torch.eq(FX1>0,Y1).float())).item()
            tra_score += score
        n = batch_idx1 + 1
        result = torch.cat([torch.tensor([tra_loss, tra_pena, tra_obj, tra_acc], device=device), tra_score]) / n
        return result.cpu().numpy()

    train = train_chunked

    def test(loader):
        model.eval()
        tes_loss = torch.zeros((), device=device)
        tes_acc  = torch.zeros((), device=device)
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(loader):
                X, Y = X.to(device), Y.to(device)
                #
                FX = model(X)
                loss = -torch.mean(F.logsigmoid(FX)*Y+F.logsigmoid(-FX)*(1-Y))
                #
                tes_loss += loss
                tes_acc  += (1.-torch.mean(torch.eq(FX>0,Y).float()))
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
        print(epoch,result[epoch,0],result[epoch,-4],result[epoch,-3])
    model.load_state_dict(torch.load("%s/1-seed-1-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch1,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch1,-2:],"@",best_epoch1)
    os.remove("%s/1-seed-1-%d.pt"%(folder,SEED))
    model.load_state_dict(torch.load("%s/1-seed-2-%d.pt"%(folder,SEED), weights_only=True))
    result[best_epoch2,-2:] = test(tes_loader)
    print('tes_loss : ', result[best_epoch2,-2:],"@",best_epoch2)
    os.remove("%s/1-seed-2-%d.pt"%(folder,SEED))
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
