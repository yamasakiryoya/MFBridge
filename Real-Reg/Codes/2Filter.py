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
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--data',  type=str)
parser.add_argument('--size',  type=int, default=100)
parser.add_argument('--score', type=int, default=0, choices=[0, 1, 2, 3, 4],
                     help='0: f_regression, 1: mutual_info_regression, 2: hsic, 3: conditional hsic, 4: hscic')
parser.add_argument('--lam',   type=float, default=.1**3.5)
parser.add_argument('--Mnum',  type=int, default=20)
parser.add_argument('--act',   type=str)
parser.add_argument('--lay',   type=int, default=4)
parser.add_argument('--seed',  type=int, default=0)
parser.add_argument('--cuda',  type=int, default=-1)
args = parser.parse_args()


SCORE_NAMES = ['f_regression', 'mutual_info_regression', 'hsic', 'chsic', 'hscic']
DATA  = args.data
SIZE  = args.size
SCORE = SCORE_NAMES[args.score]
LAM   = args.lam
M     = args.Mnum
ACT   = args.act
LAY   = args.lay
SEED  = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")
folder = "./Results/2Filter-%s/%s-%d-%d/%s/size-%d"%(SCORE,ACT,LAY,M,DATA,SIZE)
if SCORE in ('chsic', 'hscic'):
    folder = "%s/lam-%.10f"%(folder,LAM)
os.makedirs("%s/"%folder, exist_ok=True)


def gaussian_gram(V):
    if V.ndim == 1: V = V.reshape(-1, 1)
    d2 = squareform(pdist(V, 'sqeuclidean'))
    med = np.median(d2[d2 > 0])
    sigma2 = med / 2. if med > 0 else 1.
    return np.exp(-d2 / (2. * sigma2))

def hsic_score(x, y):
    n = x.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Ly = gaussian_gram(y.reshape(-1, 1))
    HLH = H @ Ly @ H
    scores = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        Kx = gaussian_gram(x[:, j])
        scores[j] = np.trace(Kx @ HLH) / (n ** 2)
    return scores

def chsic_score(x, y, eps=.1**3.5):
    n, d = x.shape
    ones_over_n = np.ones((n, n)) / n
    def centered_gram(V):
        K = gaussian_gram(V)
        return K - ones_over_n @ K - K @ ones_over_n + ones_over_n @ K @ ones_over_n
    def R(V):
        G = centered_gram(V)
        return G @ np.linalg.inv(G + n * eps * np.eye(n))
    scores = np.zeros(d)
    for j in range(d):
        Z    = np.delete(x, j, axis=1)
        Xdot = np.concatenate([x[:, j:j+1], Z], axis=1)
        Ydot = np.concatenate([y.reshape(-1, 1), Z], axis=1)
        R_Xdot = R(Xdot)
        R_Ydot = R(Ydot)
        R_Z    = R(Z)
        scores[j] = np.trace(R_Ydot @ R_Xdot
                              - 2 * R_Ydot @ R_Xdot @ R_Z
                              + R_Ydot @ R_Z @ R_Xdot @ R_Z)
    return scores

def hscic_score(x, y, lam=.1**3.5):
    n, d = x.shape
    Ky = gaussian_gram(y.reshape(-1, 1))
    scores = np.zeros(d)
    for j in range(d):
        Z  = np.delete(x, j, axis=1)
        Kx = gaussian_gram(x[:, j])
        Kz = gaussian_gram(Z)
        W  = np.linalg.inv(Kz + n * lam * np.eye(n))
        WKz = W @ Kz
        h2 = np.zeros(n)
        for i in range(n):
            a = WKz[:, i]
            term1 = a @ (Kx * Ky) @ a
            u = Kx @ a
            v = Ky @ a
            term2 = 2. * (a @ (u * v))
            term3 = (a @ Kx @ a) * (a @ Ky @ a)
            h2[i] = term1 - term2 + term3
        scores[j] = h2.mean()
    return scores


if os.path.isfile("%s/2-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("../Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)
    XYful = rd.permutation(XYful)
    XYtrava, XYtes = train_test_split(XYful, train_size=SIZE, random_state=SEED)
    XYtra, XYval = train_test_split(XYtrava, train_size=0.8, random_state=SEED)
    Xtra_np, Ytra_np = XYtra[:,:-1], XYtra[:,-1]
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.FloatTensor(XYtra[:,-1].reshape(-1,1)).to(device)
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.FloatTensor(XYval[:,-1].reshape(-1,1)).to(device)
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.FloatTensor(XYtes[:,-1].reshape(-1,1)).to(device)
    #
    Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
    for j in range(Xmea.shape[0]):
        if Xstd[j]==0: Xtra_np[:,j] = 0.; Xtra[:,j] = 0.; Xval[:,j] = 0.; Xtes[:,j] = 0.
        else:
            Xtra_np[:,j] = (Xtra_np[:,j]-Xmea[j])/Xstd[j]
            Xtra[:,j] = (Xtra[:,j]-Xmea[j])/Xstd[j]; Xval[:,j] = (Xval[:,j]-Xmea[j])/Xstd[j]; Xtes[:,j] = (Xtes[:,j]-Xmea[j])/Xstd[j]
    Ymea = np.mean(XYful[:,-1]); Ystd = np.std(XYful[:,-1])
    if Ystd==0.0: Ytra_np = 0.; Ytra = 0.; Yval = 0.; Ytes = 0.
    else:
        Ytra_np = (Ytra_np-Ymea)/Ystd
        Ytra = (Ytra-Ymea)/Ystd; Yval = (Yval-Ymea)/Ystd; Ytes = (Ytes-Ymea)/Ystd
    #
    DIM = len(XYful[0,:])-1

    if SCORE == 'f_regression':
        stat, pval = f_regression(Xtra_np, Ytra_np)
        score = stat
    elif SCORE == 'mutual_info_regression':
        score = mutual_info_regression(Xtra_np, Ytra_np, random_state=SEED)
    elif SCORE == 'hsic':
        score = hsic_score(Xtra_np, Ytra_np)
    elif SCORE == 'chsic':
        score = chsic_score(Xtra_np, Ytra_np, eps=LAM)
    elif SCORE == 'hscic':
        score = hscic_score(Xtra_np, Ytra_np, lam=LAM)
    score_idx = np.argsort(score)[::-1].copy()
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), score, delimiter=",")


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

    def train(tra_loader):
        model.train()
        tra_loss = torch.zeros((), device=device)
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
            X1, Y1 = X1.to(device), Y1.to(device)
            optimizer.zero_grad()
            #
            FX1 = model(X1)
            loss = F.mse_loss(Y1, FX1)
            #
            loss.backward()
            optimizer.step()
            #
            with torch.no_grad():
                tra_loss += loss
        n = batch_idx1 + 1
        result = tra_loss / n
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

    summary = np.zeros((DIM, 4))
    for d in range(DIM):
        nfeat = d + 1
        feat_idx = score_idx[:nfeat]
        #
        BS = 20
        tra_loader = DataLoader(TensorDataset(Xtra[:,feat_idx], Ytra), batch_size=BS, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(Xval[:,feat_idx], Yval), batch_size=BS)
        tes_loader = DataLoader(TensorDataset(Xtes[:,feat_idx], Ytes), batch_size=BS)
        #
        pt_path = "%s/1-seed-%d-%d.pt"%(folder,SEED,nfeat)
        #learning
        EPOCH = 200
        model = Net(nfeat, 100).to(device)
        optimizer = optim.Adam(model.parameters(), lr=.1**3.)
        best_epoch, best_loss, result = 0, 10.**30, np.zeros((EPOCH, 3))
        torch.save(model.state_dict(), pt_path)
        for epoch in range(EPOCH):
            result[epoch,0] = train(tra_loader)
            result[epoch,1] = test(val_loader)
            if result[epoch,1]<=best_loss:
                best_epoch, best_loss = epoch, result[epoch,1]
                torch.save(model.state_dict(), pt_path)
        model.load_state_dict(torch.load(pt_path, weights_only=True))
        result[best_epoch,2] = test(tes_loader)
        print('nfeat : ', nfeat, ' tes_loss : ', result[best_epoch,2])
        os.remove(pt_path)
        # np.savetxt("%s/1-seed-%d-%d.csv"%(folder,SEED,nfeat), result, delimiter=",")
        #
        summary[d] = [nfeat, result[best_epoch,0], result[best_epoch,1], result[best_epoch,2]]
    #
    np.savetxt("%s/2-seed-%d.csv"%(folder,SEED), summary, delimiter=",")