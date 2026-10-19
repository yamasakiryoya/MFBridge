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
#lam, seed, device
parser = argparse.ArgumentParser()
parser.add_argument('--type',  type=str)
parser.add_argument('--dim',   type=int, default=3)
parser.add_argument('--size',  type=int, default=100)
parser.add_argument('--score', type=int, default=0, choices=[0, 1, 2, 3, 4],
                     help='0: f_regression, 1: mutual_info_regression, 2: hsic, 3: conditional hsic, 4: hscic')
parser.add_argument('--lam',   type=float, default=.1**3.5,
                     help='regularization parameter for chsic (choice 3) or hscic (choice 4); '
                          'unused for choices 0-2. Passed in from bash.py, which sweeps '
                          'a grid of values for model selection.')
parser.add_argument('--Mnum',  type=int, default=20)
parser.add_argument('--act',   type=str)
parser.add_argument('--lay',   type=int, default=2)
parser.add_argument('--seed',  type=int, default=0)
parser.add_argument('--cuda',  type=int, default=-1)
args = parser.parse_args()
#
SCORE_NAMES = ['f_regression', 'mutual_info_regression', 'hsic', 'chsic', 'hscic']
TYPE  = args.type
DIM   = args.dim
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
#
folder = "./Results/2Filter-%s/%s-%d-%d/%s/dim-%d/size-%d"%(SCORE,ACT,LAY,M,TYPE,DIM,SIZE)
if SCORE in ('chsic', 'hscic'):
    #chsic and hscic require a regularization parameter; results for each
    #value of lam swept by bash.py are kept in their own subfolder, so that
    #model selection across lam (in addition to across nfeat) can be done
    #downstream from the collected results
    folder = "%s/lam-%.10f"%(folder,LAM)
os.makedirs("%s/"%folder, exist_ok=True)
print("%s/seed-%d.csv"%(folder,SEED))


#Gaussian kernel Gram matrix on an (n,k) matrix of (possibly multivariate)
#points, bandwidth by the median heuristic on the pairwise squared distances
#of the points as a whole
def gaussian_gram(V):
    if V.ndim == 1: V = V.reshape(-1, 1)
    d2 = squareform(pdist(V, 'sqeuclidean'))
    med = np.median(d2[d2 > 0])
    sigma2 = med / 2. if med > 0 else 1.
    return np.exp(-d2 / (2. * sigma2))

#Hilbert-Schmidt Independence Criterion (Gretton et al., 2005), biased estimator:
#  HSIC(x,y) = (1/n^2) tr(K H L H),  K,L Gaussian kernels, H = I - (1/n) 11^T
#bandwidth is set by the median heuristic on pairwise distances (standard choice)
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

#Kernel measure of conditional dependence (Fukumizu, Gretton, Sun, Scholkopf, NeurIPS 2007),
#Eq.(6)-(7): I^COND(X,Y|Z) = ||V_{Ydot,Xdot|Z}||_HS^2
#                           = Tr(R_Ydot R_Xdot - 2 R_Ydot R_Xdot R_Z + R_Ydot R_Z R_Xdot R_Z)
#where Xdot=(X,Z), Ydot=(Y,Z) are the extended variables, and
#R_V = G_V (G_V + n*eps*I)^{-1} for the feature-space-centered Gram matrix G_V
#of variable V. Here Z_j is taken to be all features other than j. Since Y is
#continuous, Xdot=(X_j,Z) and Ydot=(Y,Z) are formed by concatenation and a
#single Gaussian kernel is placed on the concatenated vector.
#eps_n (passed in as --lam from bash.py) is swept over a grid for model
#selection, rather than fixed at the paper's own choice of 1e-6.
#COST: O(d n^3).
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
        Z    = np.delete(x, j, axis=1)                          #conditioning set: all other features
        Xdot = np.concatenate([x[:, j:j+1], Z], axis=1)         #extended variable (X_j, Z)
        Ydot = np.concatenate([y.reshape(-1, 1), Z], axis=1)    #extended variable (Y, Z)
        R_Xdot = R(Xdot)
        R_Ydot = R(Ydot)
        R_Z    = R(Z)
        scores[j] = np.trace(R_Ydot @ R_Xdot
                              - 2 * R_Ydot @ R_Xdot @ R_Z
                              + R_Ydot @ R_Z @ R_Xdot @ R_Z)
    return scores

#Hilbert-Schmidt Conditional Independence Criterion (Park & Muandet, NeurIPS 2020),
#Definition 5.3 / Section 5.2 closed-form plug-in estimator:
#  Hhat^2_{X,Y|Z}(z_i) = a^T (K_X (*) K_Y) a - 2 a^T ((K_X a) (*) (K_Y a)) + (a^T K_X a)(a^T K_Y a)
#  where a = W @ k_Z(z_i) = (W @ K_Z)[:,i], W = (K_Z + n*lambda*I)^{-1}, (*) = elementwise product.
#Hhat^2 is a function of z; we summarize it into one feature score by averaging
#over the n training points.
#Bandwidths for X_j, Y and Z are set by the median heuristic (the original
#paper uses a fixed sigma_X=0.1 for all variables instead). lambda
#(passed in as --lam from bash.py) is swept over a grid for model
#selection, rather than fixed at the paper's own choice of 0.01.
#COST: O(d n^3) for the matrix inversions, plus O(d n^3) again for the
#per-point loop below (each of the n terms costs O(n^2)) -- can be slow.
def hscic_score(x, y, lam=.1**3.5):
    n, d = x.shape
    Ky = gaussian_gram(y.reshape(-1, 1))
    scores = np.zeros(d)
    for j in range(d):
        Z  = np.delete(x, j, axis=1)                        #conditioning set: all other features
        Kx = gaussian_gram(x[:, j])
        Kz = gaussian_gram(Z)
        W  = np.linalg.inv(Kz + n * lam * np.eye(n))
        WKz = W @ Kz                #(WKz)[:,i] = W @ k_Z(z_i), since k_Z(z_i) is the i-th column of Kz
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
    Xtra_np, Ytra_np = XYtra[:,:-1], XYtra[:,-1]
    Xtra, Ytra = torch.FloatTensor(XYtra[:,:-1]).to(device), torch.FloatTensor(XYtra[:,-1].reshape(-1,1)).to(device)
    Xval, Yval = torch.FloatTensor(XYval[:,:-1]).to(device), torch.FloatTensor(XYval[:,-1].reshape(-1,1)).to(device)
    Xtes, Ytes = torch.FloatTensor(XYtes[:,:-1]).to(device), torch.FloatTensor(XYtes[:,-1].reshape(-1,1)).to(device)

    #filter method: rank features by a model-free score computed once on the
    #training data (no model is fit here, unlike RFE/SelectFromModel), then
    #train ERM models on the top-1,...,DIM features exactly as in 2MFBridge.py
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
    #score_idx[0] = index of the most relevant feature, score_idx[1] = next, ...
    score_idx = np.argsort(score)[::-1].copy()
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), score, delimiter=",")

    #model, train, test  (penalty-free ERM, same network/train/test as 2MFBridge.py)
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
        for batch_idx1, (X1, Y1) in enumerate(tra_loader):
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
                FX = model(X)
                loss = F.mse_loss(FX, Y)
                #
                tes_loss += loss
        return (tes_loss/(batch_idx+1)).item()

    #using the top 1, 2, ..., DIM features (ranked by the filter score, most
    #relevant first), run ERM (no penalty) for each subset and record every one
    #for later model selection
    summary = np.zeros((DIM, 4))
    for d in range(DIM):
        nfeat = d + 1
        feat_idx = score_idx[:nfeat]
        #
        tra_loader = DataLoader(TensorDataset(Xtra[:,feat_idx], Ytra), batch_size=20, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(Xval[:,feat_idx], Yval), batch_size=20)
        tes_loader = DataLoader(TensorDataset(Xtes[:,feat_idx], Ytes), batch_size=20)
        #
        pt_path = "%s/1-seed-%d-%d.pt"%(folder,SEED,nfeat)
        #learning
        EPOCH = 200
        model = RegNet(nfeat, 100).to(device)
        optimizer = optim.Adam(model.parameters(), lr=.1**3.)
        best_epoch, best_loss, result = 0, 10.**8, np.zeros((EPOCH, 3))
        torch.save(model.state_dict(), pt_path)
        for epoch in range(EPOCH):
            result[epoch,0] = train(tra_loader)
            result[epoch,1] = test(val_loader)
            if result[epoch,1]<best_loss:
                best_epoch, best_loss = epoch, result[epoch,1]
                torch.save(model.state_dict(), pt_path)
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        result[best_epoch,2] = test(tes_loader)
        print('nfeat : ', nfeat, ' tes_loss : ', result[best_epoch,2], "@", best_epoch)
        os.remove(pt_path)
        #
        summary[d] = [nfeat, result[best_epoch,0], result[best_epoch,1], result[best_epoch,2]]
    #
    np.savetxt("%s/2-seed-%d.csv"%(folder,SEED), summary, delimiter=",")