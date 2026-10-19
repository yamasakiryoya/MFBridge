#import
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import itertools
import argparse
import numpy as np
import numpy.random as rd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()


DATA = args.data
SIZE = args.size
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")


folder = "./Results/1RandomForest/%s/size-%d"%(DATA,SIZE)
if os.path.isdir("%s"%folder)==False: os.makedirs("%s"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("../Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)
    XYful = rd.permutation(XYful)
    XYtrava, XYtes = train_test_split(XYful, train_size=SIZE, random_state=SEED)
    XYtra, XYval = train_test_split(XYtrava, train_size=0.8, random_state=SEED)
    Xtra, Ytra = XYtra[:,:-1], XYtra[:,-1]
    Xval, Yval = XYval[:,:-1], XYval[:,-1]
    Xtes, Ytes = XYtes[:,:-1], XYtes[:,-1]
    #
    Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
    for j in range(Xmea.shape[0]):
        if Xstd[j]==0: Xtra[:,j] = 0.; Xval[:,j] = 0.; Xtes[:,j] = 0.
        else: Xtra[:,j] = (Xtra[:,j]-Xmea[j])/Xstd[j]; Xval[:,j] = (Xval[:,j]-Xmea[j])/Xstd[j]; Xtes[:,j] = (Xtes[:,j]-Xmea[j])/Xstd[j]
    Ymea = np.mean(XYful[:,-1]); Ystd = np.std(XYful[:,-1])
    if Ystd==0.0: Ytra = 0.; Yval = 0.; Ytes = 0.
    else: Ytra = (Ytra-Ymea)/Ystd; Yval = (Yval-Ymea)/Ystd; Ytes = (Ytes-Ymea)/Ystd
    #
    DIM = len(XYful[0,:])-1

    N_ESTIMATORS = [100]
    MAX_DEPTH    = [2, 4, 8, 16, 32, 64, 128]
    grid = list(itertools.product(N_ESTIMATORS, MAX_DEPTH))

    result = np.zeros((len(grid), 3+DIM))
    best_idx, best_val = 0, np.inf
    for g, (n_estimators, max_depth) in enumerate(grid):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=SEED)
        model.fit(Xtra, Ytra)
        #
        result[g,0]    = np.mean(np.square(Ytra-model.predict(Xtra)))
        result[g,1:-2] = model.feature_importances_
        result[g,-2]   = np.mean(np.square(Yval-model.predict(Xval)))
        result[g,-1]   = np.mean(np.square(Ytes-model.predict(Xtes)))
        if result[g,-2] < best_val:
            best_idx, best_val = g, result[g,-2]

    print('best (n_estimators, max_depth) : ', grid[best_idx])
    print('result : ', result[best_idx])
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")