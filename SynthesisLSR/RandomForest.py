#import
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
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

#lam, seed, device
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--dim',  type=int,   default=3)
parser.add_argument('--size', type=int,   default=100)
parser.add_argument('--seed', type=int,   default=0)
parser.add_argument('--cuda', type=int,   default=-1)
args = parser.parse_args()
#
TYPE = args.type
DIM  = args.dim
SIZE = args.size
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/RandomForest/%s/dim-%d/size-%d"%(TYPE,DIM,SIZE)
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.zeros((SIZE+10100,DIM+1))
    for i in range(SIZE+10100):
        for j in range(DIM): XYful[i,j]  = rd.normal(0,1)
        if   TYPE=="linear":            XYful[i,-1] = XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2]+rd.normal(0,1)
        elif TYPE=="additive-sq":       XYful[i,-1] = np.square(XYful[i,0])+np.square(2.*XYful[i,1])+np.square(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sq1":  XYful[i,-1] = np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sq2":  XYful[i,-1] = np.square(XYful[i,0]+2.*XYful[i,1])+np.square(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sq3":  XYful[i,-1] = np.square(XYful[i,0])+np.square(2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="additive-sin":      XYful[i,-1] = np.sin(XYful[i,0])+np.sin(2.*XYful[i,1])+np.sin(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sin1": XYful[i,-1] = np.sin(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sin2": XYful[i,-1] = np.sin(XYful[i,0]+2.*XYful[i,1])+np.sin(3.*XYful[i,2])+rd.normal(0,1)
        elif TYPE=="non-additive-sin3": XYful[i,-1] = np.sin(XYful[i,0])+np.sin(2.*XYful[i,1]+3.*XYful[i,2])+rd.normal(0,1)
    XYful = rd.permutation(XYful)
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=SIZE, random_state=SEED) #train SIZE, valid 100
    Xtra, Ytra = XYtra[:,:-1], XYtra[:,-1]
    Xval, Yval = XYval[:,:-1], XYval[:,-1]
    Xtes, Ytes = XYtes[:,:-1], XYtes[:,-1]

    #learning
    model = RandomForestRegressor()
    model.fit(Xtra, Ytra)
    #forest = GridSearchCV(RandomForestRegressor(), {'n_estimators':[2,4,8,16,32,64],'max_depth':[2,4,8,16,32,64]}, verbose=2)
    #forest.fit(Xtra, Ytra)
    #model = forest.best_estimator_

    result = np.zeros(3+DIM)
    result[0]    = np.mean(np.square(Ytra-model.predict(Xtra)))
    result[1:-2] = model.feature_importances_
    result[-2]   = np.mean(np.square(Yval-model.predict(Xval)))
    result[-1]   = np.mean(np.square(Ytes-model.predict(Xtes)))
    print('result : ', result)
    np.savetxt("%s/seed-%d.csv"%(folder,SEED), result.reshape(1,-1), delimiter=",")
