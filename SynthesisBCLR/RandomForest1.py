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
from sklearn.ensemble import RandomForestClassifier
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
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.zeros((SIZE+10100,DIM+1))
    for i in range(SIZE+10100):
        for j in range(DIM): XYful[i,j]=rd.normal(0,1)
        if TYPE=="non-additive-sq1":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-6.))) else 0
        elif TYPE=="non-additive-sq2":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-5.))) else 0
        # if   TYPE=="linear":            XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2]))) else 0
        # elif TYPE=="additive-sq":       XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0])+np.square(2.*XYful[i,1])+np.square(3.*XYful[i,2])-10.))) else 0
        # elif TYPE=="non-additive-sq1":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-4.))) else 0
        # elif TYPE=="non-additive-sq2":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1])+np.square(3.*XYful[i,2])-10.))) else 0
        # elif TYPE=="non-additive-sq3":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0])+np.square(2.*XYful[i,1]+3.*XYful[i,2])-7.))) else 0
        # elif TYPE=="non-additive-sq4":  XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-6.))) else 0
        # elif TYPE=="additive-sin":      XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0])+np.sin(2.*XYful[i,1])+np.sin(3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin1": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin2": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0]+2.*XYful[i,1])+np.sin(3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin3": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-10.*(np.sin(XYful[i,0])+np.sin(2.*XYful[i,1]+3.*XYful[i,2])))) else 0
        # elif TYPE=="non-additive-sin4": XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.sin(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])))) else 0
    #print(np.sum(XYful[:,-1])/len(XYful[:,-1]))
    XYful = rd.permutation(XYful)
    XYtra, XYres = train_test_split(XYful, train_size=SIZE, random_state=SEED) #train SIZE
    XYval, XYtes = train_test_split(XYres, train_size=100,  random_state=SEED) #valid 100, test rest
    Xtra, Ytra = XYtra[:,:-1], XYtra[:,-1]
    Xval, Yval = XYval[:,:-1], XYval[:,-1]
    Xtes, Ytes = XYtes[:,:-1], XYtes[:,-1]

    CLA = int(np.max(XYful[:,-1])+1)

    #learning
    model = RandomForestClassifier()#criterion='log_loss'
    model.fit(Xtra, Ytra)
    #forest = GridSearchCV(RandomForestClassifier(), {'n_estimators':[2,4,8,16,32,64],'max_depth':[2,4,8,16,32,64]}, verbose=2)
    #forest.fit(Xtra, Ytra)
    #model = forest.best_estimator_

    result = np.zeros(6+DIM)
    tmp = model.predict_proba(Xtra)
    if tmp.shape[1]!=CLA: tmp = np.zeros((tmp.shape[0],CLA)); tmp[:,int(sp.stats.mode(XYful[:,-1])[0])] = 1
    #print(tmp)
    tmp1 = -np.log(.1**10+tmp); tmp2 = np.array(model.predict(Xtra), dtype=np.float32)
    result[0]    = np.mean([tmp1[i,int(Ytra[i])] for i in range(len(Ytra))])
    result[1]    = 1.-np.count_nonzero(np.equal(tmp2,np.array(Ytra, dtype=np.float32)))/len(Ytra)
    result[2:-4] = model.feature_importances_
    tmp = model.predict_proba(Xval)
    if tmp.shape[1]!=CLA: tmp = np.zeros((tmp.shape[0],CLA)); tmp[:,int(sp.stats.mode(XYful[:,-1])[0])] = 1
    #print(tmp)
    tmp1 = -np.log(.1**10+tmp); tmp2 = np.array(model.predict(Xval), dtype=np.float32)
    result[-4]   = np.mean([tmp1[i,int(Yval[i])] for i in range(len(Yval))])
    result[-3]   = 1.-np.count_nonzero(np.equal(tmp2,np.array(Yval, dtype=np.float32)))/len(Yval)
    tmp = model.predict_proba(Xtes)
    if tmp.shape[1]!=CLA: tmp = np.zeros((tmp.shape[0],CLA)); tmp[:,int(sp.stats.mode(XYful[:,-1])[0])] = 1
    #print(tmp)
    tmp1 = -np.log(.1**10+tmp); tmp2 = np.array(model.predict(Xtes), dtype=np.float32)
    result[-2]   = np.mean([tmp1[i,int(Ytes[i])] for i in range(len(Ytes))])
    result[-1]   = 1.-np.count_nonzero(np.equal(tmp2,np.array(Ytes, dtype=np.float32)))/len(Ytes)
    print('result : ', result)
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result.reshape(1,-1), delimiter=",")
