#import
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import argparse
import numpy as np
import numpy.random as rd
import scipy as sp
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
parser.add_argument('--data', type=str)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()
#

DATA = args.data
SIZE = args.size
SEED = args.seed
rd.seed(SEED)
torch.manual_seed(SEED)
if args.cuda >= 0: device = torch.device("cuda:%d" % args.cuda)
else: device = torch.device("cpu")

folder = "./Results/RandomForest/%s/size-%d"%(DATA,SIZE)
if os.path.isdir("%s/"%folder)==False: os.makedirs("%s/"%folder)
print("%s/seed-%d.csv"%(folder,SEED))
if os.path.isfile("%s/1-seed-%d.csv"%(folder,SEED))==False:
    #dataset
    XYful = np.genfromtxt("../Datasets/%s.csv"%DATA, delimiter=",", skip_header=1)
    XYful = rd.permutation(XYful)
    XYtra, XYres = train_test_split(XYful, train_size=SIZE, random_state=SEED) #train SIZE
    XYval, XYtes = train_test_split(XYres, train_size=100,  random_state=SEED) #valid 100, test rest
    Xtra, Ytra = XYtra[:,:-1], XYtra[:,-1]
    Xval, Yval = XYval[:,:-1], XYval[:,-1]
    Xtes, Ytes = XYtes[:,:-1], XYtes[:,-1]
    #
    if DATA!="naticusdroid+android+permissions":
        Xmea = np.mean(XYful[:,:-1], axis=0); Xstd = np.std(XYful[:,:-1], axis=0)
        for j in range(Xmea.shape[0]):
            if Xstd[j]!=0: Xtra = (Xtra-Xmea[j])/Xstd[j]; Xval = (Xval-Xmea[j])/Xstd[j]; Xtes = (Xtes-Xmea[j])/Xstd[j]
            else: Xtra = (Xtra-Xmea[j]); Xval = (Xval-Xmea[j]); Xtes = (Xtes-Xmea[j])
    #
    DIM = len(XYful[0,:])-1
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
    print(tmp)
    tmp1 = -np.log(.1**10+tmp); tmp2 = np.array(model.predict(Xtra), dtype=np.float32)
    result[0]    = np.mean([tmp1[i,int(Ytra[i])] for i in range(len(Ytra))])
    result[1]    = 1.-np.count_nonzero(np.equal(tmp2,np.array(Ytra, dtype=np.float32)))/len(Ytra)
    result[2:-4] = model.feature_importances_
    tmp = model.predict_proba(Xval)
    if tmp.shape[1]!=CLA: tmp = np.zeros((tmp.shape[0],CLA)); tmp[:,int(sp.stats.mode(XYful[:,-1])[0])] = 1
    print(tmp)
    tmp1 = -np.log(.1**10+tmp); tmp2 = np.array(model.predict(Xval), dtype=np.float32)
    result[-4]   = np.mean([tmp1[i,int(Yval[i])] for i in range(len(Yval))])
    result[-3]   = 1.-np.count_nonzero(np.equal(tmp2,np.array(Yval, dtype=np.float32)))/len(Yval)
    tmp = model.predict_proba(Xtes)
    if tmp.shape[1]!=CLA: tmp = np.zeros((tmp.shape[0],CLA)); tmp[:,int(sp.stats.mode(XYful[:,-1])[0])] = 1
    print(tmp)
    tmp1 = -np.log(.1**10+tmp); tmp2 = np.array(model.predict(Xtes), dtype=np.float32)
    result[-2]   = np.mean([tmp1[i,int(Ytes[i])] for i in range(len(Ytes))])
    result[-1]   = 1.-np.count_nonzero(np.equal(tmp2,np.array(Ytes, dtype=np.float32)))/len(Ytes)
    print('result : ', result)
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result.reshape(1,-1), delimiter=",")
