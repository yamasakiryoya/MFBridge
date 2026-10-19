#import
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import argparse
import itertools
import numpy as np
import numpy.random as rd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

folder = "./Results/1RandomForest/%s/dim-%d/size-%d"%(TYPE,DIM,SIZE)
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
        XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-5.))) else 0
    XYful = rd.permutation(XYful)
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED) #train:val = 4:1
    Xtra, Ytra = XYtra[:,:-1], XYtra[:,-1]
    Xval, Yval = XYval[:,:-1], XYval[:,-1]
    Xtes, Ytes = XYtes[:,:-1], XYtes[:,-1]

    CLA = int(np.max(XYful[:,-1])+1)
    #majority class, used as a fallback if predict_proba does not cover all CLA classes
    vals, counts = np.unique(XYful[:,-1], return_counts=True)
    MAJORITY_CLASS = int(vals[np.argmax(counts)])

    def logloss_and_err(model, X, Y):
        prob = model.predict_proba(X)
        if prob.shape[1]!=CLA:
            prob = np.zeros((prob.shape[0],CLA))
            prob[:,MAJORITY_CLASS] = 1
        nll  = -np.log(.1**10+prob)
        pred = np.array(model.predict(X), dtype=np.float32)
        ll   = np.mean([nll[i,int(Y[i])] for i in range(len(Y))])
        err  = 1.-np.count_nonzero(np.equal(pred,np.array(Y, dtype=np.float32)))/len(Y)
        return ll, err

    #model selection: grid search over (n_estimators, max_depth), selected by validation log-loss
    N_ESTIMATORS = [100]#[2, 4, 8, 16, 32, 64, 128]
    MAX_DEPTH    = [2, 4, 8, 16, 32, 64, 128]
    grid = list(itertools.product(N_ESTIMATORS, MAX_DEPTH))

    #one row per hyperparameter combination:
    #[tra_logloss, tra_err, importances(DIM), val_logloss, val_err, tes_logloss, tes_err]
    #(mirrors the "one row per epoch, argmin over column -4" convention used elsewhere in this codebase)
    result = np.zeros((len(grid), 6+DIM))
    best_idx, best_val = 0, np.inf
    for g, (n_estimators, max_depth) in enumerate(grid):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=SEED)
        model.fit(Xtra, Ytra)
        #
        result[g,0], result[g,1] = logloss_and_err(model, Xtra, Ytra)
        result[g,2:-4] = model.feature_importances_
        result[g,-4], result[g,-3] = logloss_and_err(model, Xval, Yval)
        result[g,-2], result[g,-1] = logloss_and_err(model, Xtes, Ytes)
        if result[g,-4] < best_val:
            best_idx, best_val = g, result[g,-4]

    print('best (n_estimators, max_depth) : ', grid[best_idx])
    print('result : ', result[best_idx])
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result, delimiter=",")
