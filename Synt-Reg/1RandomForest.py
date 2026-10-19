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
from sklearn.ensemble import RandomForestRegressor

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
        XYful[i,-1] = np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])/5+rd.normal(0,1)
    XYful = rd.permutation(XYful)
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED) #train:val = 4:1
    Xtra, Ytra = XYtra[:,:-1], XYtra[:,-1]
    Xval, Yval = XYval[:,:-1], XYval[:,-1]
    Xtes, Ytes = XYtes[:,:-1], XYtes[:,-1]

    #model selection: grid search over (n_estimators, max_depth), selected by validation MSE
    N_ESTIMATORS = [100]#[2, 4, 8, 16, 32, 64, 128]
    MAX_DEPTH    = [2, 4, 8, 16, 32, 64, 128]
    grid = list(itertools.product(N_ESTIMATORS, MAX_DEPTH))

    #one row per hyperparameter combination: [tra_mse, importances(DIM), val_mse, tes_mse]
    #(mirrors the "one row per epoch, argmin over column -2" convention used elsewhere in this codebase)
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