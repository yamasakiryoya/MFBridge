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

folder = "./Results/1Const/%s/dim-%d/size-%d"%(TYPE,DIM,SIZE)
os.makedirs("%s/"%folder, exist_ok=True)
print("%s/1-seed-%d.csv"%(folder,SEED))
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
    Ytra = XYtra[:,-1]
    Yval = XYval[:,-1]
    Ytes = XYtes[:,-1]

    #optimal constant prediction, obtained analytically rather than by
    #iterative training: for the squared loss, a constant c gives
    #  L(c) = (1/n) sum_i (y_i - c)^2,
    #whose stationary point is c* = mean(y), the training-set mean; the
    #corresponding training loss is then the training-set variance.
    #This c* is the L_infty reference used in Section 2.5.
    c_star = float(np.mean(Ytra))

    #squared error of the constant prediction c* on a label vector
    def evaluate(Y):
        return float(np.mean((Y - c_star)**2))

    tra_mse = evaluate(Ytra)
    val_mse = evaluate(Yval)
    tes_mse = evaluate(Ytes)

    print('optimal constant : %.6f  (training-set mean)'%c_star)
    print('train : MSE %.6f'%tra_mse)
    print('valid : MSE %.6f'%val_mse)
    print('test  : MSE %.6f'%tes_mse)

    #[c*, tra_mse, val_mse, tes_mse]
    result = np.array([c_star, tra_mse, val_mse, tes_mse])
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result.reshape(1,-1), delimiter=",")