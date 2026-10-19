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
        XYful[i,-1] = 1 if rd.rand()<1./(1.+np.exp(-(np.square(XYful[i,0]+2.*XYful[i,1]+3.*XYful[i,2])-5.))) else 0
    XYful = rd.permutation(XYful)
    XYres, XYtes = train_test_split(XYful, test_size=10000, random_state=SEED) #test 10000
    XYtra, XYval = train_test_split(XYres, train_size=0.8, random_state=SEED) #train:val = 4:1
    Ytra = XYtra[:,-1]
    Yval = XYval[:,-1]
    Ytes = XYtes[:,-1]

    #optimal constant prediction, obtained analytically rather than by
    #iterative training: for the logistic loss, a constant logit c gives
    #  L(c) = -(1/n) sum_i {y_i log sigma(c) + (1-y_i) log(1-sigma(c))},
    #whose stationary point is sigma(c) = p, the training-set positive rate,
    #i.e. c* = log(p/(1-p)); the corresponding training loss is the binary
    #entropy of p. This c* is the L_infty reference used in Section 2.5.
    p = float(np.mean(Ytra))
    EPS = 1e-12
    if p <= 0. or p >= 1.:
        #degenerate case: the training split contains a single class, so the
        #optimal constant diverges; we clip it and predict that class outright
        p_clip = min(max(p, EPS), 1.-EPS)
    else:
        p_clip = p
    c_star = np.log(p_clip/(1.-p_clip))

    #cross-entropy of the constant logit c* on a label vector, and the
    #misclassification rate of the induced (constant) hard prediction
    def evaluate(Y):
        #sigma(c*) = p_clip for every sample, so the CE reduces to a closed form
        ce  = -(Y*np.log(p_clip) + (1.-Y)*np.log(1.-p_clip)).mean()
        #the hard prediction is the majority class of the training split
        pred = 1. if c_star > 0. else 0.
        mcr = float(np.mean(pred != Y))
        return ce, mcr

    tra_ce, tra_mcr = evaluate(Ytra)
    val_ce, val_mcr = evaluate(Yval)
    tes_ce, tes_mcr = evaluate(Ytes)

    print('optimal constant (logit) : %.6f  (training positive rate p = %.6f)'%(c_star, p))
    print('train : CE %.6f  MCR %.6f'%(tra_ce, tra_mcr))
    print('valid : CE %.6f  MCR %.6f'%(val_ce, val_mcr))
    print('test  : CE %.6f  MCR %.6f'%(tes_ce, tes_mcr))

    #[c*, p, tra_ce, tra_mcr, val_ce, val_mcr, tes_ce, tes_mcr]
    result = np.array([c_star, p, tra_ce, tra_mcr, val_ce, val_mcr, tes_ce, tes_mcr])
    np.savetxt("%s/1-seed-%d.csv"%(folder,SEED), result.reshape(1,-1), delimiter=",")