import os
import itertools
import numpy as np
if os.path.isdir("./Results/summary")==False: os.makedirs("./Results/summary")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
args = parser.parse_args()
DATA = args.data

SEED  = 10
LAMs2 = [10.**(i/8.-6.) for i in range(81)]
LAMs3 = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [100]

def relevant_ratio1(tmp, thr=0.01):
    s = np.abs(tmp)
    if np.max(s) <= 0: return 0.0
    return np.mean(s >= thr*np.max(s))

def funcERM():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",DATA,SIZEs[si],0,se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        res2[si,se] = relevant_ratio1(tmp[np.argmin(tmp[:,-2]),3:-2])
        print(si,se,np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res1[si,se])
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %("ERM",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv" %("ERM",DATA),res2,delimiter=",")
if os.path.exists("Results/summary/%s-%s-RMSE.csv" %("ERM",DATA))==False:
    print("2ERM")
    funcERM()

def funcRF():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/%s/%s/size-%d/1-seed-%d.csv"%("1RandomForest",DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        res2[si,se] = relevant_ratio1(tmp[np.argmin(tmp[:,-2]),1:-2])
        print(si,se,np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res1[si,se])
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %("1RandomForest",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv" %("1RandomForest",DATA),res2,delimiter=",")
if os.path.exists("Results/summary/%s-%s-RMSE.csv" %("1RandomForest",DATA))==False:
    print("1RandomForest")
    funcRF()

def funcTS():
    res = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/2MFBridge/relu-4-20/%s/size-%d/2-seed-%d.csv"%(DATA,SIZEs[si],se), delimiter=",")
        if tmp.ndim == 1: tmp = tmp.reshape(1, -1)
        res[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        # res2[si,se] = relevant_ratio1(tmp[np.argmin(tmp[:,-2]),3:-2])
        print(si,se,np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res[si,se])
    np.savetxt("Results/summary/2MFBridge-%s-RMSE.csv" %DATA,res,delimiter=",")
    # np.savetxt("Results/summary/2MFBridge-%s-RMFS.csv" %DATA,res2,delimiter=",")
if os.path.exists("Results/summary/2MFBridge-%s-RMSE.csv" %DATA)==False:
    print("2MFBridge")
    funcTS()


def funcTS(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/2%s0/relu-4-20/%s/size-%d/2-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        if tmp.ndim == 1: tmp = tmp.reshape(1, -1)
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        # res2[si,se] = relevant_ratio1(tmp[np.argmin(tmp[:,-2]),3:-2])
        print(si,se,np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res1[si,se])
    np.savetxt("Results/summary/2%s0-%s-RMSE.csv" %(J,DATA),res1,delimiter=",")
    # np.savetxt("Results/summary/2%s0-%s-RMFS.csv" %(J,DATA),res2,delimiter=",")
for J in ["MFBridge","MFRisk"]:
    if os.path.exists("Results/summary/2%s0-%s-RMSE.csv" %(J,DATA))==False:
        print(J)
        funcTS(J)

def funcTS(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/%s/relu-4-20/%s/size-%d/2-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        if tmp.ndim == 1: tmp = tmp.reshape(1, -1)
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        print(si,se,np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res1[si,se])
        tmp = np.loadtxt("./Results/%s/relu-4-20/%s/size-%d/1-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        res2[si,se] = relevant_ratio1(tmp)
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv" %(J,DATA),res2,delimiter=",")
for J in ["2Filter-f_regression","2Filter-mutual_info_regression","2Filter-hsic"]:
    if os.path.exists("Results/summary/%s-%s-RMSE.csv" %(J,DATA))==False:
        print(J)
        funcTS(J)

def func(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        val = np.zeros(len(LAMs3))
        for la in range(len(LAMs3)):
            tmp = np.loadtxt("./Results/%s/relu-4-20/%s/size-%d/lam-%.10f/2-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
            if tmp.ndim == 1: tmp = tmp.reshape(1, -1)
            val[la] = np.min(tmp[:,-2])
        B = np.argmin(val)
        tmp = np.loadtxt("./Results/%s/relu-4-20/%s/size-%d/lam-%.10f/2-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[B],se), delimiter=",")
        if tmp.ndim == 1: tmp = tmp.reshape(1, -1)
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        print(si,se,LAMs3[B],np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res1[si,se])
        tmp = np.loadtxt("./Results/%s/relu-4-20/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[B],se), delimiter=",")
        res2[si,se] = relevant_ratio1(tmp)
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv" %(J,DATA),res2,delimiter=",")
for J in ["2Filter-chsic","2Filter-hscic"]:
    if os.path.exists("Results/summary/%s-%s-RMSE.csv" %(J,DATA))==False:
        print(J)
        func(J)