import os
import itertools
import numpy as np
if os.path.isdir("./Results/summary")==False: os.makedirs("./Results/summary")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
args = parser.parse_args()
DATA = args.data

SEED  = 100
LAMs2 = [10.**(i/8.-6.) for i in range(81)]
LAMs3 = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [25,50,100,200,400]

def funcOracle():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/%s/%s/dim-3/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",DATA,SIZEs[si],0,se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %("Oracle",DATA),res1,delimiter=",")
if os.path.exists("Results/summary/%s-%s-CMCR.csv"%("Oracle",DATA))==False:
    funcOracle()

def funcERM():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",DATA,SIZEs[si],0,se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-2]),3:-2])[::-1][:3]))==False: res2[si,se] += 1
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %("ERM",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv"%("ERM",DATA),res2,delimiter=",")
if os.path.exists("Results/summary/%s-%s-RMFS.csv"%("ERM",DATA))==False:
    funcERM()

def funcRF():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/1-seed-%d.csv"%("1RandomForest",DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-2]),1:-2])[::-1][:3]))==False: res2[si,se] += 1
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %("1RandomForest",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv"%("1RandomForest",DATA),res2,delimiter=",")
if os.path.exists("Results/summary/%s-%s-RMFS.csv"%("1RandomForest",DATA))==False:
    funcRF()


def funcTS():
    res = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp = np.loadtxt("./Results/2MFBridge/relu-4-20/%s/dim-10/size-%d/2-seed-%d.csv"%(DATA,SIZEs[si],se), delimiter=",")
        res[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
    np.savetxt("Results/summary/2MFBridge-%s-RMSE.csv" %DATA,res,delimiter=",")
if os.path.exists("Results/summary/2MFBridge-%s-CMCR.csv"%DATA)==False:
    funcTS()

def funcTS(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp1 = np.loadtxt("./Results/1%s-%.10f/relu-4-20/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(J,1,DATA,SIZEs[si],0,se), delimiter=",")
        tmp2 = np.loadtxt("./Results/2%s0/relu-4-20/%s/dim-10/size-%d/2-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-2]),-1]
        tmp3 = tmp1[np.argmin(tmp1[:,-2]),3:-2]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp3)[::-1][:3]))==False: res2[si,se] += 1
    np.savetxt("Results/summary/2%s0-%s-RMSE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/2%s0-%s-RMFS.csv"%(J,DATA),res2,delimiter=",")
for J in ["MFBridge","MFRisk"]:
    if os.path.exists("Results/summary/2%s0-%s-RMFS.csv"%(J,DATA))==False:
        funcTS(J)

def funcTS(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        tmp1 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/1-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        tmp2 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/2-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-2]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp1)[::-1][:3]))==False: res2[si,se] += 1
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv"%(J,DATA),res2,delimiter=",")
for J in ["2Filter-f_regression","2Filter-mutual_info_regression","2Filter-hsic"]:
    if os.path.exists("Results/summary/%s-%s-RMFS.csv"%(J,DATA))==False:
        funcTS(J)


def func(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        val = np.zeros(len(LAMs3))
        for la in range(len(LAMs3)):
            tmp = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/lam-%.10f/2-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
            val[la] = np.min(tmp[:,-2])
        B = np.argmin(val)
        tmp1 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[B],se), delimiter=",")
        tmp2 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/lam-%.10f/2-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[B],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-2]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp1)[::-1][:3]))==False: res2[si,se] += 1
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RMFS.csv"%(J,DATA),res2,delimiter=",")
#
for J in ["2Filter-chsic","2Filter-hscic"]:
    if os.path.exists("Results/summary/%s-%s-RMFS.csv"%(J,DATA))==False:
        func(J)


