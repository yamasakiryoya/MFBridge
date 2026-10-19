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
        tmp = np.loadtxt("./Results/%s/%s/dim-3/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",DATA,SIZEs[si],0,se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        res2[si,se] = tmp[np.argmin(tmp[:,-4]),-1]
    np.savetxt("Results/summary/%s-%s-CMCE.csv" %("Oracle",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMCR.csv"%("Oracle",DATA),res2,delimiter=",")
if os.path.exists("Results/summary/%s-%s-CMCR.csv"%("Oracle",DATA))==False:
    funcOracle()


def funcERM():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    res3 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",DATA,SIZEs[si],0,se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        res2[si,se] = tmp[np.argmin(tmp[:,-4]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),4:-4])[::-1][:3]))==False: res3[si,se] += 1
    np.savetxt("Results/summary/%s-%s-CMCE.csv" %("ERM",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMCR.csv"%("ERM",DATA),res2,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMFS.csv"%("ERM",DATA),res3,delimiter=",")
if os.path.exists("Results/summary/%s-%s-CMFS.csv"%("ERM",DATA))==False:
    funcERM()


def funcRF():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    res3 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/1-seed-%d.csv"%("1RandomForest",DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        res2[si,se] = tmp[np.argmin(tmp[:,-4]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),2:-4])[::-1][:3]))==False: res3[si,se] += 1
    np.savetxt("Results/summary/%s-%s-CMCE.csv" %("1RandomForest",DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMCR.csv" %("1RandomForest",DATA),res2,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMFS.csv"%("1RandomForest",DATA),res3,delimiter=",")
if os.path.exists("Results/summary/%s-%s-CMFS.csv"%("1RandomForest",DATA))==False:
    funcRF()


def funcTS():
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        tmp2 = np.loadtxt("./Results/2MFBridge/relu-4-20/%s/dim-10/size-%d/2-seed-%d.csv"%(DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-4]),-2]
        res2[si,se] = tmp2[np.argmin(tmp2[:,-4]),-1]
    np.savetxt("Results/summary/2MFBridge-%s-CMCE.csv"%DATA,res1,delimiter=",")
    np.savetxt("Results/summary/2MFBridge-%s-CMCR.csv"%DATA,res2,delimiter=",")
if os.path.exists("Results/summary/2MFBridge-%s-CMCR.csv"%DATA)==False:
    funcTS()


def funcTS(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    res3 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        tmp1 = np.loadtxt("./Results/1%s-%.10f/relu-4-20/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(J,1,DATA,SIZEs[si],0,se), delimiter=",")
        tmp2 = np.loadtxt("./Results/2%s0/relu-4-20/%s/dim-10/size-%d/2-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-4]),-2]
        res2[si,se] = tmp2[np.argmin(tmp2[:,-4]),-1]
        tmp3 = tmp1[np.argmin(tmp1[:,-4]),4:-4]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp3)[::-1][:3]))==False: res3[si,se] += 1
    np.savetxt("Results/summary/2%s0-%s-CMCE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/2%s0-%s-CMCR.csv" %(J,DATA),res2,delimiter=",")
    np.savetxt("Results/summary/2%s0-%s-CMFS.csv"%(J,DATA),res3,delimiter=",")
for J in ["MFBridge","MFRisk"]:
    if os.path.exists("Results/summary/2%s0-%s-CMFS.csv"%(J,DATA))==False:
        funcTS(J)

def funcTS(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    res3 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        tmp1 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/1-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        tmp2 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/2-seed-%d.csv"%(J,DATA,SIZEs[si],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-4]),-2]
        res2[si,se] = tmp2[np.argmin(tmp2[:,-4]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp1)[::-1][:3]))==False: res3[si,se] += 1
    np.savetxt("Results/summary/%s-%s-CMCE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMCR.csv" %(J,DATA),res2,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMFS.csv"%(J,DATA),res3,delimiter=",")
for J in ["2Filter-f_classif","2Filter-mutual_info_classif","2Filter-hsic"]:
    if os.path.exists("Results/summary/%s-%s-CMFS.csv"%(J,DATA))==False:
        funcTS(J)


def func(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    res3 = np.zeros((len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        val = np.zeros(len(LAMs3))
        for la in range(len(LAMs3)):
            tmp = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/lam-%.10f/2-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
            val[la] = np.min(tmp[:,-4])
        B = np.argmin(val)
        tmp1 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[B],se), delimiter=",")
        tmp2 = np.loadtxt("./Results/%s/relu-4-20/%s/dim-10/size-%d/lam-%.10f/2-seed-%d.csv"%(J,DATA,SIZEs[si],LAMs3[B],se), delimiter=",")
        res1[si,se] = tmp2[np.argmin(tmp2[:,-4]),-2]
        res2[si,se] = tmp2[np.argmin(tmp2[:,-4]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp1)[::-1][:3]))==False: res3[si,se] += 1
    np.savetxt("Results/summary/%s-%s-CMCE.csv" %(J,DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMCR.csv" %(J,DATA),res2,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMFS.csv"%(J,DATA),res3,delimiter=",")
for J in ["2Filter-chsic","2Filter-hscic"]:
    if os.path.exists("Results/summary/%s-%s-CMFS.csv"%(J,DATA))==False:
        func(J)
