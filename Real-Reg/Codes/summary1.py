import os
import itertools
import numpy as np
if os.path.isdir("./Results/summary")==False: os.makedirs("./Results/summary")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--j', type=int)
args = parser.parse_args()
DATA = args.data
J = args.j

SEED  = 10
LAMs  = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [100]

def relevant_ratio1(tmp, thr=0.01):
    s = np.abs(tmp)
    if s.max() <= 0: return 0.0
    return np.mean(s >= thr*s.max())
def relevant_ratio2(tmp, thr=0.01):
    tmp = np.sqrt(np.maximum(tmp**2-.1**6, 0))
    s = np.abs(tmp)
    if s.max() <= 0: return 0.0
    return np.mean(s >= thr*s.max())

def func(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    if METHs[J][0].split("-")[0]=="1MFBridge":
        res3 = np.zeros((len(METHs[J]),len(SIZEs),SEED))
        res4 = np.zeros((len(METHs[J]),len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        # print(si,se)
        val = np.zeros((len(METHs[J]),len(LAMs)))
        for me,la in itertools.product(range(len(METHs[J])),range(len(LAMs))):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[J][me],DATA,SIZEs[si],LAMs[la],se), delimiter=",")
            val[me,la] = np.min(tmp[:,-2])
        B = np.unravel_index(np.argmin(val),val.shape)
        tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[J][B[0]],DATA,SIZEs[si],LAMs[B[1]],se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-2]),-1]
        print(si,se,METHs[J][B[0]],LAMs[B[1]],np.argmin(tmp[:,-2]),np.min(tmp[:,-2]),res1[si,se])
        if J>=7 and J<=9: res2[si,se] = relevant_ratio1(tmp[np.argmin(tmp[:,-2]),3:-2])
        if J>=0 and J<=6: res2[si,se] = relevant_ratio2(tmp[np.argmin(tmp[:,-2]),3:-2])
        if METHs[J][0].split("-")[0]=="1MFBridge":
            for me in range(len(METHs[J])):
                C = np.argmin(val[me])
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[J][me],DATA,SIZEs[si],LAMs[C],se), delimiter=",")
                res3[me,si,se] = tmp[np.argmin(tmp[:,-2]),-1]
                if J>=7 and J<=9: res4[me,si,se] = relevant_ratio1(tmp[np.argmin(tmp[:,-2]),3:-2])
                if J>=0 and J<=6: res4[me,si,se] = relevant_ratio2(tmp[np.argmin(tmp[:,-2]),3:-2])
    np.savetxt("Results/summary/%s-%s-RMSE.csv" %(METHs[J][0].split("-")[0],DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-RFSR.csv" %(METHs[J][0].split("-")[0],DATA),res2,delimiter=",")
    if METHs[J][0].split("-")[0]=="1MFBridge":
        for me in range(len(METHs[J])):
            np.savetxt("Results/summary/%s-%s-RMSE.csv" %(METHs[J][me].split("/")[0],DATA),res3[me],delimiter=",")
            np.savetxt("Results/summary/%s-%s-RFSR.csv" %(METHs[J][me].split("/")[0],DATA),res4[me],delimiter=",")
#
METHs = [["1MFBridge-0.1250000000/relu-4-20","1MFBridge-0.2500000000/relu-4-20","1MFBridge-0.5000000000/relu-4-20","1MFBridge-1.0000000000/relu-4-20","1MFBridge-2.0000000000/relu-4-20"],
         ["1Bridge-0.1250000000","1Bridge-0.2500000000","1Bridge-0.5000000000","1Bridge-1.0000000000","1Bridge-2.0000000000"],
         ["1SpAM-0.1250000000/relu-4","1SpAM-0.2500000000/relu-4","1SpAM-0.5000000000/relu-4","1SpAM-1.0000000000/relu-4","1SpAM-2.0000000000/relu-4"],
         ["1MFElim-0.1250000000/relu-4-20","1MFElim-0.2500000000/relu-4-20","1MFElim-0.5000000000/relu-4-20","1MFElim-1.0000000000/relu-4-20","1MFElim-2.0000000000/relu-4-20"],
         ["1MFLoss-0.1250000000/relu-4-20","1MFLoss-0.2500000000/relu-4-20","1MFLoss-0.5000000000/relu-4-20","1MFLoss-1.0000000000/relu-4-20","1MFLoss-2.0000000000/relu-4-20"],
         ["1MFRisk-0.1250000000/relu-4-20","1MFRisk-0.2500000000/relu-4-20","1MFRisk-0.5000000000/relu-4-20","1MFRisk-1.0000000000/relu-4-20","1MFRisk-2.0000000000/relu-4-20"],
         ["1MFGrad-0.1250000000/relu-4-20","1MFGrad-0.2500000000/relu-4-20","1MFGrad-0.5000000000/relu-4-20","1MFGrad-1.0000000000/relu-4-20","1MFGrad-2.0000000000/relu-4-20"],
         ["1SPINN-0.0000000000/relu-4","1SPINN-0.2500000000/relu-4","1SPINN-0.5000000000/relu-4","1SPINN-0.7500000000/relu-4","1SPINN-1.0000000000/relu-4"],
         ["1STG-0.1250000000/relu-4","1STG-0.2500000000/relu-4","1STG-0.5000000000/relu-4","1STG-1.0000000000/relu-4","1STG-2.0000000000/relu-4"],
         ["1LassoNet-0.1000000000/relu-4","1LassoNet-0.3162277660/relu-4","1LassoNet-1.0000000000/relu-4","1LassoNet-3.1622776602/relu-4","1LassoNet-10.0000000000/relu-4"]]
if os.path.exists("Results/summary/%s-%s-RMSE.csv" %(METHs[J][0].split("-")[0],DATA))==False:
    func(J)