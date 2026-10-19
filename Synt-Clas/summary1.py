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

SEED  = 100
LAMs1 = [i/8.-6. for i in range(81)]
LAMs2 = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [25,50,100,200,400]

def func(J):
    res1 = np.zeros((len(SIZEs),SEED))
    res2 = np.zeros((len(SIZEs),SEED))
    res3 = np.zeros((len(SIZEs),SEED))
    if METHs[J][0].split("-")[0]=="1MFBridge":
        res4 = np.zeros((len(METHs[J]),len(SIZEs),SEED))
        res5 = np.zeros((len(METHs[J]),len(SIZEs),SEED))
        res6 = np.zeros((len(METHs[J]),len(SIZEs),SEED))
    for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
        print(si,se)
        val = np.zeros((len(METHs[J]),len(LAMs2)))
        for me,la in itertools.product(range(len(METHs[J])),range(len(LAMs2))):
            tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[J][me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
            val[me,la] = np.min(tmp[:,-4])
        B = np.unravel_index(np.argmin(val),val.shape)
        tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[J][B[0]],DATA,SIZEs[si],LAMs2[B[1]],se), delimiter=",")
        res1[si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        res2[si,se] = tmp[np.argmin(tmp[:,-4]),-1]
        if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),4:-4])[::-1][:3]))==False: res3[si,se] += 1
        if METHs[J][0].split("-")[0]=="1MFBridge":
            for me in range(len(METHs[J])):
                C = np.argmin(val[me])
                tmp = np.loadtxt("./Results/%s/%s/dim-10/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[J][me],DATA,SIZEs[si],LAMs2[C],se), delimiter=",")
                res4[me,si,se] = tmp[np.argmin(tmp[:,-4]),-2]
                res5[me,si,se] = tmp[np.argmin(tmp[:,-4]),-1]
                if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),4:-4])[::-1][:3]))==False: res6[me,si,se] += 1
    np.savetxt("Results/summary/%s-%s-CMCE.csv" %(METHs[J][0].split("-")[0],DATA),res1,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMCR.csv"%(METHs[J][0].split("-")[0],DATA),res2,delimiter=",")
    np.savetxt("Results/summary/%s-%s-CMFS.csv"%(METHs[J][0].split("-")[0],DATA),res3,delimiter=",")
    if METHs[J][0].split("-")[0]=="1MFBridge":
        for me in range(len(METHs[J])):
            np.savetxt("Results/summary/%s-%s-CMCE.csv" %(METHs[J][me].split("/")[0],DATA),res4[me],delimiter=",")
            np.savetxt("Results/summary/%s-%s-CMCR.csv"%(METHs[J][me].split("/")[0],DATA),res5[me],delimiter=",")
            np.savetxt("Results/summary/%s-%s-CMFS.csv"%(METHs[J][me].split("/")[0],DATA),res6[me],delimiter=",")
#
METHs = [["1Bridge-0.1250000000","1Bridge-0.2500000000","1Bridge-0.5000000000","1Bridge-1.0000000000","1Bridge-2.0000000000"],
         ["1SpAM-0.1250000000/relu-4","1SpAM-0.2500000000/relu-4","1SpAM-0.5000000000/relu-4","1SpAM-1.0000000000/relu-4","1SpAM-2.0000000000/relu-4"],
         ["1MFBridge-0.1250000000/relu-4-20","1MFBridge-0.2500000000/relu-4-20","1MFBridge-0.5000000000/relu-4-20","1MFBridge-1.0000000000/relu-4-20","1MFBridge-2.0000000000/relu-4-20"],
         ["1MFElim-0.1250000000/relu-4-20","1MFElim-0.2500000000/relu-4-20","1MFElim-0.5000000000/relu-4-20","1MFElim-1.0000000000/relu-4-20","1MFElim-2.0000000000/relu-4-20"],
         ["1MFLoss-0.1250000000/relu-4-20","1MFLoss-0.2500000000/relu-4-20","1MFLoss-0.5000000000/relu-4-20","1MFLoss-1.0000000000/relu-4-20","1MFLoss-2.0000000000/relu-4-20"],
         ["1MFRisk-0.1250000000/relu-4-20","1MFRisk-0.2500000000/relu-4-20","1MFRisk-0.5000000000/relu-4-20","1MFRisk-1.0000000000/relu-4-20","1MFRisk-2.0000000000/relu-4-20"],
         ["1MFGrad-0.1250000000/relu-4-20","1MFGrad-0.2500000000/relu-4-20","1MFGrad-0.5000000000/relu-4-20","1MFGrad-1.0000000000/relu-4-20","1MFGrad-2.0000000000/relu-4-20"],
         ["1SPINN-0.0000000000/relu-4","1SPINN-0.2500000000/relu-4","1SPINN-0.5000000000/relu-4","1SPINN-0.7500000000/relu-4","1SPINN-1.0000000000/relu-4"],
         ["1STG-0.1250000000/relu-4","1STG-0.2500000000/relu-4","1STG-0.5000000000/relu-4","1STG-1.0000000000/relu-4","1STG-2.0000000000/relu-4"],
         ["1LassoNet-0.1000000000/relu-4","1LassoNet-0.3162280000/relu-4","1LassoNet-1.0000000000/relu-4","1LassoNet-3.1622780000/relu-4","1LassoNet-10.0000000000/relu-4"]]
if os.path.exists("Results/summary/%s-%s-CMFS.csv"%(METHs[J][0].split("-")[0],DATA))==False:
    func(J)