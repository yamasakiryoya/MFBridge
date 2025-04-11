import os
import itertools
import numpy as np

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]

print(1)
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        res1 = 0
        for se in range(SEED):
            tmp1 = np.zeros((len(METHs),len(LAMs2)))
            for me in range(len(METHs)):
                for la in range(len(LAMs2)):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[me,la] = np.min(tmp[:,-2])
            idx = np.unravel_index(np.argmin(tmp1), tmp1.shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-2]),3:-4])[::-1][:3]))==False:
                res1 += 1
        print(METHs[0],DIM,SIZEs[si],res1)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
DIMs   = [10]
func(TYPEs,["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"   ],DIMs)
func(TYPEs,["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"],DIMs)
"""
func(TYPEs,["NBridge1-0.1250000000/relu-4-5" ,"NBridge1-0.2500000000/relu-4-5" ,"NBridge1-0.5000000000/relu-4-5" ,"NBridge1-1.0000000000/relu-4-5" ,"NBridge1-2.0000000000/relu-4-5" ],DIMs)
func(TYPEs,["NBridge1-0.1250000000/relu-4-20","NBridge1-0.2500000000/relu-4-20","NBridge1-0.5000000000/relu-4-20","NBridge1-1.0000000000/relu-4-20","NBridge1-2.0000000000/relu-4-20"],DIMs)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
func(TYPEs,["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"   ],DIMs)
func(TYPEs,["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"],DIMs)
DIMs  = [5,20]
func(TYPEs,METHs,DIMs)
"""

print(2)
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,me,si in itertools.product(DIMs,TYPEs,range(len(METHs)),range(len(SIZEs))):
        if METHs[me]!="RandomForest":
            res = 0
            for se in range(SEED):
                tmp1 = np.zeros(len(LAMs2))
                for la in range(len(LAMs2)):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[la] = np.min(tmp[:,-2])
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[np.argmin(tmp1)],se), delimiter=",")
                if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-2]),3:-4])[::-1][:3]))==False:
                    res += 1
            print(METHs[me],DIM,SIZEs[si],res)
        if METHs[me]=="RandomForest":
            res = 0
            for se in range(SEED):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],se), delimiter=",")
                if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[2:-4])[::-1][:3]))==False:
                    res += 1
            print(METHs[me],DIM,SIZEs[si],res)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "RandomForest"]
"""
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "NBridge1-0.1250000000/relu-4-5" ,"NBridge1-0.2500000000/relu-4-5" ,"NBridge1-0.5000000000/relu-4-5" ,"NBridge1-1.0000000000/relu-4-5" ,"NBridge1-2.0000000000/relu-4-5" ,
         "NBridge1-0.1250000000/relu-4-20","NBridge1-0.2500000000/relu-4-20","NBridge1-0.5000000000/relu-4-20","NBridge1-1.0000000000/relu-4-20","NBridge1-2.0000000000/relu-4-20",
         "RandomForest"]
"""
DIMs   = [10]
func(TYPEs,METHs,DIMs)
#
"""
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "RandomForest"]
DIMs  = [5,20]
func(TYPEs,METHs,DIMs)
"""