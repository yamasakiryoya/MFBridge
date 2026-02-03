import os
import itertools
import numpy as np

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
LAMs3 = [10.**i for i in [-5,-4,-3,-2,-1,0,1,2,3]]
SIZEs = [25,50,100,200,400]

print(1)
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        res1 = 0; res2 = 0
        for se in range(SEED):
            tmp1 = np.zeros((len(METHs),len(LAMs2))); tmp2 = np.zeros((len(METHs),len(LAMs2)))
            for me in range(len(METHs)):
                for la in range(len(LAMs2)):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[me,la] = np.min(tmp[:,-4])
                    tmp2[me,la] = np.min(tmp[:,-3])
            idx = np.unravel_index(np.argmin(tmp1), tmp1.shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),4:-4])[::-1][:3]))==False:
                res1 += 1
            idx = np.unravel_index(np.argmin(tmp2), tmp2.shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-3]),4:-4])[::-1][:3]))==False:
                res2 += 1
        print(METHs[0],DIM,SIZEs[si],res1,res2)
#
TYPEs = ["non-additive-sq2"]
DIMs   = [5,10,20]
# func(TYPEs,["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"   ],DIMs)
# func(TYPEs,["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"],DIMs)

print(2)
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,me,si in itertools.product(DIMs,TYPEs,range(len(METHs)),range(len(SIZEs))):
        if METHs[me]!="RandomForest":
            res1 = 0; res2 = 0
            for se in range(SEED):
                tmp1 = np.zeros(len(LAMs2)); tmp2 = np.zeros(len(LAMs2))
                for la in range(len(LAMs2)):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[la] = np.min(tmp[:,-4])
                    tmp2[la] = np.min(tmp[:,-3])
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[np.argmin(tmp1)],se), delimiter=",")
                if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),4:-4])[::-1][:3]))==False:
                    res1 += 1
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[np.argmin(tmp2)],se), delimiter=",")
                if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-3]),4:-4])[::-1][:3]))==False:
                    res2 += 1
            print(METHs[me],DIM,SIZEs[si],res1,res2)
        if METHs[me]=="RandomForest":
            res = 0
            for se in range(SEED):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],se), delimiter=",")
                if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[2:-4])[::-1][:3]))==False:
                    res += 1
            print(METHs[me],DIM,SIZEs[si],res)
#
TYPEs = ["non-additive-sq2"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "LBridge1-1.0000000000","ABridge1-1.0000000000/relu-4","RandomForest"]
DIMs   = [5,10,20]
# func(TYPEs,METHs,DIMs)

print(3)
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        res1 = 0; res2 = 0
        for se in range(SEED):
            tmp1 = np.zeros((len(METHs),len(LAMs2),len(LAMs3))); tmp2 = np.zeros((len(METHs),len(LAMs2),len(LAMs3)))
            for me in range(len(METHs)):
                for la in range(len(LAMs2)):
                    for la2 in range(len(LAMs3)):
                        tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],LAMs3[la2],se), delimiter=",")
                        tmp1[me,la,la2] = np.min(tmp[:,-4])
                        tmp2[me,la,la2] = np.min(tmp[:,-3])
            idx = np.unravel_index(np.argmin(tmp1), tmp1.shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],LAMs3[idx[2]],se), delimiter=",")
            if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-4]),5:-4])[::-1][:3]))==False:
                res1 += 1
            idx = np.unravel_index(np.argmin(tmp2), tmp2.shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],LAMs3[idx[2]],se), delimiter=",")
            if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp[np.argmin(tmp[:,-3]),5:-4])[::-1][:3]))==False:
                res2 += 1
        print(METHs[0],DIM,SIZEs[si],res1,res2)
#
TYPEs = ["non-additive-sq2"]
DIMs   = [10]
func(TYPEs,["NGLasso12-0.0000000000/relu-4"   ,"NGLasso12-0.2500000000/relu-4"   ,"NGLasso12-0.5000000000/relu-4"   ,"NGLasso12-0.7500000000/relu-4"   ,"NGLasso12-1.0000000000/relu-4"   ],DIMs)
