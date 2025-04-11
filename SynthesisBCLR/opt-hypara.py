#python opt-hypara.py
import os
import itertools
import numpy as np

LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]


def func(TYPEs,METHs,DIMs,SIZEs):
    for METH,TYPE,DIM,SIZE in itertools.product(METHs,TYPEs,DIMs,SIZEs):
        # load results
        result1 = np.zeros((len(LAMs2),20))
        result2 = np.zeros((len(LAMs2),20))
        for se,la in itertools.product(range(20),range(len(LAMs2))):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,TYPE,DIM,SIZE,LAMs2[la],se), delimiter=",")
            result1[la,se] = np.min(tmp[:,-4])
            result2[la,se] = np.argmin(tmp[:,-4])
        #set hypara
        print(METH,TYPE,DIM,SIZE,np.mean(result2[np.argmin(np.mean(result1, axis=1)),:]),LAMs2[np.argmin(np.mean(result1, axis=1))])

#
TYPEs = ["non-additive-sq2"]
#TYPEs = ["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10"]
#METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4",
#         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
DIMs  = [10]
func(TYPEs,METHs,DIMs,SIZEs)
