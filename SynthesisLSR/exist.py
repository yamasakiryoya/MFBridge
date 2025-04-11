#python exist-1.py -> exist-1.txt
import os
import itertools
import numpy as np

TYPEs = ["non-additive-sq2"]
#TYPEs = ["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "NBridge1-0.1250000000/relu-4-5","NBridge1-0.2500000000/relu-4-5","NBridge1-0.5000000000/relu-4-5","NBridge1-1.0000000000/relu-4-5","NBridge1-2.0000000000/relu-4-5",
         "NBridge1-0.1250000000/relu-4-20","NBridge1-0.2500000000/relu-4-20","NBridge1-0.5000000000/relu-4-20","NBridge1-1.0000000000/relu-4-20","NBridge1-2.0000000000/relu-4-20"]
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
N = 20

#"""
print(1)
for METH,TYPE,DIM,SIZE,l in itertools.product(METHs,TYPEs,[5,20],[25,50,100,200,400],range(len(LAMs2))):
  LAM = LAMs2[l]
  for SEED in range(N):
    #if not os.path.isfile("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,TYPE,DIM,SIZE,LAM,SEED)):
    print(TYPE,DIM,SIZE,LAM,SEED,METH)
    tmp = np.mean(np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,TYPE,DIM,SIZE,LAM,SEED), delimiter=","))
#"""
"""
print(2)
for METH,TYPE,DIM,SIZE in itertools.product(METHs,TYPEs,[10],[25,50,100,200,400]):
  result = np.zeros((len(LAMs2),N))
  for SEED,l in itertools.product(range(N),range(len(LAMs2))):
    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,TYPE,DIM,SIZE,LAMs2[l],SEED), delimiter=",")
    result[l,SEED] = np.min(tmp[:,-2])
  LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
  flag = 0
  for SEED,l in itertools.product(range(N),range(len(LAMs3))):
    if os.path.exists("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,TYPE,DIM,SIZE,LAMs3[l],SEED))==False: flag = 1
  if flag==1: print(METH,TYPE,DIM,SIZE)
"""