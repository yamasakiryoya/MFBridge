#nohup python bash-1.py &
#cd ~/Code/NLFS/Bin/Artificial2
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools
from multiprocessing import Pool

ACT,LAY,CUDA = "relu",4,-1
TYPEs = ["non-additive-sq2"]
SIZEs = [25,50,100,200,400]

def task():
    # ERM
    for TYPE,DIM,SIZE,SEED in itertools.product(TYPEs,[3,5,10,20],SIZEs,range(20)):
        subprocess.call("python LBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam 0. --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True)
        subprocess.call("python ABridge1.py --type %s --dim %d --size %d --alp 1.0 --lam 0. --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True)
        subprocess.call("python NGLasso1.py --type %s --dim %d --size %d --alp 0.0 --lam 0. --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True)
        subprocess.call("python NBridge1.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,1,0,ACT,LAY,SEED,CUDA), shell=True)
    # Lasso
    for TYPE,DIM,SIZE,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[10.**(i/4.-5.) for i in range(33)],range(20)):
        subprocess.call("python LBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA), shell=True)
    # SpAM
    for TYPE,DIM,SIZE,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[10.**(i/4.-5.) for i in range(33)],range(20)):
        subprocess.call("python ABridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA), shell=True)
    # random forest
    for TYPE,DIM,SIZE,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,range(20)):
        os.system("python RandomForest1.py --type %s --dim %d --size %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,SEED,CUDA))
    # NNSGLasso
    for TYPE,DIM,SIZE,ALP,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[0.0,0.25,0.5,0.75,1.0],LAM in [10.**(i/4.-5.) for i in range(33)],range(20)):
        subprocess.call("python NGLasso1.py --type %s --dim %d --size %d --alp %f --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
    # MFBridge
    for TYPE,DIM,SIZE,ALP,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[0.125,0.25,0.5,1.0,2.0],LAM in [10.**(i/4.-5.) for i in range(33)],range(20)):
        subprocess.call("python NBridge1.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
    # NNSGLasso with weight-decay
    for TYPE,DIM,SIZE,ALP,LAM1,LAM2,SEED in itertools.product(TYPEs,[10],SIZEs,[0.0,.25,0.5,0.75,1.0],[10.**(i/4.-5.) for i in range(33)],[10.**i for i in [-5,-4,-3,-2,-1,0,1,2,3]],range(20)):
        subprocess.call("python NGLasso12.py --type %s --dim %d --size %d --alp %f --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM1,LAM2,ACT,LAY,SEED,CUDA), shell=True)
    # (SCAD, MCP, Log-Sum)
    for TYPE,DIM,SIZE,SEED in itertools.product(TYPEs,[10],SIZEs,range(20)):
        for ALP,LAM in itertools.product([1.,2.,3.,4.,5.],[10.**(i/4.-5.) for i in range(33)]):
            subprocess.call("python NSCAD1.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
        for ALP,LAM in itertools.product([1.,2.,3.,4.,5.],[10.**(i/4.-5.) for i in range(33)]):
            subprocess.call("python NMCP1.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
        for ALP,LAM in itertools.product([.1**1,.1**2,.1**3,.1**4,.1**5],[10.**(i/4.-5.) for i in range(33)]):
            subprocess.call("python NLogSum1.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
        for ALP,LAM in itertools.product([.25,.5,1.,2.,4.],[10.**(i/4.-5.) for i in range(33)]):
            subprocess.call("python NAdaLasso1.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)

if __name__ == '__main__':
    task()
