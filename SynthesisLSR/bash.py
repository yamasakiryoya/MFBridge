#nohup python bash.py &
import os
import itertools
from multiprocessing import Pool

ACT,LAY,CUDA = "relu",4,-1
TYPEs = ["non-additive-sq2"]
SIZEs = [25,50,100,200,400]

def task(T):
    #"""(default; varying DIM)
    for TYPE,DIM,SIZE,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[10.**(i/4.-5.) for i in [T]],range(20)):
        os.system("python LBridge.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
    #"""
    #"""(default; varying DIM)
    for TYPE,DIM,SIZE,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[10.**(i/4.-5.) for i in [T]],range(20)):
        os.system("python ABridge.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
    #"""
    #"""(default; varying DIM)
    for TYPE,DIM,SIZE,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,range(20)):
        os.system("python RandomForest.py --type %s --dim %d --size %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,SEED,CUDA))
    #"""
    #"""(default; varying DIM)
    for TYPE,DIM,SIZE,ALP,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[0.0,.25,0.5,0.75,1.0],[10.**(i/4.-5.) for i in [T]],range(20)):
        os.system("python NGLasso.py --type %s --dim %d --size %d --alp %f --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA))
    for TYPE,DIM,SIZE,ALP,LAM,SEED in itertools.product(TYPEs,[5,10,20],SIZEs,[0.125,0.25,0.5,1.0,2.0],[10.**(i/4.-5.) for i in [T]],range(20)):
        os.system("python NBridge.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA))
    #"""
    #"""(varying Mnum)
    for TYPE,DIM,SIZE,ALP,LAM,SEED in itertools.product(TYPEs,[10],SIZEs,[0.125,0.25,0.5,1.0,2.0],[10.**(i/4.-5.) for i in [T]],range(20)):
        os.system("python NBridge.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA))
        os.system("python NBridge.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA))
    #"""
    #"""(NNSGLasso weight-decay)
    for TYPE,DIM,SIZE,ALP,LAM1,SEED,LAM2 in itertools.product(TYPEs,[10],SIZEs,[0.0,.25,0.5,0.75,1.0],[10.**(i/4.-5.) for i in [T]],[10.**i for i in [-5,-4,-3,-2,-1,0,1]]):
        #os.system("python NGLasso2.py --type %s --dim %d --size %d --alp %f --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM1,LAM2,ACT,LAY,SEED,CUDA))
    #"""

if __name__ == '__main__':
    #"""(RandomForest)
    task(0)
    #"""
    #"""(others)
    p = Pool(33)
    p.map(task, range(33))
    #"""