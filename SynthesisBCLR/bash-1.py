#nohup python bash-1.py &
import os
import itertools
from multiprocessing import Pool

ACT,LAY,CUDA = "relu",4,-1
TYPEs = ["non-additive-sq2"]
#TYPEs = ["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
SIZEs = [25,50,100,200,400]

def task(T):
    """
    for DIM,SEED,TYPE,SIZE in itertools.product([5,10,20],range(20),TYPEs,SIZEs):
        os.system("python RandomForest.py --type %s --dim %d --size %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,SEED,CUDA))
    """
    #"""
    for DIM,SEED,TYPE,SIZE,LAM in itertools.product([10],range(20),TYPEs,SIZEs,[10.**(i/4.-5.) for i in [T]]):#[10.**(i/4.-5.) for i in [T]]
        os.system("python LBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python ABridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
    #"""
    """(default)
    for DIM,SEED,TYPE,SIZE,LAM in itertools.product([10],range(20),TYPEs,SIZEs,[10.**(i/4.-5.) for i in [T]]):#[10.**(i/4.-5.) for i in [T]]
        os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.25 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.5 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.75 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #
        os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.125 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.25 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.5 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 2.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
    """
    """(varying DIM)
    for DIM,SEED,TYPE,SIZE,LAM in itertools.product([5,20],range(20),TYPEs,SIZEs,[10.**(i/4.-5.) for i in [T]]):#[10.**(i/4.-5.) for i in [T]]
        os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.25 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.5 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 0.75 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #
        os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.125 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.25 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.5 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 2.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
    """
    """(varying Mnum)
    for DIM,SEED,TYPE,SIZE,LAM in itertools.product([10],range(20),TYPEs,SIZEs,[10.**(i/4.-5.) for i in [T]]):#[10.**(i/4.-5.) for i in [T]]
        os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.125 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.25 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.5 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 2.0 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #
        os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.125 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.25 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 0.5 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 1.0 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --type %s --dim %d --size %d --alp 2.0 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM,ACT,LAY,SEED,CUDA))
    """
    """(NNSGLasso weight-decay)
    for DIM,SEED,TYPE,SIZE,LAM1,LAM2 in itertools.product([10],range(20),TYPEs,SIZEs,[10.**(i/4.-5.) for i in [T]],[10.**i for i in [-5,-4,-3,-2,-1,0,1]]):#[10.**(i/4.-5.) for i in [T]]
        #os.system("python NGLasso2.py --type %s --dim %d --size %d --alp 0.0 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --type %s --dim %d --size %d --alp 0.25 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --type %s --dim %d --size %d --alp 0.5 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --type %s --dim %d --size %d --alp 0.75 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --type %s --dim %d --size %d --alp 1.0 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
    """

if __name__ == '__main__':
    """
    task(0)
    """
    """
    p = Pool(33)
    p.map(task, range(33))
    """
    for i in range(11):
        p = Pool(3)
        p.map(task, range(i*3,(i+1)*3))