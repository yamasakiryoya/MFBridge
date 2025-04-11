#nohup python bash-2.py &
import os
import itertools
import numpy as np
from multiprocessing import Pool

ACT,LAY,CUDA = "relu",4,-1
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
LAMs3 = [-5,-4,-3,-2,-1,0,1]
LAMs4 = [10.**i for i in [-5,-4,-3,-2,-1,0,1]]
DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
SIZEs = [100]#[25,50,100,200,400]
METHs = ["NGLasso1-0.0000000000/relu-4","NBridge1-0.1250000000/relu-4-10"]
"""
METHs = [["NGLasso1-0.0000000000/relu-4","NBridge1-0.1250000000/relu-4-10"],
         ["NGLasso1-0.2500000000/relu-4","NBridge1-0.2500000000/relu-4-10"],
         ["NGLasso1-0.5000000000/relu-4","NBridge1-0.5000000000/relu-4-10"],
         ["NGLasso1-0.7500000000/relu-4","NBridge1-1.0000000000/relu-4-10"],
         ["NGLasso1-1.0000000000/relu-4","NBridge1-2.0000000000/relu-4-10"],
         ["NBridge1-0.1250000000/relu-4-5","NBridge1-0.1250000000/relu-4-20"],
         ["NBridge1-0.2500000000/relu-4-5","NBridge1-0.2500000000/relu-4-20"],
         ["NBridge1-0.5000000000/relu-4-5","NBridge1-0.5000000000/relu-4-20"],
         ["NBridge1-1.0000000000/relu-4-5","NBridge1-1.0000000000/relu-4-20"],
         ["NBridge1-2.0000000000/relu-4-5","NBridge1-2.0000000000/relu-4-20"]][0]
"""
def task(T):
    """(no regularizer)
    METH = "NGLasso1-0.000000000000/relu-4"
    for DATA,SIZE,SEED in itertools.product(DATAs,SIZEs,[T]):
        os.system("python NGLasso1.py --data %s --size %d --alp 0.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,0.,ACT,LAY,SEED,CUDA))
    """
    #"""(default; varying DIM; varying Mnum)
    for DATA,SIZE,METH in itertools.product(DATAs,SIZEs,METHs):
        result = np.zeros((len(LAMs2),20))
        for SEED,l in itertools.product(range(20),range(len(LAMs2))):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,DATA,SIZE,LAMs2[l],SEED), delimiter=",")
            result[l,SEED] = np.min(tmp[:,-2])
        #
        LAMs = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
        for SEED,LAM in itertools.product([T],LAMs):
            print(DATA,SIZE,METH,SEED,LAM)
            if   METH=="NGLasso1-0.0000000000/relu-4":    os.system("python NGLasso1.py --data %s --size %d --alp 0.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso1-0.2500000000/relu-4":    os.system("python NGLasso1.py --data %s --size %d --alp 0.25 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso1-0.5000000000/relu-4":    os.system("python NGLasso1.py --data %s --size %d --alp 0.5 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso1-0.7500000000/relu-4":    os.system("python NGLasso1.py --data %s --size %d --alp 0.75 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso1-1.0000000000/relu-4":    os.system("python NGLasso1.py --data %s --size %d --alp 1.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            #
            elif METH=="NBridge1-0.1250000000/relu-4-5":  os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-0.2500000000/relu-4-5":  os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-0.5000000000/relu-4-5":  os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-1.0000000000/relu-4-5":  os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-2.0000000000/relu-4-5":  os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            #
            elif METH=="NBridge1-0.1250000000/relu-4-10": os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-0.2500000000/relu-4-10": os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-0.5000000000/relu-4-10": os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-1.0000000000/relu-4-10": os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-2.0000000000/relu-4-10": os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            #
            elif METH=="NBridge1-0.1250000000/relu-4-20": os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-0.2500000000/relu-4-20": os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-0.5000000000/relu-4-20": os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-1.0000000000/relu-4-20": os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
            elif METH=="NBridge1-2.0000000000/relu-4-20": os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
    #
    #"""
    """(NNSGLasso weight-decay)
    for DATA,SIZE,METH in itertools.product(DATAs,SIZEs,METHs):
        result = np.zeros((len(LAMs2),len(LAMs4),20))
        for SEED,l,m in itertools.product(range(20),range(len(LAMs2)),range(len(LAMs4))):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f-%.10f/1-seed-%d.csv"%(METH,DATA,SIZE,LAMs2[l],LAMs4[m],SEED), delimiter=",")
            result[l,m,SEED] = np.min(tmp[:,-4])
        tmpA, tmpB, tmpC = 0, 0, 10.**8
        for l,m in itertools.product(range(len(LAMs2)),range(len(LAMs4))):
            if np.mean(result, axis=2)[l,m]<tmpC: tmpA=l; tmpB=m; tmpC=np.mean(result, axis=2)[l,m]
        #
        LAMsA = [10.**(i/16.+LAMs1[tmpA]) for i in [-3,-2,-1,0,1,2,3]]; LAMsB = [10.**LAMs3[tmpB]]
        for SEED,LAM1,LAM2 in itertools.product([T],LAMsA,LAMsB):
            print(DATA,SIZE,METH,SEED,LAM1,LAM2)
            if   METH=="NGLasso2-0.0000000000/relu-4": os.system("python NGLasso2.py --data %s --size %d --alp 0.0 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso2-0.2500000000/relu-4": os.system("python NGLasso2.py --data %s --size %d --alp 0.25 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso2-0.5000000000/relu-4": os.system("python NGLasso2.py --data %s --size %d --alp 0.5 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso2-0.7500000000/relu-4": os.system("python NGLasso2.py --data %s --size %d --alp 0.75 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
            elif METH=="NGLasso2-1.0000000000/relu-4": os.system("python NGLasso2.py --data %s --size %d --alp 1.0 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
    """


if __name__ == '__main__':
    p = Pool(20)
    p.map(task, range(20))