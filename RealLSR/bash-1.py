#nohup python bash-1.py &
import os
import itertools
from multiprocessing import Pool

ACT,LAY,CUDA = "relu",4,-1
DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
SIZEs = [100]#[25,50,100,200,400]

def task(T):
    for SEED,DATA,SIZE in itertools.product(range(20),DATAs,SIZEs):
        os.system("python RandomForest.py --data %s --size %d --seed %d --cuda %d"%(DATA,SIZE,SEED,CUDA))
    """(default)
    for SEED,DATA,SIZE,LAM in itertools.product(range(20),DATAs,SIZEs,[10.**(i/4.-5.) for i in [T]]):
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.25 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.5 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.75 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 1.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #
        #os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
    """
    """(varying Mnum)
    for SEED,DATA,SIZE,LAM in itertools.product(range(20),DATAs,SIZEs,[10.**(i/4.-5.) for i in [T]]):
        os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 5 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #
        os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
    """
    """(NNSGLasso weight-decay)
    for SEED,DATA,SIZE,LAM1,LAM2 in itertools.product(range(20),DATAs,SIZEs,[10.**(i/4.-5.) for i in [T]],[10.**i for i in [-5,-4,-3,-2,-1,0,1]]):
        #os.system("python NGLasso2.py --data %s --size %d --alp 0.0 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --data %s --size %d --alp 0.25 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --data %s --size %d --alp 0.5 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --data %s --size %d --alp 0.75 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso2.py --data %s --size %d --alp 1.0 --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM1,LAM2,ACT,LAY,SEED,CUDA))
    """

if __name__ == '__main__':
    task(0)
    """
    p = Pool(25)
    p.map(task, range(25))
    """