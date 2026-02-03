#nohup python bash.py &
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools

ACT,LAY,SIZE,CUDA = "relu",4,100,-1
DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]

def task():
    for DATA in DATAs:
        # ERM
        for SEED in range(20):
            subprocess.call("python RandomForest1.py --data %s --size %d --seed %d --cuda %d"%(DATA,SIZE,SEED,CUDA), shell=True)
            subprocess.call("python NGLasso1.py --data %s --size %d --alp 0. --lam 0 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NBridge1.py --data %s --size %d --alp 1. --lam 0 --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python LBridge1.py --data %s --size %d --alp 1. --lam 0 --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python ABridge1.py --data %s --size %d --alp 1. --lam 0 --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True)
        # Lasso
        for LAM,SEED in itertools.product([10.**(i/4.-5.) for i in range(33)],range(20):
            subprocess.call("python LBridge1.py --data %s --size %d --alp 1. --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA), shell=True)
        # SpAM
        for LAM,SEED in itertools.product([10.**(i/4.-5.) for i in range(33)],range(20):
            subprocess.call("python ABridge1.py --data %s --size %d --alp 1. --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA), shell=True)
        # NNSGLasso
        for ALP,LAM,SEED in itertools.product([0,0.25,0.5,0.75,1],[10.**(i/4.-5.) for i in range(33)],range(20)):
            subprocess.call("python NGLasso1.py --data %s --size %d --alp %s --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
        # MFBridge
        for ALP,LAM,SEED in itertools.product([0.125,0.25,0.5,1,2],[10.**(i/4.-5.) for i in range(33)],range(20)):
            subprocess.call("python NBridge1.py --data %s --size %d --alp %s --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True)
        # NNSGLasso with weight decay
        for ALP,LAM1,LAM2,SEED in itertools.product([0,0.25,0.5,0.75,1],[10.**(i/4.-5.) for i in range(33)],[10.**i for i in [-5,-4,-3,-2,-1,0,1,2,3]],range(20)):
            subprocess.call("python NGLasso12.py --data %s --size %d --alp %s --lam1 %s --lam2 %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM1,LAM2,ACT,LAY,SEED,CUDA), shell=True)
        # with explicit feature selection
        for ALP,SEED in itertools.product([0.9,0.95,0.99],range(20)):
            subprocess.call("python RandomForest2.py --data %s --size %d --alp %s --seed %d --cuda %d"%(DATA,SIZE,ALP,SEED,CUDA), shell=True)
            subprocess.call("python NGLasso2.py --data %s --size %d --alp %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NBridge2.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python LBridge2.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python ABridge2.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            #
            subprocess.call("python RandomForest3.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NGLasso3.py --data %s --size %d --alp %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NBridge3.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python LBridge3.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python ABridge3.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
        for ALP,SEED in itertools.product([0.25,0.5,0.75],range(20)):
            subprocess.call("python RandomForest4.py --data %s --size %d --alp %s --seed %d --cuda %d"%(DATA,SIZE,ALP,SEED,CUDA), shell=True)
            subprocess.call("python NGLasso4.py --data %s --size %d --alp %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NBridge4.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python LBridge4.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python ABridge4.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            #
            subprocess.call("python RandomForest5.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NGLasso5.py --data %s --size %d --alp %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python NBridge5.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python LBridge5.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)
            subprocess.call("python ABridge5.py --data %s --size %d --alp %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,ACT,LAY,SEED,CUDA), shell=True)

if __name__ == '__main__':
    task()
