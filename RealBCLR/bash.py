#nohup python bash.py &
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools

ACT,LAY,SIZE,CUDA = "relu",4,100,-1
DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]

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
