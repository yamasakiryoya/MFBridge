#nohup python bash-1.py &
import os
import itertools
from multiprocessing import Pool

ACT,LAY,CUDA = "relu",4,-1
DATAs = ["dis"]#["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
SIZEs = [100]#[25,50,100,200,400]

def task(T):
    """(random forest)
    for SEED,DATA,SIZE in itertools.product(range(20),DATAs,SIZEs):
        os.system("python RandomForest.py --data %s --size %d --seed %d --cuda %d"%(DATA,SIZE,SEED,CUDA))
    """
    #"""(default)
    for SEED,DATA,SIZE,LAM in itertools.product(range(20),DATAs,SIZEs,[10.**(i/4.-5.) for i in [T]]):
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.25 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.5 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 0.75 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #os.system("python NGLasso1.py --data %s --size %d --alp 1.0 --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        #
        os.system("python NBridge1.py --data %s --size %d --alp 0.125 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        os.system("python NBridge1.py --data %s --size %d --alp 0.25 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        os.system("python NBridge1.py --data %s --size %d --alp 0.5 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        os.system("python NBridge1.py --data %s --size %d --alp 1.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
        os.system("python NBridge1.py --data %s --size %d --alp 2.0 --lam %s --Mnum 10 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,LAM,ACT,LAY,SEED,CUDA))
    #"""
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
    """
    p = Pool(33)
    p.map(task, range(33))
    """
    #"""
    for i in range(11):
        p = Pool(3)
        p.map(task, range(3*i,3*(i+1)))
    #"""
    """
    task(0)
    """