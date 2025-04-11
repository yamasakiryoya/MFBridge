#python opt-hypara.py
import os
import itertools
import numpy as np

LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]


def func(DATAs,METHs,SIZEs):
    for METH,DATA,SIZE in itertools.product(METHs,DATAs,SIZEs):
        # load results
        result1 = np.zeros((len(LAMs2),20))
        result2 = np.zeros((len(LAMs2),20))
        for se,la in itertools.product(range(20),range(len(LAMs2))):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,DATA,SIZE,LAMs2[la],se), delimiter=",")
            result1[la,se] = np.min(tmp[:,-4])
            result2[la,se] = np.argmin(tmp[:,-4])
        #set hypara
        print(METH,DATA,SIZE,np.mean(result2[np.argmin(np.mean(result1, axis=1)),:]),LAMs2[np.argmin(np.mean(result1, axis=1))])

#
DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma's_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
DIMs  = [10]
func(DATAs,METHs,SIZEs)
