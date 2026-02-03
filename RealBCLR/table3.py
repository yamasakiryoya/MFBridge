import os
import itertools
import numpy as np
from scipy import stats

SEED  = 20
DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]

resM_MCEE  = np.zeros((len(DATAs),SEED)); resM_MCR  = np.zeros((len(DATAs),SEED))
resMB_MCEE = np.zeros((len(DATAs),SEED));   resMB_MCR = np.zeros((len(DATAs),SEED))
resNB_MCEE = np.zeros((len(DATAs),SEED));   resNB_MCR = np.zeros((len(DATAs),SEED))
resL1_MCEE  = np.zeros((len(DATAs),SEED));   resL1_MCR  = np.zeros((len(DATAs),SEED))
resA1_MCEE  = np.zeros((len(DATAs),SEED));   resA1_MCR  = np.zeros((len(DATAs),SEED))
resR1_MCEE  = np.zeros((len(DATAs),SEED));   resR1_MCR  = np.zeros((len(DATAs),SEED))
resL2_MCEE  = np.zeros((len(DATAs),SEED));   resL2_MCR  = np.zeros((len(DATAs),SEED))
resA2_MCEE  = np.zeros((len(DATAs),SEED));   resA2_MCR  = np.zeros((len(DATAs),SEED))
resR2_MCEE  = np.zeros((len(DATAs),SEED));   resR2_MCR  = np.zeros((len(DATAs),SEED))
resE_MCEE  = np.zeros((len(DATAs),SEED));   resE_MCR  = np.zeros((len(DATAs),SEED))

#vs N-
for ALP in [0.25,0.5,0.75]:
    print(ALP)
    for d in range(len(DATAs)):
        DATA = DATAs[d]
        #
        Qs = [0.125,0.25,0.5,1,2]
        tmpM_MCEE_va = np.zeros((SEED,len(Qs),len(LAMs2))); tmpM_MCR_va = np.zeros((SEED,len(Qs),len(LAMs2)))
        tmpM_MCEE_te = np.zeros((SEED,len(Qs),len(LAMs2))); tmpM_MCR_te = np.zeros((SEED,len(Qs),len(LAMs2)))
        for se in range(SEED):
            for q in range(len(Qs)):
                for la in range(len(LAMs2)):
                    tmp = np.loadtxt("./Results/NBridge1-%.10f/relu-4-10/%s/size-100/lam-%.10f/1-seed-%d.csv"%(Qs[q],DATA,LAMs2[la],se), delimiter=",")
                    tmpM_MCEE_va[se,q,la] = np.min(tmp[:,-4])
                    tmpM_MCEE_te[se,q,la] = tmp[np.argmin(tmp[:,-4]),-2]
                    tmpM_MCR_va[se,q,la]  = np.min(tmp[:,-3])
                    tmpM_MCR_te[se,q,la]  = tmp[np.argmin(tmp[:,-3]),-1]
            # MFBridge
            idx = np.unravel_index(np.argmin(tmpM_MCEE_va[se]), tmpM_MCEE_va[se].shape)
            resM_MCEE[d,se] = tmpM_MCEE_te[se,idx[0],idx[1]]
            idx = np.unravel_index(np.argmin(tmpM_MCR_va[se]), tmpM_MCR_va[se].shape)
            resM_MCR[d,se]  = tmpM_MCR_te[se,idx[0],idx[1]]
        # MFBridge (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/NBridge4-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resMB_MCEE[d,se]  = tmp[np.argmin(tmp[:,-4]),-2]
            resMB_MCR[d,se]   = tmp[np.argmin(tmp[:,-3]),-1]
        # ERM (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/NBridge5-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resE_MCEE[d,se] = tmp[np.argmin(tmp[:,-4]),-2]
            resE_MCR[d,se]  = tmp[np.argmin(tmp[:,-3]),-1]
        # NNSGLasso (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/NGLasso4-%.10f/relu-4/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resNB_MCEE[d,se]  = tmp[np.argmin(tmp[:,-4]),-2]
            resNB_MCR[d,se]   = tmp[np.argmin(tmp[:,-3]),-1]
        # Lasso () with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/LBridge4-%.10f/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resL1_MCEE[d,se]  = tmp[np.argmin(tmp[:,-4]),-2]
            resL1_MCR[d,se]   = tmp[np.argmin(tmp[:,-3]),-1]
        # Lasso (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/LBridge5-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resL2_MCEE[d,se]  = tmp[np.argmin(tmp[:,-4]),-2]
            resL2_MCR[d,se]   = tmp[np.argmin(tmp[:,-3]),-1]
        # SpAM () with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/ABridge4-%.10f/relu-4/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resA1_MCEE[d,se]  = tmp[np.argmin(tmp[:,-4]),-2]
            resA1_MCR[d,se]   = tmp[np.argmin(tmp[:,-3]),-1]
        # SpAM (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/ABridge5-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resA2_MCEE[d,se]  = tmp[np.argmin(tmp[:,-4]),-2]
            resA2_MCR[d,se]   = tmp[np.argmin(tmp[:,-3]),-1]
        # Random forest () with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest4-%.10f/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resR1_MCEE[d,se] = tmp[-2]
            resR1_MCR[d,se]  = tmp[-1]
        # Random forest (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest5-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resR2_MCEE[d,se] = tmp[np.argmin(tmp[:,-4]),-2]
            resR2_MCR[d,se]  = tmp[np.argmin(tmp[:,-3]),-1]

    #
    def test(p):
        print("MCEE",p)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resM_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resE_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resNB_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resL1_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resL2_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resA1_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resA2_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resR1_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCEE[d], resR2_MCEE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        print("MCR",p)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resM_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resE_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resNB_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resL1_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resL2_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resA1_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resA2_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resR1_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MCR[d], resR2_MCR[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)

    # test(0.1)
    test(0.05)
    # test(0.1/len(DATAs))
    # test(0.05/len(DATAs))