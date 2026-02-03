import os
import itertools
import numpy as np
from scipy import stats

SEED  = 20
DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]

resM_MCEE  = np.zeros((len(DATAs),SEED,5)); resM_MCR  = np.zeros((len(DATAs),SEED,5))
resMB_MCEE = np.zeros((len(DATAs),SEED));   resMB_MCR = np.zeros((len(DATAs),SEED))
resN_MCEE  = np.zeros((len(DATAs),SEED,5)); resN_MCR  = np.zeros((len(DATAs),SEED,5))
resNB_MCEE = np.zeros((len(DATAs),SEED));   resNB_MCR = np.zeros((len(DATAs),SEED))
resL_MCEE  = np.zeros((len(DATAs),SEED));   resL_MCR  = np.zeros((len(DATAs),SEED))
resA_MCEE  = np.zeros((len(DATAs),SEED));   resA_MCR  = np.zeros((len(DATAs),SEED))
resR_MCEE  = np.zeros((len(DATAs),SEED));   resR_MCR  = np.zeros((len(DATAs),SEED))
resE_MCEE  = np.zeros((len(DATAs),SEED));   resE_MCR  = np.zeros((len(DATAs),SEED))

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
            # MFBridge with q
            resM_MCEE[d,se,q] = tmpM_MCEE_te[se,q,np.argmin(tmpM_MCEE_va[se,q])]
            resM_MCR[d,se,q]  = tmpM_MCR_te[se,q,np.argmin(tmpM_MCR_va[se,q])]
        # MFBridge
        idx = np.unravel_index(np.argmin(tmpM_MCEE_va[se]), tmpM_MCEE_va[se].shape)
        resMB_MCEE[d,se] = tmpM_MCEE_te[se,idx[0],idx[1]]
        idx = np.unravel_index(np.argmin(tmpM_MCR_va[se]), tmpM_MCR_va[se].shape)
        resMB_MCR[d,se]  = tmpM_MCR_te[se,idx[0],idx[1]]
    #
    Qs = [0,0.25,0.5,0.75,1]
    tmpN_MCEE_va = np.zeros((SEED,len(Qs),len(LAMs2))); tmpN_MCR_va = np.zeros((SEED,len(Qs),len(LAMs2)))
    tmpN_MCEE_te = np.zeros((SEED,len(Qs),len(LAMs2))); tmpN_MCR_te = np.zeros((SEED,len(Qs),len(LAMs2)))
    for se in range(SEED):
        for q in range(len(Qs)):
            for la in range(len(LAMs2)):
                tmp = np.loadtxt("./Results/NGLasso1-%.10f/relu-4/%s/size-100/lam-%.10f/1-seed-%d.csv"%(Qs[q],DATA,LAMs2[la],se), delimiter=",")
                tmpN_MCEE_va[se,q,la] = np.min(tmp[:,-4])
                tmpN_MCEE_te[se,q,la] = tmp[np.argmin(tmp[:,-4]),-2]
                tmpN_MCR_va[se,q,la]  = np.min(tmp[:,-3])
                tmpN_MCR_te[se,q,la]  = tmp[np.argmin(tmp[:,-3]),-1]
            # NNSGLasso with q
            resN_MCEE[d,se,q] = tmpN_MCEE_te[se,q,np.argmin(tmpN_MCEE_va[se,q])]
            resN_MCR[d,se,q]  = tmpN_MCR_te[se,q,np.argmin(tmpN_MCR_va[se,q])]
        # NNSGLasso
        idx = np.unravel_index(np.argmin(tmpN_MCEE_va[se]), tmpN_MCEE_va[se].shape)
        resNB_MCEE[d,se] = tmpN_MCEE_te[se,idx[0],idx[1]]
        idx = np.unravel_index(np.argmin(tmpN_MCR_va[se]), tmpN_MCR_va[se].shape)
        resNB_MCR[d,se]  = tmpN_MCR_te[se,idx[0],idx[1]]
    #
    tmpL_MCEE_va = np.zeros((SEED,len(LAMs2))); tmpL_MCR_va = np.zeros((SEED,len(LAMs2)))
    tmpL_MCEE_te = np.zeros((SEED,len(LAMs2))); tmpL_MCR_te = np.zeros((SEED,len(LAMs2)))
    for se in range(SEED):
        for la in range(len(LAMs2)):
            tmp = np.loadtxt("./Results/LBridge1-%.10f/%s/size-100/lam-%.10f/1-seed-%d.csv"%(1,DATA,LAMs2[la],se), delimiter=",")
            tmpL_MCEE_va[se,la] = np.min(tmp[:,-4])
            tmpL_MCEE_te[se,la] = tmp[np.argmin(tmp[:,-4]),-2]
            tmpL_MCR_va[se,la]  = np.min(tmp[:,-3])
            tmpL_MCR_te[se,la]  = tmp[np.argmin(tmp[:,-3]),-1]
        # Lasso
        resL_MCEE[d,se] = tmpL_MCEE_te[se,np.argmin(tmpL_MCEE_va[se])]
        resL_MCR[d,se]  = tmpL_MCR_te[se,np.argmin(tmpL_MCR_va[se])]
    #
    tmpA_MCEE_va = np.zeros((SEED,len(LAMs2))); tmpA_MCR_va = np.zeros((SEED,len(LAMs2)))
    tmpA_MCEE_te = np.zeros((SEED,len(LAMs2))); tmpA_MCR_te = np.zeros((SEED,len(LAMs2)))
    for se in range(SEED):
        for la in range(len(LAMs2)):
            tmp = np.loadtxt("./Results/ABridge1-%.10f/relu-4/%s/size-100/lam-%.10f/1-seed-%d.csv"%(1,DATA,LAMs2[la],se), delimiter=",")
            tmpA_MCEE_va[se,la] = np.min(tmp[:,-4])
            tmpA_MCEE_te[se,la] = tmp[np.argmin(tmp[:,-4]),-2]
            tmpA_MCR_va[se,la]  = np.min(tmp[:,-3])
            tmpA_MCR_te[se,la]  = tmp[np.argmin(tmp[:,-3]),-1]
        # SpAM
        resA_MCEE[d,se] = tmpA_MCEE_te[se,np.argmin(tmpA_MCEE_va[se])]
        resA_MCR[d,se]  = tmpA_MCR_te[se,np.argmin(tmpA_MCR_va[se])]
    # Random forest
    for se in range(SEED):
        tmp = np.loadtxt("./Results/RandomForest1/%s/size-100/1-seed-%d.csv"%(DATA,se), delimiter=",")
        resR_MCEE[d,se] = tmp[-2]
        resR_MCR[d,se]  = tmp[-1]
    # ERM
    for se in range(SEED):
        tmp = np.loadtxt("./Results/NGLasso1-%.10f/relu-4/%s/size-100/lam-%.10f/1-seed-%d.csv"%(0,DATA,0,se), delimiter=",")
        resE_MCEE[d,se] = tmp[np.argmin(tmp[:,-4]),-2]
        resE_MCR[d,se]  = tmp[np.argmin(tmp[:,-3]),-1]

#
def test(p):
    print("MCEE",p)
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
        test = stats.mannwhitneyu(resMB_MCEE[d], resL_MCEE[d], alternative='less')
        if   test.pvalue<=p:   tmp[0]+=1
        elif test.pvalue>=1-p: tmp[2]+=1
        else:                  tmp[1]+=1
    print(tmp)
    #
    tmp = np.zeros(3)
    for d in range(len(DATAs)):
        test = stats.mannwhitneyu(resMB_MCEE[d], resA_MCEE[d], alternative='less')
        if   test.pvalue<=p:   tmp[0]+=1
        elif test.pvalue>=1-p: tmp[2]+=1
        else:                  tmp[1]+=1
    print(tmp)
    #
    tmp = np.zeros(3)
    for d in range(len(DATAs)):
        test = stats.mannwhitneyu(resMB_MCEE[d], resR_MCEE[d], alternative='less')
        if   test.pvalue<=p:   tmp[0]+=1
        elif test.pvalue>=1-p: tmp[2]+=1
        else:                  tmp[1]+=1
    print(tmp)
    #
    print("MCR",p)
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
        test = stats.mannwhitneyu(resMB_MCR[d], resL_MCR[d], alternative='less')
        if   test.pvalue<=p:   tmp[0]+=1
        elif test.pvalue>=1-p: tmp[2]+=1
        else:                  tmp[1]+=1
    print(tmp)
    #
    tmp = np.zeros(3)
    for d in range(len(DATAs)):
        test = stats.mannwhitneyu(resMB_MCR[d], resA_MCR[d], alternative='less')
        if   test.pvalue<=p:   tmp[0]+=1
        elif test.pvalue>=1-p: tmp[2]+=1
        else:                  tmp[1]+=1
    print(tmp)
    #
    tmp = np.zeros(3)
    for d in range(len(DATAs)):
        test = stats.mannwhitneyu(resMB_MCR[d], resR_MCR[d], alternative='less')
        if   test.pvalue<=p:   tmp[0]+=1
        elif test.pvalue>=1-p: tmp[2]+=1
        else:                  tmp[1]+=1
    print(tmp)

# test(0.1)
test(0.05)
# test(0.1/len(DATAs))
# test(0.05/len(DATAs))